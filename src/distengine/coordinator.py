"""Multi-prefill/multi-decode scheduling, with one operation per worker.

Prefill, decode and transfers progress across actors without global barriers.
Actors are exclusively owned by this coordinator until drained or closed by
its caller. Ordinary workers use the same scheduler synchronously for CPU
verification; Ray actors execute independently. Cached capacity is authoritative
only while that actor is idle.
"""

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import math
import time
from typing import Any, Iterable, cast

import ray

from .metric import MetricsRecorder, _MetricsHooks
from .worker import PrefillWorker, DecodeWorker, scheduled_state, run_scheduled, import_scheduled


OperationRef = Future[dict[str, Any]] | ray.ObjectRef[dict[str, Any]]


@dataclass(frozen=True)
class Submission:
    request_id: str
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int


class Coordinator(_MetricsHooks):
    """Own lists of prefill/decode replicas using one shared scheduling policy.

    Pass ordinary workers for synchronous CPU checks, or Ray actor handles for
    independent execution. Both paths support batching and optional stealing.
    """

    def __init__(self, sources: Iterable[Any], targets: Iterable[Any], *,
                 metrics: MetricsRecorder | None = None, decode_batch_size=1,
                 poll_timeout_s=.002, stealing_enabled=False):
        if type(stealing_enabled) is not bool:
            raise ValueError("stealing_enabled must be a boolean")
        self.stealing_enabled = stealing_enabled
        self.steal_events: list[dict[str, Any]] = []
        self.prefill_assignments: dict[str, int] = {}
        self._prefill_started: set[str] = set()
        if not math.isfinite(poll_timeout_s) or poll_timeout_s <= 0:
            raise ValueError("poll_timeout_s must be finite and positive")
        if type(decode_batch_size) is not int or decode_batch_size <= 0:
            raise ValueError("decode_batch_size must be a positive integer")
        self.sources, self.targets = list(sources), list(targets)
        if not self.sources or not self.targets:
            raise ValueError("At least one prefill and one decode worker are required")
        workers = self.sources + self.targets
        local = [isinstance(w, (PrefillWorker, DecodeWorker)) for w in workers]
        if any(local) and not all(local):
            raise ValueError("Do not mix ordinary workers and Ray actors")
        self._local = all(local)
        self._states: list[dict[str, Any]]
        if self._local:
            if (any(not isinstance(w, PrefillWorker) for w in self.sources)
                    or any(not isinstance(w, DecodeWorker) for w in self.targets)):
                raise ValueError("Workers must match their prefill/decode roles")
            if len({id(w) for w in workers}) != len(workers):
                raise ValueError("All workers must be distinct")
            if len({id(w.runner.storage) for w in workers}) != len(workers):
                raise ValueError("Workers must have independent KV storage")
            if len({id(w.prefill_pool) for w in self.sources}) != len(self.sources):
                raise ValueError("Prefill workers must have independent holding pools")
            self._states = [scheduled_state(w) for w in workers]
        else:
            state_refs: list[ray.ObjectRef[dict[str, Any]]] = [w.scheduled_state.remote() for w in workers]
            identity_refs: list[ray.ObjectRef[dict[str, Any]]] = [w.ready.remote() for w in workers]
            self._states = ray.get(state_refs)
            identities = ray.get(identity_refs)
            if len({w["actor_id"] for w in identities}) != len(workers):
                raise ValueError("All actors must be distinct")
        if any(w["closed"] for w in self._states):
            raise RuntimeError("Worker is closed")
        if any(w["active_request_ids"] or w["num_free_blocks"] != w["num_blocks"]
               or w["held_requests"] for w in self._states):
            raise ValueError("Coordinator requires fresh, drained workers")
        if (any(w["role"] != "PrefillWorker" for w in self._states[:len(self.sources)])
                or any(w["role"] != "DecodeWorker" for w in self._states[len(self.sources):])):
            raise ValueError("Workers must match their prefill/decode roles")
        self.metrics = metrics
        self.decode_batch_size = decode_batch_size
        self.poll_timeout_s = poll_timeout_s
        self.waiting: list[deque[Submission]] = [deque() for _ in self.sources]
        self.decode_ready: list[deque[str]] = [deque() for _ in self.targets]
        self.completed: deque[dict[str, Any]] = deque()
        self.seen_ids: set[str] = set()
        self.decode_assignments: dict[str, int] = {}
        self.pending_releases: dict[str, int] = {}
        self.next_submit = self.next_pull = self.next_target = 0
        self._observed_at = [time.perf_counter()] * len(self._states)
        self._busy: dict[int, OperationRef] = {}  # global worker index -> operation ref
        self._operations: dict[OperationRef, tuple[int, str, Any]] = {}  # ref -> (worker index, operation, context)
        self._transfers: dict[str, tuple[int, int]] = {}  # request id -> (source index, target index)
        self._last_target_operation: list[str | None] = [None] * len(self.targets)
        self._fatal_error: Exception | None = None

    def _check_open(self):
        if self._fatal_error is not None:
            raise self._fatal_error
        if any(state["closed"] for state in self._states) or (
                self._local and any(w.closed for w in (*self.sources, *self.targets))):
            raise RuntimeError("Worker is closed")

    def drain_completed(self):
        results = list(self.completed)
        self.completed.clear()
        return results

    def submit(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        max_new_tokens: int,
        eos_token_id: int,
    ) -> None:
        """Validate and enqueue a copied submission without allocating KV."""
        self._check_open()
        metadata = (self._states[self.next_submit], *self._states[len(self.sources):])

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be a nonempty string")

        if request_id in self.seen_ids:
            raise ValueError("Request ID has already been submitted")

        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise ValueError("prompt_token_ids must be a nonempty list")

        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")

        if type(eos_token_id) is not int:
            raise ValueError("eos_token_id must be an integer")

        for token_id in prompt_token_ids:
            if type(token_id) is not int:
                raise ValueError("Token IDs must be integers")

            for info in metadata:
                if not 0 <= token_id < info["vocab_size"]:
                    raise ValueError("Token ID is outside the model vocabulary")

        maximum = len(prompt_token_ids) + max_new_tokens - 1
        for info in metadata:
            if maximum > info["num_blocks"] * info["block_size"]:
                raise ValueError("Request exceeds worker KV capacity")

        submission = Submission(
            request_id=request_id,
            prompt_token_ids=tuple(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
        )
        self._record_submission(request_id)
        self.waiting[self.next_submit].append(submission)
        self.seen_ids.add(request_id)
        self.next_submit = (self.next_submit + 1) % len(self.sources)

    def cached_workers(self):
        """Last completed-operation snapshots; sampling never waits on actors."""
        now = time.perf_counter()
        return [dict(state, snapshot_age_s=now - observed,
                     operation_in_flight=index in self._busy)
                for index, (state, observed) in enumerate(zip(self._states, self._observed_at))]

    def has_pending(self):
        return bool(self._operations or self._transfers or self.pending_releases
                    or any(self.waiting) or any(self.decode_ready)
                    or any(s.get("held_descriptors") for s in self._states[:len(self.sources)]))

    def _launch(self, index: int, operation: str, context: Any, *args: Any) -> OperationRef:
        if index in self._busy:
            raise RuntimeError("Worker already has an outstanding operation")
        worker = (*self.sources, *self.targets)[index]
        ref: OperationRef
        if self._local:
            ref = Future[dict[str, Any]]()
            try:
                envelope = (import_scheduled(worker, args[0].result()) if operation == "import"
                            else run_scheduled(worker, operation, *args))
                ref.set_result(envelope)
            except Exception as exc:
                ref.set_exception(exc)
        else:
            ref = (worker.scheduled_import.remote(*args) if operation == "import"
                   else worker.scheduled.remote(operation, *args))
        self._busy[index] = ref
        self._operations[ref] = (index, operation, context)
        return ref

    @staticmethod
    def _resolve_operation(ref: OperationRef) -> dict[str, Any]:
        """Resolve one scheduled envelope, rather than Ray's list overload."""
        if isinstance(ref, Future):
            return ref.result()
        return cast(dict[str, Any], ray.get(ref))

    def _reconcile(self, ready: Iterable[OperationRef]) -> None:
        errors: list[Exception] = []
        for ref in ready:
            index, operation, context = self._operations.pop(ref)
            del self._busy[index]
            # Export payloads remain in Ray; the destination consumes their refs.
            # Export does not change source ownership or capacity.
            if operation == "export_request":
                continue
            try:
                envelope = self._resolve_operation(ref)
            except Exception as exc:
                # Actor/task failure has an unknown outcome. Stop dispatching;
                # caller cleanup must close/kill all exclusively owned actors.
                self._fatal_error = exc
                errors.append(exc)
                continue
            self._states[index] = envelope["state"]
            self._observed_at[index] = time.perf_counter()
            error = envelope["error"]
            result = envelope["result"]
            if operation == "prefill":
                submission = self.waiting[index].popleft()
                if error is None:
                    self._record_result(result)
                    if result["status"] == "FINISHED":
                        self.completed.append(result)
                else:
                    self._record_retired(submission.request_id, error)
            elif operation in ("step", "step_batch"):
                target = index - len(self.sources)
                if error is None:
                    results = [result] if operation == "step" else result
                    for rid, item in zip(context, results):
                        self.decode_ready[target].remove(rid)
                        if item["status"] == "FINISHED":
                            self.completed.append(item)
                        else:
                            self.decode_ready[target].append(rid)
                        self._record_result(item)
                else:
                    for rid in context:
                        if rid not in envelope["state"]["active_request_ids"]:
                            self.decode_ready[target].remove(rid)
                            self._record_retired(rid, error)
            elif operation == "import":
                source, target = self._transfers.pop(context)
                if error is None:
                    self.decode_assignments[context] = target
                    self.decode_ready[target].append(context)
                    self.pending_releases[context] = source
            elif operation == "release":
                if error is None or context not in envelope["state"]["active_request_ids"]:
                    del self.pending_releases[context]
            if error is not None:
                errors.append(error)
        if errors:
            raise errors[0]

    def _can_start_prefill(self, source, request):
        state = self._states[source]
        size = state["block_size"]
        blocks = (len(request.prompt_token_ids) + size - 1) // size
        maximum = len(request.prompt_token_ids) + request.max_new_tokens - 1
        return (not state["closed"]
                and maximum <= state["num_blocks"] * size
                and all(0 <= token < state["vocab_size"] for token in request.prompt_token_ids)
                and state["held_requests"] < state["max_held_requests"]
                and blocks <= state["num_free_blocks"])

    def _steal_prefill(self, receiver):
        """Move one unstarted whole request to an idle, empty source queue.

        Prefer the donor with most queued prompt tokens. Steal from its tail,
        preserving the donor's FIFO head and excluding its in-flight request.
        No model state or KV exists for any eligible request.
        """
        if not self.stealing_enabled or receiver in self._busy or self.waiting[receiver]:
            return
        candidates = []
        for donor, queue in enumerate(self.waiting):
            if donor == receiver:
                continue
            unstarted = [r for r in queue if r.request_id not in self._prefill_started]
            eligible = next((r for r in reversed(unstarted)
                             if self._can_start_prefill(receiver, r)), None)
            if eligible is not None:
                candidates.append((sum(len(r.prompt_token_ids) for r in unstarted), donor, eligible))
        if not candidates:
            return
        _, donor, request = max(candidates, key=lambda item: (item[0], -item[1]))
        self.waiting[donor].remove(request)
        self.waiting[receiver].append(request)
        self.steal_events.append({"request_id": request.request_id, "from_source": donor,
                                  "to_source": receiver, "prompt_tokens": len(request.prompt_token_ids),
                                  "observed_s": self.metrics.elapsed() if self.metrics is not None else None})

    def _dispatch_available(self):
        count = len(self.sources)
        # Acknowledged imports release source KV before further prefills.
        for rid, source in list(self.pending_releases.items()):
            if source not in self._busy:
                self._launch(source, "release", rid, rid)

        start = self.next_target
        self.next_target = (start + 1) % len(self.targets)
        for offset in range(len(self.targets)):
            target = (start + offset) % len(self.targets)
            index = count + target
            if index in self._busy:
                continue
            queue = self.decode_ready[target]
            # At most one import between decode batches while requests are active.
            try_import = not queue or self._last_target_operation[target] != "import"
            selected: tuple[int, dict[str, Any]] | None = None
            if try_import:
                state = self._states[index]
                for delta in range(count):
                    source = (self.next_pull + delta) % count
                    if source in self._busy:
                        continue
                    held = self._states[source]["held_descriptors"]
                    candidate = next((d for d in held if d["request_id"] not in self._transfers
                                      and d["request_id"] not in self.pending_releases), None)
                    if candidate is None:
                        continue
                    tokens = candidate["num_prompt_tokens"] + candidate["max_new_tokens"] - 1
                    blocks = (tokens + state["block_size"] - 1) // state["block_size"]
                    if blocks <= state["num_free_blocks"]:
                        selected = (source, candidate)
                        break
            if selected is not None:
                source, descriptor = selected
                rid = descriptor["request_id"]
                self._transfers[rid] = (source, target)
                exported = self._launch(source, "export_request", rid, rid)
                self._launch(index, "import", rid, exported)
                self._last_target_operation[target] = "import"
                self.next_pull = (source + 1) % count
            elif queue:
                ids = list(queue)[:self.decode_batch_size]
                operation = "step" if self.decode_batch_size == 1 else "step_batch"
                self._launch(index, operation, ids, ids[0] if operation == "step" else ids)
                self._last_target_operation[target] = "decode"

        for source, waiting in enumerate(self.waiting):
            if source in self._busy:
                continue
            self._steal_prefill(source)
            if not waiting:
                continue
            request = waiting[0]
            if self._can_start_prefill(source, request):
                self._prefill_started.add(request.request_id)
                self.prefill_assignments[request.request_id] = source
                self._record_prefill_dispatch(request.request_id)
                self._launch(source, "prefill", request.request_id, request.request_id,
                             list(request.prompt_token_ids), request.max_new_tokens, request.eos_token_id)

    def _wait(self, count: int, timeout: float) -> tuple[list[OperationRef], list[OperationRef]]:
        refs = list(self._operations)
        if self._local:
            return refs[:count], refs[count:]
        actor_refs: list[ray.ObjectRef | ray.ObjectRefGenerator] = [
            cast(ray.ObjectRef[dict[str, Any]], ref) for ref in refs]
        ready, remaining = ray.wait(actor_refs, num_returns=count, timeout=timeout)
        # Only ObjectRefs were submitted; Ray's annotation also allows generators.
        return ([cast(OperationRef, ref) for ref in ready],
                [cast(OperationRef, ref) for ref in remaining])

    def step(self):
        self._check_open()
        if self._operations:
            ready, _ = self._wait(len(self._operations), 0)
            self._reconcile(ready)
        try:
            self._dispatch_available()
        except Exception as exc:
            self._fatal_error = exc
            raise
        if self._operations:
            ready, _ = self._wait(1, self.poll_timeout_s)
            self._reconcile(ready)
        elif self.has_pending():
            raise RuntimeError("Pending requests cannot make progress")
        return self.drain_completed()
