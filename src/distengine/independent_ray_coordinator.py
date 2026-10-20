"""Event-driven Ray scheduling, with one outstanding operation per worker.

Prefill, decode and transfers progress across actors without global barriers.
Actors are exclusively owned by this coordinator until drained or closed by
its caller. Cached capacity is authoritative only while that actor is idle.
"""

import math
import time

import ray

from .multi_decode_ray_coordinator import MultiDecodeRayCoordinator


class IndependentRayCoordinator(MultiDecodeRayCoordinator):
    def __init__(self, sources, targets, *, metrics=None, decode_batch_size=1,
                 poll_timeout_s=.002, stealing_enabled=False):
        if type(stealing_enabled) is not bool:
            raise ValueError("stealing_enabled must be a boolean")
        self.stealing_enabled = stealing_enabled
        self.steal_events = []
        self.prefill_assignments = {}
        self._prefill_started = set()
        if not math.isfinite(poll_timeout_s) or poll_timeout_s <= 0:
            raise ValueError("poll_timeout_s must be finite and positive")
        super().__init__(sources, targets, metrics=metrics, decode_batch_size=decode_batch_size)
        self.poll_timeout_s = poll_timeout_s
        self._states = ray.get([w.scheduled_state.remote() for w in (*self.sources, *self.targets)])
        self._observed_at = [time.perf_counter()] * len(self._states)
        self._busy = {}  # global worker index -> operation ref
        self._operations = {}  # ref -> (worker index, operation, context)
        self._transfers = {}  # request id -> (source index, target index)
        self._last_target_operation = [None] * len(self.targets)
        self._fatal_error = None

    def _check_open(self):
        # Parent construction still performs the full fresh-actor validation.
        if not hasattr(self, "_states"):
            return super()._check_open()
        if self._fatal_error is not None:
            raise self._fatal_error
        if any(state["closed"] for state in self._states):
            raise RuntimeError("Worker is closed")
        return self._metadata

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

    def _launch(self, index, operation, context, *args):
        if index in self._busy:
            raise RuntimeError("Worker already has an outstanding operation")
        worker = (*self.sources, *self.targets)[index]
        ref = (worker.scheduled_import.remote(*args) if operation == "import"
               else worker.scheduled.remote(operation, *args))
        self._busy[index] = ref
        self._operations[ref] = (index, operation, context)
        return ref

    def _reconcile(self, ready):
        errors = []
        for ref in ready:
            index, operation, context = self._operations.pop(ref)
            del self._busy[index]
            # Export payloads remain in Ray; the destination consumes their refs.
            # Export does not change source ownership or capacity.
            if operation == "export_request":
                continue
            try:
                envelope = ray.get(ref)
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
            descriptor = None
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
                        descriptor = candidate
                        break
            if descriptor is not None:
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

    def step(self):
        self._check_open()
        if self._operations:
            ready, _ = ray.wait(list(self._operations), num_returns=len(self._operations), timeout=0)
            self._reconcile(ready)
        try:
            self._dispatch_available()
        except Exception as exc:
            self._fatal_error = exc
            raise
        if self._operations:
            ready, _ = ray.wait(list(self._operations), num_returns=1, timeout=self.poll_timeout_s)
            self._reconcile(ready)
        elif self.has_pending():
            raise RuntimeError("Pending requests cannot make progress")
        return self.drain_completed()
