"""Multiple prefill/decode workers with placement fixed at KV admission."""

from collections import deque

from .multi_prefill_coordinator import MultiPrefillCoordinator


class _MultiDecodeScheduling:
    """Shared driver policy; subclasses supply local or Ray worker calls."""

    def _init_decode(self):
        self.decode_ready = [deque() for _ in self.targets]
        # Lifetime placement history; a request is never migrated or reassigned.
        self.decode_assignments: dict[str, int] = {}
        self.next_target = 0
        # Successful imports are tracked before source release can fail.
        self.pending_releases: dict[str, int] = {}

    def _release_sources(self):
        first_error = None
        for request_id, source_index in list(self.pending_releases.items()):
            source = self.sources[source_index]
            try:
                if request_id in self._info(source)["active_request_ids"]:
                    self._call(source, "release", request_id)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            else:
                del self.pending_releases[request_id]
        if first_error is not None:
            raise first_error

    def _decode_targets(self):
        first_error = None
        for index, request_id, result, error in self._decode_outcomes():
            if error is not None:
                self._discard_failed_decode(index, request_id, error)
                if first_error is None:
                    first_error = error
            else:
                self.decode_ready[index].remove(request_id)
                if result["status"] == "FINISHED":
                    self.completed.append(result)
                else:
                    self.decode_ready[index].append(request_id)
                self._record_result(result)
        if first_error is not None:
            raise first_error

    def _discard_failed_decode(self, index, request_id, error):
        try:
            info = self._info(self.targets[index])
        except Exception:
            return  # Keep ownership if an actor's outcome is unknown.
        if request_id not in info["active_request_ids"]:
            self.decode_ready[index].remove(request_id)
            self._record_retired(request_id, error)

    def _pull_targets(self):
        first_error = None
        start = self.next_target
        self.next_target = (start + 1) % len(self.targets)
        for offset in range(len(self.targets)):
            target_index = (start + offset) % len(self.targets)
            target = self.targets[target_index]
            source_start = self.next_pull
            for source_offset in range(len(self.sources)):
                index = (source_start + source_offset) % len(self.sources)
                # Never re-import held KV after destination acknowledgement.
                if index in self.pending_releases.values():
                    continue
                source = self.sources[index]
                try:
                    descriptor = self._call(source, "peek_ready")
                    if descriptor is None or not self._call(
                        target, "can_admit", descriptor["num_prompt_tokens"], descriptor["max_new_tokens"],
                    ):
                        continue
                    request_id = descriptor["request_id"]
                    if request_id in self.decode_assignments:
                        raise RuntimeError("Held request already has a decode owner")
                    payload = self._export(source, request_id)
                    result = self._call(target, "import_request", payload, reserve_full=True)
                    self.decode_assignments[request_id] = target_index
                    self.decode_ready[target_index].append(result["request_id"])
                    self.pending_releases[request_id] = index
                    self.next_pull = (index + 1) % len(self.sources)
                    self._release_sources()
                except Exception as exc:
                    if first_error is None:
                        first_error = exc
                    # Explicit caller retry on the next tick; try other targets.
                    break
                else:
                    break  # At most one admission per target per tick.
        if first_error is not None:
            raise first_error

    def step(self):
        self._ensure_all_open()
        self._release_sources()
        self._decode_targets()
        self._pull_targets()
        self._prefill_all()
        return self.drain_completed()

    def has_pending(self):
        return bool(any(self.decode_ready) or self.pending_releases
                    or any(self.waiting) or self._has_held())


class MultiDecodeCoordinator(_MultiDecodeScheduling, MultiPrefillCoordinator):
    """Ordinary workers: round-robin sources and capacity-aware decode pulls.

    Each target owns a round-robin active queue. Every submission must fit each
    target individually. Worker models must use compatible immutable weights,
    as required by the existing KV handoff contract.
    """

    def __init__(self, sources, targets, *, metrics=None, decode_batch_size=1):
        if type(decode_batch_size) is not int or decode_batch_size <= 0:
            raise ValueError("decode_batch_size must be a positive integer")
        self.decode_batch_size = decode_batch_size
        self.targets = list(targets)
        sources = list(sources)
        if not self.targets:
            raise ValueError("At least one decode worker is required")
        workers = [*sources, *self.targets]
        if len({id(w) for w in workers}) != len(workers):
            raise ValueError("All workers must be distinct")
        from .workers import DecodeWorker, PrefillWorker
        if any(not isinstance(w, PrefillWorker) for w in sources) or any(
            not isinstance(w, DecodeWorker) for w in self.targets
        ):
            raise ValueError("Workers must match their prefill/decode roles")
        for w in workers:
            w._ensure_open()
            if w.requests or w.runner.caches or w.runner.block_manager.request_blocks:
                raise ValueError("Coordinator requires fresh workers")
        if len({id(w.runner.storage) for w in workers}) != len(workers):
            raise ValueError("Workers must have independent KV storage")
        super().__init__(sources, self.targets[0], metrics=metrics)
        self._init_decode()

    def _submission_targets(self):
        return self.targets

    def _ensure_open(self):
        self._ensure_all_open()

    def _ensure_all_open(self):
        for worker in (*self.sources, *self.targets):
            worker._ensure_open()

    @staticmethod
    def _call(worker, method, *args, **kwargs):
        return getattr(worker, method)(*args, **kwargs)

    _dispatch = _call

    @staticmethod
    def _resolve(result):
        return result

    @staticmethod
    def _info(worker):
        return worker.info()

    def _export(self, source, request_id):
        return source.export_request(request_id)

    def _decode_outcomes(self):
        # Yield immediately after each local call, before the next worker runs.
        for index, target in enumerate(self.targets):
            if not self.decode_ready[index]:
                continue
            ids = list(self.decode_ready[index])[:self.decode_batch_size]
            try:
                results = ([target.step(ids[0])] if self.decode_batch_size == 1
                           else target.step_batch(ids))
            except Exception as exc:
                for request_id in ids:
                    yield index, request_id, None, exc
            else:
                for request_id, result in zip(ids, results):
                    yield index, request_id, result, None

    def _prefill_all(self):
        first_error = None
        for index in range(len(self.sources)):
            try:
                result = self._prefill_one(index)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            else:
                if result is not None:
                    self.completed.append(result)
        if first_error is not None:
            raise first_error

    def _has_held(self):
        return any(len(source.prefill_pool) for source in self.sources)
