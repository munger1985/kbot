"""解析计算用例：有界持久事件、单代执行及可恢复确认。"""

import asyncio
import os
import time
from hashlib import sha256

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.parsing.events import ParseEvent, ParseEventType, ParseStreamValidator


class ParseComputationService:
    def __init__(self, store, executor, *, clock=time.time, max_jobs=2, max_ttl_seconds=86400,
                 max_unacked_events=32):
        require(max_jobs > 0 and max_ttl_seconds > 0, "计算并发或期限预算无效")
        self.store, self.executor, self.clock = store, executor, clock
        self.max_jobs, self.max_ttl_seconds = max_jobs, max_ttl_seconds
        require(max_unacked_events > 0, "未确认事件窗口无效")
        self.max_unacked_events = max_unacked_events
        self._tasks = {}
        self._failures = {}
        self._submit_lock = asyncio.Lock()

    def _validator(self, job):
        validator = ParseStreamValidator(job.run, max_events=job.profile.max_objects * 4 + 4,
            max_objects=job.profile.max_objects, max_payload_bytes=job.profile.max_event_bytes)
        while True:
            page = self.store.events(job.access, job.computation_ref, after_sequence=validator.event_count,
                                     limit=100, now=self.clock())
            if not page:
                return validator
            for event in page:
                validator.accept(event)

    async def submit(self, job, source):
        require(job.profile.max_input_bytes <= 67108864 and job.profile.max_text_bytes <= 67108864
                and job.profile.max_objects <= 100000 and job.profile.max_event_bytes <= 1048576
                and job.profile.timeout_seconds <= 600, "解析 Profile 超过服务硬上限")
        require(self.clock() < job.expires_at <= self.clock() + self.max_ttl_seconds, "计算期限超出服务预算")
        require(source.descriptor == job.source, "受控源与任务不一致")
        async with self._submit_lock:
            meta = self.store.prepare(job)
            task = self._tasks.get(job.computation_ref)
            if task is not None:
                return job.computation_ref
            if meta["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                return job.computation_ref
            require(len(self._tasks) < self.max_jobs, "解析计算并发已达到上限")
            execution_fd = self.store.acquire_execution(job)
            if execution_fd is None:
                # 其他 API 进程仍持有执行权；不能把其活跃任务当作重启中断。
                return job.computation_ref
            try:
                meta = self.store.recover(job)
                if meta["status"] in {"COMPLETED", "FAILED", "CANCELLED"}:
                    self.store.release_execution(execution_fd)
                    return job.computation_ref
                if meta["status"] == "RUNNING":
                    self._fail(job, self._validator(job), ErrorCode.CHANNEL_FAILED, "executor_interrupted")
                    self.store.release_execution(execution_fd)
                    return job.computation_ref
                self.store.start(job)
            except BaseException:
                self.store.release_execution(execution_fd)
                raise
            task = asyncio.create_task(self._execute(job, source))
            self._tasks[job.computation_ref] = task
            def finished(done):
                self.store.release_execution(execution_fd)
                self._tasks.pop(job.computation_ref, None)
                if not done.cancelled() and done.exception() is not None:
                    if len(self._failures) >= 1000:
                        self._failures.pop(next(iter(self._failures)))
                    self._failures[job.computation_ref] = True
            task.add_done_callback(finished)
            return job.computation_ref

    def _fail(self, job, validator, code, stage):
        _, _, meta = self.store.load(job.access, job.computation_ref, self.clock(), allow_expired=True)
        if meta["status"] in {"CANCELLED", "COMPLETED", "FAILED"}:
            return
        if not validator.event_count:
            event = ParseEvent.create(run=job.run, sequence=1, batch_key="batch-1",
                event_type=ParseEventType.RUN_STARTED,
                payload={"profile_hash": job.profile.profile_hash, "parser_revision": "executor/v1"})
            validator.accept(event)
            self.store.append(job, event)
        sequence = validator.event_count + 1
        event = ParseEvent.create(run=job.run, sequence=sequence, batch_key=f"batch-{sequence}",
            event_type=ParseEventType.RUN_FAILED, payload={"error_code": code.value, "stage": stage})
        validator.accept(event)
        self.store.append(job, event)

    async def _execute(self, job, source):
        try:
            validator = self._validator(job)
            async with asyncio.timeout(job.profile.timeout_seconds):
                async for event in self.executor.events(job, source, self.store.directory(job.access, job.computation_ref)):
                    self.store.load(job.access, job.computation_ref, self.clock())
                    validator.accept(event)
                    self.store.append(job, event)
                    # 磁盘暂存不能替代背压；窗口满后不继续读取子进程 stdout。
                    while event.event_type not in {ParseEventType.RUN_COMPLETED, ParseEventType.RUN_FAILED}:
                        _, _, checkpoint = self.store.load(job.access, job.computation_ref, self.clock())
                        if checkpoint["status"] == "CANCELLED":
                            return
                        if checkpoint["produced_sequence"] - checkpoint["acked_sequence"] < self.max_unacked_events:
                            break
                        await asyncio.sleep(0.02)
                _, _, meta = self.store.load(job.access, job.computation_ref, self.clock(), allow_expired=True)
                if meta["status"] not in {"COMPLETED", "FAILED", "CANCELLED"}:
                    self._fail(job, validator, ErrorCode.MANIFEST_INVALID, "executor_incomplete")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            code = exc.code if isinstance(exc, AlgorithmError) else (
                ErrorCode.TIMEOUT if isinstance(exc, TimeoutError) else ErrorCode.CHANNEL_FAILED)
            self.store.recover(job)
            self._fail(job, self._validator(job), code, "executor")

    def events(self, access, computation_ref, *, after_sequence, limit):
        self.store.load(access, computation_ref, self.clock())
        if computation_ref in self._failures:
            raise AlgorithmError(ErrorCode.CHANNEL_FAILED, "计算事件暂存不可用", stage="compute")
        return self.store.events(access, computation_ref, after_sequence=after_sequence, limit=limit, now=self.clock())

    def acknowledge(self, access, computation_ref, *, sequence, event_digest):
        self.store.acknowledge(access, computation_ref, sequence=sequence, event_digest=event_digest, now=self.clock())

    async def cancel(self, access, computation_ref):
        self.store.cancel(access, computation_ref, self.clock())
        task = self._tasks.get(computation_ref)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def artifact(self, access, computation_ref, artifact_ref, *, max_bytes):
        directory, job, _ = self.store.load(access, computation_ref, self.clock())
        found, after = None, 0
        while True:
            page = self.events(access, computation_ref, after_sequence=after, limit=100)
            if not page:
                break
            for event in page:
                for artifact in event.artifact_handles:
                    if artifact.handle_ref == artifact_ref:
                        require(found in {None, artifact}, "制品句柄绑定冲突")
                        found = artifact
            after = page[-1].sequence
        require(found is not None, "制品不属于当前计算任务")
        if found.expires_at <= self.clock():
            raise AlgorithmError(ErrorCode.ARTIFACT_EXPIRED, "计算制品已过期", stage="artifact")
        require(0 < found.byte_count <= max_bytes, "制品大小超过服务预算")
        # 文件名只由验证过的摘要生成；句柄永远不拼接为路径。
        fd = os.open(directory / ("artifact-" + found.sha256), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            require(os.fstat(fd).st_size == found.byte_count, "制品大小失效")
            digest, count = sha256(), 0
            while block := os.read(fd, 65536):
                self.store.load(access, computation_ref, self.clock())
                if found.expires_at <= self.clock():
                    raise AlgorithmError(ErrorCode.ARTIFACT_EXPIRED, "计算制品已过期", stage="artifact")
                count += len(block)
                require(count <= found.byte_count, "制品流超过声明大小")
                digest.update(block)
                yield block
                await asyncio.sleep(0)
            require(count == found.byte_count and digest.hexdigest() == found.sha256, "制品摘要不一致")
        finally:
            os.close(fd)

    async def close(self):
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
