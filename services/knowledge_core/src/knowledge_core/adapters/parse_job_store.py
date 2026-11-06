"""按 App/运行隔离的有界计算事件暂存；不保存可检索正文表。"""

from contextlib import contextmanager
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import re

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.wire import decode, encode
from knowledge_algorithms.parsing.wire import event_from_wire, event_to_wire
from knowledge_core.domain.parse_jobs import ParseJob


class ParseJobStore:
    def __init__(self, root: Path, *, max_page_bytes=8388608, max_job_bytes=134217728):
        require(max_page_bytes > 0 and max_job_bytes > 0, "计算暂存预算无效")
        self.max_page_bytes, self.max_job_bytes = max_page_bytes, max_job_bytes
        self._sweep_after = {}
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    def directory(self, access, computation_ref):
        require(access.owner in {"knowledge_retrieval", "aiops"}
                and re.fullmatch(r"[0-9a-f]{64}", computation_ref), "计算任务身份无效")
        owner_root = self.root / access.owner
        owner_root.mkdir(exist_ok=True, mode=0o700)
        require(not owner_root.is_symlink(), "计算命名空间不能为符号链接")
        result = owner_root / computation_ref
        require(not result.is_symlink(), "计算任务目录不能为符号链接")
        return result

    @contextmanager
    def locked(self, directory):
        gate_fd = os.open(directory.parent / ".namespace.lock",
                          os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        fd = None
        try:
            fcntl.flock(gate_fd, fcntl.LOCK_SH)
            directory.mkdir(exist_ok=True, mode=0o700)
            fd = os.open(directory / "lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            if fd is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            fcntl.flock(gate_fd, fcntl.LOCK_UN)
            os.close(gate_fd)

    def write_json(self, path, value):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        temporary = path.parent / ("pending-" + os.urandom(16).hex())
        fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "wb") as writer:
                writer.write(encoded)
                writer.flush()
                os.fsync(writer.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def read_json(self, path, *, max_bytes):
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as reader:
            encoded = reader.read(max_bytes + 1)
        require(len(encoded) <= max_bytes, "计算暂存元数据超过预算")
        return json.loads(encoded)

    def prepare(self, job):
        directory = self.directory(job.access, job.computation_ref)
        with self.locked(directory):
            meta = directory / "task.json"
            if meta.exists():
                existing = self.read_json(meta, max_bytes=1048576)
                previous = decode(ParseJob, existing["job"])
                require(previous.computation_ref == job.computation_ref, "计算幂等身份冲突")
                existing["job"]["expires_at"] = max(previous.expires_at, job.expires_at)
                self.write_json(meta, existing)
                return existing
            value = {"job": encode(job), "status": "RECEIVED", "produced_sequence": 0, "acked_sequence": 0}
            self.write_json(meta, value)
            return value

    def load(self, access, computation_ref, now, *, allow_expired=False):
        directory = self.directory(access, computation_ref)
        value = self.read_json(directory / "task.json", max_bytes=1048576)
        job = decode(ParseJob, value["job"])
        require(job.access == access and job.computation_ref == computation_ref, "计算任务不属于当前主体或范围")
        if not allow_expired and job.expires_at <= now:
            raise AlgorithmError(ErrorCode.ARTIFACT_EXPIRED, "计算任务已过期", stage="compute")
        return directory, job, value

    def start(self, job):
        directory = self.directory(job.access, job.computation_ref)
        with self.locked(directory):
            meta = self.read_json(directory / "task.json", max_bytes=1048576)
            require(meta["status"] == "RECEIVED", "计算任务已经启动")
            meta["status"] = "RUNNING"
            self.write_json(directory / "task.json", meta)

    def acquire_execution(self, job):
        directory = self.directory(job.access, job.computation_ref)
        fd = os.open(directory / "executor.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            os.close(fd)
            return None

    def release_execution(self, fd):
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    def recover(self, job):
        directory = self.directory(job.access, job.computation_ref)
        with self.locked(directory):
            meta = self.read_json(directory / "task.json", max_bytes=1048576)
            while (directory / f"event-{meta['produced_sequence'] + 1}.json").exists():
                path = directory / f"event-{meta['produced_sequence'] + 1}.json"
                event = event_from_wire(self.read_json(path, max_bytes=job.profile.max_event_bytes * 2 + 8192))
                require(event.run == job.run and event.sequence == meta["produced_sequence"] + 1,
                        "事件恢复身份不一致")
                meta["produced_sequence"] = event.sequence
                meta["produced_bytes"] = meta.get("produced_bytes", 0) + path.stat().st_size
                if event.event_type.value in {"RUN_COMPLETED", "RUN_FAILED"} and meta["status"] != "CANCELLED":
                    meta["status"] = "COMPLETED" if event.event_type.value == "RUN_COMPLETED" else "FAILED"
            self.write_json(directory / "task.json", meta)
            return meta

    def events(self, access, computation_ref, *, after_sequence, limit, now):
        require(type(after_sequence) is int and after_sequence >= 0 and 1 <= limit <= 1000, "事件分页无效")
        directory, job, value = self.load(access, computation_ref, now)
        result, page_bytes = [], 0
        # 文件先持久、元数据后推进；崩溃留下的单个事件仍能被重放。
        for sequence in range(after_sequence + 1, after_sequence + limit + 1):
            path = directory / f"event-{sequence}.json"
            if not path.exists():
                break
            size = path.stat(follow_symlinks=False).st_size
            require(size <= self.max_page_bytes, "单事件超过分页字节预算")
            if page_bytes + size > self.max_page_bytes:
                break
            page_bytes += size
            result.append(event_from_wire(self.read_json(path, max_bytes=job.profile.max_event_bytes * 2 + 8192)))
        return tuple(result)

    def append(self, job, event):
        require(event.run == job.run, "计算事件代次不一致")
        require(all(re.fullmatch(r"[A-Za-z0-9_-]{1,256}", artifact.handle_ref)
                    for artifact in event.artifact_handles), "HTTP 制品句柄必须为受控不透明标识")
        directory = self.directory(job.access, job.computation_ref)
        with self.locked(directory):
            meta = self.read_json(directory / "task.json", max_bytes=1048576)
            require(meta["status"] not in {"CANCELLED", "COMPLETED", "FAILED"}, "计算任务已封闭")
            path = directory / f"event-{event.sequence}.json"
            wire = event_to_wire(event)
            if path.exists():
                require(self.read_json(path, max_bytes=job.profile.max_event_bytes * 2 + 8192) == wire, "同序号事件冲突")
            else:
                require(event.sequence == meta["produced_sequence"] + 1, "计算事件不连续")
                size = len(json.dumps(wire, ensure_ascii=False).encode())
                require(meta.get("produced_bytes", 0) + size <= self.max_job_bytes
                        or event.event_type.value == "RUN_FAILED", "计算事件暂存超过总预算")
                self.write_json(path, wire)
                meta["produced_bytes"] = meta.get("produced_bytes", 0) + size
            meta["produced_sequence"] = event.sequence
            if event.event_type.value in {"RUN_COMPLETED", "RUN_FAILED"}:
                meta["status"] = "COMPLETED" if event.event_type.value == "RUN_COMPLETED" else "FAILED"
            else:
                meta["status"] = "RUNNING"
            self.write_json(directory / "task.json", meta)

    def acknowledge(self, access, computation_ref, *, sequence, event_digest, now):
        directory, job, _ = self.load(access, computation_ref, now)
        with self.locked(directory):
            meta = self.read_json(directory / "task.json", max_bytes=1048576)
            require(meta["status"] != "CANCELLED" and type(sequence) is int and sequence > 0,
                    "确认超出已产出范围")
            # 恢复事件已 fsync、元数据尚未推进的崩溃窗口；不跳过缺失序号。
            while (directory / f"event-{meta['produced_sequence'] + 1}.json").exists():
                next_event = event_from_wire(self.read_json(
                    directory / f"event-{meta['produced_sequence'] + 1}.json",
                    max_bytes=job.profile.max_event_bytes * 2 + 8192))
                require(next_event.run == job.run and next_event.sequence == meta["produced_sequence"] + 1,
                        "暂存事件恢复身份不一致")
                meta["produced_sequence"] = next_event.sequence
                if next_event.event_type.value in {"RUN_COMPLETED", "RUN_FAILED"}:
                    meta["status"] = "COMPLETED" if next_event.event_type.value == "RUN_COMPLETED" else "FAILED"
            require(sequence <= meta["produced_sequence"], "确认超出已产出范围")
            event = event_from_wire(self.read_json(directory / f"event-{sequence}.json",
                max_bytes=job.profile.max_event_bytes * 2 + 8192))
            require(event.event_digest == event_digest, "确认摘要不一致")
            meta["acked_sequence"] = max(meta["acked_sequence"], sequence)
            self.write_json(directory / "task.json", meta)

    def cancel(self, access, computation_ref, now):
        directory, _, _ = self.load(access, computation_ref, now, allow_expired=True)
        with self.locked(directory):
            meta = self.read_json(directory / "task.json", max_bytes=1048576)
            meta["status"] = "CANCELLED"
            self.write_json(directory / "task.json", meta)
    def sweep_expired(self, now, *, limit=32, max_examined=256, grace_seconds=3600):
        """只清理已核对身份并过期的计算目录；执行中任务与未知目录均不触碰。"""
        import heapq
        import shutil
        require(limit > 0 and max_examined >= limit and grace_seconds >= 0, "计算回收预算无效")
        require(shutil.rmtree.avoids_symlink_attacks, "当前系统不支持安全目录回收")
        removed = 0
        for owner in ("aiops", "knowledge_retrieval"):
            owner_root = self.root / owner
            if not owner_root.is_dir() or owner_root.is_symlink():
                continue
            gate_fd = os.open(owner_root / ".namespace.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            try:
                try:
                    fcntl.flock(gate_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                after = self._sweep_after.get(owner, "")
                with os.scandir(owner_root) as entries:
                    names = heapq.nsmallest(max_examined, (entry.name for entry in entries
                        if re.fullmatch(r"[0-9a-f]{64}", entry.name) and entry.name > after
                        and entry.is_dir(follow_symlinks=False)))
                if not names:
                    self._sweep_after[owner] = ""
                    continue
                for name in names:
                    self._sweep_after[owner] = name
                    directory = owner_root / name
                    execution_fd = None
                    try:
                        meta = self.read_json(directory / "task.json", max_bytes=1048576)
                        job = decode(ParseJob, meta["job"])
                        if (job.access.owner != owner or job.computation_ref != name
                            or job.expires_at + grace_seconds > now):
                            continue
                        execution_fd = self.acquire_execution(job)
                        if execution_fd is None:
                            continue
                        # 元数据写入持有共享命名空间锁，独占锁下不能续期或重新提交。
                        if directory.is_symlink() or directory.parent != owner_root:
                            continue
                        shutil.rmtree(directory)
                        removed += 1
                        if removed >= limit:
                            return removed
                    except (OSError, KeyError, TypeError, ValueError, AlgorithmError):
                        # 无法证明归属或有效期时保留，不猜测删除。
                        continue
                    finally:
                        if execution_fd is not None:
                            self.release_execution(execution_fd)
            finally:
                fcntl.flock(gate_fd, fcntl.LOCK_UN)
                os.close(gate_fd)
        return removed
