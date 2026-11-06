"""隔离转换子进程；限制地址空间、CPU、文件、墙钟和事件输出。"""

import asyncio
from hashlib import sha256
import json
import os
import signal
import sys

from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.wire import event_from_wire


class BoundedParseExecutor:
    def __init__(self, *, memory_bytes=536870912, cpu_seconds=120, max_file_bytes=67108864,
                 max_artifact_bytes=134217728, converter_artifacts_path=None, ocr=None):
        require(memory_bytes >= 67108864 and cpu_seconds > 0 and max_file_bytes > 0, "执行器资源预算无效")
        self.memory_bytes, self.cpu_seconds, self.max_file_bytes = memory_bytes, cpu_seconds, max_file_bytes
        require(max_artifact_bytes > 0, "制品总预算无效")
        self.max_artifact_bytes, self.converter_artifacts_path = max_artifact_bytes, converter_artifacts_path
        self.ocr = ocr

    async def events(self, job, source, directory):
        from .document_parsing import DOCUMENT_TYPES, IMAGE_TYPES
        require(job.source.mime_type in {"text/plain"} | set(DOCUMENT_TYPES) | IMAGE_TYPES,
                "该执行器未登记此格式适配器")
        process = None
        try:
            async with asyncio.timeout(job.profile.timeout_seconds):
                fd = os.open(directory / "source.bin", os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
                digest, count = sha256(), 0
                iterator = source.chunks()
                try:
                    async for block in iterator:
                        require(type(block) is bytes and 0 < len(block) <= 65536, "源数据块超过执行器预算")
                        count += len(block)
                        require(count <= min(job.source.byte_count, job.profile.max_input_bytes, self.max_file_bytes),
                                "源大小超过执行器预算")
                        digest.update(block)
                        view = memoryview(block)
                        while view:
                            written = os.write(fd, view)
                            require(written > 0, "暂存源写入未取得进展")
                            view = view[written:]
                        await asyncio.sleep(0)
                    require(count == job.source.byte_count and digest.hexdigest() == job.source.sha256,
                            "受控源大小或摘要不匹配")
                    os.fsync(fd)
                finally:
                    os.close(fd)
                    close = getattr(iterator, "aclose", None)
                    if close:
                        await close()
                line_limit = job.profile.max_event_bytes * 2 + 8192
                process = await asyncio.create_subprocess_exec(sys.executable, "-m", "knowledge_core.workers.compute_parser",
                    str(directory), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL, start_new_session=True, limit=line_limit)
                payload = json.dumps({"job": encode(job), "memory_bytes": self.memory_bytes,
                    "cpu_seconds": self.cpu_seconds, "max_file_bytes": self.max_file_bytes,
                    "max_artifact_bytes": self.max_artifact_bytes,
                    "converter_artifacts_path": self.converter_artifacts_path, "ocr": self.ocr}).encode()
                require(len(payload) <= 1048576, "子进程启动合同超过预算")
                process.stdin.write(payload)
                await process.stdin.drain()
                process.stdin.close()
                while line := await process.stdout.readline():
                    require(len(line) <= line_limit, "子进程事件超过预算")
                    yield event_from_wire(json.loads(line))
                require(await process.wait() == 0, "解析子进程异常退出")
        finally:
            if process is not None and process.returncode is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(process.wait(), timeout=2)
                except TimeoutError:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
