"""隔离转换子进程；限制地址空间、CPU、文件、墙钟和事件输出。"""

import asyncio
from base64 import b64decode
from hashlib import sha256
import json
import os
import re
import signal
import sys
from time import monotonic
from uuid import uuid4

from loguru import logger

from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode, require
from knowledge_algorithms.contracts.wire import encode
from knowledge_algorithms.parsing.wire import event_from_wire
from .model_transcription import (
    IMAGE_DIMENSIONS_ERROR, MAX_REQUEST_LINE, OPTIONAL_BUDGET_ERROR,
    ImageDimensionsUnsupported, TranscriptionBudgetExhausted,
)


async def _stderr_tail(stream):
    tail = b""
    while block := await stream.read(4096):
        tail = (tail + block)[-4096:]
    return tail


def _exit_error_code(returncode, stderr):
    if returncode in {-signal.SIGXCPU, -signal.SIGXFSZ} or (
        b"OpenBLAS error: Memory allocation still failed" in stderr
    ):
        return ErrorCode.RESOURCE_LIMIT
    return ErrorCode.CHANNEL_FAILED


def _safe_label(value):
    return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_]{1,64}", value) else "UNKNOWN"


class OptionalTranscriptionBudget:
    """可选区域不得耗尽任务的整形、物化与事件交付时间。"""

    def __init__(self, *, timeout_seconds, deadline):
        self.deadline = deadline - min(15.0, timeout_seconds * 0.15)
        self.remaining = timeout_seconds * 0.20

    async def run(self, transcribe):
        clock = asyncio.get_running_loop().time
        started = clock()
        available = min(self.remaining, self.deadline - started)
        if available <= 0:
            raise TranscriptionBudgetExhausted()
        window = asyncio.timeout_at(started + available)
        try:
            async with window:
                return await transcribe()
        except TimeoutError:
            if window.expired():
                raise TranscriptionBudgetExhausted() from None
            raise
        finally:
            self.remaining -= clock() - started


class BoundedParseExecutor:
    def __init__(self, *, memory_bytes=536870912, cpu_seconds=120, max_file_bytes=67108864,
                 max_artifact_bytes=134217728, converter_artifacts_path=None, ocr=None, model_transcription=None):
        require(memory_bytes >= 67108864 and cpu_seconds > 0 and max_file_bytes > 0, "执行器资源预算无效")
        self.memory_bytes, self.cpu_seconds, self.max_file_bytes = memory_bytes, cpu_seconds, max_file_bytes
        require(max_artifact_bytes > 0, "制品总预算无效")
        self.max_artifact_bytes, self.converter_artifacts_path = max_artifact_bytes, converter_artifacts_path
        self.ocr = ocr
        self.model_transcription = model_transcription

    async def events(self, job, source, directory):
        enhanced = job.profile.ocr_model is not None
        require(not enhanced or (self.model_transcription is not None and getattr(source, "context", None) is not None),
                "当前执行器没有受认证的模型转写端口")
        from .document_parsing import DOCUMENT_TYPES, IMAGE_TYPES
        require(job.source.mime_type in {"text/plain"} | set(DOCUMENT_TYPES) | IMAGE_TYPES,
                "该执行器未登记此格式适配器")
        process = stderr_task = None
        try:
            async with asyncio.timeout(job.profile.timeout_seconds) as deadline:
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
                event_limit = job.profile.max_event_bytes * 2 + 8192
                line_limit = max(event_limit, MAX_REQUEST_LINE) if enhanced else event_limit
                # 在运行库导入前限线程；只约束解析子进程，不修改服务环境。
                child_env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                             "MKL_NUM_THREADS": "1", "DOCLING_NUM_THREADS": "1", "OPENCV_FOR_THREADS_NUM": "1",
                             "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "TRANSFORMERS_OFFLINE": "1"}
                process = await asyncio.create_subprocess_exec(sys.executable, "-m", "knowledge_core.workers.compute_parser",
                    str(directory), stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, start_new_session=True, limit=line_limit, env=child_env)
                stderr_task = asyncio.create_task(_stderr_tail(process.stderr))
                payload = json.dumps({"job": encode(job), "memory_bytes": self.memory_bytes,
                    "cpu_seconds": self.cpu_seconds, "max_file_bytes": self.max_file_bytes,
                    "max_artifact_bytes": self.max_artifact_bytes,
                    "converter_artifacts_path": self.converter_artifacts_path, "ocr": self.ocr}).encode()
                require(len(payload) <= 1048576, "子进程启动合同超过预算")
                process.stdin.write(payload + b"\n")
                await process.stdin.drain()
                if not enhanced:
                    process.stdin.close()
                remaining = min(job.profile.max_objects, job.profile.max_pages * 4 + 32)
                optional_budget = OptionalTranscriptionBudget(timeout_seconds=job.profile.timeout_seconds,
                                                              deadline=deadline.when())
                while line := await process.stdout.readline():
                    require(len(line) <= line_limit, "子进程事件超过预算")
                    value = json.loads(line)
                    if isinstance(value, dict) and set(value) == {"diagnostic"}:
                        diagnostic = value["diagnostic"]
                        require(isinstance(diagnostic, list) and len(diagnostic) == 3, "解析诊断合同无效")
                        logger.warning("解析编排失败：job_id={} run_id={} stage={} code={} type={}",
                            job.run.job_ref, job.run.parse_run_ref, *map(_safe_label, diagnostic))
                        continue
                    if isinstance(value, dict) and set(value) == {"model_request"}:
                        require(enhanced, "该任务没有冻结转写模型")
                        request = value["model_request"]
                        optional = self._optional_request(job.profile, request)
                        try:
                            if optional and remaining <= 0:
                                raise TranscriptionBudgetExhausted()
                            require(remaining > 0, "模型转写次数超过任务预算")
                            remaining -= 1
                            if optional:
                                text = await optional_budget.run(lambda: self._transcribe(job, source.context, request))
                            else:
                                text = await self._transcribe(job, source.context, request)
                            reply = {"text": text}
                        except ImageDimensionsUnsupported:
                            reply = {"error_code": IMAGE_DIMENSIONS_ERROR}
                        except TranscriptionBudgetExhausted:
                            logger.info("可选图片补识别预算耗尽，保留原图和已完成内容：job_id={} run_id={}",
                                        job.run.job_ref, job.run.parse_run_ref)
                            reply = {"error_code": OPTIONAL_BUDGET_ERROR}
                        process.stdin.write(json.dumps(reply, ensure_ascii=False).encode() + b"\n")
                        await process.stdin.drain()
                        continue
                    require(len(line) <= event_limit, "子进程事件超过预算")
                    yield event_from_wire(value)
                returncode = await process.wait()
                stderr = await stderr_task
                if returncode != 0:
                    code = _exit_error_code(returncode, stderr)
                    # 错误输出可能含有输入内容，不写入运行日志或失败事件。
                    logger.warning("解析子进程异常退出：job_id={} run_id={} exit_code={} code={}",
                                   job.run.job_ref, job.run.parse_run_ref, returncode, code.value)
                    raise AlgorithmError(code, "解析子进程异常退出", stage="executor")
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
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
                await asyncio.gather(stderr_task, return_exceptions=True)

    @staticmethod
    def _optional_request(profile, request):
        require(isinstance(request, dict) and set(request) == {"role", "region", "image", "location"},
                "解析模型子进程协议无效", stage="model_transcription")
        location = request["location"]
        require(isinstance(location, dict) and set(location) == {"object", "page", "origin", "width", "height", "optional"}
                and type(location["optional"]) is bool, "解析模型区域诊断无效")
        optional = location["optional"]
        require(not optional or (request["region"] == "figure" and location["origin"] == "CONVERTER_FIGURE"
                                and profile.allow_partial), "原图或整页转写不得使用可选预算")
        return optional

    async def _transcribe(self, job, context, request):
        role, region = request["role"], request["region"]
        require(role in {"ocr", "vlm"} and region in {"page", "figure"}, "解析模型区域无效")
        location = request["location"]
        image = request["image"]
        require(isinstance(image, str) and len(image) >= 32, "解析模型图像诊断无效")
        # 只读 PNG 头的前 24 字节，记录缩放后的实际尺寸，不打开或记录整图。
        header = b64decode(image[:32], validate=True)
        require(header[:16] == b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR", "解析模型图像头无效")
        request_id = str(uuid4())
        key = location["object"]
        metadata = dict(job_id=job.run.job_ref, run_id=job.run.parse_run_ref, request_id=request_id,
                        role=role, region=region, origin=_safe_label(location["origin"]), optional=location["optional"],
                        node=key if isinstance(key, str) and re.fullmatch(r"[a-f0-9]{64}", key) else None,
                        page=location["page"] if type(location["page"]) is int else None,
                        original=tuple(location[axis] if type(location[axis]) is int else None
                                       for axis in ("width", "height")),
                        input_size=(int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")))
        started = monotonic()
        logger.info("开始解析模型转写：{}", metadata)
        try:
            return await self.model_transcription(job.profile, context.model_copy(update={"request_id": request_id}),
                {field: request[field] for field in ("role", "region", "image")})
        except ImageDimensionsUnsupported:
            logger.info("视觉模型不接受区域图像尺寸，保留原图和已识别文字：{} code={}", metadata, IMAGE_DIMENSIONS_ERROR)
            raise
        except (Exception, asyncio.CancelledError) as exc:
            level = "INFO" if location["optional"] and isinstance(exc, asyncio.CancelledError) else "WARNING"
            logger.log(level, "解析模型转写未完成：{} code={} type={} elapsed_ms={}", metadata,
                exc.code.value if isinstance(exc, AlgorithmError) else "MODEL_CALL_ABORTED",
                _safe_label(type(exc).__name__), round((monotonic() - started) * 1000))
            raise
