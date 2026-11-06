"""仅扫描 App 提供的允许文本副本；固定二进制、无 shell、无递归目录参数。"""

import asyncio
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import time

from knowledge_algorithms.contracts.errors import AlgorithmError, require
from knowledge_algorithms.contracts.identity import SourceSpan
from knowledge_algorithms.ports.text_scan import TextMatch, TextScanBatch


class ScanLimit(Exception):
    pass


class RipgrepTextScanner:
    def __init__(self, *, binding, provider, executable, temporary_root):
        self.binding, self.provider = binding, provider
        self.executable = str(Path(executable).resolve(strict=True))
        require(Path(self.executable).is_file() and os.access(self.executable, os.X_OK),
                "扫描二进制不可执行")
        self.root = Path(temporary_root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        require(not self.root.is_symlink(), "扫描临时根不能是符号链接")

    async def scan(self, request, scope, snapshot):
        self.binding.check_owner(scope.owner_binding_ref)
        snapshot.validate_scope(self.binding, scope)
        for key in request.candidate_keys:
            self.binding.check_owner(key.owner_binding_ref)
        await self.provider.validate(scope, snapshot)
        matches, scanned, seen, consumed, total_output = [], [], set(), 0, 0
        reason, status = None, "COMPLETE"
        deadline = time.monotonic() + request.timeout_seconds
        iterator = self.provider.sources(scope, snapshot)
        try:
            with tempfile.TemporaryDirectory(prefix="scan-", dir=self.root) as temporary:
                async with asyncio.timeout(request.timeout_seconds):
                    async for source in iterator:
                        self.binding.check_owner(source.key.owner_binding_ref)
                        source.validate()
                        require(source.key not in seen, "授权文本来源重复")
                        seen.add(source.key)
                        if request.mode == "WITHIN_CANDIDATES" and source.key not in request.candidate_keys:
                            continue
                        raw = source.text.encode("utf-8")
                        if len(scanned) >= request.max_sources or consumed + len(raw) > request.max_bytes:
                            raise ScanLimit("SOURCE_BUDGET")
                        consumed += len(raw)
                        path = Path(temporary) / ("source-" + source.key.stable_key)
                        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o400)
                        with os.fdopen(fd, "wb") as output:
                            output.write(raw)
                        require(not path.is_symlink() and sha256(path.read_bytes()).hexdigest() == source.text_hash,
                                "授权文本副本发生改变")
                        output_limit = request.max_output_bytes - total_output
                        if output_limit <= 0:
                            raise ScanLimit("OUTPUT_BUDGET")
                        matched, used = await self._run(path, request, timeout=max(0.001, deadline - time.monotonic()),
                                                        output_limit=output_limit)
                        total_output += used
                        if total_output > request.max_output_bytes:
                            raise ScanLimit("OUTPUT_BUDGET")
                        scanned.append(source.key)
                        if not matched:
                            continue
                        import re
                        flags = 0 if request.case_sensitive else re.IGNORECASE
                        for index, pattern in enumerate(request.patterns):
                            # 在同一已授权 UTF-8 副本中核验每个模式，避免 rg 重叠模式丢覆盖。
                            for occurrence in re.finditer(re.escape(pattern), source.text, flags):
                                start, end = occurrence.span()
                                spans = [span for span in source.sources
                                         if span.unit_start <= start < end <= span.unit_end]
                                if len(spans) != 1:
                                    reason, status = "SOURCE_MAP_GAP", "PARTIAL"
                                    continue
                                if len(matches) >= request.max_matches:
                                    raise ScanLimit("MATCH_BUDGET")
                                span = spans[0]
                                source_start = span.source_start + start - span.unit_start
                                matches.append(TextMatch(source.key, index, len(source.text[:start].encode()),
                                    len(source.text[:end].encode()), span.source_key,
                                    SourceSpan(span.source_key.object_key, source_start, source_start + end - start)))
        except ScanLimit as exc:
            status, reason = "PARTIAL", str(exc)
        except TimeoutError:
            status, reason = "PARTIAL", "TIMEOUT"
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, AlgorithmError):
            status, reason = ("PARTIAL" if matches else "FAILED"), "SCAN_FAILED"
        finally:
            await iterator.aclose()
        await self.provider.validate(scope, snapshot)
        return TextScanBatch(tuple(matches), status, status == "COMPLETE", tuple(scanned), reason, request.mode)

    async def _run(self, path, request, *, timeout, output_limit):
        arguments = [self.executable, "--no-config", "--json", "--text", "--fixed-strings",
                     "--no-ignore", "--hidden", "--line-buffered"]
        if not request.case_sensitive:
            arguments.append("--ignore-case")
        for pattern in request.patterns:
            arguments.extend(("-e", pattern))
        arguments.extend(("--", str(path)))
        process = await asyncio.create_subprocess_exec(*arguments, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=min(output_limit + 1, 1048576))
        found, output_bytes = False, 0
        async def stderr():
            size = 0
            while block := await process.stderr.read(4096):
                size += len(block)
                if size > 16384:
                    raise ScanLimit("STDERR_BUDGET")
        error_task = asyncio.create_task(stderr())
        try:
            async with asyncio.timeout(timeout):
                while line := await process.stdout.readline():
                    output_bytes += len(line)
                    if output_bytes > output_limit:
                        raise ScanLimit("OUTPUT_BUDGET")
                    value = json.loads(line)
                    require(value.get("type") in {"begin", "end", "match", "context", "summary"}, "扫描输出协议无效")
                    if value["type"] == "match":
                        found = True
                code = await process.wait()
                await error_task
                require(code in {0, 1} and (code == 0) == found, "扫描退出码或结果不一致")
                return found, output_bytes
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 0.5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
            if not error_task.done():
                error_task.cancel()
            await asyncio.gather(error_task, return_exceptions=True)
