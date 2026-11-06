"""纯计算子进程入口；不导入数据库、服务启动配置或模型凭据。"""

import asyncio
import json
import os
from pathlib import Path
import resource
import sys

from knowledge_algorithms.contracts.identity import AppBackendBinding
from knowledge_algorithms.contracts.wire import decode
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
from knowledge_algorithms.parsing.wire import event_to_wire
from knowledge_core.domain.parse_jobs import ParseJob
from knowledge_core.adapters.document_parsing import WorkerArtifactWriter, NativeImageAdapter, DoclingParsingAdapter
from knowledge_core.adapters.local_ocr import TesseractTranscriber


class LocalSource:
    def __init__(self, descriptor, directory):
        self.descriptor, self.directory = descriptor, directory

    async def chunks(self):
        fd = os.open(self.directory / "source.bin", os.O_RDONLY | os.O_NOFOLLOW)
        try:
            while block := os.read(fd, 65536):
                yield block
                await asyncio.sleep(0)
        finally:
            os.close(fd)


async def run(value, directory):
    job = decode(ParseJob, value["job"])
    for key, budget in ((resource.RLIMIT_AS, value["memory_bytes"]),
                       (resource.RLIMIT_CPU, value["cpu_seconds"]),
                       (resource.RLIMIT_FSIZE, value["max_file_bytes"])):
        resource.setrlimit(key, (budget, budget))
    writer = WorkerArtifactWriter(directory, expires_at=job.expires_at,
        max_bytes=value["max_file_bytes"], max_total_bytes=value["max_artifact_bytes"])
    ocr = value.get("ocr")
    transcriber = TesseractTranscriber(**ocr) if ocr is not None else None
    def artifacts(kind, payload):
        return tuple(writer.artifacts[item["local_key"]] for item in payload.get("objects", ())
                     if item.get("local_key") in writer.artifacts)
    engine = ParseEngine(AppBackendBinding(job.access.owner, job.access.owner),
        (PlainTextAdapter(), NativeImageAdapter(writer, transcriber=transcriber),
         DoclingParsingAdapter(writer, artifacts_path=value.get("converter_artifacts_path"), transcriber=transcriber)),
        artifact_provider=artifacts)
    async for event in engine.parse_events(LocalSource(job.source, directory), job.profile, job.run):
        sys.stdout.write(json.dumps(event_to_wire(event), ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    raw = sys.stdin.buffer.read(1048577)
    if len(raw) > 1048576:
        raise SystemExit(2)
    asyncio.run(run(json.loads(raw), Path(sys.argv[1])))
