"""Lease-based KC Parser Worker runtime."""

import asyncio
from contextlib import suppress
import hashlib
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse

from loguru import logger

from knowledge_core.parsing import canonical_json_hash
from knowledge_core.parsing.converter import KcDoclingConverter
from knowledge_core.parsing.pipeline import KcParsingPipeline
from .client import KcParseClient, KcParserProtocolError, ParseTask


_MIME_SUFFIX = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "text/markdown": ".md",
    "text/plain": ".md",
    "text/csv": ".csv",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
}


class KcParserWorker:
    def __init__(
        self, *, client: KcParseClient, converter: KcDoclingConverter,
        pipeline: KcParsingPipeline, worker_id: str, lease_seconds: int,
        poll_interval: float, evidence_batch_size: int, visual_enricher=None,
    ):
        self._client = client
        self._converter = converter
        self._pipeline = pipeline
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._poll_interval = poll_interval
        self._evidence_batch_size = evidence_batch_size
        self._visual_enricher = visual_enricher
        self._stop = asyncio.Event()
        self._run_task: asyncio.Task | None = None

    async def start(self) -> None:
        if self._run_task is not None:
            return
        await self._client.__aenter__()
        self._run_task = asyncio.create_task(self._run(), name="kc-parser-worker")

    async def stop(self) -> None:
        self._stop.set()
        if self._run_task is not None:
            self._run_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._run_task
            self._run_task = None
        await self._client.__aexit__(None, None, None)

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                tasks = await self._client.claim(
                    worker_id=self._worker_id, lease_seconds=self._lease_seconds,
                )
                if not tasks:
                    await asyncio.sleep(self._poll_interval)
                    continue
                await self._process(tasks[0])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("KC Parser Worker loop failed")
                await asyncio.sleep(self._poll_interval)

    async def _process(self, task: ParseTask) -> None:
        heartbeat = asyncio.create_task(self._heartbeat(task))
        try:
            with tempfile.TemporaryDirectory(prefix=f"kbot-kc-parse-{task.job_id}-") as directory:
                source_path = await self._materialize_source(task, Path(directory))
                policy = task.policy_snapshot
                document = await self._converter.convert(
                    source_path=source_path,
                    do_ocr=bool(policy.get("do_ocr", True)),
                    ocr_engine=str(policy.get("ocr_engine", "tesseract")),
                    image_scale=float(policy.get("image_scale", 2.0)),
                )
                if self._visual_enricher is not None:
                    await self._visual_enricher.enrich(
                        document, model_name=policy.get("vlm_model"),
                        prompt=str(policy.get("visual_description_prompt", "")),
                    )
                output = self._pipeline.parse(
                    document_version_id=task.document_version_id,
                    parse_view_id=task.parse_view_id,
                    document=document,
                )
                artifact_manifest = {}
                for name, payload in output.artifacts.items():
                    artifact_manifest[name] = await self._client.upload_artifact(
                        task, name=name, payload=payload,
                        sha256=canonical_json_hash(payload),
                        schema=output.artifact_schemas[name],
                        generator=f"kc-parser/{self._pipeline.parser_version}",
                    )
                if not output.quality_report.passed:
                    await self._client.fail(
                        task, failure_class="POLICY", failure_code="QUALITY_REJECTED",
                        failure_message="; ".join(output.quality_report.hard_failures),
                        artifact_manifest=artifact_manifest,
                    )
                    return
                evidence_dicts = [evidence.as_dict() for evidence in output.evidences]
                for offset in range(0, len(evidence_dicts), self._evidence_batch_size):
                    await self._client.submit_evidence(
                        task, evidence_dicts[offset:offset + self._evidence_batch_size],
                    )
                await self._client.complete(
                    task, artifact_manifest=artifact_manifest,
                    output_fingerprint=output.output_fingerprint,
                    quality_report=output.quality_report.as_dict(), quality_score=1.0,
                )
                logger.info("KC 解析任务 {} 已完成，共生成 {} 条 Evidence", task.job_id, len(evidence_dicts))
        except KcParserProtocolError as exc:
            if exc.code in {"JOB_LEASE_INVALID", "JOB_STALE"}:
                logger.warning("KC 解析任务 {} 的租约已失效", task.job_id)
                return
            await self._safe_fail(task, "TRANSIENT", exc.code, str(exc))
        except (ValueError, FileNotFoundError) as exc:
            await self._safe_fail(task, "PERMANENT", "PARSE_INPUT_INVALID", str(exc))
        except Exception as exc:
            logger.exception("KC parse job {} failed", task.job_id)
            await self._safe_fail(task, "TRANSIENT", "PARSER_UNEXPECTED", str(exc))
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat

    async def _heartbeat(self, task: ParseTask) -> None:
        while True:
            await asyncio.sleep(max(10, self._lease_seconds / 3))
            await self._client.heartbeat(task, lease_seconds=self._lease_seconds)

    async def _materialize_source(self, task: ParseTask, directory: Path) -> Path:
        suffix = _MIME_SUFFIX.get(task.detected_mime_type.lower())
        if suffix is None:
            raise ValueError(f"unsupported MIME type: {task.detected_mime_type}")
        target = directory / f"source{suffix}"
        parsed = urlparse(task.source_read_url)
        if parsed.scheme in {"http", "https"}:
            target.write_bytes(await self._client.download_source(task))
        else:
            source = Path(task.source_read_url)
            if not source.is_file():
                raise FileNotFoundError(task.source_read_url)
            await asyncio.to_thread(shutil.copyfile, source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != task.input_fingerprint:
            raise ValueError("source content hash does not match claimed Document Version")
        return target

    async def _safe_fail(
        self, task: ParseTask, failure_class: str, failure_code: str, message: str,
    ) -> None:
        with suppress(Exception):
            await self._client.fail(
                task, failure_class=failure_class,
                failure_code=failure_code, failure_message=message,
            )
