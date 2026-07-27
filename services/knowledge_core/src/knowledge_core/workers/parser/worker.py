"""Lease-based KC Parser Worker runtime."""

import asyncio
from contextlib import suppress
import hashlib
import base64
from io import BytesIO
from pathlib import Path
import shutil
import tempfile
from urllib.parse import urlparse
from uuid import UUID

from loguru import logger

from knowledge_core.parsing import canonical_json_hash
from knowledge_core.parsing.converter import KcDoclingConverter
from knowledge_core.parsing.pipeline import KcParsingPipeline
from knowledge_core.workers.job_wait import AdaptiveJobWait
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
        job_wait: AdaptiveJobWait,
        evidence_batch_size: int, visual_enricher=None,
        deepseek_ocr_enricher=None, model_config_client=None,
    ):
        self._client = client
        self._converter = converter
        self._pipeline = pipeline
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._job_wait = job_wait
        self._evidence_batch_size = evidence_batch_size
        self._visual_enricher = visual_enricher
        self._deepseek_ocr_enricher = deepseek_ocr_enricher
        self._model_config_client = model_config_client
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
        await self._job_wait.close()

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                tasks = await self._client.claim(
                    worker_id=self._worker_id, lease_seconds=self._lease_seconds,
                )
                if not tasks:
                    await self._job_wait.wait()
                    continue
                self._job_wait.reset()
                await self._process(tasks[0])
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("KC Parser Worker 循环执行失败")
                await self._job_wait.wait()

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
                ocr_enrichment = None
                if self._deepseek_ocr_enricher is not None:
                    ocr_enrichment = (
                        await self._deepseek_ocr_enricher.enrich(
                            document,
                            served_model_name=policy.get("ocr_model"),
                        )
                    )
                visual_enrichment = None
                if self._visual_enricher is not None:
                    vlm_model_name = await self._resolve_vlm_model(policy)
                    visual_enrichment = await self._visual_enricher.enrich(
                        document,
                        served_model_name=vlm_model_name,
                        prompt=str(policy.get("visual_description_prompt", "")),
                        full_page_prompt=str(
                            policy.get("full_page_visual_prompt", "")
                        ),
                        strategy=str(policy.get("parse_strategy", "AUTO")),
                        min_text_characters=int(
                            policy.get("visual_min_text_characters", 80)
                        ),
                        min_mean_confidence=float(
                            policy.get("visual_min_mean_confidence", 0.65)
                        ),
                        max_gibberish_ratio=float(
                            policy.get("visual_max_gibberish_ratio", 0.08)
                        ),
                        max_concurrency=int(
                            policy.get("visual_max_concurrency", 2)
                        ),
                    )
                output = self._pipeline.parse(
                    document_version_id=task.document_version_id,
                    parse_view_id=task.parse_view_id,
                    document=document,
                    ocr_enrichment=ocr_enrichment,
                    visual_enrichment=visual_enrichment,
                    visual_embedding_enabled=bool(
                        policy.get("models", {}).get("visual_embedding")
                    ),
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
                if policy.get("models", {}).get("visual_embedding"):
                    visual_assets = self._visual_assets(
                        document, visual_enrichment
                    )
                    for offset in range(
                        0, len(visual_assets), self._evidence_batch_size
                    ):
                        await self._client.submit_visual_assets(
                            task,
                            visual_assets[
                                offset : offset + self._evidence_batch_size
                            ],
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
            logger.exception("KC 解析任务 {} 执行失败", task.job_id)
            await self._safe_fail(task, "TRANSIENT", "PARSER_UNEXPECTED", str(exc))
        finally:
            heartbeat.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await heartbeat

    async def _resolve_vlm_model(self, policy: dict) -> str | None:
        """从 Collection 冻结的模型 ID 解析 VLM 调用名称。"""
        model_id = policy.get("models", {}).get("parser_vlm")
        if not model_id:
            return None
        if self._model_config_client is None:
            raise RuntimeError("KC Parser 未配置模型目录客户端")
        model = await self._model_config_client.get_model(UUID(str(model_id)))
        if int(model.get("category") or 0) != 5:
            raise RuntimeError("Collection models.parser_vlm 不是 VLM")
        if int(model.get("status") or 0) != 1:
            raise RuntimeError("Collection 绑定的 Parser VLM 不可用")
        served_name = str(model.get("served_model_name") or "").strip()
        if not served_name:
            raise RuntimeError("Collection Parser VLM 缺少 served_model_name")
        return served_name

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
        try:
            await self._client.fail(
                task, failure_class=failure_class,
                failure_code=failure_code, failure_message=message,
            )
        except Exception:
            logger.exception("KC 解析任务 {} 上报失败状态失败", task.job_id)

    @staticmethod
    def _visual_assets(document, enrichment) -> list[dict]:
        """导出整页和 Figure 原图；视觉向量由后续 INDEX 阶段生成。"""
        page_descriptions = {
            int(item.page_no): item.markdown
            for item in getattr(enrichment, "page_results", ())
        }
        assets: list[dict] = []

        def encode(image) -> tuple[str, str]:
            stream = BytesIO()
            image.convert("RGB").save(stream, format="PNG")
            payload = stream.getvalue()
            return (
                base64.b64encode(payload).decode("ascii"),
                hashlib.sha256(payload).hexdigest(),
            )

        for page_key, page in sorted(document.pages.items()):
            image = getattr(getattr(page, "image", None), "pil_image", None)
            if image is None:
                continue
            page_no = int(getattr(page, "page_no", page_key))
            content, digest = encode(image)
            assets.append(
                {
                    "asset_key": f"page:{page_no}:{digest[:24]}",
                    "asset_type": "PAGE",
                    "page_no": page_no,
                    "source_item_ref": None,
                    "bbox": None,
                    "mime_type": "image/png",
                    "content_base64": content,
                    "content_sha256": digest,
                    "description": page_descriptions.get(page_no),
                }
            )
        for picture in getattr(document, "pictures", ()):
            image = getattr(getattr(picture, "image", None), "pil_image", None)
            source_ref = str(getattr(picture, "self_ref", "") or "")
            provenance = list(getattr(picture, "prov", ()) or ())
            if image is None or not source_ref:
                continue
            content, digest = encode(image)
            first = provenance[0] if provenance else None
            bbox = None
            if first is not None and getattr(first, "bbox", None) is not None:
                raw_bbox = first.bbox
                bbox = {
                    "l": float(raw_bbox.l),
                    "t": float(raw_bbox.t),
                    "r": float(raw_bbox.r),
                    "b": float(raw_bbox.b),
                }
            descriptions = [
                str(getattr(item, "text", "") or "").strip()
                for item in getattr(picture, "annotations", ())
                if str(getattr(item, "text", "") or "").strip()
            ]
            assets.append(
                {
                    "asset_key": (
                        f"figure:{hashlib.sha256(source_ref.encode()).hexdigest()[:24]}"
                    ),
                    "asset_type": "FIGURE",
                    "page_no": (
                        int(first.page_no) if first is not None else None
                    ),
                    "source_item_ref": source_ref,
                    "bbox": bbox,
                    "mime_type": "image/png",
                    "content_base64": content,
                    "content_sha256": digest,
                    "description": "\n".join(descriptions) or None,
                }
            )
        return assets
