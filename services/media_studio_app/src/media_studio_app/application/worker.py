"""多媒体创作工作台生成 Worker。"""

from __future__ import annotations

import asyncio
import hashlib
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from loguru import logger

from media_studio_app.application.runtime import (
    IMAGE_KIND,
    TERMINAL_STATUSES,
    already_attempted,
    append_event,
    mark_upstream_attempted,
)
from media_studio_app.entities import MediaStudioMediaAssetEntity
from platform_clients.generative import GenerativeResponsesClientError
from platform_core.contracts import ImageGenerationRequest
from platform_core.identity import uuid7


MIME_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
_IMAGE_FAIL_MESSAGES = {
    "CONTENT_REJECTED": "内容安全策略拒绝生成",
    "PROVIDER_UNSUPPORTED_TOOL": "上游拒绝图片生成工具",
    "PROVIDER_QUOTA_EXHAUSTED": "上游配额已耗尽",
    "PROVIDER_TIMEOUT": "上游图片生成超时",
    "PROVIDER_UNAVAILABLE": "上游图片服务不可用",
}


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    run_id: UUID
    domain_id: int
    kind: str
    status: str
    actor_id: str
    model_id: UUID
    request_json: dict[str, Any]
    result_json: dict[str, Any] | None
    provider_request_id: str | None
    lease_token: UUID
    prompt_revision_id: UUID | None
    trace_id: str


class MediaStudioRunWorker:
    def __init__(self, *, uow_factory, generative_client, object_store, poll_seconds: float = 5, lease_seconds: int = 360, worker_id: str | None = None):
        self._uow_factory = uow_factory
        self._generative = generative_client
        self._store = object_store
        self._poll_seconds = poll_seconds
        self._lease_seconds = lease_seconds
        self._worker_id = worker_id or f"{socket.gethostname()}:{uuid7()}"

    async def run_forever(self) -> None:
        logger.info("多媒体创作 Worker 开始运行：{}", self._worker_id)
        while True:
            handled = await self.process_once()
            if not handled:
                await asyncio.sleep(self._poll_seconds)

    async def process_once(self) -> bool:
        snapshot = await self._claim()
        if snapshot is None:
            return False
        try:
            await self._process(snapshot)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("多媒体生成 Run 处理失败：run_id={}", snapshot.run_id)
            await self._fail(snapshot, code="PROVIDER_UNAVAILABLE", message="处理 Run 时发生未预期错误")
        return True

    async def _claim(self) -> RunSnapshot | None:
        lease_until = datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds)
        async with self._uow_factory() as uow:
            assert uow.runs is not None
            row = await uow.runs.claim(worker_id=self._worker_id, lease_until=lease_until)
            snapshot = self._snapshot(row) if row is not None else None
            await uow.commit()
            return snapshot

    async def _process(self, snapshot: RunSnapshot) -> None:
        async with self._uow_factory() as uow:
            row = await self._locked(uow, snapshot)
            if row is None or row.status in TERMINAL_STATUSES:
                return
            if already_attempted(row):
                row.error_code = "PROVIDER_UNAVAILABLE"
                row.error_message = "检测到未完成的上游尝试，已阻止二次调用"
                await append_event(uow, row, stage="FAILED", message=row.error_message)
                await uow.commit()
                return
            mark_upstream_attempted(row)
            await append_event(uow, row, stage="GENERATING", message="正在调用模型服务")
            await uow.commit()
        try:
            result = await self._generative.generate_image(self._image_request(snapshot))
            await self._complete(snapshot, result)
        except GenerativeResponsesClientError as exc:
            await self._fail(snapshot, code=exc.code, message=exc.message)

    async def _complete(self, snapshot: RunSnapshot, result) -> None:
        async with self._uow_factory() as uow:
            row = await self._locked(uow, snapshot)
            if row is None:
                return
            row.provider_request_id = result.provider_request_id
            row.usage_json = result.usage.model_dump(mode="json") if result.usage else None
            if result.status == "REJECTED":
                row.result_json = {"asset_ids": []}
                row.error_code = result.error_code or "CONTENT_REJECTED"
                row.error_message = "内容安全策略拒绝生成"
                await append_event(uow, row, stage="REJECTED", message=row.error_message)
                await uow.commit()
                return
            if result.status != "COMPLETED":
                row.error_code = result.error_code or "PROVIDER_UNAVAILABLE"
                row.error_message = result.error_message or _IMAGE_FAIL_MESSAGES.get(row.error_code, "图片生成未完成")
                await append_event(uow, row, stage="FAILED", message=row.error_message, payload={"error_code": row.error_code})
                await uow.commit()
                return
            assert uow.media_assets is not None
            asset_ids: list[str] = []
            now = datetime.now(timezone.utc)
            for artifact in result.artifacts:
                asset_id = uuid7()
                extension = MIME_EXTENSIONS.get(artifact.mime_type, "png")
                object_key = f"media/{int(row.domain_id)}/{row.run_id}/{asset_id}.{extension}"
                await self._store.put(object_key, artifact.content)
                await uow.media_assets.add(MediaStudioMediaAssetEntity(
                    asset_id=asset_id, domain_id=int(row.domain_id), run_id=row.run_id,
                    prompt_revision_id=row.prompt_revision_id, object_key=object_key,
                    mime_type=artifact.mime_type, byte_size=len(artifact.content),
                    content_sha256=hashlib.sha256(artifact.content).hexdigest(),
                    width=artifact.width, height=artifact.height, status="READY",
                    provider_artifact_id=artifact.provider_artifact_id,
                    created_by=row.actor_id, created_at=now,
                ))
                asset_ids.append(str(asset_id))
            row.result_json = {"asset_ids": asset_ids}
            await append_event(uow, row, stage="COMPLETED", message="图片已写入对象存储", payload={"artifact_count": len(asset_ids)})
            await uow.commit()

    async def _fail(self, snapshot: RunSnapshot, *, code: str, message: str) -> None:
        async with self._uow_factory() as uow:
            row = await self._locked(uow, snapshot)
            if row is None or row.status in TERMINAL_STATUSES:
                return
            row.error_code = code
            row.error_message = message
            await append_event(uow, row, stage="FAILED", message=message, payload={"error_code": code})
            await uow.commit()

    async def _locked(self, uow, snapshot: RunSnapshot):
        assert uow.runs is not None
        row = await uow.runs.get(domain_id=snapshot.domain_id, run_id=snapshot.run_id, lock=True)
        if row is None or row.lease_token != snapshot.lease_token:
            logger.warning("放弃写入：租约已失效 run_id={}", snapshot.run_id)
            return None
        return row

    @staticmethod
    def _snapshot(row) -> RunSnapshot:
        if row.lease_token is None:
            raise RuntimeError("认领 Run 后缺少 lease_token")
        return RunSnapshot(
            run_id=row.run_id, domain_id=int(row.domain_id), kind=row.kind, status=row.status,
            actor_id=row.actor_id, model_id=row.model_id, request_json=dict(row.request_json or {}),
            result_json=dict(row.result_json) if row.result_json else None,
            provider_request_id=row.provider_request_id, lease_token=row.lease_token,
            prompt_revision_id=row.prompt_revision_id, trace_id=row.trace_id,
        )

    @staticmethod
    def _image_request(snapshot: RunSnapshot) -> ImageGenerationRequest:
        payload = dict(snapshot.request_json)
        payload["model_id"] = str(snapshot.model_id)
        payload["trace_id"] = payload.get("trace_id") or snapshot.trace_id
        return ImageGenerationRequest.model_validate(payload)
