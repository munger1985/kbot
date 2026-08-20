"""KM Asset 持久任务处理器。"""

import asyncio
import hashlib
import json
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid4

import aiohttp
from loguru import logger

from km_asset_app.application.assets import KmAssetService
from km_asset_app.application.credentials import KmCredentialService
from km_asset_app.entities import KmAttachmentEntity, KmJobEntity
from km_asset_app.integrations import (
    AssetMetaDbClient,
    SharePointClient,
    SharePointDownloadError,
)
from platform_clients import KnowledgeCoreClient
from platform_core.contracts import AuthContext, PrincipalKind


@dataclass(frozen=True, slots=True)
class _JobSnapshot:
    job_id: UUID
    job_type: str
    domain_id: int
    source_id: UUID | None
    km_asset_id: UUID | None
    asset_revision_id: UUID | None
    payload_json: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _SourceSnapshot:
    source_id: UUID
    domain_id: int
    metadb_endpoint: str
    batch_size: int
    sharepoint_site_path: str
    collection_id: UUID
    auto_sync_enabled: bool


class _KcRevisionPending(Exception):
    """KC Revision 仍在正常处理，当前任务应延后再次检查。"""


class _KcReindexFailed(Exception):
    """KC Discovery 重新索引已经进入失败终态。"""


class KmAssetWorker:
    def __init__(self, *, uow_factory, credential_service: KmCredentialService, asset_service: KmAssetService, knowledge_core_client: KnowledgeCoreClient, poll_seconds: float = 5, lease_seconds: int = 120):
        self._uow_factory = uow_factory
        self._credentials = credential_service
        self._assets = asset_service
        self._kc = knowledge_core_client
        self._poll_seconds = poll_seconds
        self._lease_seconds = lease_seconds
        self._worker_id = f"{socket.gethostname()}:{uuid4()}"

    async def run_forever(self) -> None:
        logger.info("KM Asset Worker 开始运行：{}", self._worker_id)
        while True:
            job = await self._claim()
            if job is None:
                await self._schedule_active_sources()
                await asyncio.sleep(self._poll_seconds)
                continue
            try:
                await self._dispatch(job)
                await self._complete(job.job_id, succeeded=True)
            except asyncio.CancelledError:
                raise
            except _KcRevisionPending as exc:
                logger.debug("KC Revision 尚未完成：job_id={} status={}", job.job_id, exc)
                await self._defer_kc_status_sync(job.job_id)
            except _KcReindexFailed as exc:
                logger.warning("KC 重新索引失败：job_id={} error={}", job.job_id, exc)
                await self._complete(
                    job.job_id,
                    succeeded=False,
                    error=exc,
                    terminal=True,
                )
            except Exception as exc:
                logger.exception("KM Asset 任务失败：job_id={} type={}", job.job_id, job.job_type)
                await self._complete(job.job_id, succeeded=False, error=exc)

    async def _claim(self):
        async with self._uow_factory() as uow:
            row = await uow.assets.claim_job(worker_id=self._worker_id, lease_until=datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds))
            snapshot = self._job_snapshot(row) if row is not None else None
            await uow.commit()
            return snapshot

    async def _schedule_active_sources(self) -> None:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            for source in await uow.assets.list_auto_sync_sources():
                interval = max(10, int(source.poll_interval_seconds))
                bucket = int(now.timestamp()) // interval
                key = f"source-sync:{source.source_id}:{bucket}"
                if await uow.assets.find_job_by_key(domain_id=int(source.domain_id), idempotency_key=key) is not None:
                    continue
                await uow.assets.add(KmJobEntity(domain_id=source.domain_id, source_id=source.source_id, job_type="SOURCE_SYNC", idempotency_key=key, payload_json={"source_id": str(source.source_id), "trigger": "AUTO"}, status="PENDING", created_by=self._worker_id))
            await uow.commit()

    async def _dispatch(self, job: _JobSnapshot) -> None:
        if job.job_type == "SOURCE_SYNC":
            await self._source_sync(job)
        elif job.job_type in {"ATTACHMENT_DOWNLOAD", "RETRY"}:
            await self._download_and_ingest(job)
        elif job.job_type == "SOURCE_STATUS_UPDATE":
            await self._source_status_update(job)
        elif job.job_type == "KC_STATUS_SYNC":
            await self._kc_status_sync(job)
        else:
            raise RuntimeError(f"不支持的 KM Asset 任务类型：{job.job_type}")

    async def _source_sync(self, job: _JobSnapshot) -> None:
        source, metadb_values = await self._source_and_credentials(job, "METADB_BASIC")
        if (
            job.payload_json.get("trigger") == "AUTO"
            or job.payload_json.get("scheduled") is True
        ) and not source.auto_sync_enabled:
            logger.info("KM 来源后台同步已关闭，跳过自动任务：source_id={}", source.source_id)
            return
        client = AssetMetaDbClient(endpoint=source.metadb_endpoint, username=str(metadb_values["username"]), password=str(metadb_values["password"]))
        rows = await client.list_assets(offset=0, limit=source.batch_size, processed="N")
        for payload in rows:
            await self._assets.ingest_raw_asset(domain_id=int(source.domain_id), source_id=source.source_id, payload=payload, actor_id=self._worker_id)
        async with self._uow_factory() as uow:
            locked = await uow.assets.get_source(domain_id=int(source.domain_id), source_id=source.source_id, lock=True)
            locked.last_sync_at = datetime.now(timezone.utc)
            locked.error_code = None
            locked.error_message = None
            await uow.commit()

    async def _download_and_ingest(self, job: _JobSnapshot) -> None:
        if job.km_asset_id is None or job.asset_revision_id is None:
            raise RuntimeError("附件任务缺少 Asset 定位信息")
        async with self._uow_factory() as uow:
            asset = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
            source = await uow.assets.get_source(domain_id=int(job.domain_id), source_id=job.source_id)
            revision = await uow.assets.get_revision(asset_revision_id=job.asset_revision_id)
            if asset is None or source is None or revision is None:
                raise RuntimeError("附件任务引用的资源不存在")
            asset.ingestion_status = "DOWNLOADING"
            source_site_path = str(source.sharepoint_site_path)
            source_collection_id = source.collection_id
            raw_metadata = dict(asset.raw_metadata_json)
            urls = [item.strip() for field in ("first_sp_url", "second_sp_url") for item in str(raw_metadata.get(field) or "").split("^^^") if item.strip()]
            values = (
                await self._credentials.read(
                    uow=uow,
                    domain_id=int(job.domain_id),
                    credential_id=source.sharepoint_credential_id,
                    credential_kind="SHAREPOINT_GRAPH",
                    source_id=source.source_id,
                )
                if urls
                else None
            )
            external_asset_id = str(asset.external_asset_id)
            asset_title = asset.asset_title
            normalized_metadata = dict(asset.normalized_metadata_json)
            persisted_asset_id = asset.km_asset_id
            source_revision = str(revision.source_revision)
            revision_snapshot_hash = str(revision.snapshot_hash)
            await uow.commit()
        sp = (
            SharePointClient(
                tenant_id=str(values["tenant_id"]),
                client_id=str(values["client_id"]),
                client_secret=str(values["client_secret"]),
                site_path=source_site_path,
            )
            if values is not None
            else None
        )
        downloaded = []
        download_failures = []
        for ordinal, url in enumerate(urls):
            try:
                item = await sp.download(url)
                downloaded.append((ordinal, url, item, hashlib.sha256(item.content).hexdigest()))
            except SharePointDownloadError as exc:
                failure = {
                    "external_document_id": self._unavailable_document_id(url),
                    "source_url": url,
                    "declared_name": self._attachment_name(url, ordinal),
                    "ordinal": ordinal,
                    "failure_code": "SOURCE_DOWNLOAD_FAILED",
                    "failure_message": str(exc)[:1000] or "SharePoint 附件下载失败",
                }
                download_failures.append(failure)
                logger.warning(
                    "KM Asset 附件下载失败，继续提交元数据：asset_id={} ordinal={} error={}",
                    external_asset_id,
                    ordinal,
                    failure["failure_message"],
                )
        form = aiohttp.MultipartWriter("form-data")
        bundle = {
            "source_id": external_asset_id,
            "source_revision": source_revision,
            "title": asset_title or "Untitled Asset",
            "canonical_url": raw_metadata.get("osn_link") or None,
            "security_level": 1,
            "facet": {
                key: value
                for key, value in {
                    "product": raw_metadata.get("asset_product"),
                    "sub_type": raw_metadata.get("sub_type"),
                    "industry": raw_metadata.get("industry_id"),
                    "solution": raw_metadata.get("asset_solution"),
                    "language": raw_metadata.get("asset_language"),
                    "asset_type": raw_metadata.get("asset_type"),
                    "content_category": raw_metadata.get("content_category"),
                    "pillar": raw_metadata.get("pillar"),
                    "pillar_category": raw_metadata.get("pillar_category"),
                }.items()
                if value not in (None, "")
            },
            "metadata": {
                **normalized_metadata,
                "metadata_schema": "km_asset/v1",
                "km_asset_id": str(persisted_asset_id),
            },
        }
        declarations = []
        for ordinal, source_url, item, digest in downloaded:
            part_name = f"attachment_{ordinal}"
            declarations.append({"part_name": part_name, "external_document_id": item.external_document_id, "role": "ATTACHMENT", "source_url": source_url, "declared_name": item.name, "declared_mime_type": item.mime_type, "ordinal": ordinal, "required_flag": False, "byte_size": len(item.content), "content_sha256": digest})
        for name, value in (("bundle", bundle), ("documents", declarations), ("document_failures", download_failures)):
            part = form.append(json.dumps(value, ensure_ascii=False))
            part.set_content_disposition("form-data", name=name)
        for declaration, (_, _, item, _) in zip(declarations, downloaded):
            part = form.append(item.content, {"Content-Type": item.mime_type})
            part.set_content_disposition("form-data", name=declaration["part_name"], filename=item.name)
        context = self._auth_context(domain_id=int(job.domain_id))
        response = await self._kc.ingest_multipart(domain_id=int(job.domain_id), collection_id=source_collection_id, intake_kind="km-assets", content_type=form.content_type, body=form, idempotency_key=f"km-{revision_snapshot_hash}", auth_context=context)
        accepted = dict(response.payload)
        async with self._uow_factory() as uow:
            locked = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
            locked.ingestion_status = "KC_ACCEPTED"
            locked.kc_bundle_id = UUID(str(accepted["bundle_id"]))
            locked.kc_bundle_revision_id = UUID(str(accepted["bundle_revision_id"]))
            locked.failure_stage = None
            locked.error_code = None
            locked.error_message = None
            revision_row = await uow.assets.get_revision(asset_revision_id=job.asset_revision_id)
            revision_row.status = "PROCESSING"
            revision_row.kc_bundle_revision_id = locked.kc_bundle_revision_id
            for declaration in declarations:
                attachment = await uow.assets.find_attachment(
                    asset_revision_id=job.asset_revision_id,
                    external_document_id=declaration["external_document_id"],
                )
                if attachment is None:
                    attachment = KmAttachmentEntity(
                        asset_revision_id=job.asset_revision_id,
                        external_document_id=declaration["external_document_id"],
                        source_url=declaration["source_url"],
                        file_name=declaration["declared_name"],
                        mime_type=declaration["declared_mime_type"],
                        ordinal_no=declaration["ordinal"],
                        byte_size=declaration["byte_size"],
                        content_sha256=declaration["content_sha256"],
                        status="AVAILABLE",
                    )
                    await uow.assets.add(attachment)
                else:
                    attachment.source_url = declaration["source_url"]
                    attachment.file_name = declaration["declared_name"]
                    attachment.mime_type = declaration["declared_mime_type"]
                    attachment.ordinal_no = declaration["ordinal"]
                    attachment.byte_size = declaration["byte_size"]
                    attachment.content_sha256 = declaration["content_sha256"]
                    attachment.status = "AVAILABLE"
                    attachment.error_code = None
                    attachment.error_message = None
            for failure in download_failures:
                attachment = await uow.assets.find_attachment(
                    asset_revision_id=job.asset_revision_id,
                    external_document_id=failure["external_document_id"],
                )
                if attachment is None:
                    attachment = KmAttachmentEntity(
                        asset_revision_id=job.asset_revision_id,
                        external_document_id=failure["external_document_id"],
                        source_url=failure["source_url"],
                        file_name=failure["declared_name"],
                        ordinal_no=failure["ordinal"],
                        status="FAILED",
                        error_code=failure["failure_code"],
                        error_message=failure["failure_message"],
                    )
                    await uow.assets.add(attachment)
                else:
                    attachment.source_url = failure["source_url"]
                    attachment.file_name = failure["declared_name"]
                    attachment.ordinal_no = failure["ordinal"]
                    attachment.status = "FAILED"
                    attachment.error_code = failure["failure_code"]
                    attachment.error_message = failure["failure_message"]
            await uow.assets.add(KmJobEntity(domain_id=job.domain_id, source_id=job.source_id, km_asset_id=job.km_asset_id, asset_revision_id=job.asset_revision_id, job_type="KC_STATUS_SYNC", idempotency_key=f"kc-status:{accepted['bundle_revision_id']}", payload_json={"bundle_id": accepted["bundle_id"], "bundle_revision_id": accepted["bundle_revision_id"]}, status="PENDING", max_attempts=120, available_at=datetime.now(timezone.utc) + timedelta(seconds=10), created_by=self._worker_id))
            await uow.commit()

    @staticmethod
    def _unavailable_document_id(source_url: str) -> str:
        """为无法取得 Graph ID 的附件生成稳定文档标识。"""
        digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()
        return f"unavailable:{digest}"

    @staticmethod
    def _attachment_name(source_url: str, ordinal: int) -> str:
        """从错误链接提取仅用于展示的附件名。"""
        path = unquote(urlparse(source_url).path).rstrip("/")
        name = path.rsplit("/", 1)[-1].strip() if path else ""
        return (name or f"attachment-{ordinal + 1}")[:512]

    async def _source_status_update(self, job: _JobSnapshot) -> None:
        source, values = await self._source_and_credentials(job, "METADB_BASIC")
        await AssetMetaDbClient(endpoint=source.metadb_endpoint, username=str(values["username"]), password=str(values["password"])).set_processed(asset_id=str(job.payload_json["asset_id"]), processed=str(job.payload_json["processed"]))
        async with self._uow_factory() as uow:
            asset = None
            if job.km_asset_id is not None:
                asset = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
                if asset is None:
                    raise RuntimeError("状态更新任务引用的 Asset 不存在")
                asset.source_status = str(job.payload_json["processed"])
            if job.payload_json.get("next_job_type") == "RETRY":
                if asset is None:
                    raise RuntimeError("重试任务缺少本地 Asset")
                await uow.assets.add(KmJobEntity(domain_id=job.domain_id, source_id=job.source_id, km_asset_id=job.km_asset_id, asset_revision_id=job.asset_revision_id, job_type="RETRY", idempotency_key=f"retry:{job.asset_revision_id}:{asset.row_version}", payload_json={}, status="PENDING", priority=90, created_by=self._worker_id))
            elif job.payload_json.get("next_job_type") == "SOURCE_SYNC":
                await uow.assets.add(KmJobEntity(
                    domain_id=job.domain_id,
                    source_id=job.source_id,
                    job_type="SOURCE_SYNC",
                    idempotency_key=f"source-sync-after-reset:{job.job_id}",
                    payload_json={"source_id": str(job.source_id), "reset_job_id": str(job.job_id)},
                    status="PENDING",
                    priority=90,
                    created_by=self._worker_id,
                ))
            await uow.commit()

    async def _kc_status_sync(self, job: _JobSnapshot) -> None:
        bundle_id = UUID(str(job.payload_json["bundle_id"]))
        bundle_revision_id = UUID(str(job.payload_json["bundle_revision_id"]))
        if job.payload_json.get("operation_type") == "DISCOVERY_REINDEX":
            operation = await self._kc.get_reindex_discovery_status(
                domain_id=int(job.domain_id),
                bundle_id=bundle_id,
                bundle_revision_id=bundle_revision_id,
                generation=UUID(str(job.payload_json["reindex_generation"])),
                auth_context=self._auth_context(domain_id=int(job.domain_id)),
            )
            operation_status = str(operation.get("status") or "").upper()
            if operation_status in {"PENDING", "RUNNING"}:
                raise _KcRevisionPending(f"REINDEX_{operation_status}")
            if operation_status == "FAILED":
                failed = next(
                    (
                        item for item in operation.get("jobs") or []
                        if item.get("job_status") == "FAILED"
                    ),
                    {},
                )
                raise _KcReindexFailed(
                    str(failed.get("failure_message") or "KC Discovery 重新索引失败")
                )
            if operation_status != "SUCCEEDED":
                raise RuntimeError(
                    f"KC 重新索引返回未知状态：{operation_status or 'EMPTY'}"
                )
            return
        status = await self._kc.get_revision_status(
            domain_id=int(job.domain_id),
            bundle_id=bundle_id,
            bundle_revision_id=bundle_revision_id,
            include_members=False,
            auth_context=self._auth_context(domain_id=int(job.domain_id)),
        )
        value = str(status.get("status") or "").upper()
        if not value:
            raise RuntimeError("KC Revision 状态响应缺少 status")
        if value in {"ACCEPTED", "PENDING_REVIEW", "PROCESSING"}:
            raise _KcRevisionPending(value)
        if value not in {"READY", "PARTIAL", "FAILED", "REJECTED"}:
            raise RuntimeError(f"KC Revision 返回未知状态：{value}")
        if value in {"READY", "PARTIAL"}:
            bundle = await self._kc.get_bundle_status(
                domain_id=int(job.domain_id),
                bundle_id=bundle_id,
                auth_context=self._auth_context(domain_id=int(job.domain_id)),
            )
            published_revision_id = bundle.get("current_revision_id")
            availability = str(
                bundle.get("availability_status") or ""
            ).upper()
            expected_bundle_row_version = job.payload_json.get(
                "expected_bundle_row_version"
            )
            if (
                str(published_revision_id or "") != str(bundle_revision_id)
                or availability not in {"READY", "PARTIAL"}
                or (
                    expected_bundle_row_version is not None
                    and int(bundle.get("row_version") or 0)
                    < int(expected_bundle_row_version)
                )
            ):
                raise _KcRevisionPending("DISCOVERY_PUBLISHING")
        async with self._uow_factory() as uow:
            asset = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
            if value in {"READY", "PARTIAL"}:
                asset.ingestion_status = "READY"
                asset.completed_at = datetime.now(timezone.utc)
                revision = await uow.assets.get_revision(asset_revision_id=job.asset_revision_id)
                revision.status = "READY"
                await uow.assets.add(KmJobEntity(domain_id=job.domain_id, source_id=job.source_id, km_asset_id=job.km_asset_id, asset_revision_id=job.asset_revision_id, job_type="SOURCE_STATUS_UPDATE", idempotency_key=f"source-ready:{job.asset_revision_id}", payload_json={"asset_id": asset.external_asset_id, "processed": "Y", "next_job_type": None}, status="PENDING", created_by=self._worker_id))
            else:
                asset.ingestion_status = "FAILED"
                asset.failure_stage = "KC_PARSE"
                asset.error_code = "KC_BUNDLE_FAILED"
                revision = await uow.assets.get_revision(asset_revision_id=job.asset_revision_id)
                revision.status = "FAILED"
                await uow.assets.add(KmJobEntity(
                    domain_id=job.domain_id,
                    source_id=job.source_id,
                    km_asset_id=job.km_asset_id,
                    asset_revision_id=job.asset_revision_id,
                    job_type="SOURCE_STATUS_UPDATE",
                    idempotency_key=f"source-failed:{job.asset_revision_id}:KC_PARSE",
                    payload_json={"asset_id": asset.external_asset_id, "processed": "F", "next_job_type": None},
                    status="PENDING",
                    priority=100,
                    created_by=self._worker_id,
                ))
            asset.row_version += 1
            await uow.commit()

    async def _source_and_credentials(self, job: _JobSnapshot, credential_kind: str):
        async with self._uow_factory() as uow:
            source = await uow.assets.get_source(domain_id=int(job.domain_id), source_id=job.source_id)
            if source is None:
                raise RuntimeError("KM Asset 来源不存在")
            credential_id = source.metadb_credential_id if credential_kind == "METADB_BASIC" else source.sharepoint_credential_id
            values = await self._credentials.read(uow=uow, domain_id=int(job.domain_id), credential_id=credential_id, credential_kind=credential_kind, source_id=source.source_id)
            snapshot = _SourceSnapshot(
                source_id=source.source_id,
                domain_id=int(source.domain_id),
                metadb_endpoint=str(source.metadb_endpoint),
                batch_size=int(source.batch_size),
                sharepoint_site_path=str(source.sharepoint_site_path),
                collection_id=source.collection_id,
                auto_sync_enabled=bool(source.auto_sync_enabled),
            )
            return snapshot, dict(values)

    @staticmethod
    def _job_snapshot(row: KmJobEntity) -> _JobSnapshot:
        """在任务领取事务内复制 Worker 后续需要的全部字段。"""
        return _JobSnapshot(
            job_id=row.job_id,
            job_type=str(row.job_type),
            domain_id=int(row.domain_id),
            source_id=row.source_id,
            km_asset_id=row.km_asset_id,
            asset_revision_id=row.asset_revision_id,
            payload_json=dict(row.payload_json or {}),
        )

    async def _complete(
        self,
        job_id: UUID,
        *,
        succeeded: bool,
        error: Exception | None = None,
        terminal: bool = False,
    ) -> None:
        async with self._uow_factory() as uow:
            job = await uow.assets.get_job_by_id(job_id=job_id, lock=True)
            if job is None:
                return
            if succeeded:
                job.status = "SUCCEEDED"
                job.completed_at = datetime.now(timezone.utc)
                job.error_code = None
                job.error_message = None
            elif not terminal and job.attempt_count < job.max_attempts:
                job.status = "RETRY_WAIT"
                job.available_at = datetime.now(timezone.utc) + timedelta(seconds=min(300, 2 ** job.attempt_count * 5))
                job.error_code = type(error).__name__
                job.error_message = str(error)[:1000]
            else:
                job.status = "FAILED"
                job.completed_at = datetime.now(timezone.utc)
                job.error_code = type(error).__name__
                job.error_message = str(error)[:1000]
                is_reindex = (
                    (job.payload_json or {}).get("operation_type")
                    == "DISCOVERY_REINDEX"
                )
                if job.km_asset_id is not None and not is_reindex:
                    asset = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
                    if asset is not None:
                        asset.ingestion_status = "FAILED"
                        asset.failure_stage = job.job_type
                        asset.error_code = type(error).__name__
                        asset.error_message = str(error)[:1000]
                        asset.row_version += 1
                        if job.job_type != "SOURCE_STATUS_UPDATE":
                            await uow.assets.add(KmJobEntity(domain_id=job.domain_id, source_id=job.source_id, km_asset_id=job.km_asset_id, asset_revision_id=job.asset_revision_id, job_type="SOURCE_STATUS_UPDATE", idempotency_key=f"source-failed:{job.asset_revision_id}:{job.job_type}", payload_json={"asset_id": asset.external_asset_id, "processed": "F", "next_job_type": None}, status="PENDING", priority=100, created_by=self._worker_id))
            job.lease_owner = None
            job.lease_until = None
            await uow.commit()

    async def _defer_kc_status_sync(self, job_id: UUID) -> None:
        """延后正常进行中的 KC 状态检查，不消耗失败重试额度。"""
        async with self._uow_factory() as uow:
            job = await uow.assets.get_job_by_id(job_id=job_id, lock=True)
            if job is None:
                return
            job.status = "RETRY_WAIT"
            job.available_at = datetime.now(timezone.utc) + timedelta(seconds=10)
            job.attempt_count = max(0, job.attempt_count - 1)
            job.error_code = None
            job.error_message = None
            job.lease_owner = None
            job.lease_until = None
            await uow.commit()

    @staticmethod
    def _auth_context(*, domain_id: int) -> AuthContext:
        request_id = str(uuid4())
        return AuthContext(principal_kind=PrincipalKind.SERVICE, client_id="kbot-km-asset-app-worker", calling_service="kbot-km-asset-app-worker", request_id=request_id, trace_id=request_id, domain_id=str(domain_id), asserted_user_id="svc:km-asset")
