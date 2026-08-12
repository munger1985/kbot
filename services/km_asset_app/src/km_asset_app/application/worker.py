"""KM Asset 持久任务处理器。"""

import asyncio
import hashlib
import json
import socket
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import aiohttp
from loguru import logger

from km_asset_app.application.assets import KmAssetService
from km_asset_app.application.credentials import KmCredentialService
from km_asset_app.entities import KmAttachmentEntity, KmJobEntity
from km_asset_app.integrations import AssetMetaDbClient, SharePointClient
from platform_clients import KnowledgeCoreClient
from platform_core.contracts import AuthContext, PrincipalKind


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
            except Exception as exc:
                logger.exception("KM Asset 任务失败：job_id={} type={}", job.job_id, job.job_type)
                await self._complete(job.job_id, succeeded=False, error=exc)

    async def _claim(self):
        async with self._uow_factory() as uow:
            row = await uow.assets.claim_job(worker_id=self._worker_id, lease_until=datetime.now(timezone.utc) + timedelta(seconds=self._lease_seconds))
            await uow.commit()
            return row

    async def _schedule_active_sources(self) -> None:
        now = datetime.now(timezone.utc)
        async with self._uow_factory() as uow:
            for source in await uow.assets.list_active_sources():
                interval = max(10, int(source.poll_interval_seconds))
                bucket = int(now.timestamp()) // interval
                key = f"source-sync:{source.source_id}:{bucket}"
                if await uow.assets.find_job_by_key(domain_id=int(source.domain_id), idempotency_key=key) is not None:
                    continue
                await uow.assets.add(KmJobEntity(domain_id=source.domain_id, source_id=source.source_id, job_type="SOURCE_SYNC", idempotency_key=key, payload_json={"source_id": str(source.source_id), "scheduled": True}, status="PENDING", created_by=self._worker_id))
            await uow.commit()

    async def _dispatch(self, job: KmJobEntity) -> None:
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

    async def _source_sync(self, job: KmJobEntity) -> None:
        source, metadb_values = await self._source_and_credentials(job, "METADB_BASIC")
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

    async def _download_and_ingest(self, job: KmJobEntity) -> None:
        if job.km_asset_id is None or job.asset_revision_id is None:
            raise RuntimeError("附件任务缺少 Asset 定位信息")
        async with self._uow_factory() as uow:
            asset = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
            source = await uow.assets.get_source(domain_id=int(job.domain_id), source_id=job.source_id)
            revision = await uow.assets.get_revision(asset_revision_id=job.asset_revision_id)
            if asset is None or source is None or revision is None:
                raise RuntimeError("附件任务引用的资源不存在")
            values = await self._credentials.read(uow=uow, domain_id=int(job.domain_id), credential_id=source.sharepoint_credential_id, credential_kind="SHAREPOINT_GRAPH", source_id=source.source_id)
            asset.ingestion_status = "DOWNLOADING"
            await uow.commit()
            raw_metadata = dict(asset.raw_metadata_json)
        urls = [item.strip() for field in ("first_sp_url", "second_sp_url") for item in str(raw_metadata.get(field) or "").split("^^^") if item.strip()]
        if not urls:
            raise RuntimeError("Asset 没有可下载的 SharePoint 附件")
        sp = SharePointClient(
            tenant_id=str(values["tenant_id"]),
            client_id=str(values["client_id"]),
            client_secret=str(values["client_secret"]),
            site_path=source.sharepoint_site_path,
        )
        downloaded = []
        for ordinal, url in enumerate(urls):
            item = await sp.download(url)
            downloaded.append((ordinal, url, item, hashlib.sha256(item.content).hexdigest()))
        form = aiohttp.MultipartWriter("form-data")
        bundle = {
            "source_id": asset.external_asset_id,
            "source_revision": revision.source_revision,
            "title": asset.asset_title or "Untitled Asset",
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
                **asset.normalized_metadata_json,
                "metadata_schema": "km_asset/v1",
                "km_asset_id": str(asset.km_asset_id),
            },
        }
        declarations = []
        for ordinal, source_url, item, digest in downloaded:
            part_name = f"attachment_{ordinal}"
            declarations.append({"part_name": part_name, "external_document_id": item.external_document_id, "role": "ATTACHMENT", "source_url": source_url, "declared_name": item.name, "declared_mime_type": item.mime_type, "ordinal": ordinal, "required_flag": False, "byte_size": len(item.content), "content_sha256": digest})
        for name, value in (("bundle", bundle), ("documents", declarations), ("document_failures", [])):
            part = form.append(json.dumps(value, ensure_ascii=False))
            part.set_content_disposition("form-data", name=name)
        for declaration, (_, _, item, _) in zip(declarations, downloaded):
            part = form.append(item.content, {"Content-Type": item.mime_type})
            part.set_content_disposition("form-data", name=declaration["part_name"], filename=item.name)
        context = self._auth_context(domain_id=int(job.domain_id))
        response = await self._kc.ingest_multipart(domain_id=int(job.domain_id), collection_id=source.collection_id, intake_kind="km-assets", content_type=form.content_type, body=form, idempotency_key=f"km-{revision.snapshot_hash}", auth_context=context)
        accepted = dict(response.payload)
        async with self._uow_factory() as uow:
            locked = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
            locked.ingestion_status = "KC_ACCEPTED"
            locked.kc_bundle_id = UUID(str(accepted["bundle_id"]))
            locked.kc_bundle_revision_id = UUID(str(accepted["bundle_revision_id"]))
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
            await uow.assets.add(KmJobEntity(domain_id=job.domain_id, source_id=job.source_id, km_asset_id=job.km_asset_id, asset_revision_id=job.asset_revision_id, job_type="KC_STATUS_SYNC", idempotency_key=f"kc-status:{accepted['bundle_revision_id']}", payload_json={"bundle_id": accepted["bundle_id"], "bundle_revision_id": accepted["bundle_revision_id"]}, status="PENDING", max_attempts=120, available_at=datetime.now(timezone.utc) + timedelta(seconds=10), created_by=self._worker_id))
            await uow.commit()

    async def _source_status_update(self, job: KmJobEntity) -> None:
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

    async def _kc_status_sync(self, job: KmJobEntity) -> None:
        bundle_id = UUID(str(job.payload_json["bundle_id"]))
        status = await self._kc.get_bundle_status(domain_id=int(job.domain_id), bundle_id=bundle_id, auth_context=self._auth_context(domain_id=int(job.domain_id)))
        value = str(status.get("status") or status.get("processing_status") or "").upper()
        if value not in {"READY", "FAILED", "REJECTED"}:
            raise RuntimeError(f"KC Bundle 尚未完成：{value or 'UNKNOWN'}")
        async with self._uow_factory() as uow:
            asset = await uow.assets.get_asset(domain_id=int(job.domain_id), km_asset_id=job.km_asset_id, lock=True)
            if value == "READY":
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

    async def _source_and_credentials(self, job: KmJobEntity, credential_kind: str):
        async with self._uow_factory() as uow:
            source = await uow.assets.get_source(domain_id=int(job.domain_id), source_id=job.source_id)
            if source is None:
                raise RuntimeError("KM Asset 来源不存在")
            credential_id = source.metadb_credential_id if credential_kind == "METADB_BASIC" else source.sharepoint_credential_id
            values = await self._credentials.read(uow=uow, domain_id=int(job.domain_id), credential_id=credential_id, credential_kind=credential_kind, source_id=source.source_id)
            return source, values

    async def _complete(self, job_id: UUID, *, succeeded: bool, error: Exception | None = None) -> None:
        async with self._uow_factory() as uow:
            job = await uow.assets.get_job_by_id(job_id=job_id, lock=True)
            if job is None:
                return
            if succeeded:
                job.status = "SUCCEEDED"
                job.completed_at = datetime.now(timezone.utc)
            elif job.attempt_count < job.max_attempts:
                job.status = "RETRY_WAIT"
                job.available_at = datetime.now(timezone.utc) + timedelta(seconds=min(300, 2 ** job.attempt_count * 5))
                job.error_code = type(error).__name__
                job.error_message = str(error)[:1000]
            else:
                job.status = "FAILED"
                job.completed_at = datetime.now(timezone.utc)
                job.error_code = type(error).__name__
                job.error_message = str(error)[:1000]
                if job.km_asset_id is not None:
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

    @staticmethod
    def _auth_context(*, domain_id: int) -> AuthContext:
        request_id = str(uuid4())
        return AuthContext(principal_kind=PrincipalKind.SERVICE, client_id="kbot-km-asset-app-worker", calling_service="kbot-km-asset-app-worker", request_id=request_id, trace_id=request_id, domain_id=str(domain_id), asserted_user_id="svc:km-asset")
