"""KM Asset 来源、快照、同步任务和失败重试用例。"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import UUID

from km_asset_app.application.credentials import KmCredentialService
from km_asset_app.entities import KmAssetEntity, KmAssetRevisionEntity, KmJobEntity, KmSourceEntity
from km_asset_app.integrations import AssetMetaDbClient, AssetMetaDbError
from platform_core.contracts import AuthContext, PrincipalKind
from platform_core.identity import uuid7


class KmAssetApplicationError(RuntimeError):
    def __init__(self, *, status_code: int, code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


@dataclass(frozen=True)
class CreateSourceCommand:
    domain_id: int
    display_name: str
    metadb_endpoint: str
    metadb_credentials: dict[str, str]
    sharepoint_credentials: dict[str, str]
    sharepoint_site_path: str
    collection_id: UUID
    poll_interval_seconds: int
    batch_size: int
    actor_id: str


class KmAssetService:
    def __init__(self, *, uow_factory: Callable, credential_service: KmCredentialService, data_query_client=None, knowledge_core_client=None):
        self._uow_factory = uow_factory
        self._credentials = credential_service
        self._data_query = data_query_client
        self._knowledge_core = knowledge_core_client

    async def create_source(self, command: CreateSourceCommand) -> dict[str, Any]:
        self._require_data_query()
        managed = await self._data_query.reconcile_km_asset_dataset(
            auth_context=self._auth_context(
                domain_id=command.domain_id,
                actor_id=command.actor_id,
            )
        )
        async with self._uow_factory() as uow:
            source_id = uuid7()
            metadb_credential = await self._credentials.put(uow=uow, domain_id=command.domain_id, source_id=source_id, credential_kind="METADB_BASIC", values=command.metadb_credentials, actor_id=command.actor_id)
            sharepoint_credential = await self._credentials.put(uow=uow, domain_id=command.domain_id, source_id=source_id, credential_kind="SHAREPOINT_GRAPH", values=command.sharepoint_credentials, actor_id=command.actor_id)
            row = KmSourceEntity(
                source_id=source_id,
                domain_id=command.domain_id,
                display_name=command.display_name.strip(),
                metadb_endpoint=str(command.metadb_endpoint).strip(),
                metadb_credential_id=metadb_credential.credential_id,
                sharepoint_credential_id=sharepoint_credential.credential_id,
                sharepoint_site_path=command.sharepoint_site_path.strip(),
                collection_id=command.collection_id,
                semantic_model_id=UUID(str(managed["semantic_model_id"])),
                policy_binding_id=UUID(str(managed["policy_binding_id"])),
                model_catalog_hash=str(managed["catalog_hash"]),
                model_status="READY",
                status="DRAFT",
                auto_sync_enabled=0,
                poll_interval_seconds=command.poll_interval_seconds,
                batch_size=command.batch_size,
                created_by=command.actor_id,
                updated_by=command.actor_id,
            )
            await uow.assets.add(row)
            await uow.commit()
            return self._source(row)

    async def list_sources(self, *, domain_id: int):
        async with self._uow_factory() as uow:
            return [self._source(row) for row in await uow.assets.list_sources(domain_id=domain_id)]

    async def update_source(
        self,
        *,
        domain_id: int,
        source_id: UUID,
        expected_row_version: int,
        changes: dict[str, Any],
        actor_id: str,
    ) -> dict[str, Any]:
        """更新来源运行配置，并支持显式轮换托管凭据。"""
        async with self._uow_factory() as uow:
            row = await uow.assets.get_source(
                domain_id=domain_id,
                source_id=source_id,
                lock=True,
            )
            if row is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            if int(row.row_version) != expected_row_version:
                raise KmAssetApplicationError(
                    status_code=409,
                    code="ROW_VERSION_CONFLICT",
                    message="来源配置已被其他请求修改",
                )
            metadb_credentials = changes.pop("metadb_credentials", None)
            sharepoint_credentials = changes.pop("sharepoint_credentials", None)
            if metadb_credentials is not None:
                await self._credentials.put(
                    uow=uow,
                    domain_id=domain_id,
                    source_id=source_id,
                    credential_kind="METADB_BASIC",
                    values=metadb_credentials,
                    actor_id=actor_id,
                )
            if sharepoint_credentials is not None:
                await self._credentials.put(
                    uow=uow,
                    domain_id=domain_id,
                    source_id=source_id,
                    credential_kind="SHAREPOINT_GRAPH",
                    values=sharepoint_credentials,
                    actor_id=actor_id,
                )
            for field, value in changes.items():
                if field in {"display_name", "sharepoint_site_path"}:
                    value = str(value).strip()
                elif field == "metadb_endpoint":
                    value = str(value).strip()
                elif field == "auto_sync_enabled":
                    value = 1 if value else 0
                setattr(row, field, value)
            row.row_version += 1
            row.updated_by = actor_id
            await uow.commit()
            return self._source(row)

    async def list_metadb_assets(self, *, domain_id: int, source_id: UUID, processed: str, offset: int, limit: int):
        async with self._uow_factory() as uow:
            source = await uow.assets.get_source(domain_id=domain_id, source_id=source_id)
            if source is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            metadb_endpoint = str(source.metadb_endpoint)
            metadb_credential_id = source.metadb_credential_id
            persisted_source_id = source.source_id
            values = await self._credentials.read(
                uow=uow,
                domain_id=domain_id,
                credential_id=metadb_credential_id,
                credential_kind="METADB_BASIC",
                source_id=persisted_source_id,
            )
        try:
            rows = await AssetMetaDbClient(
                endpoint=metadb_endpoint,
                username=str(values["username"]),
                password=str(values["password"]),
            ).list_assets(offset=offset, limit=limit, processed=processed)
        except AssetMetaDbError as exc:
            raise KmAssetApplicationError(status_code=503, code="KM_METADB_UNAVAILABLE", message=str(exc)) from exc
        return {"source_id": source_id, "processed": processed, "offset": offset, "limit": limit, "items": rows}

    async def retry_metadb_asset(self, *, domain_id: int, source_id: UUID, external_asset_id: str, actor_id: str):
        asset_id = external_asset_id.strip()
        if not asset_id:
            raise KmAssetApplicationError(status_code=422, code="ASSET_ID_REQUIRED", message="Asset ID 不能为空")
        asset_key = hashlib.sha256(asset_id.encode("utf-8")).hexdigest()[:32]
        async with self._uow_factory() as uow:
            source = await uow.assets.get_source(domain_id=domain_id, source_id=source_id)
            if source is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            if source.status != "ACTIVE":
                raise KmAssetApplicationError(status_code=409, code="KM_SOURCE_NOT_ACTIVE", message="KM Asset 来源未激活")
            job = KmJobEntity(
                domain_id=domain_id,
                source_id=source_id,
                job_type="SOURCE_STATUS_UPDATE",
                idempotency_key=f"raw-retry:{source_id}:{asset_key}:{uuid7()}",
                payload_json={"asset_id": asset_id, "processed": "N", "next_job_type": "SOURCE_SYNC"},
                status="PENDING",
                priority=100,
                created_by=actor_id,
            )
            await uow.assets.add(job)
            await uow.commit()
            return self._job(job)

    async def activate_source(self, *, domain_id: int, source_id: UUID, expected_row_version: int, actor_id: str):
        self._require_data_query()
        managed = await self._data_query.reconcile_km_asset_dataset(
            auth_context=self._auth_context(domain_id=domain_id, actor_id=actor_id)
        )
        async with self._uow_factory() as uow:
            row = await uow.assets.get_source(domain_id=domain_id, source_id=source_id, lock=True)
            if row is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            if row.row_version != expected_row_version:
                raise KmAssetApplicationError(status_code=409, code="ROW_VERSION_CONFLICT", message="来源配置已被其他请求修改")
            row.semantic_model_id = UUID(str(managed["semantic_model_id"]))
            row.policy_binding_id = UUID(str(managed["policy_binding_id"]))
            row.model_catalog_hash = str(managed["catalog_hash"])
            row.model_status = "READY"
            if row.model_status != "READY" or row.semantic_model_id is None or row.policy_binding_id is None:
                raise KmAssetApplicationError(status_code=409, code="MANAGED_MODEL_NOT_READY", message="系统托管问数模型尚未就绪")
            row.status = "ACTIVE"
            row.row_version += 1
            row.updated_by = actor_id
            await uow.commit()
            return self._source(row)

    async def reconcile_model(self, *, domain_id: int, source_id: UUID, actor_id: str):
        self._require_data_query()
        managed = await self._data_query.reconcile_km_asset_dataset(
            auth_context=self._auth_context(domain_id=domain_id, actor_id=actor_id)
        )
        async with self._uow_factory() as uow:
            row = await uow.assets.get_source(domain_id=domain_id, source_id=source_id, lock=True)
            if row is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            row.semantic_model_id = UUID(str(managed["semantic_model_id"]))
            row.policy_binding_id = UUID(str(managed["policy_binding_id"]))
            row.model_catalog_hash = str(managed["catalog_hash"])
            row.model_status = "READY"
            row.error_code = None
            row.error_message = None
            row.row_version += 1
            row.updated_by = actor_id
            await uow.commit()
            return {"status": row.model_status, "catalog_hash": row.model_catalog_hash, "semantic_model_id": row.semantic_model_id, "policy_binding_id": row.policy_binding_id, "managed_by": "data_query"}

    async def managed_model(self, *, domain_id: int, source_id: UUID):
        async with self._uow_factory() as uow:
            row = await uow.assets.get_source(domain_id=domain_id, source_id=source_id)
            if row is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            return {"status": row.model_status, "catalog_hash": row.model_catalog_hash, "semantic_model_id": row.semantic_model_id, "policy_binding_id": row.policy_binding_id, "managed_by": "data_query"}

    async def submit_sync(self, *, domain_id: int, source_id: UUID, actor_id: str):
        async with self._uow_factory() as uow:
            source = await uow.assets.get_source(domain_id=domain_id, source_id=source_id)
            if source is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            if source.status != "ACTIVE":
                raise KmAssetApplicationError(status_code=409, code="KM_SOURCE_NOT_ACTIVE", message="KM Asset 来源未激活")
            bucket = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
            job = KmJobEntity(domain_id=domain_id, source_id=source_id, job_type="SOURCE_SYNC", idempotency_key=f"source-sync:{source_id}:{bucket}", payload_json={"source_id": str(source_id), "trigger": "MANUAL"}, status="PENDING", created_by=actor_id)
            await uow.assets.add(job)
            await uow.commit()
            return self._job(job)

    async def ingest_raw_asset(self, *, domain_id: int, source_id: UUID, payload: dict[str, Any], actor_id: str):
        external_id = str(payload.get("asset_id") or "").strip()
        if not external_id:
            raise KmAssetApplicationError(status_code=422, code="ASSET_ID_REQUIRED", message="MetaDB 记录缺少 asset_id")
        normalized = self._normalize(payload)
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        snapshot_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        source_revision = str(payload.get("last_update_time") or snapshot_hash)
        async with self._uow_factory() as uow:
            source = await uow.assets.get_source(domain_id=domain_id, source_id=source_id)
            if source is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            asset = await uow.assets.find_asset(source_id=source_id, external_asset_id=external_id, lock=True)
            if asset is not None and asset.snapshot_hash == snapshot_hash:
                return self._asset(asset)
            if asset is None:
                source_status = str(payload.get("processed") or "UNKNOWN").upper()
                if source_status not in {"N", "Y", "F"}:
                    source_status = "UNKNOWN"
                asset = KmAssetEntity(domain_id=domain_id, source_id=source_id, external_asset_id=external_id, snapshot_hash=snapshot_hash, source_revision=source_revision, source_status=source_status, ingestion_status="DISCOVERED", raw_metadata_json=payload, normalized_metadata_json=normalized, created_by=actor_id, updated_by=actor_id, **self._columns(normalized))
                await uow.assets.add(asset)
            else:
                asset.snapshot_hash = snapshot_hash
                asset.source_revision = source_revision
                source_status = str(payload.get("processed") or "UNKNOWN").upper()
                asset.source_status = source_status if source_status in {"N", "Y", "F"} else "UNKNOWN"
                asset.ingestion_status = "DISCOVERED"
                asset.raw_metadata_json = payload
                asset.normalized_metadata_json = normalized
                asset.updated_by = actor_id
                asset.row_version += 1
                for key, value in self._columns(normalized).items():
                    setattr(asset, key, value)
            revision = KmAssetRevisionEntity(km_asset_id=asset.km_asset_id, domain_id=domain_id, revision_no=await uow.assets.next_revision_no(km_asset_id=asset.km_asset_id), source_revision=source_revision, snapshot_hash=snapshot_hash, raw_payload_json=payload, normalized_payload_json=normalized, status="DISCOVERED", created_by=actor_id)
            await uow.assets.add(revision)
            asset.current_revision_id = revision.asset_revision_id
            asset.ingestion_status = "SYNC_PENDING"
            job = KmJobEntity(domain_id=domain_id, source_id=source_id, km_asset_id=asset.km_asset_id, asset_revision_id=revision.asset_revision_id, job_type="ATTACHMENT_DOWNLOAD", idempotency_key=f"download:{revision.asset_revision_id}", payload_json={"asset_revision_id": str(revision.asset_revision_id)}, status="PENDING", created_by=actor_id)
            await uow.assets.add(job)
            await uow.commit()
            return self._asset(asset)

    async def list_assets(self, *, domain_id: int, source_id: UUID | None, ingestion_status: str | None, offset: int, limit: int):
        async with self._uow_factory() as uow:
            return [self._asset(row) for row in await uow.assets.list_assets(domain_id=domain_id, source_id=source_id, ingestion_status=ingestion_status, offset=offset, limit=limit)]

    async def get_asset(self, *, domain_id: int, km_asset_id: UUID):
        async with self._uow_factory() as uow:
            row = await uow.assets.get_asset(domain_id=domain_id, km_asset_id=km_asset_id)
            if row is None:
                self._not_found("KM_ASSET_NOT_FOUND", "KM Asset 不存在")
            result = self._asset(row)
            result["raw_metadata"] = row.raw_metadata_json
            result["normalized_metadata"] = row.normalized_metadata_json
            result["attachments"] = [self._attachment(item) for item in await uow.assets.list_attachments(asset_revision_id=row.current_revision_id)] if row.current_revision_id else []
            return result

    async def retry_asset(self, *, domain_id: int, km_asset_id: UUID, expected_row_version: int, actor_id: str):
        async with self._uow_factory() as uow:
            row = await uow.assets.get_asset(domain_id=domain_id, km_asset_id=km_asset_id, lock=True)
            if row is None:
                self._not_found("KM_ASSET_NOT_FOUND", "KM Asset 不存在")
            if row.row_version != expected_row_version:
                raise KmAssetApplicationError(status_code=409, code="ROW_VERSION_CONFLICT", message="Asset 已被其他请求修改")
            if row.ingestion_status not in {"FAILED", "DOWNLOAD_FAILED"} and row.source_status != "F":
                raise KmAssetApplicationError(status_code=409, code="KM_ASSET_NOT_RETRYABLE", message="只有失败的 Asset 可以重试")
            row.ingestion_status = "RETRY_PENDING"
            row.error_code = None
            row.error_message = None
            row.failure_stage = None
            row.row_version += 1
            row.updated_by = actor_id
            job = KmJobEntity(domain_id=domain_id, source_id=row.source_id, km_asset_id=row.km_asset_id, asset_revision_id=row.current_revision_id, job_type="SOURCE_STATUS_UPDATE", idempotency_key=f"retry-reset:{row.km_asset_id}:{row.row_version}", payload_json={"asset_id": row.external_asset_id, "processed": "N", "next_job_type": "RETRY"}, status="PENDING", priority=100, created_by=actor_id)
            await uow.assets.add(job)
            await uow.commit()
            return {"asset": self._asset(row), "job": self._job(job)}

    async def reindex_asset(
        self,
        *,
        domain_id: int,
        km_asset_id: UUID,
        expected_row_version: int,
        actor_id: str,
    ):
        """重建 KC Revision 的全文与向量 Discovery Profile。"""
        if self._knowledge_core is None:
            raise KmAssetApplicationError(
                status_code=503,
                code="KNOWLEDGE_CORE_UNAVAILABLE",
                message="Knowledge Core 服务未配置",
            )
        async with self._uow_factory() as uow:
            row = await uow.assets.get_asset(
                domain_id=domain_id,
                km_asset_id=km_asset_id,
                lock=True,
            )
            if row is None:
                self._not_found("KM_ASSET_NOT_FOUND", "KM Asset 不存在")
            if int(row.row_version) != expected_row_version:
                raise KmAssetApplicationError(
                    status_code=409,
                    code="ROW_VERSION_CONFLICT",
                    message="Asset 已被其他请求修改",
                )
            if row.kc_bundle_id is None or row.kc_bundle_revision_id is None:
                raise KmAssetApplicationError(
                    status_code=409,
                    code="KM_ASSET_KC_REVISION_MISSING",
                    message="Asset 尚未形成可重新索引的 KC Revision",
                )
            source = await uow.assets.get_source(
                domain_id=domain_id,
                source_id=row.source_id,
            )
            if source is None:
                self._not_found("KM_SOURCE_NOT_FOUND", "KM Asset 来源不存在")
            collection_id = source.collection_id
            bundle_id = row.kc_bundle_id
            bundle_revision_id = row.kc_bundle_revision_id
        receipt = await self._knowledge_core.reindex_discovery(
            domain_id=domain_id,
            collection_id=collection_id,
            bundle_id=bundle_id,
            bundle_revision_id=bundle_revision_id,
            auth_context=self._auth_context(
                domain_id=domain_id,
                actor_id=actor_id,
            ),
        )
        async with self._uow_factory() as uow:
            row = await uow.assets.get_asset(
                domain_id=domain_id,
                km_asset_id=km_asset_id,
                lock=True,
            )
            if row is None or int(row.row_version) != expected_row_version:
                raise KmAssetApplicationError(
                    status_code=409,
                    code="ROW_VERSION_CONFLICT",
                    message="Asset 已被其他请求修改",
                )
            row.ingestion_status = "PARSING"
            row.completed_at = None
            row.error_code = None
            row.error_message = None
            row.failure_stage = None
            row.row_version += 1
            row.updated_by = actor_id
            job = KmJobEntity(
                domain_id=domain_id,
                source_id=row.source_id,
                km_asset_id=row.km_asset_id,
                asset_revision_id=row.current_revision_id,
                job_type="KC_STATUS_SYNC",
                idempotency_key=(
                    f"kc-reindex:{bundle_revision_id}:{row.row_version}"
                ),
                payload_json={
                    "bundle_id": str(bundle_id),
                    "bundle_revision_id": str(bundle_revision_id),
                    "expected_bundle_row_version": int(
                        receipt["expected_bundle_row_version"]
                    ),
                },
                status="PENDING",
                max_attempts=120,
                available_at=datetime.now(timezone.utc) + timedelta(seconds=10),
                created_by=actor_id,
            )
            await uow.assets.add(job)
            await uow.commit()
            return {
                "asset": self._asset(row),
                "job": self._job(job),
                "kc_reindex": receipt,
            }

    async def list_jobs(self, *, domain_id: int, source_id: UUID | None, limit: int):
        async with self._uow_factory() as uow:
            return [self._job(row) for row in await uow.assets.list_jobs(domain_id=domain_id, source_id=source_id, limit=limit)]

    def _require_data_query(self) -> None:
        if self._data_query is None:
            raise KmAssetApplicationError(
                status_code=503,
                code="KM_MANAGED_MODEL_SERVICE_UNAVAILABLE",
                message="KM 托管问数模型服务未配置",
            )

    @staticmethod
    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        return {str(key).strip().lower(): value for key, value in payload.items()}

    @staticmethod
    def _columns(data: dict[str, Any]) -> dict[str, Any]:
        mapping = {"asset_title": "asset_title", "author_mail": "author_mail", "asset_product": "asset_product", "asset_solution": "asset_solution", "industry_id": "industry_id", "content_category": "content_category", "asset_status": "asset_status", "publish_date": "publish_date", "last_update_time": "last_update_time"}
        return {target: None if data.get(source) is None else str(data.get(source)) for source, target in mapping.items()}

    @staticmethod
    def _source(row):
        return {"source_id": row.source_id, "domain_id": int(row.domain_id), "display_name": row.display_name, "metadb_endpoint": row.metadb_endpoint, "sharepoint_site_path": row.sharepoint_site_path, "collection_id": row.collection_id, "semantic_model_id": row.semantic_model_id, "policy_binding_id": row.policy_binding_id, "model_status": row.model_status, "catalog_hash": row.model_catalog_hash, "status": row.status, "auto_sync_enabled": bool(row.auto_sync_enabled), "poll_interval_seconds": row.poll_interval_seconds, "batch_size": row.batch_size, "last_sync_at": row.last_sync_at, "error_code": row.error_code, "error_message": row.error_message, "row_version": row.row_version}

    @staticmethod
    def _asset(row):
        return {"km_asset_id": row.km_asset_id, "source_id": row.source_id, "external_asset_id": row.external_asset_id, "source_revision": row.source_revision, "source_status": row.source_status, "ingestion_status": row.ingestion_status, "asset_title": row.asset_title, "author_mail": row.author_mail, "asset_product": row.asset_product, "asset_solution": row.asset_solution, "industry_id": row.industry_id, "content_category": row.content_category, "asset_status": row.asset_status, "publish_date": row.publish_date, "last_update_time": row.last_update_time, "kc_bundle_id": row.kc_bundle_id, "kc_bundle_revision_id": row.kc_bundle_revision_id, "failure_stage": row.failure_stage, "error_code": row.error_code, "error_message": row.error_message, "attempt_count": row.attempt_count, "synced_at": row.synced_at, "completed_at": row.completed_at, "row_version": row.row_version}

    @staticmethod
    def _attachment(row):
        return {"attachment_id": row.attachment_id, "external_document_id": row.external_document_id, "source_url": row.source_url, "file_name": row.file_name, "mime_type": row.mime_type, "ordinal": row.ordinal_no, "byte_size": row.byte_size, "content_sha256": row.content_sha256, "status": row.status, "error_code": row.error_code, "error_message": row.error_message}

    @staticmethod
    def _job(row):
        return {"job_id": row.job_id, "job_type": row.job_type, "source_id": row.source_id, "km_asset_id": row.km_asset_id, "status": row.status, "attempt_count": row.attempt_count, "max_attempts": row.max_attempts, "available_at": row.available_at, "error_code": row.error_code, "error_message": row.error_message, "created_at": row.created_at, "completed_at": row.completed_at}

    @staticmethod
    def _not_found(code: str, message: str):
        raise KmAssetApplicationError(status_code=404, code=code, message=message)

    @staticmethod
    def _auth_context(*, domain_id: int, actor_id: str) -> AuthContext:
        token = str(uuid7())
        return AuthContext(principal_kind=PrincipalKind.SERVICE, client_id="kbot-km-asset-app-api", calling_service="kbot-km-asset-app-api", request_id=token, trace_id=token, domain_id=str(domain_id), asserted_user_id=actor_id)
