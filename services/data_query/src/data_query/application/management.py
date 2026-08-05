"""面向内部 HTTP 的 Data Query 管理面服务。"""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from datetime import UTC, datetime
from uuid import UUID

from data_query.application.semantic_models import publish_semantic_model_version
from data_query.application.semantic_models import return_semantic_model_version_for_revision
from data_query.application.semantic_models import retire_semantic_model_version
from data_query.application.semantic_models import submit_semantic_model_version_for_review
from data_query.application.semantic_models import update_semantic_model_draft
from data_query.application.model_validation import create_model_validation_run, get_model_validation_result
from data_query.application.sources import (
    DataQueryManagementError,
    create_data_source,
    update_data_source,
    create_agent_binding,
    create_policy_binding,
    create_semantic_model_draft,
    request_schema_snapshot,
)
from data_query.application.schema_metadata import (
    confirm_snapshot_selection,
    generate_semantic_candidate,
    enrich_semantic_candidate,
    retry_snapshot_object,
    supply_manual_metadata,
)
from data_query.contracts import (
    DataSourceCreate,
    DataSourceUpdate,
    DataSourceConnectionTest,
    DataSourceConnectionTestResult,
    DataSourceCredentialStatus,
    DataSourceDetail,
    DataSourceEndpoint,
    DataSourcePage,
    DataSourceStatusChange,
    DataQueryAuditPage,
    DataQueryAuditView,
    DataSourceView,
    AgentBindingCreate,
    AgentBindingPage,
    AgentBindingStatusChange,
    AgentBindingView,
    PolicyBindingCreate,
    PolicyBindingDetail,
    PolicyBindingPage,
    PolicyBindingStatusChange,
    PolicyBindingView,
    PublishSemanticModelCommand,
    ReturnSemanticModelForRevisionCommand,
    RetireSemanticModelVersionCommand,
    DeleteSemanticModelCommand,
    SubmitSemanticModelReviewCommand,
    SchemaSnapshotReceipt,
    SchemaSnapshotDetail,
    SchemaSnapshotSummary,
    SchemaSnapshotPage,
    SchemaSnapshotObjectView,
    SchemaObjectSelection,
    ManualSchemaDefinition,
    SemanticModelCandidateRequest,
    SemanticModelCandidate,
    SemanticModelGenerationReceipt,
    SemanticModelGenerationView,
    SemanticModelDraftUpdate,
    SemanticModelValidationRequest,
    SemanticModelValidationReceipt,
    SemanticModelValidationResult,
    SemanticModelDraftCreate,
    SemanticModelDetail,
    SemanticModelDraftView,
    SemanticModelPage,
    SemanticModelSearch,
    SemanticModelVersionView,
    SemanticModelView,
    VerifiedQueryPage,
    VerifiedQueryView,
    PromoteVerifiedQueryCommand,
)
from data_query.domain import DataSourceStatus, can_transition
from data_query.persistence import DataQueryUnitOfWork
from data_query.entities import SemanticModelGenerationJobEntity, VerifiedQueryEntity
from data_query.adapters import DatabaseCredentialService


class DataQueryManagementService:
    def __init__(self, *, uow_factory: Callable[[], DataQueryUnitOfWork], credential_service: DatabaseCredentialService, connection_tester, model_config_client=None, model_client=None) -> None:
        self._uow_factory = uow_factory
        self._credential_service = credential_service
        self._connection_tester = connection_tester
        self._model_config_client = model_config_client
        self._model_client = model_client

    async def create_source(self, *, domain_id: int, actor_id: str, command: DataSourceCreate) -> DataSourceView:
        entity = await create_data_source(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id, command=command,
            credential_service=self._credential_service,
        )
        return self._source_view(entity)

    async def update_source(
        self, *, domain_id: int, actor_id: str, data_source_id: UUID,
        command: DataSourceUpdate,
    ) -> DataSourceView:
        entity = await update_data_source(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            data_source_id=data_source_id, command=command,
            credential_service=self._credential_service,
        )
        return self._source_view(entity)

    async def test_source_connection(
        self, *, domain_id: int, command: DataSourceConnectionTest
    ) -> DataSourceConnectionTestResult:
        del domain_id
        return await self._connection_tester(command=command)

    async def list_sources(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> DataSourcePage:
        async with self._uow_factory() as uow:
            assert uow.data_sources
            rows = await uow.data_sources.list_by_domain(
                domain_id=domain_id, after_id=after_id, limit=limit + 1
            )
            await uow.commit()
        visible = rows[:limit]
        return DataSourcePage(
            items=tuple(self._source_view(row) for row in visible),
            next_cursor=rows[limit].data_source_id if len(rows) > limit else None,
        )

    async def get_source(self, *, domain_id: int, data_source_id: UUID) -> DataSourceDetail:
        async with self._uow_factory() as uow:
            assert uow.data_sources and uow.credentials
            row = await uow.data_sources.get_by_id(data_source_id=data_source_id)
            credential = None
            if row is not None and row.domain_id == domain_id:
                credential = await uow.credentials.get_scoped(
                    credential_id=row.credential_id,
                    domain_id=domain_id,
                    data_source_id=row.data_source_id,
                    active_only=True,
                )
            await uow.commit()
        if row is None or row.domain_id != domain_id or credential is None:
            raise DataQueryManagementError("DATA_SOURCE_NOT_FOUND")
        try:
            endpoint = DataSourceEndpoint.model_validate(row.configuration_json)
        except ValueError as exc:
            raise DataQueryManagementError("DATA_SOURCE_CONFIGURATION_INVALID") from exc
        return DataSourceDetail(
            **self._source_view(row).model_dump(), endpoint=endpoint,
            credential=DataSourceCredentialStatus(
                configured=True,
                key_version=credential.key_version,
                updated_at=credential.updated_at,
            ),
            capabilities=row.capabilities_json, error_code=row.error_code,
            updated_at=row.updated_at,
        )

    async def change_source_status(
        self, *, domain_id: int, actor_id: str, data_source_id: UUID,
        command: DataSourceStatusChange,
    ) -> DataSourceView:
        async with self._uow_factory() as uow:
            assert uow.data_sources
            row = await uow.data_sources.get_by_id(data_source_id=data_source_id, lock=True)
            if row is None or row.domain_id != domain_id:
                raise DataQueryManagementError("DATA_SOURCE_NOT_FOUND")
            if int(row.row_version) != command.expected_row_version:
                raise DataQueryManagementError("ROW_VERSION_CONFLICT")
            target = DataSourceStatus(command.status)
            if row.status != target.value and not can_transition(DataSourceStatus(row.status), target):
                raise DataQueryManagementError("DATA_SOURCE_STATUS_TRANSITION_DENIED")
            row.status = target.value
            row.updated_by = actor_id
            await uow.commit()
            return self._source_view(row)

    @staticmethod
    def _source_view(entity) -> DataSourceView:
        return DataSourceView(
            data_source_id=entity.data_source_id,
            display_name=entity.display_name, source_type=entity.source_type,
            status=entity.status, current_version=entity.current_version,
            row_version=int(entity.row_version),
        )

    async def request_snapshot(
        self, *, domain_id: int, data_source_id: UUID, actor_id: str
    ) -> SchemaSnapshotReceipt:
        async with self._uow_factory() as uow:
            assert uow.data_sources
            source = await uow.data_sources.get_by_id(data_source_id=data_source_id)
            await uow.commit()
        if source is None or source.domain_id != domain_id:
            raise DataQueryManagementError("DATA_SOURCE_NOT_FOUND")
        entity = await request_schema_snapshot(
            uow_factory=self._uow_factory, data_source_id=data_source_id, actor_id=actor_id
        )
        return SchemaSnapshotReceipt(
            schema_snapshot_id=entity.schema_snapshot_id,
            data_source_id=entity.data_source_id,
            status=entity.status,
            source_version=entity.source_version,
        )

    async def list_snapshots(self, *, domain_id: int, data_source_id: UUID) -> SchemaSnapshotPage:
        async with self._uow_factory() as uow:
            assert uow.data_sources and uow.schema_snapshots and uow.schema_snapshot_objects
            source = await uow.data_sources.get_by_id(data_source_id=data_source_id)
            if source is None or source.domain_id != domain_id:
                raise DataQueryManagementError("DATA_SOURCE_NOT_FOUND")
            snapshots = await uow.schema_snapshots.list_by_source(data_source_id=data_source_id)
            details = []
            for snapshot in snapshots:
                objects = await uow.schema_snapshot_objects.list_by_snapshot(
                    schema_snapshot_id=snapshot.schema_snapshot_id
                )
                details.append(self._snapshot_summary(snapshot, objects))
            await uow.commit()
        return SchemaSnapshotPage(items=tuple(details))

    async def get_snapshot(self, *, domain_id: int, snapshot_id: UUID) -> SchemaSnapshotDetail:
        async with self._uow_factory() as uow:
            assert uow.data_sources and uow.schema_snapshots and uow.schema_snapshot_objects
            snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
            source = None if snapshot is None else await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
            if snapshot is None or source is None or source.domain_id != domain_id:
                raise DataQueryManagementError("SCHEMA_SNAPSHOT_NOT_FOUND")
            objects = await uow.schema_snapshot_objects.list_by_snapshot(schema_snapshot_id=snapshot_id)
            await uow.commit()
        return self._snapshot_detail(snapshot, objects)

    async def select_snapshot_objects(
        self, *, domain_id: int, actor_id: str, snapshot_id: UUID, command: SchemaObjectSelection,
    ) -> SchemaSnapshotDetail:
        await confirm_snapshot_selection(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            snapshot_id=snapshot_id, object_ids=command.object_ids,
        )
        return await self.get_snapshot(domain_id=domain_id, snapshot_id=snapshot_id)

    async def retry_snapshot_object(
        self, *, domain_id: int, actor_id: str, snapshot_id: UUID, object_id: UUID,
    ) -> SchemaSnapshotDetail:
        await retry_snapshot_object(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            snapshot_id=snapshot_id, object_id=object_id,
        )
        return await self.get_snapshot(domain_id=domain_id, snapshot_id=snapshot_id)

    async def supply_manual_schema(
        self, *, domain_id: int, actor_id: str, snapshot_id: UUID, object_id: UUID,
        command: ManualSchemaDefinition,
    ) -> SchemaSnapshotDetail:
        await supply_manual_metadata(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            snapshot_id=snapshot_id, object_id=object_id, ddl=command.ddl,
        )
        return await self.get_snapshot(domain_id=domain_id, snapshot_id=snapshot_id)

    async def generate_semantic_candidate(
        self, *, domain_id: int, snapshot_id: UUID,
        command: SemanticModelCandidateRequest,
    ) -> SemanticModelCandidate:
        candidate = await generate_semantic_candidate(
            uow_factory=self._uow_factory, domain_id=domain_id,
            snapshot_id=snapshot_id, command=command,
        )
        if command.ai_model_id is None:
            return candidate
        if not command.allow_ai_metadata:
            return candidate.model_copy(update={
                "warnings": tuple(candidate.warnings) + ("未确认向所选模型处理结构元数据，已跳过 AI 增强。",),
            })
        if self._model_config_client is None or self._model_client is None:
            return candidate.model_copy(update={
                "warnings": tuple(candidate.warnings) + ("AI 增强服务未配置，本次保留规则生成结果。",),
            })
        return await enrich_semantic_candidate(
            candidate=candidate, command=command, uow_factory=self._uow_factory,
            model_config_client=self._model_config_client,
            model_client=self._model_client,
        )

    async def generate_model_draft(
        self, *, domain_id: int, actor_id: str, snapshot_id: UUID,
        command: SemanticModelCandidateRequest,
    ) -> SemanticModelGenerationReceipt:
        """持久化生成请求并立即返回；独立 Worker 完成 AI 增强和草稿落库。"""
        if command.ai_model_id is not None and not command.allow_ai_metadata:
            raise DataQueryManagementError("SEMANTIC_MODEL_AI_METADATA_NOT_APPROVED")
        async with self._uow_factory() as uow:
            assert uow.schema_snapshots and uow.data_sources
            assert uow.semantic_model_generation_jobs
            snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
            source = None if snapshot is None else await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
            if snapshot is None or source is None or source.domain_id != domain_id:
                raise DataQueryManagementError("SCHEMA_SNAPSHOT_NOT_FOUND")
            job = SemanticModelGenerationJobEntity(
                domain_id=domain_id,
                schema_snapshot_id=snapshot_id,
                requested_by=actor_id,
                request_json=command.model_dump(mode="json"),
                status="QUEUED",
            )
            await uow.semantic_model_generation_jobs.add(job)
            await uow.commit()
        return SemanticModelGenerationReceipt(generation_job_id=job.generation_job_id, status="QUEUED")

    async def get_model_generation_job(self, *, domain_id: int, generation_job_id: UUID) -> SemanticModelGenerationView:
        async with self._uow_factory() as uow:
            assert uow.semantic_model_generation_jobs and uow.schema_snapshots
            job = await uow.semantic_model_generation_jobs.get_by_id(generation_job_id=generation_job_id)
            snapshot = None if job is None else await uow.schema_snapshots.get_by_id(
                schema_snapshot_id=job.schema_snapshot_id,
            )
            await uow.commit()
        if job is None or snapshot is None or job.domain_id != domain_id:
            raise DataQueryManagementError("SEMANTIC_MODEL_GENERATION_JOB_NOT_FOUND")
        return SemanticModelGenerationView(
            generation_job_id=job.generation_job_id, status=job.status,
            data_source_id=snapshot.data_source_id,
            schema_snapshot_id=job.schema_snapshot_id,
            semantic_model_id=job.semantic_model_id,
            semantic_model_version_id=job.semantic_model_version_id,
            error_code=job.error_code,
        )

    async def update_model_draft(
        self, *, domain_id: int, semantic_model_id: UUID,
        semantic_model_version_id: UUID, command: SemanticModelDraftUpdate,
    ) -> SemanticModelDraftView:
        row_version = await update_semantic_model_draft(
            uow_factory=self._uow_factory, domain_id=domain_id,
            semantic_model_id=semantic_model_id,
            semantic_model_version_id=semantic_model_version_id,
            definition=command.definition,
            expected_row_version=command.expected_row_version,
        )
        async with self._uow_factory() as uow:
            assert uow.semantic_model_versions
            entity = await uow.semantic_model_versions.get_by_id(
                semantic_model_version_id=semantic_model_version_id,
            )
            await uow.commit()
        if entity is None:
            raise DataQueryManagementError("MODEL_VERSION_NOT_FOUND")
        return SemanticModelDraftView(
            semantic_model_id=entity.semantic_model_id,
            semantic_model_version_id=entity.semantic_model_version_id,
            version_no=entity.version_no, status=entity.status,
            row_version=row_version,
        )

    async def create_model_validation(
        self, *, domain_id: int, actor_id: str,
        semantic_model_id: UUID, semantic_model_version_id: UUID,
        command: SemanticModelValidationRequest,
    ) -> SemanticModelValidationReceipt:
        if self._model_config_client is None or self._model_client is None:
            raise DataQueryManagementError("MODEL_VALIDATION_AI_NOT_CONFIGURED")
        return await create_model_validation_run(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            semantic_model_id=semantic_model_id,
            semantic_model_version_id=semantic_model_version_id,
            command=command, model_config_client=self._model_config_client,
            model_client=self._model_client,
        )

    async def get_model_validation(
        self, *, domain_id: int, semantic_model_id: UUID,
        semantic_model_version_id: UUID, run_id: UUID,
    ) -> SemanticModelValidationResult:
        return await get_model_validation_result(
            uow_factory=self._uow_factory, domain_id=domain_id,
            semantic_model_id=semantic_model_id,
            semantic_model_version_id=semantic_model_version_id,
            run_id=run_id,
        )

    @staticmethod
    def _snapshot_summary(snapshot, objects) -> SchemaSnapshotSummary:
        return SchemaSnapshotSummary(
            schema_snapshot_id=snapshot.schema_snapshot_id,
            data_source_id=snapshot.data_source_id, status=snapshot.status,
            source_version=snapshot.source_version,
            created_at=snapshot.created_at,
            discovered_count=len(objects),
            selected_count=sum(1 for item in objects if item.selected),
            succeeded_count=sum(1 for item in objects if item.status in {"READY", "MANUAL"}),
            failed_count=sum(1 for item in objects if item.status == "FAILED"),
            completed_at=snapshot.completed_at,
        )

    @classmethod
    def _snapshot_detail(cls, snapshot, objects) -> SchemaSnapshotDetail:
        def object_view(item) -> SchemaSnapshotObjectView:
            details = item.metadata_json.get("column_details", []) if isinstance(item.metadata_json, dict) else []
            return SchemaSnapshotObjectView(
                schema_snapshot_object_id=item.schema_snapshot_object_id,
                schema_name=item.schema_name, object_name=item.object_name,
                object_type=item.object_type, selected=item.selected,
                status=item.status, attempt_count=item.attempt_count,
                metadata_source=item.metadata_source,
                column_count=len(details) if isinstance(details, list) else 0,
                error_code=item.error_code, error_message=item.error_message,
            )
        return SchemaSnapshotDetail(
            **cls._snapshot_summary(snapshot, objects).model_dump(),
            objects=tuple(object_view(item) for item in objects),
        )

    async def create_model_draft(self, *, domain_id: int, actor_id: str, command: SemanticModelDraftCreate) -> SemanticModelDraftView:
        entity = await create_semantic_model_draft(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id, command=command
        )
        return SemanticModelDraftView(
            semantic_model_id=entity.semantic_model_id,
            semantic_model_version_id=entity.semantic_model_version_id,
            version_no=entity.version_no,
            status=entity.status,
            row_version=int(entity.row_version),
        )

    async def list_semantic_models(self, *, domain_id: int, after_id: UUID | None, limit: int) -> SemanticModelPage:
        async with self._uow_factory() as uow:
            assert uow.semantic_models
            rows = await uow.semantic_models.list_by_domain(domain_id=domain_id, after_id=after_id, limit=limit + 1)
            await uow.commit()
        return SemanticModelPage(
            items=tuple(self._semantic_model_view(item) for item in rows[:limit]),
            next_cursor=rows[limit].semantic_model_id if len(rows) > limit else None,
        )

    async def search_semantic_models(
        self, *, domain_id: int, search: SemanticModelSearch,
    ) -> SemanticModelPage:
        async with self._uow_factory() as uow:
            assert uow.semantic_models
            rows = await uow.semantic_models.search_by_ids(
                domain_id=domain_id,
                semantic_model_ids=search.semantic_model_ids,
                query=search.query.strip() if search.query else None,
                publication_status=search.publication_status,
                after_id=search.cursor,
                limit=search.limit + 1,
            )
            await uow.commit()
        items = rows[:search.limit]
        return SemanticModelPage(
            items=tuple(self._semantic_model_view(item) for item in items),
            next_cursor=items[-1].semantic_model_id if len(rows) > search.limit else None,
        )

    async def get_semantic_model(self, *, domain_id: int, semantic_model_id: UUID) -> SemanticModelDetail:
        async with self._uow_factory() as uow:
            assert uow.semantic_models and uow.semantic_model_versions
            model = await uow.semantic_models.get_by_id(semantic_model_id=semantic_model_id)
            if model is None or model.domain_id != domain_id:
                raise DataQueryManagementError("SEMANTIC_MODEL_NOT_FOUND")
            versions = await uow.semantic_model_versions.list_by_model(semantic_model_id=semantic_model_id)
            await uow.commit()
        return SemanticModelDetail(
            **self._semantic_model_view(model).model_dump(),
            versions=tuple(self._semantic_model_version_view(item) for item in versions),
            updated_at=model.updated_at,
        )

    async def retire_model_version(
        self, *, domain_id: int, actor_id: str, semantic_model_id: UUID,
        semantic_model_version_id: UUID, command: RetireSemanticModelVersionCommand,
    ) -> None:
        await retire_semantic_model_version(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            semantic_model_id=semantic_model_id,
            semantic_model_version_id=semantic_model_version_id,
            expected_row_version=command.expected_row_version,
        )

    async def delete_semantic_model(
        self, *, domain_id: int, semantic_model_id: UUID,
        command: DeleteSemanticModelCommand,
    ) -> None:
        """仅删除已停止使用且没有治理引用的模型，拒绝隐式级联。"""
        async with self._uow_factory() as uow:
            assert (
                uow.semantic_models and uow.semantic_model_versions
                and uow.policy_bindings and uow.agent_bindings and uow.verified_queries
            )
            model = await uow.semantic_models.get_by_id(
                semantic_model_id=semantic_model_id, lock=True,
            )
            if model is None or model.domain_id != domain_id:
                raise DataQueryManagementError("SEMANTIC_MODEL_NOT_FOUND")
            if int(model.row_version) != command.expected_row_version:
                raise DataQueryManagementError("ROW_VERSION_CONFLICT")
            if model.active_version is not None:
                raise DataQueryManagementError("SEMANTIC_MODEL_MUST_BE_RETIRED")
            if await uow.agent_bindings.references_model(
                domain_id=domain_id, semantic_model_id=semantic_model_id,
            ):
                raise DataQueryManagementError("SEMANTIC_MODEL_AGENT_BINDING_EXISTS")
            if await uow.policy_bindings.references_model(
                domain_id=domain_id, semantic_model_id=semantic_model_id,
            ):
                raise DataQueryManagementError("SEMANTIC_MODEL_POLICY_BINDING_EXISTS")
            if await uow.verified_queries.references_model(semantic_model_id=semantic_model_id):
                raise DataQueryManagementError("SEMANTIC_MODEL_VERIFIED_QUERY_EXISTS")
            await uow.semantic_model_versions.delete_by_model(semantic_model_id=semantic_model_id)
            await uow.semantic_models.delete(model)
            await uow.commit()

    async def list_verified_queries(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> VerifiedQueryPage:
        async with self._uow_factory() as uow:
            assert uow.verified_queries
            rows = await uow.verified_queries.list_by_domain(
                domain_id=domain_id, after_id=after_id, limit=limit + 1
            )
            await uow.commit()
        return VerifiedQueryPage(
            items=tuple(self._verified_query_view(item) for item in rows[:limit]),
            next_cursor=rows[limit].verified_query_id if len(rows) > limit else None,
        )

    async def promote_verified_query(self, *, domain_id: int, actor_id: str, command: PromoteVerifiedQueryCommand) -> VerifiedQueryView:
        async with self._uow_factory() as uow:
            assert uow.runs and uow.semantic_model_versions and uow.verified_queries
            run = await uow.runs.get_by_id(data_query_run_id=command.data_query_run_id)
            if run is None or run.domain_id != domain_id or run.status not in {"COMPLETED", "COMPLETED_EMPTY"}:
                raise DataQueryManagementError("RUN_NOT_VERIFIABLE")
            snapshot = run.semantic_model_snapshot_json or {}
            model_id = snapshot.get("model_id"); version_no = snapshot.get("version")
            if not isinstance(model_id, str) or not isinstance(version_no, int) or not isinstance(run.plan_snapshot_json, dict):
                raise DataQueryManagementError("RUN_SNAPSHOT_INVALID")
            version = await uow.semantic_model_versions.get_by_model_version(semantic_model_id=UUID(model_id), version_no=version_no)
            if version is None:
                raise DataQueryManagementError("MODEL_VERSION_NOT_FOUND")
            question_hash = hashlib.sha256(run.standalone_query.encode("utf-8")).hexdigest()
            existing = await uow.verified_queries.get_by_question_hash(semantic_model_version_id=version.semantic_model_version_id, question_hash=question_hash)
            if existing is not None:
                await uow.commit(); return self._verified_query_view(existing)
            entity = VerifiedQueryEntity(semantic_model_version_id=version.semantic_model_version_id, question=run.standalone_query, question_hash=question_hash, query_plan_json=run.plan_snapshot_json, assertion_json=command.assertion, status="VERIFIED", verified_by=actor_id, verified_at=datetime.now(UTC))
            await uow.verified_queries.add(entity); await uow.commit()
            return self._verified_query_view(entity)

    async def list_audits(
        self, *, domain_id: int, after_id: UUID | None, limit: int
    ) -> DataQueryAuditPage:
        async with self._uow_factory() as uow:
            assert uow.audits
            rows = await uow.audits.list_by_domain(
                domain_id=domain_id, after_id=after_id, limit=limit + 1
            )
            await uow.commit()
        return DataQueryAuditPage(
            items=tuple(self._audit_view(item) for item in rows[:limit]),
            next_cursor=rows[limit].audit_id if len(rows) > limit else None,
        )

    async def publish_model_version(self, *, domain_id: int, actor_id: str, command: PublishSemanticModelCommand) -> None:
        await publish_semantic_model_version(
            uow_factory=self._uow_factory,
            domain_id=domain_id,
            semantic_model_id=command.semantic_model_id,
            semantic_model_version_id=command.semantic_model_version_id,
            schema_snapshot_id=command.schema_snapshot_id,
            expected_row_version=command.expected_row_version,
            actor_id=actor_id,
        )

    async def submit_model_review(
        self, *, domain_id: int, actor_id: str, semantic_model_id: UUID,
        semantic_model_version_id: UUID, command: SubmitSemanticModelReviewCommand,
    ) -> None:
        await submit_semantic_model_version_for_review(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            semantic_model_id=semantic_model_id, semantic_model_version_id=semantic_model_version_id,
            expected_row_version=command.expected_row_version,
        )

    async def return_model_for_revision(
        self, *, domain_id: int, actor_id: str, semantic_model_id: UUID,
        semantic_model_version_id: UUID, command: ReturnSemanticModelForRevisionCommand,
    ) -> None:
        await return_semantic_model_version_for_revision(
            uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id,
            semantic_model_id=semantic_model_id, semantic_model_version_id=semantic_model_version_id,
            review_comment=command.review_comment, expected_row_version=command.expected_row_version,
        )

    async def create_policy(self, *, domain_id: int, actor_id: str, command: PolicyBindingCreate) -> PolicyBindingView:
        entity = await create_policy_binding(uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id, command=command)
        return PolicyBindingView(policy_binding_id=entity.policy_binding_id, status=entity.status, row_version=int(entity.row_version))

    async def list_policies(self, *, domain_id: int, after_id: UUID | None, limit: int) -> PolicyBindingPage:
        async with self._uow_factory() as uow:
            assert uow.policy_bindings
            rows = await uow.policy_bindings.list_by_domain(domain_id=domain_id, after_id=after_id, limit=limit + 1)
            await uow.commit()
        return PolicyBindingPage(
            items=tuple(self._policy_detail(item) for item in rows[:limit]),
            next_cursor=rows[limit].policy_binding_id if len(rows) > limit else None,
        )

    async def get_policy(self, *, domain_id: int, policy_binding_id: UUID) -> PolicyBindingDetail:
        async with self._uow_factory() as uow:
            assert uow.policy_bindings
            row = await uow.policy_bindings.get_by_id(policy_binding_id=policy_binding_id, lock=True)
            await uow.commit()
        if row is None or row.domain_id != domain_id:
            raise DataQueryManagementError("POLICY_NOT_FOUND")
        return self._policy_detail(row)

    async def change_policy_status(
        self, *, domain_id: int, actor_id: str, policy_binding_id: UUID,
        command: PolicyBindingStatusChange,
    ) -> PolicyBindingView:
        async with self._uow_factory() as uow:
            assert uow.policy_bindings
            row = await uow.policy_bindings.get_by_id(policy_binding_id=policy_binding_id, lock=True)
            if row is None or row.domain_id != domain_id:
                raise DataQueryManagementError("POLICY_NOT_FOUND")
            if int(row.row_version) != command.expected_row_version:
                raise DataQueryManagementError("ROW_VERSION_CONFLICT")
            row.status = command.status
            row.updated_by = actor_id
            await uow.commit()
            return self._policy_view(row)

    async def create_agent_binding(self, *, domain_id: int, actor_id: str, command: AgentBindingCreate) -> AgentBindingView:
        entity = await create_agent_binding(uow_factory=self._uow_factory, domain_id=domain_id, actor_id=actor_id, command=command)
        return AgentBindingView(
            agent_binding_id=entity.agent_binding_id, agent_id=entity.agent_id,
            semantic_model_id=entity.semantic_model_id, policy_binding_id=entity.policy_binding_id,
            status=entity.status, row_version=int(entity.row_version),
        )

    async def list_agent_bindings(self, *, domain_id: int, after_id: UUID | None, limit: int) -> AgentBindingPage:
        async with self._uow_factory() as uow:
            assert uow.agent_bindings
            rows = await uow.agent_bindings.list_by_domain(domain_id=domain_id, after_id=after_id, limit=limit + 1)
            await uow.commit()
        return AgentBindingPage(
            items=tuple(self._agent_binding_view(item) for item in rows[:limit]),
            next_cursor=rows[limit].agent_binding_id if len(rows) > limit else None,
        )

    async def change_agent_binding_status(
        self, *, domain_id: int, actor_id: str, agent_binding_id: UUID,
        command: AgentBindingStatusChange,
    ) -> AgentBindingView:
        async with self._uow_factory() as uow:
            assert uow.agent_bindings
            row = await uow.agent_bindings.get_by_id(agent_binding_id=agent_binding_id, lock=True)
            if row is None or row.domain_id != domain_id:
                raise DataQueryManagementError("AGENT_BINDING_NOT_FOUND")
            if int(row.row_version) != command.expected_row_version:
                raise DataQueryManagementError("ROW_VERSION_CONFLICT")
            row.status = command.status
            row.updated_by = actor_id
            await uow.commit()
            return self._agent_binding_view(row)

    @staticmethod
    def _policy_view(entity) -> PolicyBindingView:
        return PolicyBindingView(policy_binding_id=entity.policy_binding_id, status=entity.status, row_version=int(entity.row_version))

    @classmethod
    def _policy_detail(cls, entity) -> PolicyBindingDetail:
        budget = entity.policy_json.get("budget") if isinstance(entity.policy_json, dict) else None
        if not isinstance(budget, dict):
            raise DataQueryManagementError("POLICY_INVALID")
        return PolicyBindingDetail(
            **cls._policy_view(entity).model_dump(),
            semantic_model_ids=tuple(
                UUID(item) for item in entity.semantic_model_ids_json
            ),
            budget=budget,
            updated_at=entity.updated_at,
        )

    @staticmethod
    def _agent_binding_view(entity) -> AgentBindingView:
        return AgentBindingView(
            agent_binding_id=entity.agent_binding_id, agent_id=entity.agent_id,
            semantic_model_id=entity.semantic_model_id, policy_binding_id=entity.policy_binding_id,
            status=entity.status, row_version=int(entity.row_version),
        )

    @staticmethod
    def _semantic_model_view(entity) -> SemanticModelView:
        return SemanticModelView(
            semantic_model_id=entity.semantic_model_id,
            display_name=entity.display_name, description=entity.description,
            active_version=entity.active_version, row_version=int(entity.row_version),
        )

    @staticmethod
    def _semantic_model_version_view(entity) -> SemanticModelVersionView:
        return SemanticModelVersionView(
            semantic_model_version_id=entity.semantic_model_version_id,
            semantic_model_id=entity.semantic_model_id, version_no=entity.version_no,
            data_source_id=entity.data_source_id, schema_snapshot_id=entity.schema_snapshot_id,
            status=entity.status, row_version=int(entity.row_version),
            review_comment=entity.review_comment,
            definition=entity.definition_json,
        )

    @staticmethod
    def _verified_query_view(entity) -> VerifiedQueryView:
        return VerifiedQueryView(
            verified_query_id=entity.verified_query_id,
            semantic_model_version_id=entity.semantic_model_version_id,
            question_summary=entity.question[:256], status=entity.status,
            verified_by=entity.verified_by, verified_at=entity.verified_at,
            row_version=int(entity.row_version),
        )

    @staticmethod
    def _audit_view(entity) -> DataQueryAuditView:
        return DataQueryAuditView(
            audit_id=entity.audit_id, data_query_run_id=entity.data_query_run_id,
            actor_id=entity.actor_id, trace_id=entity.trace_id,
            action=entity.action, created_at=entity.created_at,
        )
