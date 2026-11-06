"""会话与报告案例输出的冻结输入处理，复用既有脱敏和 KC 对账。"""

from copy import deepcopy
from dataclasses import dataclass
from uuid import UUID

from aiops_agent.application.case_closure import (
    case_from_frozen_input, execute_prepared_conversation_case,
    freeze_conversation_case_input, prepare_conversation_case, rejected_closure,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.report_case_output import (
    execute_prepared_report_case, report_case_from_frozen_input,
    validate_prepared_report_case, validate_report_case_result,
)
from aiops_agent.application.case_knowledge_projection import case_projection_is_current
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from aiops_agent.workers.output_jobs import OutputAdvance
from platform_core.contracts.aiops.output_jobs import OutputJobSource
from platform_core.identity import uuid7


@dataclass(frozen=True)
class CaseExecutionOptions:
    context: object
    entity_scope: dict
    kc_scope: dict
    model_catalog: object
    model_scope: dict
    technical_name_field: str


async def submit_conversation_case(*, knowledge_service, jobs, scope, conversation_id,
    command, idempotency_key):
    """受理只读现存原件，不调用模型、KC 或现场取证。"""
    async def freeze(uow, root):
        if str(root.created_by) != str(scope.actor_id):
            raise rejected_closure()
        prepared = await prepare_conversation_case(uow=uow, domain_id=scope.domain_id,
            conversation_id=conversation_id, actor_id=scope.actor_id, command=command,
            plan_capture_reader=knowledge_service._plan_capture_reader)
        return await freeze_conversation_case_input(prepared=prepared, uow=uow, domain_id=scope.domain_id)

    return await jobs.submit(scope=scope, job_type="CASE_EXTRACTION",
        source=OutputJobSource(kind="CONVERSATION", source_id=conversation_id),
        idempotency_key=idempotency_key, freeze=freeze)


class CaseOutputProcessor:
    """每次执行都核对当前来源；受理、入库、索引和人工发布是不同阶段。"""

    def __init__(self, *, knowledge_service, options_factory):
        self._knowledge = knowledge_service
        self._options_factory = options_factory

    async def advance(self, execution, *, authority, checkpoint, record_receipt):
        if execution.job_type != "CASE_EXTRACTION" or execution.frozen.source.kind not in {"CONVERSATION", "REPORT"}:
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_KIND_UNSUPPORTED",
                message="案例处理器不能处理此任务类型", status_code=409)
        ref = deepcopy(execution.processing_ref)
        options = self._options_factory(execution, authority)
        if str(getattr(options.context, "principal_kind", "")).upper() != "SERVICE":
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_EXECUTOR_IDENTITY_INVALID",
                message="输出任务必须以服务身份执行，不能伪造登录用户", status_code=403)
        report_source = execution.frozen.source.kind == "REPORT"
        if report_source:
            prepared = report_case_from_frozen_input(execution.frozen)
            execute = execute_prepared_report_case
        else:
            if "closure_artifact_id" not in ref:
                ref["closure_artifact_id"] = str(uuid7())
                await checkpoint(stage="REDACTING", processing_ref=ref)
            prepared = case_from_frozen_input(execution.frozen,
                closure_artifact_id=UUID(ref["closure_artifact_id"]))
            execute = execute_prepared_conversation_case
        if prepared.actor_id != str(execution.scope.actor_id):
            raise rejected_closure()

        async def before_external():
            # checkpoint 重新查询当前授权并检查租约；不能把领取快照用于整个模型阶段。
            stage = "INDEXING" if "case_result" in ref else "INGESTING" if "intake" in ref else "REDACTING"
            await checkpoint(stage=stage, processing_ref=ref)
            async with self._knowledge._uow_factory() as uow:
                agents = set()
                for source in execution.frozen.artifacts:
                    run = await uow.runs.get_run_scoped(
                        ops_run_id=source.run_id, domain_id=execution.scope.domain_id)
                    if run is None:
                        raise rejected_closure()
                    agents.add(str(run.agent_id))
                    agent = await uow.agents.get(domain_id=execution.scope.domain_id, agent_id=run.agent_id)
                    if agent is None or agent.status != "ACTIVE" or agent.current_version_id is None:
                        raise rejected_closure()
                if agents != {str(value) for value in authority.request.source_agent_ids}:
                    raise rejected_closure()
                if report_source:
                    await validate_prepared_report_case(uow=uow, domain_id=execution.scope.domain_id,
                        prepared=prepared, plan_capture_reader=self._knowledge._plan_capture_reader)

        async def save_projection(recovery):
            ref["intake"] = recovery.model_dump(mode="json")
            await checkpoint(stage="INGESTING", processing_ref=ref)

        async def save_receipt(recovery):
            if recovery is None:
                ref.pop("intake", None)
            else:
                ref["intake"] = recovery.model_dump(mode="json")
            await record_receipt(processing_ref=ref)

        if "case_result" not in ref:
            result = await execute(self._knowledge,
                domain_id=execution.scope.domain_id, prepared=prepared,
                context=options.context, entity_scope=options.entity_scope, kc_scope=options.kc_scope,
                model_catalog=options.model_catalog, model_scope=options.model_scope,
                technical_name_field=options.technical_name_field, actor_id=execution.scope.actor_id,
                recovery=ref.get("intake"), checkpoint=save_projection, record_receipt=save_receipt,
                source_preflight_hook=before_external)
            ref["case_result"] = {"asset_id": result["asset"]["asset_id"],
                "asset_version_id": result["version"]["asset_version_id"]}
            await checkpoint(stage="INDEXING", processing_ref=ref)
        version_id = UUID(ref["case_result"]["asset_version_id"])
        await before_external()
        if not await self._knowledge._case_index_source_fence(
            domain_id=execution.scope.domain_id, version_id=version_id, actor=execution.scope.actor_id):
            raise rejected_closure()
        detail = await self._knowledge.reconcile_version(
            domain_id=execution.scope.domain_id, asset_version_id=version_id,
            context=options.context, actor_id=execution.scope.actor_id,
            **({"tenant_id": execution.scope.tenant_id} if execution.scope.tenant_id is not None else {}))
        if (str(detail["asset"]["asset_id"]) != str(ref["case_result"]["asset_id"])
            or not report_source and (detail["version"].get("profile") or {}).get("source", {}).get("request_hash") != prepared.request_hash):
            raise rejected_closure()
        if report_source:
            async with self._knowledge._uow_factory() as uow:
                await validate_report_case_result(uow=uow, domain_id=execution.scope.domain_id,
                    prepared=prepared, asset_id=UUID(ref["case_result"]["asset_id"]), version_id=version_id,
                    plan_capture_reader=self._knowledge._plan_capture_reader)
        status = detail["version"]["status"]
        indexes = detail.get("indexes") or ()
        if status in {"PROCESSING", "TAGGING"}:
            return OutputAdvance(stage="INDEXING", processing_ref=ref)
        if (status not in {"DRAFT", "REVIEW_REQUIRED", "PUBLISHED"}
            or not indexes or any(row["status"] != "READY" for row in indexes)):
            raise AIOpsApplicationError(code="AIOPS_CASE_OUTPUT_INDEX_FAILED",
                message="案例索引未形成可审阅草稿", status_code=409)
        profile = DiagnosisCaseProfile.model_validate(detail["version"]["profile"])
        manifest = profile.knowledge_projection
        if manifest is None or not case_projection_is_current(title=manifest.title, profile=profile):
            raise rejected_closure()
        index = indexes[0]
        access = self._knowledge._documents.access(options.context, actor_id=execution.scope.actor_id)
        record = await self._knowledge._documents.inspect(access, UUID(index["document_version_id"]))
        if (len(indexes) != 1 or record["processing_status"] != "PARSED"
            or record["source_hash"] != manifest.body_sha256
            or record["parse_run_id"] != index["parse_run_id"]
            or record["generation_id"] != index["index_generation_id"]
            or record["status"] == "WITHDRAWN"):
            raise AIOpsApplicationError(code="AIOPS_CASE_OUTPUT_INDEX_FAILED",
                message="案例正文尚未形成可引用的确切索引", status_code=409)
        if not await self._knowledge._case_index_source_fence(
            domain_id=execution.scope.domain_id, version_id=version_id, actor=execution.scope.actor_id):
            raise rejected_closure()
        return OutputAdvance(stage="FINALIZING", processing_ref=ref,
            finalization={"case_result": ref["case_result"], "request_hash": prepared.request_hash,
                "version_row_version": int(detail["version"]["row_version"]),
                "asset_row_version": int(detail["asset"]["row_version"]),
                "index_stamp": {**{key: index[key] for key in
                    ("document_id", "document_version_id", "ingestion_job_id", "parse_run_id", "index_generation_id")},
                    "body_sha256": manifest.body_sha256}})
