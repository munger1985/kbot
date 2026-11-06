"""输出任务的事务终态：复核确切产物，与通知 Outbox 一同提交。"""

from uuid import UUID

from pydantic import ValidationError

from aiops_agent.application.case_closure import (
    case_from_frozen_input, register_conversation_case_intake,
)
from aiops_agent.application.case_intake_recovery import CaseIntakeRecovery
from aiops_agent.application.case_source_validation import validate_case_effect, validate_case_sources
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.report_case_output import (
    register_report_case_intake, report_case_from_frozen_input, validate_report_case_result,
)
from aiops_agent.application.knowledge_withdrawal import assert_case_not_withdrawn, queue_case_withdrawals
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput, OutputJobResult


def result_changed():
    return AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_RESULT_CHANGED",
        message="输出产物或来源在完成前发生变化", status_code=409, retryable=True)


class OutputJobFinalizer:
    def __init__(self, *, plan_capture_reader=None, runtime_service=None):
        self._plan_capture_reader = plan_capture_reader
        self._runtime_service = runtime_service

    async def succeeded(self, *, uow, job, execution, authority, material, now):
        if job.job_type == "REPORT_GENERATION":
            if self._runtime_service is None:
                raise AIOpsApplicationError(code="AIOPS_REPORT_OUTPUT_NOT_CONFIGURED",
                    message="报告后台处理依赖未配置", status_code=503)
            from aiops_agent.application.report_output import finalize_report_output
            return await finalize_report_output(runtime_service=self._runtime_service,
                uow=uow, job=job, execution=execution, material=material, now=now,
                plan_capture_reader=self._plan_capture_reader)
        if job.job_type != "CASE_EXTRACTION":
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_KIND_UNSUPPORTED",
                message="该任务类型的事务完成处理尚不可用", status_code=409)
        result = OutputJobResult.model_validate(material["case_result"])
        if result.report_id is not None or result.asset_id is None or result.asset_version_id is None:
            raise result_changed()
        version = await uow.operations_knowledge.get_version(domain_id=job.domain_id,
            asset_version_id=result.asset_version_id, for_update=True)
        if version is None:
            raise result_changed()
        asset = await uow.operations_knowledge.get_asset(domain_id=job.domain_id,
            asset_id=result.asset_id, for_update=True)
        profile = version.profile_json or {}
        source = profile.get("source") or {}
        frozen = execution.frozen
        if (asset is None or version.asset_id != result.asset_id or asset.asset_kind != "DIAGNOSIS_CASE"
            or getattr(asset, "tenant_id", None) != execution.scope.tenant_id
            or int(asset.security_level) < max(row.security_level for row in frozen.artifacts)
            or version.status not in {"DRAFT", "REVIEW_REQUIRED", "PUBLISHED"}
            or (version.status == "PUBLISHED" and asset.current_version_id != version.asset_version_id)
            or (frozen.source.kind == "CONVERSATION" and (
                source.get("request_hash") != material["request_hash"]
                or source.get("source_fingerprint") != frozen.source_fingerprint))
            or int(version.row_version) != material["version_row_version"]
            or int(asset.row_version) != material["asset_row_version"]):
            raise result_changed()
        if frozen.source.kind == "REPORT":
            prepared = report_case_from_frozen_input(frozen)
            if prepared.actor_id != str(job.actor_id) or prepared.request_hash != material["request_hash"]:
                raise result_changed()
            await validate_report_case_result(uow=uow, domain_id=job.domain_id, prepared=prepared,
                asset_id=result.asset_id, version_id=result.asset_version_id,
                plan_capture_reader=self._plan_capture_reader)
        elif frozen.source.kind != "CONVERSATION":
            raise result_changed()
        relations = await uow.operations_knowledge.get_version_relations(
            asset_version_id=version.asset_version_id)
        indexes = relations["indexes"]
        stamp = material["index_stamp"]
        manifest = profile.get("knowledge_projection") or {}
        if (len(indexes) != 1 or indexes[0].index_status != "READY"
            or any(str(getattr(indexes[0], key)) != stamp[key]
                for key in ("document_id", "document_version_id", "ingestion_job_id", "parse_run_id", "index_generation_id"))
            or manifest.get("body_sha256") != stamp["body_sha256"]):
            raise result_changed()
        await assert_case_not_withdrawn(uow=uow, indexes=indexes)
        if (not await validate_case_sources(uow=uow, domain_id=job.domain_id,
                version_id=version.asset_version_id, plan_capture_reader=self._plan_capture_reader)
            or not await validate_case_effect(uow=uow, domain_id=job.domain_id, profile=profile)):
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_SOURCE_INVALID",
                message="案例完成前来源已失效", status_code=409)
        return result

    async def notify_succeeded(self, *, uow, job, result, now):
        """任务成功已刷新后再冻结通知来源版本，仍与产物共用一个事务。"""
        if job.status != "SUCCEEDED":
            raise result_changed()
        if job.job_type == "CASE_EXTRACTION":
            await uow.platform_notifications.emit_output_job_event(
                job=job, event_type="aiops.case.draft_ready", result=result, now=now)
            return
        report = await uow.inspections.get_report_scoped(
            domain_id=job.domain_id, report_id=result.report_id)
        if report is None or str(report.report_version) != result.report_version:
            raise result_changed()
        run = await uow.runs.get_run(ops_run_id=report.ops_run_id)
        if (run is None or run.domain_id != job.domain_id
            or getattr(run, "tenant_id", None) != getattr(job, "tenant_id", None)):
            raise result_changed()
        await uow.platform_notifications.emit_report_ready(
            run=run, report=report, actor_id=job.actor_id, output_job=job, now=now)

    async def failed(self, *, uow, job, execution, now):
        """已受理但未登记的案例也须恢复确切 Registry，非瞬时失败才撤回。"""
        ref = job.processing_ref_json or {}
        if job.job_type == "CASE_EXTRACTION" and ref.get("intake"):
            try:
                frozen = FrozenOutputInput.model_validate(job.input_snapshot_json)
                if canonical_hash(frozen.model_dump(mode="json", exclude={"frozen_at"})) != job.input_hash:
                    raise ValueError("冻结输入封印不一致")
                recovery = CaseIntakeRecovery.model_validate(ref["intake"])
                if frozen.source.kind == "REPORT":
                    prepared = report_case_from_frozen_input(frozen)
                    register_intake = register_report_case_intake
                elif frozen.source.kind == "CONVERSATION":
                    prepared = case_from_frozen_input(frozen,
                        closure_artifact_id=UUID(ref["closure_artifact_id"]))
                    register_intake = register_conversation_case_intake
                else:
                    raise ValueError("任务案例来源类型无效")
            except (KeyError, TypeError, ValueError, ValidationError, AIOpsApplicationError):
                # 损坏封套不能授权创建 Registry；原恢复材料保留，失败通知仍须可见。
                recovery = None
            if recovery is not None and recovery.ingestion_registered:
                entity_scope = {"tenant_id": execution.scope.tenant_id} if execution.scope.tenant_id else {}
                exact = await uow.operations_knowledge.find_case_by_intake_revision(
                    domain_id=job.domain_id, revision_id=recovery.document_version_id)
                if exact is None:
                    raise result_changed()
                asset, version = exact
                # 登记幂等重放可能返回未加锁的既有版本。与人工发布使用相同锁顺序，
                # 必须重新锁定后判定是否允许清理，不能撤掉并发完成发布的版本。
                version = await uow.operations_knowledge.get_version(
                    domain_id=job.domain_id, asset_version_id=version.asset_version_id, for_update=True)
                if version is None:
                    raise result_changed()
                asset = await uow.operations_knowledge.get_asset(
                    domain_id=job.domain_id, asset_id=version.asset_id, for_update=True)
                if asset is None or getattr(asset, "tenant_id", None) != execution.scope.tenant_id:
                    raise result_changed()
                # 已发布的幂等重放结果不属于本次失败新产物，不能撤掉既有知识。
                if (not job.retryable and version.status not in {"PUBLISHED", "RETIRED"}
                    and asset.current_version_id != version.asset_version_id):
                    version.status = "REVIEW_REQUIRED"
                    if asset.current_version_id is None:
                        asset.status = "REVIEW_REQUIRED"
                    await queue_case_withdrawals(uow=uow, asset=asset, version=version)
        await uow.flush()
        await uow.platform_notifications.emit_output_job_event(
            job=job, event_type="aiops.output_job.failed", now=now)
