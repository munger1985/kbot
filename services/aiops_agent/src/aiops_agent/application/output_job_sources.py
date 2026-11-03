"""输出任务读取与重试的来源边界；普通状态读取不泄露冻结正文。"""

from uuid import UUID

from pydantic import ValidationError

from aiops_agent.application.case_closure import (
    case_from_frozen_input, rejected_closure, validate_closure_sources,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput
from platform_core.identity import uuid7


class OutputJobSourceValidator:
    def __init__(self, *, plan_capture_reader=None):
        self._plan_capture_reader = plan_capture_reader

    async def read(self, uow, job):
        # 发起人／Domain／Tenant 由 OutputJobService 与可信 API 上下文核验。
        # 状态回执不含正文、输入或 KC 定位。来源失效后仍可查看自己的失败记录，
        # 产物正文和深链目标须使用对应业务接口的现时授权。
        return None

    async def retry(self, uow, job):
        try:
            frozen = FrozenOutputInput.model_validate(job.input_snapshot_json)
            if (canonical_hash(frozen.model_dump(mode="json", exclude={"frozen_at"})) != job.input_hash
                or frozen.source.model_dump(mode="json") != job.source_json):
                raise ValueError("冻结输入已变化")
        except (TypeError, ValueError, ValidationError) as exc:
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_INPUT_INVALID",
                message="任务冻结来源损坏，不能重试", status_code=409) from exc
        if job.job_type == "REPORT_GENERATION":
            from aiops_agent.application.output_jobs import OutputJobScope
            from aiops_agent.application.report_output import validate_report_input
            await validate_report_input(uow=uow, frozen=frozen,
                scope=OutputJobScope(domain_id=job.domain_id, actor_id=job.actor_id,
                    tenant_id=getattr(job, "tenant_id", None), trace_id=job.trace_id),
                plan_capture_reader=self._plan_capture_reader)
            return
        if job.job_type == "CASE_EXTRACTION" and frozen.source.kind == "CONVERSATION":
            ref = job.processing_ref_json or {}
            try:
                prepared = case_from_frozen_input(frozen,
                    closure_artifact_id=UUID(ref["closure_artifact_id"]) if ref.get("closure_artifact_id") else uuid7())
            except (AttributeError, KeyError, TypeError, ValueError) as exc:
                raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_INPUT_INVALID",
                    message="任务恢复定位损坏，不能重试", status_code=409) from exc
            if prepared.closure.closed_by != str(job.actor_id):
                raise rejected_closure()
            run = await uow.runs.get_run_scoped(
                ops_run_id=prepared.anchor_run_id, domain_id=job.domain_id)
            if run is None or not await validate_closure_sources(
                uow=uow, domain_id=job.domain_id, run=run, artifact=prepared.artifact(),
                plan_capture_reader=self._plan_capture_reader):
                raise rejected_closure()
            return
        if job.job_type == "CASE_EXTRACTION" and frozen.source.kind == "REPORT":
            from aiops_agent.application.report_case_output import (
                report_case_from_frozen_input, validate_prepared_report_case,
            )
            prepared = report_case_from_frozen_input(frozen)
            if prepared.actor_id != str(job.actor_id):
                raise rejected_closure()
            await validate_prepared_report_case(uow=uow, domain_id=job.domain_id, prepared=prepared,
                plan_capture_reader=self._plan_capture_reader)
            return
        raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_KIND_UNSUPPORTED",
            message="该来源的输出重试处理尚未配置", status_code=409)
