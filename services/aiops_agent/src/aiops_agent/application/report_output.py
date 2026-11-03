"""正式报告的持久化异步受理、冻结来源重放和事务内装配。"""

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, ValidationError

from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found
from aiops_agent.application.report_preparation import PreparedReport, prepare_conversation_report, prepare_user_report
from aiops_agent.application.reporting import ReportTemplate
from aiops_agent.application.report_lifecycle import report_input_hash
from aiops_agent.application.report_source_revision import report_source_is_usable
from aiops_agent.application.source_authorization import source_scope_is_current
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.output_jobs import OutputAdvance
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput, OutputJobSource, OutputJobResult, OutputSourceArtifact
from platform_core.contracts.aiops.types import AIOpsContract, UUIDv7, UtcDatetime, Sha256Digest


class ReportRunStamp(AIOpsContract):
    run_id: UUIDv7
    target_id: UUIDv7
    agent_id: UUIDv7
    plan_hash: Sha256Digest
    target_version: int = Field(ge=1)
    final_artifact_id: UUIDv7 | None = None
    trigger_type: str
    status: str


class FrozenReportMaterial(AIOpsContract):
    run_id: UUIDv7
    task_id: UUIDv7
    source_artifact_id: UUIDv7
    source: dict
    runs: tuple[ReportRunStamp, ...] = Field(min_length=1, max_length=512)
    source_reports: dict[str, UUIDv7] = Field(default_factory=dict)
    period_kind: Literal["AD_HOC", "DAILY", "MONTHLY", "QUARTERLY", "ANNUAL"]
    period_start: UtcDatetime
    period_end: UtcDatetime
    template_version_id: UUIDv7 | None = None


def invalid_report_source():
    return AIOpsApplicationError(code="AIOPS_REPORT_OUTPUT_SOURCE_INVALID",
        message="报告冻结来源、权限或引用已变化，请核验后重新提交", status_code=409)


async def _current_run(uow, *, scope, run_id):
    run = await uow.runs.get_run_scoped(ops_run_id=run_id, domain_id=scope.domain_id)
    if (run is None or run.domain_id != scope.domain_id or getattr(run, "tenant_id", None) != scope.tenant_id
        or run.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}):
        raise invalid_report_source()
    target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=scope.domain_id)
    agent = await uow.agents.get(domain_id=scope.domain_id, agent_id=run.agent_id)
    if (target is None or target.status != "ENABLED"
        or agent is None or agent.status != "ACTIVE" or agent.current_version_id is None
        or not await uow.agents.version_has_target(agent_version_id=agent.current_version_id, target_id=run.target_id)
        or not await source_scope_is_current(uow=uow, run=run, domain_id=scope.domain_id)):
        raise invalid_report_source()
    frozen_sources = (run.plan_snapshot_json or {}).get("capability_snapshot", {}).get("source_snapshots") or ()
    current_sources = await uow.agents.version_source_ids(
        agent_version_id=agent.current_version_id, target_id=run.target_id)
    if not {str(row["source_id"]) for row in frozen_sources} <= {str(value) for value in current_sources}:
        raise invalid_report_source()
    return run, target


async def freeze_report_input(*, prepared, uow, scope, source, template_version_id=None,
    plan_capture_reader=None):
    """所有来源清单与源报告都在受理事务内定位，不借新取证补齐缺口。"""
    artifacts = {(prepared.run.ops_run_id, prepared.source_artifact.artifact_id): prepared.source_artifact}
    for run_id, artifact in prepared.extra_artifacts:
        artifacts[(run_id, artifact.artifact_id)] = artifact
    for reference in prepared.source.get("evidence_refs") or ():
        if not isinstance(reference, dict) or not reference.get("artifact_id"):
            continue
        try:
            artifact_id = UUID(str(reference["artifact_id"]))
        except (TypeError, ValueError) as exc:
            raise invalid_report_source() from exc
        artifact = await uow.runs.get_artifact(artifact_id=artifact_id)
        if (artifact is None
            or reference.get("source_run_id") and str(artifact.ops_run_id) != str(reference["source_run_id"])
            or reference.get("content_hash") and artifact.content_hash != reference["content_hash"]):
            raise invalid_report_source()
        artifacts[(artifact.ops_run_id, artifact.artifact_id)] = artifact
    if len(artifacts) > 512:
        raise AIOpsApplicationError(code="AIOPS_REPORT_OUTPUT_SOURCE_LIMIT",
            message="报告来源超过单任务处理范围，请缩小报告范围", status_code=422)
    stamps, refs, source_reports, observed = {}, [], {}, []
    for (run_id, artifact_id), artifact in sorted(artifacts.items(), key=lambda row: (str(row[0][0]), str(row[0][1]))):
        run, target = await _current_run(uow, scope=scope, run_id=run_id)
        if (str(artifact.ops_run_id) != str(run_id)
            or canonical_hash(artifact.payload_json) != artifact.content_hash):
            raise invalid_report_source()
        stamps[str(run_id)] = ReportRunStamp(run_id=run_id, target_id=run.target_id,
            agent_id=run.agent_id, plan_hash=canonical_hash(run.plan_snapshot_json or {}),
            target_version=int(target.row_version), final_artifact_id=run.final_artifact_id,
            trigger_type=run.trigger_type, status=run.status)
        turn_id = None
        if source.kind == "CONVERSATION":
            link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=run_id)
            turn = await uow.turns.get_turn(domain_id=scope.domain_id, turn_id=link.turn_id) if link else None
            if (turn is None or turn.conversation_id != source.source_id
                or turn.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}):
                raise invalid_report_source()
            turn_id = turn.turn_id
        refs.append(OutputSourceArtifact(run_id=run_id, artifact_id=artifact_id, turn_id=turn_id,
            schema_version=artifact.schema_version, content_hash=artifact.content_hash,
            security_level=int(artifact.security_level)))
        observed.append(run.completed_at or run.created_at)
        if artifact.schema_version == "REPORT_CONTENT.v1":
            report = await uow.inspections.get_report_by_content_scoped(
                domain_id=scope.domain_id, content_artifact_id=artifact_id)
            if (report is None or report.content_hash != artifact.content_hash
                or report.ops_run_id != run_id
                or not await report_source_is_usable(uow=uow, domain_id=scope.domain_id, report=report)):
                raise invalid_report_source()
            source_reports[str(artifact_id)] = report.report_id
    material = FrozenReportMaterial(run_id=prepared.run.ops_run_id, task_id=prepared.task.ops_task_id,
        source_artifact_id=prepared.source_artifact.artifact_id, source=deepcopy(prepared.source),
        runs=tuple(stamps.values()), source_reports=source_reports,
        period_kind=prepared.period_kind, period_start=prepared.period_start,
        # 点击时钟不属于业务输入；AD_HOC 的观察范围结束于被冻结原件的结束时间。
        period_end=max(observed) if prepared.period_kind == "AD_HOC" else prepared.period_end,
        template_version_id=template_version_id)
    fingerprint = canonical_hash({"artifacts": [row.model_dump(mode="json") for row in refs],
        "report": material.model_dump(mode="json")})
    frozen = FrozenOutputInput(source=source, artifacts=tuple(refs), source_fingerprint=fingerprint,
        template={**asdict(prepared.template), "content_hash": prepared.template.content_hash},
        parameters={"report": material.model_dump(mode="json"), "authority": {
            "origin": "HUMAN", "source_agent_ids": sorted({str(row.agent_id) for row in stamps.values()})}},
        frozen_at=await uow.runs.database_now())
    await validate_report_input(uow=uow, scope=scope, frozen=frozen,
        plan_capture_reader=plan_capture_reader)
    return frozen


def report_from_frozen_input(frozen):
    """重建确切模板和范围，不解析当前模板指针，也不读取最新轮次。"""
    try:
        material = FrozenReportMaterial.model_validate(frozen.parameters["report"])
        snapshot = deepcopy(frozen.template)
        content_hash = snapshot.pop("content_hash")
        for key in ("sections", "applicable_source_kinds", "allowed_period_kinds"):
            snapshot[key] = tuple(snapshot[key])
        template = ReportTemplate(**snapshot)
        expected = canonical_hash({"artifacts": [row.model_dump(mode="json") for row in frozen.artifacts],
            "report": material.model_dump(mode="json")})
        identities = {(row.run_id, row.artifact_id) for row in frozen.artifacts}
        if (frozen.source.kind not in {"CONVERSATION", "RUN"} or content_hash != template.content_hash
            or expected != frozen.source_fingerprint or len(identities) != len(frozen.artifacts)
            or len({row.run_id for row in material.runs}) != len(material.runs)
            or {row.run_id for row in material.runs} != {row.run_id for row in frozen.artifacts}
            or (material.run_id, material.source_artifact_id) not in identities
            or material.period_end < material.period_start
            or frozen.source.kind == "RUN" and frozen.source.source_id != material.run_id
            or frozen.source.kind == "CONVERSATION"
                and str(material.source.get("conversation", {}).get("conversation_id")) != str(frozen.source.source_id)):
            raise ValueError("冻结报告范围不一致")
        return material, template
    except (AttributeError, KeyError, TypeError, ValueError, ValidationError) as exc:
        raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_INPUT_INVALID",
            message="报告冻结材料损坏，不能继续生成", status_code=409) from exc


async def validate_report_input(*, uow, scope, frozen, plan_capture_reader=None):
    """允许无关后续轮次；已选判断被反驳、原件更正或授权撤回仍阻断。"""
    material, template = report_from_frozen_input(frozen)
    runs, artifacts = {}, {}
    if frozen.source.kind == "CONVERSATION":
        conversation = await uow.conversations.get_conversation(
            conversation_id=frozen.source.source_id, domain_id=scope.domain_id)
        if conversation is None or str(conversation.created_by) != str(scope.actor_id):
            raise invalid_report_source()
    if template.template_ref.startswith("domain:"):
        template_id = UUID(template.template_ref.removeprefix("domain:"))
        root = await uow.inspections.get_session_report_template(
            domain_id=scope.domain_id, template_id=template_id)
        version = await uow.inspections.get_session_report_template_version(
            template_version_id=material.template_version_id)
        if (root is None or root.status != "ACTIVE" or version is None
            or version.template_id != template_id
            or str(version.version_no) != str(template.version)
            or version.content_hash != template.content_hash
            or canonical_hash(version.definition_json) != template.content_hash):
            raise invalid_report_source()
    for stamp in material.runs:
        run, target = await _current_run(uow, scope=scope, run_id=stamp.run_id)
        if (run.target_id != stamp.target_id or run.agent_id != stamp.agent_id
            or canonical_hash(run.plan_snapshot_json or {}) != stamp.plan_hash
            or run.final_artifact_id != stamp.final_artifact_id
            or int(target.row_version) != stamp.target_version
            or run.trigger_type != stamp.trigger_type):
            raise invalid_report_source()
        if run.status != stamp.status:
            raise invalid_report_source()
        runs[stamp.run_id] = run
    agents = sorted({str(row.agent_id) for row in material.runs})
    if frozen.parameters.get("authority") != {"origin": "HUMAN", "source_agent_ids": agents}:
        raise invalid_report_source()
    for ref in frozen.artifacts:
        run = runs[ref.run_id]
        if frozen.source.kind == "CONVERSATION":
            link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=ref.run_id)
            turn = await uow.turns.get_turn(domain_id=scope.domain_id, turn_id=ref.turn_id) if ref.turn_id else None
            if (link is None or link.turn_id != ref.turn_id or turn is None
                or turn.conversation_id != frozen.source.source_id
                or turn.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}):
                raise invalid_report_source()
        artifact = await uow.runs.get_artifact(artifact_id=ref.artifact_id)
        if (artifact is None or artifact.ops_run_id != ref.run_id
            or artifact.content_hash != ref.content_hash or artifact.schema_version != ref.schema_version
            or int(artifact.security_level) != ref.security_level
            or canonical_hash(artifact.payload_json) != ref.content_hash):
            raise invalid_report_source()
        artifacts[ref.artifact_id] = artifact
        if ref.schema_version == "REPORT_CONTENT.v1":
            report_id = material.source_reports.get(str(ref.artifact_id))
            report = await uow.inspections.get_report_scoped(
                domain_id=scope.domain_id, report_id=report_id) if report_id else None
            if (report is None or report.content_artifact_id != ref.artifact_id
                or report.content_hash != ref.content_hash
                or not await report_source_is_usable(uow=uow, domain_id=scope.domain_id, report=report)):
                raise invalid_report_source()
        if ref.schema_version == "DIAGNOSIS_CLOSURE.v1":
            from aiops_agent.application.case_closure import validate_closure_sources
            if not await validate_closure_sources(uow=uow, domain_id=scope.domain_id,
                run=run, artifact=artifact, plan_capture_reader=plan_capture_reader):
                raise invalid_report_source()
        else:
            from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
            if not await validate_investigation_sources(uow=uow, run=run,
                payload=dict(artifact.payload_json or {}), plan_capture_reader=plan_capture_reader):
                raise invalid_report_source()
            record = (artifact.payload_json or {}).get("investigation_record")
            if record:
                from aiops_agent.contracts.investigation import InvestigationRecord
                from aiops_agent.application.case_source_validation import _case_claims_still_current
                selected = {row["claim_id"] for row in material.source.get("facts", ())
                    if isinstance(row, dict) and row.get("claim_id")
                    and row.get("source_run_id") == str(run.ops_run_id)
                    and row.get("record_id") == record.get("record_id")}
                if selected and not await _case_claims_still_current(uow=uow, domain_id=scope.domain_id,
                    run=run, artifact=artifact, record=InvestigationRecord.model_validate(record), claim_ids=selected):
                    raise invalid_report_source()
    tasks = await uow.runs.list_tasks(ops_run_id=material.run_id)
    if runs[material.run_id].status not in {"COMPLETED", "PARTIAL"}:
        raise invalid_report_source()
    task = next((row for row in tasks if row.ops_task_id == material.task_id), None)
    if task is None:
        raise invalid_report_source()
    return material, template, runs[material.run_id], artifacts[material.source_artifact_id], task


async def submit_report_output(*, runtime_service, jobs, scope, conversation_id, ops_run_id,
    template, template_version_id, period_kind, idempotency_key):
    source = OutputJobSource(kind="CONVERSATION" if conversation_id else "RUN",
        source_id=conversation_id or ops_run_id)

    async def freeze(uow, root):
        if conversation_id:
            if str(root.created_by) != str(scope.actor_id):
                raise resource_not_found("Conversation")
            prepared = await prepare_conversation_report(runtime_service, uow=uow,
                domain_id=scope.domain_id, actor_id=scope.actor_id, conversation_id=conversation_id,
                template=template, trace_id=scope.trace_id)
        else:
            prepared = await prepare_user_report(runtime_service, uow=uow,
                domain_id=scope.domain_id, actor_id=scope.actor_id, ops_run_id=ops_run_id,
                template=template, period_kind=period_kind, trace_id=scope.trace_id)
        version_id = template_version_id
        if prepared.template.template_ref.startswith("domain:"):
            template_id = UUID(prepared.template.template_ref.removeprefix("domain:"))
            row = await uow.inspections.get_session_report_template(
                domain_id=scope.domain_id, template_id=template_id)
            if row is None or row.status != "ACTIVE":
                raise invalid_report_source()
            version = await uow.inspections.get_session_report_template_version(
                template_version_id=template_version_id or row.current_version_id)
            if (version is None or version.template_id != template_id
                or str(version.version_no) != prepared.template.version):
                raise invalid_report_source()
            version_id = version.template_version_id
        return await freeze_report_input(prepared=prepared, uow=uow, scope=scope,
            source=source, template_version_id=version_id,
            plan_capture_reader=getattr(runtime_service, "_plan_capture_reader", None))

    return await jobs.submit(scope=scope, job_type="REPORT_GENERATION",
        source=source, idempotency_key=idempotency_key, freeze=freeze)


class ReportOutputProcessor:
    def __init__(self, *, uow_factory, plan_capture_reader=None):
        self._uow_factory = uow_factory
        self._plan_capture_reader = plan_capture_reader

    async def advance(self, execution, *, authority, checkpoint, record_receipt):
        if execution.job_type != "REPORT_GENERATION":
            raise invalid_report_source()
        await checkpoint(stage="ASSEMBLING", processing_ref={})
        async with self._uow_factory() as uow:
            await validate_report_input(uow=uow, scope=execution.scope, frozen=execution.frozen,
                plan_capture_reader=self._plan_capture_reader)
        return OutputAdvance(stage="FINALIZING", processing_ref={},
            finalization={"source_fingerprint": execution.frozen.source_fingerprint})


async def finalize_report_output(*, runtime_service, uow, job, execution, material, now,
    plan_capture_reader=None):
    """产物、成功终态、报告通知共用当前 Worker 的同一事务。"""
    if material != {"source_fingerprint": execution.frozen.source_fingerprint}:
        raise invalid_report_source()
    root = await uow.output_jobs.lock_source(domain_id=job.domain_id,
        source_kind=execution.frozen.source.kind, source_id=execution.frozen.source.source_id)
    if root is None or getattr(root, "tenant_id", None) != execution.scope.tenant_id:
        raise invalid_report_source()
    frozen, template, run, artifact, task = await validate_report_input(
        uow=uow, scope=execution.scope, frozen=execution.frozen, plan_capture_reader=plan_capture_reader)
    prepared = PreparedReport(run=run, task=task, source_artifact=artifact, source=frozen.source,
        template=template, period_kind=frozen.period_kind,
        period_start=frozen.period_start, period_end=frozen.period_end)
    report = await publish_prepared_report(runtime_service, uow=uow, prepared=prepared,
        source=execution.frozen.source, scope=execution.scope, job=job, now=now,
        minimum_security_level=max(row.security_level for row in execution.frozen.artifacts))
    return OutputJobResult(report_id=report.report_id, report_version=str(report.report_version))


async def publish_prepared_report(runtime_service, *, uow, prepared, source, scope, job, now,
    minimum_security_level):
    """在持有来源锁的终态事务内幂等装配；不承担受理、授权或获取新来源。"""
    template, run, artifact, task = prepared.template, prepared.run, prepared.source_artifact, prepared.task
    current = (await uow.inspections.get_current_report_for_conversation_template(
        conversation_id=source.source_id, template_id=template.template_ref)
        if source.kind == "CONVERSATION" else
        await uow.inspections.get_current_report_for_run_template(
            ops_run_id=run.ops_run_id, template_id=template.template_ref))
    digest = report_input_hash(source_artifact=artifact, source=prepared.source, template=template,
        period_kind=prepared.period_kind, period_start=prepared.period_start, period_end=prepared.period_end)
    if current is not None and current.content_artifact_id is not None:
        from aiops_agent.application.runtime.service import report_content_reference_is_complete
        content = await uow.runs.get_artifact(artifact_id=current.content_artifact_id)
        if (report_content_reference_is_complete(current, content)
            and current.security_level >= minimum_security_level
            and content.security_level >= minimum_security_level
            and (content.payload_json.get("provenance") or {}).get("report_input_hash") == digest):
            return current
    return await runtime_service._publish_diagnosis_report(uow=uow, run=run, task=task,
        source_artifact=artifact, now=now, trace_id=scope.trace_id,
        template=template, actor_id=scope.actor_id, source_override=prepared.source,
        period_start_override=prepared.period_start, period_end_override=prepared.period_end,
        period_kind=prepared.period_kind, template_is_frozen=True, output_job=job,
        minimum_security_level=minimum_security_level)
