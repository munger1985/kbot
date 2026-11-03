"""报告案例的冻结来源适配；与会话案例共用 KC 管线，不在请求内调用模型。"""

from copy import deepcopy
from typing import Literal
from uuid import UUID

from pydantic import Field

from aiops_agent.application.case_redaction import CaseRedactor
from aiops_agent.application.case_report import load_case_report, profile_from_diagnosis_report, rejected_source
from aiops_agent.application.effect_case import profile_from_effect_report
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.report_source_revision import report_ancestors, report_source_is_usable
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from aiops_agent.playbooks import canonical_hash
from platform_core.contracts.aiops.output_jobs import FrozenOutputInput, OutputJobSource, OutputSourceArtifact
from platform_core.contracts.aiops.types import AIOpsContract, Sha256Digest, UUIDv7
from platform_core.identity import uuid7


class ReportCaseEvidence(AIOpsContract):
    run_id: UUIDv7
    artifact_id: UUIDv7
    content_hash: Sha256Digest
    source_kind: str


class FrozenReportCase(AIOpsContract):
    schema_version: Literal["OPS_FROZEN_REPORT_CASE.v1"] = "OPS_FROZEN_REPORT_CASE.v1"
    report_id: UUIDv7
    report_version: int = Field(ge=1)
    run_id: UUIDv7
    content_artifact_id: UUIDv7
    content_hash: Sha256Digest
    target_id: UUIDv7
    target_version: int = Field(ge=1)
    report_type: str
    security_level: int = Field(ge=1, le=5)
    title: str = Field(min_length=1, max_length=256)
    actor_id: str
    profile: DiagnosisCaseProfile
    evidence: tuple[ReportCaseEvidence, ...] = Field(min_length=1, max_length=511)
    artifacts: tuple[OutputSourceArtifact, ...] = Field(min_length=1, max_length=512)
    run_stamps: dict[str, dict]
    source_fingerprint: Sha256Digest
    request_hash: Sha256Digest

    @property
    def source_kind(self): return "REPORT"

    @property
    def source_id(self): return self.report_id

    @property
    def intake_id(self): return self.report_id


def _report_evidence(sources):
    return tuple(ReportCaseEvidence(run_id=row.ops_run_id, artifact_id=row.source_artifact_id,
        content_hash=row.content_hash, source_kind=row.source_kind)
        for row in sorted(sources, key=lambda row: (str(row.ops_run_id), str(row.source_artifact_id))))


async def _report_run_stamps(*, uow, domain_id, artifacts):
    stamps = {}
    for ref in artifacts:
        run = await uow.runs.get_run_scoped(ops_run_id=ref.run_id, domain_id=domain_id)
        if run is None or run.status not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}:
            raise rejected_source()
        agent = await uow.agents.get(domain_id=domain_id, agent_id=run.agent_id)
        if agent is None or agent.status != "ACTIVE" or agent.current_version_id is None:
            raise rejected_source()
        stamps[str(ref.run_id)] = {"agent_id": str(run.agent_id), "target_id": str(run.target_id),
            "status": run.status, "final_artifact_id": str(run.final_artifact_id) if run.final_artifact_id else None,
            "plan_hash": canonical_hash(run.plan_snapshot_json or {})}
    return stamps


async def prepare_report_case(*, uow, domain_id, report_id, actor_id, plan_capture_reader=None):
    """只读取并冻结已登记报告及原件；模型脱敏和 KC 上传留给 Worker。"""
    from aiops_agent.application.operations_knowledge import _diagnosis_case_eligibility
    report, content, sources = await load_case_report(uow=uow, domain_id=domain_id,
        report_id=report_id, plan_capture_reader=plan_capture_reader)
    evidence = _report_evidence(sources)
    effect_profile = await profile_from_effect_report(uow=uow, domain_id=domain_id,
        report=report, content=content, sources=sources, plan_capture_reader=plan_capture_reader)
    eligibility = {"eligible": True} if effect_profile is not None else _diagnosis_case_eligibility(
        report=report, report_content=content, evidence_refs=[str(row.artifact_id) for row in evidence])
    if not eligibility["eligible"]:
        raise AIOpsApplicationError(code="AIOPS_KNOWLEDGE_SOURCE_REJECTED", status_code=422,
            message="；".join(row["message"] for row in eligibility["reasons"]))
    target = await uow.targets.get_scoped(target_id=report.target_id, domain_id=domain_id)
    if target is None or target.status != "ENABLED":
        raise rejected_source()
    profile = CaseRedactor().profile(effect_profile or await profile_from_diagnosis_report(
        uow=uow, report=report, content=content, sources=sources, target=target))
    artifacts = []
    identities = {(report.ops_run_id, report.content_artifact_id),
        *((row.run_id, row.artifact_id) for row in evidence)}
    for run_id, artifact_id in sorted(identities, key=lambda row: (str(row[0]), str(row[1]))):
        artifact = await uow.runs.get_artifact(artifact_id=artifact_id)
        if artifact is None or artifact.ops_run_id != run_id:
            raise rejected_source()
        artifacts.append(OutputSourceArtifact(run_id=run_id, artifact_id=artifact_id,
            schema_version=artifact.schema_version, content_hash=artifact.content_hash,
            security_level=int(artifact.security_level)))
    if any(row.security_level > int(report.security_level) for row in artifacts):
        raise rejected_source()
    content_ref = next(row for row in artifacts if row.artifact_id == report.content_artifact_id)
    if content_ref.security_level != int(report.security_level):
        raise rejected_source()
    stamps = await _report_run_stamps(uow=uow, domain_id=domain_id, artifacts=artifacts)
    fingerprint = canonical_hash({"report_id": str(report_id), "version": int(report.report_version),
        "content_hash": report.content_hash, "target_id": str(report.target_id),
        "target_version": int(target.row_version), "environment": profile.environment,
        "security_level": int(report.security_level), "runs": stamps,
        "evidence": [row.model_dump(mode="json") for row in evidence],
        "artifacts": [row.model_dump(mode="json") for row in artifacts]})
    request_hash = canonical_hash({"source_fingerprint": fingerprint, "profile": profile.model_dump(mode="json")})
    profile = profile.model_copy(update={"source": {**profile.source,
        "source_fingerprint": fingerprint, "request_hash": request_hash}})
    return FrozenReportCase(report_id=report_id, report_version=int(report.report_version),
        run_id=report.ops_run_id, content_artifact_id=report.content_artifact_id,
        content_hash=report.content_hash, target_id=report.target_id, target_version=int(target.row_version),
        report_type=report.report_type, security_level=int(report.security_level), title=report.title,
        actor_id=str(actor_id), profile=profile, evidence=evidence, artifacts=tuple(artifacts),
        run_stamps=stamps, source_fingerprint=fingerprint, request_hash=request_hash)


async def freeze_report_case_input(*, prepared, uow):
    return FrozenOutputInput(source=OutputJobSource(kind="REPORT", source_id=prepared.report_id),
        artifacts=prepared.artifacts, source_fingerprint=prepared.source_fingerprint,
        parameters={"report_case": prepared.model_dump(mode="json"), "authority": {"origin": "HUMAN",
            "source_agent_ids": sorted({row["agent_id"] for row in prepared.run_stamps.values()})}},
        frozen_at=await uow.runs.database_now())


def report_case_from_frozen_input(frozen):
    try:
        prepared = FrozenReportCase.model_validate(deepcopy(frozen.parameters["report_case"]))
        if (frozen.source != OutputJobSource(kind="REPORT", source_id=prepared.report_id)
            or frozen.source_fingerprint != prepared.source_fingerprint or frozen.artifacts != prepared.artifacts
            or prepared.profile.source.get("request_hash") != prepared.request_hash
            or prepared.profile.source.get("source_fingerprint") != prepared.source_fingerprint):
            raise rejected_source()
        return prepared
    except (KeyError, TypeError, ValueError) as exc:
        raise rejected_source() from exc


async def validate_prepared_report_case(*, uow, domain_id, prepared, plan_capture_reader=None):
    current = await prepare_report_case(uow=uow, domain_id=domain_id, report_id=prepared.report_id,
        actor_id=prepared.actor_id, plan_capture_reader=plan_capture_reader)
    if current != prepared:
        raise rejected_source()


async def find_prepared_report_case(*, uow, domain_id, prepared):
    existing = await uow.operations_knowledge.find_by_source_hash(domain_id=domain_id,
        asset_kind="DIAGNOSIS_CASE", source_hash=prepared.content_hash)
    if existing is not None:
        return existing
    report = await uow.inspections.get_report_scoped(report_id=prepared.report_id, domain_id=domain_id)
    if report is None:
        return None
    for ancestor in await report_ancestors(uow=uow, domain_id=domain_id, report=report):
        previous = await uow.operations_knowledge.find_case_by_report_id(
            domain_id=domain_id, report_id=ancestor.report_id)
        if previous is not None:
            if await report_source_is_usable(uow=uow, domain_id=domain_id, report=ancestor):
                return previous
            break
    closure_id = (await uow.runs.get_artifact(artifact_id=prepared.content_artifact_id)).payload_json.get("scope", {}).get("closure_artifact_id")
    if closure_id:
        return await uow.operations_knowledge.find_case_by_artifact_id(domain_id=domain_id, artifact_id=UUID(str(closure_id)))
    return None


async def validate_report_case_result(*, uow, domain_id, prepared, asset_id, version_id, plan_capture_reader=None):
    """来源等价的旧案例可复用，但不能改写它的历史指纹或绕过当前报告围栏。"""
    await validate_prepared_report_case(uow=uow, domain_id=domain_id, prepared=prepared,
        plan_capture_reader=plan_capture_reader)
    existing = await find_prepared_report_case(uow=uow, domain_id=domain_id, prepared=prepared)
    if existing is None or existing[0].asset_id != asset_id or existing[1].asset_version_id != version_id:
        raise rejected_source()


async def submit_report_case(*, knowledge_service, jobs, scope, report_id, idempotency_key):
    async def freeze(uow, root):
        prepared = await prepare_report_case(uow=uow, domain_id=scope.domain_id, report_id=report_id,
            actor_id=scope.actor_id, plan_capture_reader=knowledge_service._plan_capture_reader)
        return await freeze_report_case_input(prepared=prepared, uow=uow)
    return await jobs.submit(scope=scope, job_type="CASE_EXTRACTION",
        source=OutputJobSource(kind="REPORT", source_id=report_id),
        idempotency_key=idempotency_key, freeze=freeze)


async def execute_prepared_report_case(service, *, domain_id, prepared, **options):
    from aiops_agent.application.case_intake import execute_case_intake
    return await execute_case_intake(service, domain_id=domain_id, prepared=prepared,
        source_validator=validate_prepared_report_case, replay_finder=find_prepared_report_case,
        register_intake=register_report_case_intake, reject=rejected_source, **options)


async def register_report_case_intake(*, uow, domain_id, prepared, recovery, entity_scope, actor):
    """先登记已受理 Revision，再重验资格；撤权也不能丢失外部对象的撤回定位。"""
    from aiops_agent.application.case_intake_recovery import CaseIntakeRecovery
    from aiops_agent.application.knowledge_withdrawal import queue_case_withdrawals
    from aiops_agent.application.operations_knowledge import _normalize, _now
    from aiops_agent.entities import (OperationsKnowledgeAssetEntity, OperationsKnowledgeVersionEntity,
        OperationsKnowledgeSourceEntity, OperationsKnowledgeScopeEntity, OperationsKnowledgeIndexEntity,
        OperationsKnowledgeReviewEntity)
    recovery = CaseIntakeRecovery.model_validate(recovery)
    recovery.body(prepared=prepared, collection_id=recovery.collection_id)
    if recovery.bundle_revision_id is None or prepared.actor_id != str(actor):
        raise rejected_source()
    await uow.inspections.get_report_scoped(report_id=prepared.report_id, domain_id=domain_id, lock=True)
    exact = await uow.operations_knowledge.find_case_by_intake_revision(
        domain_id=domain_id, revision_id=recovery.bundle_revision_id)
    if exact is not None:
        if (exact[1].source_hash != prepared.content_hash
            or exact[1].profile_json.get("source") != recovery.profile.source):
            raise rejected_source()
        if exact[1].status == "RETIRED":
            replay = await uow.operations_knowledge.find_by_source_hash(domain_id=domain_id,
                asset_kind="DIAGNOSIS_CASE", source_hash=prepared.content_hash)
            if replay is None:
                raise rejected_source()
            return exact[0], exact[1], replay, False
        return exact[0], exact[1], None, True
    # 此处只找来源身份，不把来源可用性作为保存外部受理回执的前置条件。
    replay = await uow.operations_knowledge.find_by_source_hash(domain_id=domain_id,
        asset_kind="DIAGNOSIS_CASE", source_hash=prepared.content_hash)
    if replay is not None:
        relations = await uow.operations_knowledge.get_version_relations(asset_version_id=replay[1].asset_version_id)
        if any(row.bundle_revision_id == recovery.bundle_revision_id for row in relations["indexes"]):
            return replay[0], replay[1], None, True
    previous = replay
    if previous is None:
        report = await uow.inspections.get_report_scoped(report_id=prepared.report_id, domain_id=domain_id)
        if report is not None:
            for ancestor in await report_ancestors(uow=uow, domain_id=domain_id, report=report):
                previous = await uow.operations_knowledge.find_case_by_report_id(
                    domain_id=domain_id, report_id=ancestor.report_id)
                if previous is not None:
                    if await report_source_is_usable(uow=uow, domain_id=domain_id, report=ancestor):
                        replay = previous
                    break
        artifact = await uow.runs.get_artifact(artifact_id=prepared.content_artifact_id)
        closure_id = (artifact.payload_json.get("scope") or {}).get("closure_artifact_id") if artifact else None
        if closure_id:
            closure_case = await uow.operations_knowledge.find_case_by_artifact_id(
                domain_id=domain_id, artifact_id=UUID(str(closure_id)))
            if closure_case is not None:
                previous = replay = closure_case
    asset = previous[0] if previous else OperationsKnowledgeAssetEntity(**entity_scope,
        domain_id=domain_id, asset_id=uuid7(), asset_kind="DIAGNOSIS_CASE", display_name=recovery.title,
        status="PROCESSING", security_level=prepared.security_level, created_by=actor, updated_by=actor)
    if previous:
        asset = await uow.operations_knowledge.get_asset(domain_id=domain_id, asset_id=asset.asset_id, for_update=True)
        if asset is None:
            raise rejected_source()
    scope = {**entity_scope, **({"domain_id": domain_id} if entity_scope else {})}
    versions = await uow.operations_knowledge.list_versions(domain_id=domain_id, asset_id=asset.asset_id) if previous else []
    version_id = uuid7()
    version = OperationsKnowledgeVersionEntity(**scope, asset_version_id=version_id, asset_id=asset.asset_id,
        version_no=max((row.version_no for row in versions), default=0) + 1,
        status="RETIRED" if replay else "PROCESSING", source_hash=prepared.content_hash,
        profile_schema_version=recovery.profile.schema_version, profile_json=recovery.profile.model_dump(mode="json"),
        extraction_warnings_json=[], created_by=actor)
    sources = [OperationsKnowledgeSourceEntity(**scope, asset_version_id=version_id,
        source_kind="CASE_DUPLICATE" if replay else "REPORT", source_report_id=prepared.report_id,
        source_run_id=prepared.run_id, source_artifact_id=prepared.content_artifact_id,
        content_hash=prepared.content_hash, source_locator_json={"report_version": prepared.report_version}),
        *(OperationsKnowledgeSourceEntity(**scope, asset_version_id=version_id,
            source_kind="CASE_DUPLICATE_EVIDENCE" if replay else "REPORT_EVIDENCE", source_report_id=prepared.report_id,
            source_run_id=row.run_id, source_artifact_id=row.artifact_id, content_hash=row.content_hash,
            source_locator_json={"source_kind": row.source_kind}) for row in prepared.evidence)]
    environment = prepared.profile.environment
    scopes = [OperationsKnowledgeScopeEntity(**scope, asset_version_id=version_id, scope_kind=kind,
        scope_value=str(value), normalized_value=_normalize(str(value)), source_kind="REPORT_FACT",
        source_locator_json={"report_id": str(prepared.report_id)}) for kind, value in (
            ("TARGET_ID", prepared.target_id), ("DATABASE_TYPE", environment.get("database_type")),
            ("DATABASE_VERSION", environment.get("database_version")), ("PLATFORM", environment.get("platform")),
            ("PROBLEM_CLASS", prepared.report_type), ("TOPOLOGY", environment.get("topology"))) if value]
    index = OperationsKnowledgeIndexEntity(**scope, asset_version_id=version_id, collection_id=recovery.collection_id,
        bundle_id=recovery.bundle_id, bundle_revision_id=recovery.bundle_revision_id, index_status="PENDING_APPROVAL")
    if previous:
        if not replay:
            if asset.current_version_id is not None:
                old = await uow.operations_knowledge.get_version(domain_id=domain_id,
                    asset_version_id=asset.current_version_id, for_update=True)
                if old is not None:
                    before = old.status
                    old.status, old.retired_at = "RETIRED", _now()
                    await queue_case_withdrawals(uow=uow, asset=asset, version=old)
                    await uow.operations_knowledge.add_review(OperationsKnowledgeReviewEntity(**scope,
                        asset_version_id=old.asset_version_id, decision="SOURCE_CORRECTED", reviewer_id=actor,
                        comment_text="来源报告已发布更正版，旧案例停止参与检索", before_status=before, after_status="RETIRED"))
                asset.current_version_id = None
            asset.display_name, asset.status, asset.updated_by = recovery.title, "PROCESSING", actor
        asset.security_level = max(int(asset.security_level), prepared.security_level)
        await uow.operations_knowledge.add_version(version=version, scopes=scopes, sources=sources, indexes=[index])
    else:
        await uow.operations_knowledge.add_asset_version(asset=asset, version=version, scopes=scopes, sources=sources, indexes=[index])
    if replay:
        await queue_case_withdrawals(uow=uow, asset=asset, version=version)
    return asset, version, replay, False
