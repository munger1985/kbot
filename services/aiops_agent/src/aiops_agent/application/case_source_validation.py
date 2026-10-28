"""发布与检索时重新核对案例来源；只读校验不改写历史知识版本。"""

from aiops_agent.playbooks import canonical_hash
from aiops_agent.application.source_authorization import source_scope_is_current


async def validate_case_sources(*, uow, domain_id, version_id, agent_id=None):
    relations = await uow.operations_knowledge.get_version_relations(asset_version_id=version_id)
    sources = relations.get("sources") or ()
    if not sources:
        return False
    for source in sources:
        if source.source_report_id is not None:
            report = await uow.inspections.get_report_scoped(
                report_id=source.source_report_id, domain_id=domain_id,
            )
            if report is None or not report.is_current:
                return False
        if source.source_artifact_id is None or source.source_run_id is None:
            return False
        artifact = await uow.runs.get_artifact(artifact_id=source.source_artifact_id)
        run = await uow.runs.get_run_scoped(ops_run_id=source.source_run_id, domain_id=domain_id)
        if (
            artifact is None or run is None
            or str(artifact.ops_run_id) != str(run.ops_run_id)
            or artifact.content_hash != source.content_hash
            or canonical_hash(artifact.payload_json) != artifact.content_hash
        ):
            return False
        try:
            selected_agent = type(run.agent_id)(str(agent_id)) if agent_id is not None else run.agent_id
        except (TypeError, ValueError):
            return False
        active_agent = await uow.agents.get(domain_id=domain_id, agent_id=selected_agent)
        if active_agent is None or active_agent.current_version_id is None:
            return False
        if not await uow.agents.version_has_target(
            agent_version_id=active_agent.current_version_id, target_id=run.target_id,
        ):
            return False
        old_sources = dict(dict(run.plan_snapshot_json or {}).get("capability_snapshot") or {}).get("source_snapshots") or ()
        current_sources = await uow.agents.version_source_ids(
            agent_version_id=active_agent.current_version_id, target_id=run.target_id,
        )
        if not {str(item["source_id"]) for item in old_sources} <= {str(item) for item in current_sources}:
            return False
        if not await source_scope_is_current(uow=uow, run=run, domain_id=domain_id):
            return False
        payload = dict(artifact.payload_json or {})
        root_id = dict(getattr(artifact, "provenance_json", None) or {}).get("source_artifact_id")
        if root_id and payload.get("schema_version") == "DIAGNOSIS_CLOSURE.v1":
            try:
                root = await uow.runs.get_artifact(artifact_id=type(source.source_artifact_id)(str(root_id)))
            except (TypeError, ValueError):
                return False
            if (root is None or str(root.ops_run_id) != str(run.ops_run_id)
                or canonical_hash(root.payload_json) != root.content_hash):
                return False
            artifact = root
            payload = dict(root.payload_json or {})
        record = payload.get("investigation_record")
        if record:
            from aiops_agent.application.diagnosis.monitoring_calculations import validate_monitoring_trace_sources
            if not await validate_monitoring_trace_sources(uow=uow, run=run, payload=payload):
                return False
            from aiops_agent.contracts.investigation import InvestigationRecord
            try:
                value = InvestigationRecord.model_validate(record)
            except ValueError:
                return False
            if value.target_id != str(run.target_id) or value.run_id != str(run.ops_run_id):
                return False
            if not any(item.revision_state == "ACTIVE" and item.level in {"OBSERVED", "SUPPORTED"} for item in value.claims):
                return False
            if not await _case_claims_still_current(uow=uow, domain_id=domain_id, run=run, artifact=artifact, record=value):
                return False
    return True

async def _case_claims_still_current(*, uow, domain_id, run, artifact, record):
    """沿同会话后续的确切修订追踪反证；已知错误不进入新检索。"""
    link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=run.ops_run_id)
    if link is None:
        return True
    turn = await uow.turns.get_turn(domain_id=domain_id, turn_id=link.turn_id)
    if turn is None:
        return False
    successors = await uow.turns.list_turns(
        conversation_id=turn.conversation_id, after_turn_no=turn.turn_no, limit=101,
    )
    if len(successors) > 100:
        return False
    from aiops_agent.contracts.investigation import InvestigationRecord
    tracked = {(str(artifact.artifact_id), claim.claim_id) for claim in record.claims if claim.revision_state == "ACTIVE"}
    for following in successors:
        later_link = await uow.turns.get_run_link(turn_id=following.turn_id, purpose="PRIMARY")
        if later_link is None:
            continue
        later_run = await uow.runs.get_run_scoped(ops_run_id=later_link.ops_run_id, domain_id=domain_id)
        if later_run is None or str(later_run.target_id) != record.target_id:
            continue
        if later_run.final_artifact_id is None:
            continue
        later_artifact = await uow.runs.get_artifact(artifact_id=later_run.final_artifact_id)
        if later_artifact is None or later_artifact.schema_version != "AIOPS_TURN_RESULT.v1":
            continue
        if str(later_artifact.ops_run_id) != str(later_run.ops_run_id):
            return False
        if canonical_hash(later_artifact.payload_json) != later_artifact.content_hash:
            return False
        data = dict(later_artifact.payload_json or {}).get("investigation_record")
        if not data:
            continue
        try:
            later = InvestigationRecord.model_validate(data)
        except ValueError:
            return False
        if later.record_id != record.record_id or later.parent is None:
            continue
        if later.target_id != record.target_id or later.run_id != str(later_run.ops_run_id):
            return False
        for change in later.claim_revisions:
            origin = change.prior_artifact_id or later.parent.artifact_id
            if (origin, change.prior_claim_id) not in tracked:
                continue
            if change.relation in {"REFUTES", "SUPERSEDES"}:
                return False
            tracked.add((str(later_artifact.artifact_id), change.current_claim_id))
    return True

async def validate_case_effect(*, uow, domain_id, profile):
    """案例中的肯定效果必须来自受控比较产物，不能只信任 profile 标记。"""
    from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
    try:
        profile = DiagnosisCaseProfile.model_validate(profile)
    except ValueError:
        return False
    if profile.case_kind not in {"VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION"}:
        return True
    from uuid import UUID
    from aiops_agent.contracts.report.comparison import ComparisonResult
    reference = str(profile.verification.get("controlled_verification_ref") or "")
    if not reference.startswith("artifact:") or "#" in reference:
        return False
    try:
        identity = UUID(reference.removeprefix("artifact:"))
    except ValueError:
        return False
    artifact = await uow.runs.get_artifact(artifact_id=str(identity))
    if (artifact is None or artifact.schema_version != "COMPARISON_RESULT.v1"
        or artifact.trust_level != "SOURCE_VERIFIED" or not isinstance(artifact.payload_json, dict)
        or canonical_hash(artifact.payload_json) != artifact.content_hash):
        return False
    run = await uow.runs.get_run_scoped(ops_run_id=artifact.ops_run_id, domain_id=domain_id)
    if run is None:
        return False
    try:
        result = ComparisonResult.model_validate(artifact.payload_json)
    except ValueError:
        return False
    if str(result.source_run_id) not in {str(value) for value in profile.source.get("run_ids", ())}:
        return False
    if not (
        result.measured_effect == "IMPROVED" and result.guardrails_passed is True
        and result.attribution_strength == "SUPPORTED" and not result.gap_codes
        and result.primary_signals and all(value is True for value in result.primary_signals.values())
        and result.guardrail_signals and all(value is True for value in result.guardrail_signals.values())
        and result.evidence_hashes
    ):
        return False
    if profile.case_kind == "VERIFIED_RESOLUTION" and result.symptom_recovered is not True:
        return False
    verification = await uow.runs.get_artifact(artifact_id=result.verification_artifact_id)
    receipt = await uow.runs.get_artifact(artifact_id=result.source_result_artifact_id)
    plan = await uow.runs.get_artifact(artifact_id=result.comparison_plan_artifact_id)
    for source in (verification, receipt, plan):
        if (source is None or not isinstance(source.payload_json, dict)
            or canonical_hash(source.payload_json) != source.content_hash):
            return False
        source_run = await uow.runs.get_run_scoped(ops_run_id=source.ops_run_id, domain_id=domain_id)
        if source_run is None or str(source_run.target_id) != str(run.target_id):
            return False
        if str(source.payload_json.get("proposal_id")) != str(result.proposal_id):
            return False
    # 标记不能替代原始窗口复算；没有执行前冻结条件的比较不能背书已验证案例。
    from datetime import datetime
    from aiops_agent.application.effect_measurement import evaluate_effect
    from aiops_agent.contracts.change import ActionVerification
    from aiops_agent.contracts.report import ComparisonPlan
    try:
        checked_plan = ComparisonPlan.model_validate(plan.payload_json)
        if checked_plan.business_effect is None or checked_plan.result_rule_version != "business-effect.v1":
            return False
        frozen = checked_plan.business_effect.model_dump(mode="json")
        checked = {}
        for ref in result.measurement_artifact_refs:
            native = await uow.runs.get_artifact(artifact_id=UUID(ref["artifact_id"]))
            if native is None or native.content_hash != ref["content_hash"] or canonical_hash(native.payload_json) != native.content_hash:
                return False
            native_run = await uow.runs.get_run_scoped(ops_run_id=native.ops_run_id, domain_id=domain_id)
            if native_run is None or str(native_run.target_id) != str(run.target_id) or not await source_scope_is_current(uow=uow, run=native_run, domain_id=domain_id):
                return False
            checked[str(native.artifact_id)] = native
        baseline = checked[str(frozen["baseline_artifact_id"])]
        after = [item for item in checked.values() if item.schema_version == "AIOPS_WORKLOAD_REPORT.v2" and str(item.artifact_id) != str(baseline.artifact_id)]
        if len(after) != 1 or baseline.content_hash != frozen["baseline_hash"]:
            return False
        action = ActionVerification.model_validate(verification.payload_json)
        if action.gap_codes or receipt.schema_version != "EXECUTION_RESULT.v1" or receipt.trust_level != "SOURCE_VERIFIED":
            return False
        recomputed = evaluate_effect(
            frozen=frozen, baseline=baseline.payload_json, after=after[0].payload_json,
            occurred_at=datetime.fromisoformat(receipt.payload_json["occurred_at"].replace("Z", "+00:00")),
            direct_status=action.status, controlled_execution=True,
            baseline_ref=f"artifact:{baseline.artifact_id}",
            after_ref=f"artifact:{after[0].artifact_id}",
        )
        for key in ("measured_effect", "symptom_recovered", "guardrails_passed", "attribution_strength", "measurements"):
            if recomputed[key] != getattr(result, key):
                return False
    except (KeyError, TypeError, ValueError):
        return False
    return True
