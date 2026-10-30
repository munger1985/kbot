"""发布与检索时重新核对案例来源；只读校验不改写历史知识版本。"""

from aiops_agent.playbooks import canonical_hash
from aiops_agent.application.source_authorization import source_scope_is_current
from aiops_agent.application.report_source_revision import report_source_is_usable


async def validate_case_sources(*, uow, domain_id, version_id, agent_id=None, plan_capture_reader=None):
    relations = await uow.operations_knowledge.get_version_relations(asset_version_id=version_id)
    sources = relations.get("sources") or ()
    return await validate_case_source_rows(
        uow=uow, domain_id=domain_id, sources=sources, agent_id=agent_id,
        plan_capture_reader=plan_capture_reader,
    )


async def validate_case_source_rows(*, uow, domain_id, sources, agent_id=None, plan_capture_reader=None):
    """新提炼与既有版本复用同一来源、授权和反证校验，不以幂等命中绕过。"""
    if not sources:
        return False
    for source in sources:
        report = None
        if source.source_report_id is not None:
            report = await uow.inspections.get_report_scoped(
                report_id=source.source_report_id, domain_id=domain_id,
            )
            if not await report_source_is_usable(uow=uow, domain_id=domain_id, report=report):
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
        if getattr(artifact, "schema_version", None) == "REPORT_CONTENT.v1" and (
            report is None or report.content_artifact_id != artifact.artifact_id
            or report.content_hash != artifact.content_hash
            or str(report.ops_run_id) != str(run.ops_run_id)
            or str(report.target_id) != str(run.target_id)
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
        if record or any(item.get("source_id") == "prior_turn.sql_observation" or item.get("followup_run_id") or item.get("pagination") or item.get("sql_plan")
            or item.get("tool_id") in {"evidence.sql_plan.controlled_estimate", "metric.query_range",
                "monitor.event.query", "monitor.discover_metrics", "monitor.query_native_metric", "loki.query_range"}
            or item.get("source_identity_witnesses") for item in payload.get("evidence", ())):
            from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
            if not await validate_investigation_sources(uow=uow, run=run, payload=payload, plan_capture_reader=plan_capture_reader):
                return False
        if record:
            from aiops_agent.contracts.investigation import InvestigationRecord
            try:
                value = InvestigationRecord.model_validate(record)
                from aiops_agent.contracts.turn_answer import TurnEvidenceFact
                value.validate_observations(tuple(
                    TurnEvidenceFact.model_validate(item) for item in payload.get("evidence", ())
                ))
                from aiops_agent.application.diagnosis.investigation_record import validate_diagnosis
                from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
                validate_diagnosis(
                    DiagnosisAnswerDraft(claims=value.claims, recommendations=value.recommendations, goal_satisfied=False),
                    tuple(TurnEvidenceFact.model_validate(item) for item in payload.get("evidence", ())),
                )
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
    """所有效果案例均重放来源；参考与失败不能绕过完整性，VERIFIED 仍须严格成功。"""
    from uuid import UUID
    from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
    from aiops_agent.application.changes.effect_replay import is_supported_effect, replay_effect
    from aiops_agent.application.effect_case import effect_case_kind, effect_case_verification
    try:
        profile = DiagnosisCaseProfile.model_validate(profile)
        verified = profile.case_kind in {"VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION"}
        reference = str(profile.verification.get("effect_result_ref") or profile.verification.get("controlled_verification_ref") or "")
        source_id = profile.source.get("effect_result_artifact_id")
        if not reference:
            return not verified and not source_id
        if not reference.startswith("artifact:") or "#" in reference:
            return False
        if source_id and reference != f"artifact:{source_id}":
            return False
        artifact = await uow.runs.get_artifact(artifact_id=UUID(reference.removeprefix("artifact:")))
        if artifact is None:
            return False
        run = await uow.runs.get_run_scoped(ops_run_id=artifact.ops_run_id, domain_id=domain_id)
        if run is None or str(run.ops_run_id) not in {str(value) for value in profile.source.get("run_ids", ())}:
            return False
        result = await replay_effect(uow=uow, run=run, artifact=artifact)
        if result is None:
            return False
        if verified and not is_supported_effect(result):
            return False
        if profile.case_kind == "VERIFIED_RESOLUTION" and result.symptom_recovered is not True:
            return False
        if profile.case_kind == "NEGATIVE_EXPERIENCE" and effect_case_kind(result) != "NEGATIVE_EXPERIENCE":
            return False
        if profile.verification.get("effect_result_ref"):
            plan = await uow.runs.get_artifact(artifact_id=result.comparison_plan_artifact_id)
            if (plan is None or plan.content_hash != result.measurement_artifact_refs[0]["content_hash"]
                or canonical_hash(plan.payload_json) != plan.content_hash):
                return False
            expected = effect_case_verification(result=result, artifact_id=artifact.artifact_id,
                frozen_plan=plan.payload_json["business_effect"])
            if any(profile.verification.get(key) != value for key, value in expected.items()):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False
