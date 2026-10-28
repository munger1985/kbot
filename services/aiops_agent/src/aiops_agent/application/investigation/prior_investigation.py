"""重新授权后读取上一轮冻结判断；历史判断始终不是本轮原始观测。"""

from __future__ import annotations

from aiops_agent.contracts.investigation import InvestigationParent, InvestigationRecord
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.playbooks import canonical_hash
from aiops_agent.application.source_authorization import source_scope_is_current


async def load_parent_investigation(*, uow, turn, run, target, capabilities, sources):
    if int(getattr(turn, "turn_no", 1)) <= 1:
        return None
    previous = await uow.turns.get_previous_turn(
        conversation_id=turn.conversation_id, before_turn_no=int(turn.turn_no),
    )
    if previous is None or str(previous.status) not in {"COMPLETED", "PARTIAL"}:
        return None
    if any(str(getattr(previous, key)) != str(getattr(turn, key))
           for key in ("tenant_id", "domain_id", "conversation_id", "created_by")):
        return None
    link = await uow.turns.get_run_link(turn_id=previous.turn_id, purpose="PRIMARY")
    prior_run = await uow.runs.get_run(ops_run_id=link.ops_run_id) if link else None
    if prior_run is None or any(
        str(getattr(prior_run, key)) != str(getattr(run, key))
        for key in ("tenant_id", "domain_id", "target_id", "agent_id", "agent_version_id", "actor_id")
    ):
        return None
    if not await uow.agents.version_has_target(
        agent_version_id=run.agent_version_id, target_id=run.target_id,
    ):
        return None
    snapshot = dict(prior_run.plan_snapshot_json or {})
    prior_target = dict(snapshot.get("target") or {})
    if str(prior_target.get("row_version")) != str(target.row_version) or (
        str(prior_target.get("db_type") or "").upper() != str(target.db_type).upper()
    ):
        return None
    prior_caps = dict(snapshot.get("capability_snapshot") or {})
    # 授权或配置变化时不继承旧文本，避免撤权后仍可通过模型输入读取。
    current_sources = capabilities.model_dump(mode="json").get("source_snapshots", [])
    old_sources = prior_caps.get("source_snapshots")
    def key(values):
        # 健康检查时钟不改变授权；配置、许可和范围仍须完全一致。
        transient = {"checked_at", "expires_at", "health_state"}
        return sorted(
            ({name: value for name, value in item.items() if name not in transient} for item in values),
            key=lambda item: str(item["source_id"]),
        )
    if old_sources is None or key(old_sources) != key(current_sources):
        return None
    completed_at = getattr(prior_run, "completed_at", None)
    if any(
        completed_at is None or getattr(source, "updated_at", None) is None
        or source.updated_at > completed_at for source in sources
    ):
        return None
    if prior_run.final_artifact_id is None:
        return None
    if not await source_scope_is_current(uow=uow, run=prior_run, domain_id=run.domain_id):
        return None
    artifact = await uow.runs.get_artifact(artifact_id=prior_run.final_artifact_id)
    if artifact is None or str(artifact.ops_run_id) != str(prior_run.ops_run_id) or (
        artifact.schema_version != "AIOPS_TURN_RESULT.v1"
    ):
        return None
    try:
        payload = dict(artifact.payload_json)
        if canonical_hash(payload) != artifact.content_hash:
            return None
        record = InvestigationRecord.model_validate(payload.get("investigation_record"))
        evidence = tuple(TurnEvidenceFact.model_validate(item) for item in payload.get("evidence", ()))
        record.validate_observations(evidence)
        from aiops_agent.application.diagnosis.investigation_record import validate_diagnosis
        validate_diagnosis(record, evidence)
        if record.run_id != str(prior_run.ops_run_id) or record.target_id != str(run.target_id):
            return None
        from aiops_agent.application.diagnosis.revisions import active_historical_claims
        history = active_historical_claims(record)
        if len(history) > 128 or len({entry.artifact_id for entry in history}) > 16:
            return None
        verified = {}
        for entry in history:
            if entry.artifact_id not in verified:
                historical_run = await uow.runs.get_run(ops_run_id=type(prior_run.ops_run_id)(entry.run_id))
                historical_artifact = await uow.runs.get_artifact(artifact_id=type(artifact.artifact_id)(entry.artifact_id))
                if historical_run is None or historical_artifact is None or any(
                    str(getattr(historical_run, field)) != str(getattr(run, field))
                    for field in ("tenant_id", "domain_id", "target_id", "agent_id", "agent_version_id", "actor_id")
                ):
                    return None
                old_snapshot = dict(historical_run.plan_snapshot_json or {})
                if not await source_scope_is_current(uow=uow, run=historical_run, domain_id=run.domain_id):
                    return None
                old_target = dict(old_snapshot.get("target") or {})
                old_caps = dict(old_snapshot.get("capability_snapshot") or {})
                if (str(old_target.get("row_version")) != str(target.row_version)
                    or str(old_target.get("db_type", "")).upper() != str(target.db_type).upper()
                    or key(old_caps.get("source_snapshots", ())) != key(current_sources)):
                    return None
                finished = historical_run.completed_at
                if any(finished is None or source.updated_at is None or source.updated_at > finished for source in sources):
                    return None
                if (str(historical_artifact.ops_run_id) != entry.run_id
                    or historical_artifact.schema_version != "AIOPS_TURN_RESULT.v1"
                    or historical_artifact.content_hash != entry.content_hash
                    or canonical_hash(historical_artifact.payload_json) != entry.content_hash):
                    return None
                old_payload = dict(historical_artifact.payload_json)
                historical_record = InvestigationRecord.model_validate(old_payload.get("investigation_record"))
                old_evidence = tuple(TurnEvidenceFact.model_validate(item) for item in old_payload.get("evidence", ()))
                historical_record.validate_observations(old_evidence)
                validate_diagnosis(historical_record, old_evidence)
                if (historical_record.record_id != record.record_id
                    or historical_record.run_id != entry.run_id or historical_record.target_id != record.target_id):
                    return None
                verified[entry.artifact_id] = historical_record
            if entry.claim not in verified[entry.artifact_id].claims:
                return None
        return InvestigationParent(
            record_id=record.record_id, revision=record.revision, target_id=record.target_id,
            run_id=record.run_id, artifact_id=str(artifact.artifact_id), content_hash=artifact.content_hash,
            claims=tuple(item for item in record.claims if item.revision_state == "ACTIVE"),
            historical_claims=history,
        ).model_dump(mode="json")
    except (ValueError, TypeError, KeyError):
        return None
