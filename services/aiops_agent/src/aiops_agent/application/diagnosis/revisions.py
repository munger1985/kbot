"""按不可变父版本显式合并判断，不根据措辞或最高等级猜测修订。"""

from aiops_agent.application.diagnosis.investigation_record import (
    investigation_report_source, render_investigation,
)
from aiops_agent.contracts.investigation import InvestigationRecord


def active_historical_claims(record):
    """保留上一版本中尚未被明确修订的历史判断，不按文本或等级合并。"""
    if record.parent is None:
        return ()
    from aiops_agent.contracts.investigation import InvestigationHistoricalClaim
    entries = (
        *record.parent.historical_claims,
        *(InvestigationHistoricalClaim(
            artifact_id=record.parent.artifact_id, content_hash=record.parent.content_hash,
            run_id=record.parent.run_id, claim=claim,
        ) for claim in record.parent.claims if claim.revision_state == "ACTIVE"),
    )
    retired = {
        (change.prior_artifact_id or record.parent.artifact_id, change.prior_claim_id)
        for change in record.claim_revisions
    }
    active = {(item.artifact_id, item.claim.claim_id): item for item in entries
              if (item.artifact_id, item.claim.claim_id) not in retired}
    return tuple(active[key] for key in sorted(active))


def merge_investigation_sources(rows):
    """输入顺序为轮次；保留原始记录，当前投影移除已被明确修订的判断和关联建议。"""
    by_artifact = {str(artifact.artifact_id): (run, artifact, source) for run, artifact, source in rows}
    retired = {}
    history = []
    for run, artifact, source in rows:
        data = source.get("investigation_record")
        if not data:
            continue
        record = InvestigationRecord.model_validate(data)
        if record.run_id != str(run.ops_run_id) or record.target_id != str(run.target_id):
            raise ValueError("报告调查记录与来源 Run/Target 不一致")
        if record.parent is None:
            continue
        parent_entry = by_artifact.get(record.parent.artifact_id)
        if parent_entry is None:
            # 报告只选本轮时不越界读取未选中的旧记录。
            continue
        parent_run, parent_artifact, parent_source = parent_entry
        previous = InvestigationRecord.model_validate(parent_source.get("investigation_record"))
        if (
            parent_artifact.content_hash != record.parent.content_hash
            or previous.record_id != record.record_id
            or previous.revision != record.revision - 1
            or previous.target_id != record.target_id
            or previous.run_id != record.parent.run_id
            or tuple(item for item in previous.claims if item.revision_state == "ACTIVE") != record.parent.claims
            or active_historical_claims(previous) != record.parent.historical_claims
        ):
            raise ValueError("报告修订链与确切父版本不一致")
        for change in record.claim_revisions:
            origin_artifact = change.prior_artifact_id or record.parent.artifact_id
            retired.setdefault(origin_artifact, set()).add(change.prior_claim_id)
            history.append({
                "previous_artifact_id": origin_artifact,
                "current_artifact_id": str(artifact.artifact_id),
                **change.model_dump(mode="json"),
            })
    projected = {}
    for run, artifact, source in rows:
        artifact_id = str(artifact.artifact_id)
        old_ids = retired.get(artifact_id, set())
        if not old_ids:
            projected[artifact_id] = source
            continue
        record = InvestigationRecord.model_validate(source["investigation_record"])
        remaining = tuple(item for item in record.claims if item.claim_id not in old_ids)
        actions = tuple(item for item in record.recommendations if not set(item.conclusion_ids) & old_ids)
        view = record.model_copy(update={"claims": remaining, "recommendations": actions})
        analysis, solution, refs = render_investigation(view)
        facts = tuple(item for item in source.get("facts", ()) if item.get("claim_id") not in old_ids)
        projected[artifact_id] = {
            **source, "facts": facts, "diagnosis_rationale": analysis,
            "solution": {"long_term_remediations": (solution,) if solution else ()},
            "evidence_refs": refs,
            "root_cause": {"effective_level": "INCONCLUSIVE"},
        }
    return projected, tuple(history)
