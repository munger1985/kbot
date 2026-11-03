"""知识业务重排；有界加分不能放宽精确发布、来源或措施资格门禁。"""

from datetime import UTC, datetime
import math


RANKING_POLICY = "OPS_KNOWLEDGE_RANKING.v1"
# 首版离线策略，不是训练或真实质量评测所得权重；调整须同步合同测试与质量集。
FACTOR_WEIGHTS = {
    "current_applicability": 0.06,
    "evidence_structure": 0.04,
    "verification_linkage": 0.04,
    "purpose_fit": 0.04,
    "version_freshness": 0.02,
}
FRESHNESS_HALF_LIFE_DAYS = 365


def _ratio(values):
    return sum(values) / len(values) if values else 0.0


def _case_factors(profile, decision, purpose):
    steps = list(profile.get("investigation_steps") or ())
    evidence = _ratio([step.get("level") in {"OBSERVED", "SUPPORTED"}
        and bool((step.get("source") or {}).get("evidence_refs")) for step in steps])
    verification = profile.get("verification") or {}
    controlled = str(verification.get("controlled_verification_ref") or "").removeprefix("artifact:")
    effects = {item["action_ordinal"]: item for item in profile.get("action_effects") or ()}
    # 不奖励“成功”标签或任意方法名称；失败测量与正向测量使用相同的闭合要求。
    comparable = bool(controlled) and verification.get("comparability") == "COMPARABLE"
    measured = []
    for ordinal, _ in enumerate(profile.get("actions") or (), 1):
        effect = effects.get(ordinal) or {}
        measured.append(comparable and bool(effect.get("method"))
            and effect.get("result") in {"RESOLVED", "IMPROVED", "NOT_ACHIEVED", "ADVERSE", "DEGRADED", "UNCHANGED"}
            and controlled in {str(ref).removeprefix("artifact:") for ref in effect.get("evidence_refs") or ()}
            and not effect.get("limitations"))
    linkage = _ratio(measured)
    status = decision.get("status", "UNKNOWN")
    # 请求环境提示、空过滤器推导的 MATCHED 均不能充当当前可信目标判断。
    applicability = 1.0 if status == "MATCHED" else 0.0
    if purpose in {"DIAGNOSE", "EXPLAIN"}:
        fit = float(bool(evidence))
    elif purpose in {"PLAN", "CHANGE"}:
        fit = float(decision.get("action_recommendation_allowed") is True
            or decision.get("action_use") == "WARNING_ONLY")
    elif purpose == "VERIFY":
        fit = float(bool(linkage))
    else:
        fit = 0.0
    return evidence, linkage, applicability, fit, status


def _manual_factors(profile, purpose):
    procedures = list(profile.get("procedures") or ())
    steps = [step for item in procedures for step in item.get("steps") or ()]
    source_ids = set(profile.get("source_evidence_ids") or ())
    grounded = lambda step: bool(source_ids.intersection(step.get("source_evidence_ids") or ()))
    evidence = _ratio([grounded(step) for step in steps])
    linked_procedures = [item for item in procedures
        if any(grounded(step) for step in item.get("steps") or ())]
    linkage = _ratio([bool(item.get("validations")) for item in linked_procedures])
    if purpose in {"DIAGNOSE", "EXPLAIN"}:
        fit = float(bool(source_ids))
    elif purpose == "VERIFY":
        fit = float(bool(linkage))
    else:
        fit = float(any(item.get("intent") == purpose for item in linked_procedures))
    # 资料现有范围筛选不是当前环境的语义判定；本策略不替它凭空宣布相符。
    return evidence, linkage, 0.0, fit, "NOT_ASSESSED"


def knowledge_rank_scorecard(*, candidate, result, kc_relevance, purpose, now):
    """各因子可解释且总增益不超过 20%；零相关性不能凭新鲜度或类型制造命中。"""
    profile = candidate["profile"]
    if candidate["asset_kind"] == "DIAGNOSIS_CASE":
        values = _case_factors(profile, result.get("applicability_decision") or {}, purpose)
    else:
        values = _manual_factors(profile, purpose)
    evidence, linkage, applicability, fit, status = values
    created = candidate.get("created_at")
    age = None
    if isinstance(created, datetime) and created.tzinfo is not None and created.utcoffset() is not None:
        seconds = (now - created.astimezone(UTC)).total_seconds()
        if seconds >= 0:
            age = seconds / 86400
    freshness = 1.0 / (1.0 + age / FRESHNESS_HALF_LIFE_DAYS) if age is not None else 0.0
    factors = {"current_applicability": applicability, "evidence_structure": evidence,
        "verification_linkage": linkage, "purpose_fit": fit, "version_freshness": freshness}
    try:
        relevance = float(kc_relevance)
    except (TypeError, ValueError):
        relevance = 0.0
    if not math.isfinite(relevance) or relevance < 0:
        relevance = 0.0
    bonus = sum(FACTOR_WEIGHTS[key] * value for key, value in factors.items())
    return {
        "schema_version": RANKING_POLICY,
        "kc_relevance": relevance, "factors": factors, "weights": dict(FACTOR_WEIGHTS),
        "bonus": bonus, "score": relevance * (1.0 + bonus),
        "applicability_state": status, "version_age_days": age,
        "freshness_basis": "ASSET_VERSION_CREATED_AT",
        "freshness_half_life_days": FRESHNESS_HALF_LIFE_DAYS,
        "purpose": purpose, "assessed_at": now.isoformat(),
        "quality_empirically_validated": False,
        "historical_knowledge_is_current_evidence": False, "execution_authorized": False,
    }


def knowledge_rank_key(result):
    """相同分数以原相关性及稳定版本 ID 决胜，不偏置资料／案例或重复引用数量。"""
    ranking = result["ranking"]
    return (-ranking["score"], -ranking["kc_relevance"], result["asset_version_id"])
