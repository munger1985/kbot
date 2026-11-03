"""当前目标与案例复用判断；模型仅判断语义条件，身份和环境仍由服务端核验。"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import re
from uuid import UUID

from aiops_agent.adapters.model_serving import AIOpsModelError
from aiops_agent.application.case_redaction import CaseRedactor
from aiops_agent.contracts.case_applicability import CaseApplicabilityAssessment
from aiops_agent.playbooks import canonical_hash
from platform_clients.model import AIModelConfigClientError
from platform_core.prompts import PromptIntegrityError


ENVIRONMENT_FIELDS = ("database_type", "database_version", "topology")
ACTION_PURPOSES = frozenset({"PLAN", "CHANGE", "VERIFY"})


@dataclass(frozen=True)
class CaseSearchContext:
    fingerprint: str
    agent_version_id: str
    model_id: str | None
    facts: tuple[dict, ...]


def _text(value):
    text = value.strip() if isinstance(value, str) else ""
    return text if text and text.upper() != "UNKNOWN" else None


async def load_case_search_context(*, uow, domain_id, agent_id, target_id):
    """只读取当前授权 Agent 和 Target 的安全技术字段，不接收请求字段冒充现场事实。"""
    agent = await uow.agents.get(domain_id=domain_id, agent_id=UUID(str(agent_id)))
    if agent is None or agent.status != "ACTIVE" or agent.current_version_id is None:
        return None
    version = await uow.agents.version(agent_id=UUID(str(agent_id)), agent_version_id=agent.current_version_id)
    if version is None:
        return None
    model_id = dict(version.models_json or {}).get("planner_llm")
    identity = {"agent_version_id": str(agent.current_version_id), "model_id": str(model_id) if model_id else None}
    facts = []
    if target_id is not None:
        target = await uow.targets.get_scoped(domain_id=domain_id, target_id=target_id)
        if target is None or target.status != "ENABLED":
            return None
        capabilities = dict(target.capabilities_json or {})
        values = {
            "database_type": target.db_type, "database_version": target.version_code,
            "topology": capabilities.get("topology") or getattr(target, "deployment_type", None),
            "database_role": getattr(target, "db_role", None),
            "deployment_type": getattr(target, "deployment_type", None),
        }
        facts = [{"fact_id": "target." + key, "field": key, "value": value, "authority": "CURRENT_TARGET"}
            for key, raw in values.items() if (value := _text(raw)) is not None and value.upper() != "UNKNOWN"]
        identity.update(target_id=str(target_id), target_row_version=target.row_version)
    identity["facts"] = facts
    return CaseSearchContext(canonical_hash(identity), str(agent.current_version_id),
        str(model_id) if model_id else None, tuple(facts))


def case_conditions(profile):
    """保留每条前提、禁忌及验证限制的独立身份；不做关键词分类或合并丢失警告。"""
    values = []
    for kind, field in (("REQUIRED", "applicability"), ("FORBIDDEN", "contraindications")):
        values.extend({"condition_id": f"{field}:{index}", "kind": kind, "text": str(text)}
            for index, text in enumerate(profile.get(field) or (), 1))
    values.extend({"condition_id": f"verification:{index}", "kind": "LIMITATION", "text": str(text)}
        for index, text in enumerate((profile.get("verification") or {}).get("causal_limitations") or (), 1))
    for effect in profile.get("action_effects") or ():
        values.extend({"condition_id": f"action:{effect['action_ordinal']}:limitation:{index}",
            "kind": "LIMITATION", "text": str(text)}
            for index, text in enumerate(effect.get("limitations") or (), 1))
    return values


def _normalize(value):
    return " ".join(str(value).upper().split())


def _version(value):
    normalized = _normalize(value)
    match = re.fullmatch(r"([0-9]+)(?:\.[0-9]+)*[A-Z]*", normalized)
    return match[1] if match else None


def case_reuse_decision(*, profile, context, judgments=(), semantic_status="NOT_ASSESSED", semantic_provenance=None):
    """匹配不授予执行权限；明确冲突与未知条件不得作为当前处置推荐。"""
    facts = {fact["field"]: fact for fact in context.facts}
    environment = dict(profile.get("environment") or {})
    dimensions = []
    for field in ENVIRONMENT_FIELDS:
        expected = _text(environment.get(field))
        fact = facts.get(field)
        actual = fact["value"] if fact else None
        normalize = _version if field == "database_version" else _normalize
        expected_key = normalize(expected) if expected is not None else None
        actual_key = normalize(actual) if actual is not None else None
        state = "UNKNOWN" if expected_key is None or actual_key is None else (
            "MATCHED" if expected_key == actual_key else "CONFLICT")
        dimensions.append({"dimension": field, "state": state, "case_value": expected,
            "current_value": actual, "fact_id": fact["fact_id"] if fact else None})
    conditions = case_conditions(profile)
    by_id = {item["condition_id"]: item for item in judgments}
    evaluated = [{**item, **by_id.get(item["condition_id"], {
        "state": "UNKNOWN", "fact_ids": [], "explanation": "本条尚无可核验的当前条件判断",
    })} for item in conditions]
    # 缺少登记不等于无风险；LIMITATION 描述限制已存在，不由模型宣布限制消失。
    unknown = any(item["state"] == "UNKNOWN" for item in dimensions) or any(
        item["state"] == "UNKNOWN" or item["kind"] == "LIMITATION" for item in evaluated)
    unknown = unknown or not profile.get("applicability") or not profile.get("contraindications") or bool(profile.get("unresolved_questions"))
    conflict = any(item["state"] == "CONFLICT" for item in dimensions) or any(
        item["kind"] == "REQUIRED" and item["state"] == "NOT_MET" for item in evaluated)
    prohibited = any(item["kind"] == "FORBIDDEN" and item["state"] == "MET" for item in evaluated)
    status = "PROHIBITED" if prohibited else "CONFLICT" if conflict else "UNKNOWN" if unknown else "MATCHED"
    kind = profile["case_kind"]
    verified = status == "MATCHED" and kind in {"VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION"}
    controlled = str((profile.get("verification") or {}).get("controlled_verification_ref") or "").removeprefix("artifact:")
    effects = {item["action_ordinal"]: item for item in profile.get("action_effects") or ()}
    actions = []
    for ordinal, _action in enumerate(profile.get("actions") or (), 1):
        effect = effects.get(ordinal) or {}
        linked = bool(controlled) and controlled in {
            str(value).removeprefix("artifact:") for value in effect.get("evidence_refs") or ()}
        eligible = verified and linked and effect.get("result") in {"RESOLVED", "IMPROVED"} and not effect.get("limitations")
        actions.append({"action_ordinal": ordinal, "action_recommendation_allowed": eligible,
            "effect_result": effect.get("result", "NOT_VERIFIED"),
            "controlled_effect_linked": linked, "execution_authorized": False})
    allowed = any(item["action_recommendation_allowed"] for item in actions)
    action_use = "WARNING_ONLY" if kind == "NEGATIVE_EXPERIENCE" else (
        "REFERENCE_REQUIRES_LOCAL_VALIDATION" if allowed else "REFERENCE_ONLY_NO_ACTION_RECOMMENDATION")
    return {
        "schema_version": "OPS_CASE_REUSE_DECISION.v1", "context_fingerprint": context.fingerprint,
        "agent_version_id": context.agent_version_id, "status": status, "dimensions": dimensions,
        "conditions": evaluated, "semantic_status": semantic_status,
        "semantic_provenance": semantic_provenance or {},
        "action_recommendation_allowed": allowed, "action_use": action_use,
        "actions": actions,
        "unresolved_questions": list(profile.get("unresolved_questions") or ()),
        "requires_current_verification": True, "execution_authorized": False,
        "historical_knowledge_is_current_evidence": False,
    }


def _validate_assessment(*, output, context, selected):
    if output.context_fingerprint != context.fingerprint:
        raise ValueError("适用性判断没有绑定当前目标快照")
    expected = {item["asset_version_id"]: item["conditions"] for item in selected}
    actual = {item.asset_version_id: item for item in output.cases}
    if len(actual) != len(output.cases) or set(actual) != set(expected):
        raise ValueError("适用性判断必须覆盖且只能覆盖所选案例")
    known_facts = {fact["fact_id"] for fact in context.facts}
    result = {}
    for identity, conditions in expected.items():
        allowed = {item["condition_id"]: item for item in conditions}
        judgments = actual[identity].conditions
        if len({item.condition_id for item in judgments}) != len(judgments) or {item.condition_id for item in judgments} != set(allowed):
            raise ValueError("适用性判断缺少、重复或增加条件")
        for judgment in judgments:
            if judgment.quote != allowed[judgment.condition_id]["text"]:
                raise ValueError("适用性判断必须逐字引用原条件")
            if not set(judgment.fact_ids) <= known_facts or (
                judgment.state != "UNKNOWN" and not judgment.fact_ids
            ):
                raise ValueError("已知判断必须引用当前可信事实，用户问题和历史案例不能充当现场事实")
        result[identity] = [item.model_dump(mode="json") for item in judgments]
    return result


async def assess_case_candidates(service, *, candidates, context, request, model_scope, max_cases):
    """以当前 Agent 显式 planner_llm 单批判断；失效或超过预算时保持未知，不借用其他模型。"""
    selected, excluded = [], []
    for item in candidates:
        conditions = case_conditions(item["profile"])
        if conditions and len(conditions) <= 96 and len(selected) < max_cases:
            selected.append({"asset_version_id": item["asset_version_id"], "conditions": conditions})
        else:
            excluded.append(item["asset_version_id"])
    if not selected:
        return {}, "NOT_REQUIRED", bool(excluded), {}
    if not context.facts:
        return {}, "NO_CURRENT_TARGET", True, {}
    structured_model = getattr(service, "_structured_model", None)
    prompts = getattr(service, "_prompt_registry", None)
    catalog = getattr(service, "_case_model_catalog", None)
    if not context.model_id or structured_model is None or prompts is None or catalog is None:
        return {}, "UNAVAILABLE", True, {}
    try:
        definition = await catalog.get_model(UUID(context.model_id))
        name = _text(definition.get(service._case_model_name_field))
        if name is None:
            raise ValueError("当前 Agent 的规划模型没有可用技术名称")
        prompt = await prompts.resolve("case_applicability")
        payload = {"context_fingerprint": context.fingerprint, "current_facts": list(context.facts),
            "query_hint_not_current_evidence": CaseRedactor().text(request.query), "cases": selected}
        if len(str(payload)) > 100_000 or sum(len(item["conditions"]) for item in selected) > 512:
            return {}, "BUDGET_EXCEEDED", True, {}
        model_ref = {"model_id": context.model_id, "technical_name": name, "revision": context.model_id,
            **{key: str(value) for key, value in model_scope.items()}}
        generated = await asyncio.wait_for(structured_model.generate_structured(
            purpose="operations_knowledge.case_applicability", output_model=CaseApplicabilityAssessment,
            model_snapshot=model_ref, prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload=payload, deadline=datetime.now(UTC) + timedelta(seconds=30),
            idempotency_key="case-applicability:" + canonical_hash({
                "input": payload, "model": model_ref, "prompt": prompt.ref(),
            }),
        ), timeout=30)
        output = CaseApplicabilityAssessment.model_validate(generated.output)
        return _validate_assessment(output=output, context=context, selected=selected), "ASSESSED", bool(excluded), {
            "model": model_ref, "prompt": prompt.ref(), "input_hash": canonical_hash(payload),
            "output_hash": canonical_hash(output.model_dump(mode="json")),
        }
    except (AIOpsModelError, AIModelConfigClientError, PromptIntegrityError, TimeoutError, KeyError, TypeError, ValueError, LookupError):
        return {}, "UNAVAILABLE", True, {}
