"""将脱敏案例编译为稳定、可引用的 KC 正文；不调用模型或读取现场。"""

from dataclasses import dataclass
import hashlib
import json
import re

from aiops_agent.application.case_redaction import CaseRedactor
from aiops_agent.contracts.case_knowledge import CaseClaimSource, CaseInvestigationStep, CaseKnowledgeManifest
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from aiops_agent.playbooks import canonical_hash


PURPOSE_LABELS = {
    "DIAGNOSTIC_REFERENCE": "诊断参考，不代表已验证处置",
    "VERIFIED_RESOLUTION": "已验证解决案例，复用仍须现场核验",
    "VERIFIED_OPTIMIZATION": "已验证优化案例，改善不自动证明根因",
    "NEGATIVE_EXPERIENCE": "失败或无效经验；其中动作不作为推荐措施",
}
RESULT_LABELS = {
    "NOT_VERIFIED": "未验证", "RESOLVED": "人工登记已恢复",
    "IMPROVED": "观察到改善", "NOT_ACHIEVED": "未达预期",
    "ADVERSE": "出现不利效果", "DEGRADED": "退化", "UNCHANGED": "无明显变化",
    "INCONCLUSIVE": "尚不能下结论", "UNKNOWN": "未知",
}
LEVEL_LABELS = {
    "OBSERVED": "直接观测", "SUPPORTED": "有证据支持",
    "HYPOTHESIS": "待验证假设", "UNKNOWN": "未知",
}


@dataclass(frozen=True)
class CaseKnowledgeUnit:
    unit_id: str
    kind: str
    start: int
    end: int
    content_hash: str
    related_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class CaseKnowledgeProjection:
    body: bytes
    content_hash: str
    profile_hash: str
    units: tuple[CaseKnowledgeUnit, ...]
    title: str

    def manifest(self):
        """位置采用正文 Unicode 字符偏移；外部索引仍绑定精确 KC Revision。"""
        return {
            "schema_version": "OPS_CASE_KNOWLEDGE_PROJECTION.v1",
            "title": self.title,
            "body_sha256": self.content_hash,
            "profile_hash": self.profile_hash,
            "offset_unit": "UNICODE_CHARACTER",
            "units": [{
                "unit_id": unit.unit_id, "kind": unit.kind,
                "start": unit.start, "end": unit.end,
                "content_hash": unit.content_hash,
                "related_unit_ids": list(unit.related_unit_ids),
            } for unit in self.units],
        }


def investigation_knowledge(*, artifact, claim_ids):
    """只保留已选择的有效发现及公开限制，不复制日志、观测原文或模型推理字段。"""
    from aiops_agent.contracts.investigation import InvestigationRecord

    record = InvestigationRecord.model_validate(artifact.payload_json["investigation_record"])
    selected = set(claim_ids)
    steps = []
    for claim in record.claims:
        if claim.claim_id not in selected or claim.revision_state != "ACTIVE":
            continue
        source = CaseClaimSource(
            artifact_id=str(artifact.artifact_id), content_hash=artifact.content_hash,
            run_id=record.run_id, record_id=record.record_id, record_revision=record.revision,
            claim_id=claim.claim_id,
            evidence_refs=tuple(dict.fromkeys(ref.evidence_ref for ref in claim.evidence_fields)),
            counterevidence_refs=tuple(dict.fromkeys(ref.evidence_ref for ref in claim.counterevidence_fields)),
        )
        identity = canonical_hash({
            "artifact_id": source.artifact_id, "record_revision": source.record_revision,
            "claim_id": source.claim_id,
        })[:24]
        steps.append(CaseInvestigationStep(
            step_id=f"check-{identity}", kind=claim.kind, subject=claim.subject,
            finding=claim.statement, level=claim.level, applicable_time=claim.applicable_time,
            limitations=claim.limitations, source=source,
        ))
    if {step.source.claim_id for step in steps} != selected:
        raise ValueError("案例检查路径包含未选择或已失效的判断")
    return tuple(steps), tuple(record.unresolved_questions)


def _text(value):
    """正文仅承载字面内容，来源文本不能注入新标题、链接或 HTML。"""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for character in ("\\", "`", "*", "_", "[", "]", "#", "|"):
        text = text.replace(character, "\\" + character)
    return " / ".join(line.strip() for line in text.split("\n") if line.strip())


def _items(values, *, missing):
    values = [str(value) for value in values if str(value).strip()]
    return ["- " + _text(value) for value in values] or [missing]


def _verification_lines(profile):
    from aiops_agent.application.report_lifecycle import report_action_items
    verification = profile.verification
    result = str(verification.get("result") or "NOT_VERIFIED")
    method = str(verification.get("method") or "UNKNOWN")
    lines = [
        "案例整体结果：" + _text(RESULT_LABELS.get(result, result)) + "；不据此推断每项动作的独立效果。",
        "原验证方式：" + _text("人工报告，不等于受控测量" if method == "USER_REPORTED" else method),
        "归因强度：" + _text(verification.get("attribution_strength") or "未确认"),
        "负载／口径可比性：" + _text(verification.get("comparability") or "未确认"),
    ]
    lines.extend(_text(line) for line in report_action_items({
        "executed_actions": [], "verification": verification,
        "action_source": "USER_PROVIDED" if any(
            action.get("source_kind") == "USER_PROVIDED" for action in profile.actions
        ) else None,
    }))
    if verification.get("symptom_recovered") is not None:
        lines.append("原目标症状恢复：" + ("是" if verification["symptom_recovered"] is True else "否"))
    if verification.get("guardrails_passed") is not None:
        lines.append("原验证护栏通过：" + ("是" if verification["guardrails_passed"] is True else "否"))
    for name, measurement in (verification.get("measurements") or {}).items():
        if not isinstance(measurement, dict):
            continue
        values = {key: measurement[key] for key in
            ("role", "metric", "baseline", "after", "relative_change", "unit", "passed")
            if key in measurement}
        lines.append("- 原测量 " + _text(name) + "：" + _text(json.dumps(values, ensure_ascii=False, sort_keys=True)))
    lines.extend(_items(verification.get("causal_limitations") or (), missing="原记录未提供额外归因限制，不代表不存在限制。"))
    return lines


def compile_case_knowledge(*, title, profile):
    """每项动作正文自带前提、整体结果和禁忌；关联表仍供精确上下文闭包使用。"""
    profile = DiagnosisCaseProfile.model_validate(profile)
    purpose = PURPOSE_LABELS[profile.case_kind]
    bounds = [
        "案例用途：" + purpose + "。",
        "适用前提：",
        *_items(profile.applicability, missing="适用条件未完整登记；复用前必须核对当前环境。"),
        "禁忌与风险：",
        *_items(profile.contraindications, missing="禁忌未完整登记；不能视为无风险。"),
        "历史经验不代替本次现场证据，不授予工具或变更权限。",
    ]
    definitions = []
    def add(identity, kind, heading, lines, related=()):
        definitions.append((identity, kind, heading, lines, tuple(related)))

    signature, environment = profile.problem_signature, profile.environment
    add("case-problem", "PROBLEM", "问题与环境", [
        "案例用途：" + purpose + "。",
        *_items(signature.get("normalized_symptoms") or (), missing="未登记问题特征。"),
        *[f"- {_text(key)}：{_text(value)}" for key, value in environment.items() if value is not None],
        *[f"- {_text(key)}：{_text(signature[key])}" for key in
          ("problem_class", "component", "error_codes", "signal_names") if signature.get(key)],
    ], ("case-boundaries",))
    add("case-conclusion", "CONCLUSION", "调查结论与原因强度", [
        "案例用途：" + purpose + "。",
        _text(profile.root_cause.get("summary") or "未确认原因。"),
        "原因等级：" + _text(profile.root_cause.get("effective_level") or "INCONCLUSIVE"),
        "结论来源：" + _text(profile.root_cause.get("trust_level") or "冻结调查；不代表因果已确认"),
    ], ("case-problem", "case-boundaries"))
    for step in profile.investigation_steps:
        add(step.step_id, "INVESTIGATION", "检查与发现：" + _text(step.subject), [
            "发现强度：" + LEVEL_LABELS[step.level] + "；判断类型：" + step.kind,
            _text(step.finding),
            "原适用时间：" + _text(step.applicable_time),
            f"支持观测：{len(step.source.evidence_refs)} 项；反证观测：{len(step.source.counterevidence_refs)} 项。",
            *_items(step.limitations, missing="原记录未列额外限制；仍需现场核验。"),
        ], ("case-conclusion", "case-boundaries"))
    effects = {effect.action_ordinal: effect for effect in profile.action_effects}
    for ordinal, action in enumerate(profile.actions, start=1):
        effect = effects.get(ordinal)
        lines = [
            "案例用途：" + purpose + "。",
            _text(action.get("summary") or "动作描述缺失"),
            "动作来源：" + ("人工登记，不能视为平台受控执行。" if action.get("source_kind") == "USER_PROVIDED"
                else _text(action.get("source_kind") or "未确认")),
            "动作引用：" + ("已登记来源；引用不自动证明受控执行。" if action.get("evidence_refs") else "未登记实际执行证据；不能当作已执行措施。"),
        ]
        if effect is None:
            lines.append("逐项效果：未建立该动作与效果的独立对应关系，不能将整体改善归因于本项。")
        else:
            lines.extend([
                "逐项结果：" + _text(RESULT_LABELS[effect.result]),
                "逐项验证方式：" + _text(effect.method),
                _text(effect.summary),
                *_items(effect.limitations, missing="本项仍需核对原验证条件和当前环境。"),
            ])
        add(f"case-action-{ordinal}", "ACTION_EFFECT", f"登记措施与效果 {ordinal}", [
            *lines, *_verification_lines(profile), *bounds,
        ], ("case-verification", "case-boundaries", "case-problem"))
    if not profile.actions:
        add("case-no-action", "ACTION_EFFECT", "登记的处置动作", [
            "未记录已执行动作，仅作为诊断参考。", *bounds,
        ], ("case-verification", "case-boundaries"))
    add("case-verification", "VERIFICATION", "原结果与验证边界", _verification_lines(profile), ("case-boundaries",))
    add("case-boundaries", "BOUNDARIES", "复用前提与禁忌", bounds)
    add("case-unresolved", "UNRESOLVED", "未决问题与现场核验", [
        *_items(profile.unresolved_questions, missing="未决项目未额外登记；不代表本次现场问题已解决。"),
        "后续诊断须独立采集当前证据；禁止直接执行历史参数或失败措施。",
    ], ("case-boundaries",))

    body = "# " + _text(title) + "\n\n"
    units = []
    for identity, kind, heading, lines, related in definitions:
        text = f"## [{identity}] {heading}\n\n" + "\n\n".join(lines) + "\n\n"
        start = len(body)
        body += text
        units.append(CaseKnowledgeUnit(
            identity, kind, start, len(body),
            hashlib.sha256(text.encode("utf-8")).hexdigest(), related,
        ))
    encoded = body.encode("utf-8")
    shared = profile.model_dump(mode="json", exclude={"source", "redaction", "knowledge_projection"})
    return CaseKnowledgeProjection(
        encoded, hashlib.sha256(encoded).hexdigest(), canonical_hash(shared), tuple(units), title,
    )


def case_knowledge_markdown(*, title, profile):
    """所有生成／修订入口共用；正文不含内部血缘原文，不改写只读来源。"""
    redactor = CaseRedactor()
    redacted_title = redactor.text(title)
    redacted = redactor.profile(profile)
    projection = compile_case_knowledge(title=redacted_title, profile=redacted)
    profile.knowledge_projection = CaseKnowledgeManifest.model_validate(projection.manifest())
    return projection.body


def case_projection_is_current(*, title, profile):
    """发布及读取前核对正文、定位清单和叙述，旧或改动后未重新索引的版本不能冒用清单。"""
    if profile.knowledge_projection is None:
        return False
    expected = CaseKnowledgeManifest.model_validate(
        compile_case_knowledge(title=title, profile=profile).manifest()
    )
    return profile.knowledge_projection == expected


def case_context_for_citation(*, profile, citation, document_version_id,
    max_units=64, max_characters=64000):
    """从命中标题形成语义闭包；只引用已核对 KC 原文件哈希的确切版本。"""
    profile = DiagnosisCaseProfile.model_validate(profile)
    manifest = profile.knowledge_projection
    if manifest is None or not case_projection_is_current(title=manifest.title, profile=profile):
        raise ValueError("案例正文清单未就绪")
    projection = compile_case_knowledge(title=manifest.title, profile=profile)
    unit_map = {unit.unit_id: unit for unit in manifest.units}
    headings = []
    for item in citation.get("items") or ():
        evidence = item.get("evidence") or {}
        headings.extend(evidence.get("heading_path") or ())
    # 这是编译器的结构标识，不按故障关键词猜测命中章节。
    matched = {identity for heading in headings
        for identity in re.findall(r"\[([a-z0-9-]+)\]", str(heading)) if identity in unit_map}
    if not matched:
        raise ValueError("案例命中缺少可追溯的语义单元标题")
    wanted = matched | {"case-problem", "case-conclusion", "case-verification", "case-boundaries", "case-unresolved"}
    pending = list(wanted)
    while pending:
        identity = pending.pop()
        for related in unit_map[identity].related_unit_ids:
            if related not in wanted:
                wanted.add(related)
                pending.append(related)
    selected = [unit for unit in manifest.units if unit.unit_id in wanted]
    if len(selected) > max_units or sum(unit.end - unit.start for unit in selected) > max_characters:
        raise ValueError("案例上下文超过本次读取预算，不能省略前提或禁忌后输出措施")
    body = projection.body.decode("utf-8")
    return {
        "schema_version": "OPS_CASE_CONTEXT.v1",
        "role": "HISTORICAL_KNOWLEDGE_NOT_CURRENT_EVIDENCE",
        "body_sha256": manifest.body_sha256,
        "bundle_revision_id": str(citation["bundle_revision_id"]),
        "document_version_id": str(document_version_id),
        "offset_unit": "UNICODE_CHARACTER",
        "units": [{
            "unit_id": unit.unit_id, "kind": unit.kind, "start": unit.start, "end": unit.end,
            "content_hash": unit.content_hash, "matched": unit.unit_id in matched,
            "quote": body[unit.start:unit.end],
        } for unit in selected],
        "requires_current_verification": True,
        "action_use": "WARNING_ONLY" if profile.case_kind == "NEGATIVE_EXPERIENCE"
            else "REFERENCE_REQUIRES_LOCAL_VALIDATION",
    }
