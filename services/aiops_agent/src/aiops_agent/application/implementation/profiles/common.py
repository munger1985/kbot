"""数据库实施档案的公共确定性编译能力。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from aiops_agent.application.implementation.artifacts import (
    GeneratedRunbookArtifact,
    attach_generated_artifacts,
)
from aiops_agent.contracts.implementation import (
    ImplementationRunbook,
    RunbookApplicability,
    RunbookCommand,
    RunbookCommandType,
    RunbookExecutor,
    RunbookFactSource,
    RunbookMissingFact,
    RunbookParameterStatus,
    RunbookPhase,
    RunbookResolvedParameter,
    RunbookRiskLevel,
    RunbookStatus,
    RunbookStep,
)
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import ImplementationProfile


@dataclass(frozen=True)
class ProfileSpec:
    profile: ImplementationProfile
    title: str
    policy_id: str
    fact_tool_id: str
    required_facts: tuple[tuple[str, str, RunbookFactSource, tuple[str, ...]], ...]
    phases: tuple[tuple[str, str, tuple[str, ...]], ...]
    stop_conditions: tuple[str, ...]
    manual_only: bool = False


def first_row(
    evidence: Iterable[TurnEvidenceFact], tool_id: str
) -> tuple[dict[str, Any], str | None]:
    for fact in evidence:
        if fact.tool_id != tool_id or not fact.rows:
            continue
        names = [str(item.get("name") or "").lower() for item in fact.columns]
        return dict(zip(names, fact.rows[0], strict=False)), fact.evidence_ref
    return {}, None


def value(context: dict[str, Any], facts: dict[str, Any], key: str) -> str:
    """按显式参数、拓扑事实、数据库事实的顺序解析非秘密值。"""
    names = (key, key.lower())
    sources = (
        dict(context.get("implementation_parameters") or {}),
        dict(context.get("deployment_topology") or {}),
        dict(context.get("policy_parameters") or {}),
        facts,
    )
    for source in sources:
        for name in names:
            raw = source.get(name)
            if raw is not None and str(raw).strip():
                candidate = str(raw).strip()
                if _safe_parameter_value(key, candidate):
                    return candidate
    return ""


def _safe_parameter_value(key: str, candidate: str) -> bool:
    """限制进入 SQL、RMAN 和 Shell 的事实字符集，拒绝控制符和命令拼接。"""
    if not candidate or len(candidate) > 2048:
        return False
    if any(marker in candidate for marker in ("\n", "\r", "\x00", "`", "$", ";", "|", "&", "<", ">")):
        return False
    normalized_key = key.upper()
    if normalized_key == "RECOVERY_TARGET_SCN":
        return candidate.isdigit()
    if normalized_key == "RECOVERY_TARGET_TIME":
        return bool(re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", candidate))
    if normalized_key == "SOURCE_SCHEMAS":
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]*(?:,[A-Za-z][A-Za-z0-9_$#]*)*", candidate))
    return bool(re.fullmatch(r"[A-Za-z0-9_+./:@,#=() *-]+", candidate))


def command(
    command_id: str,
    title: str,
    content: str,
    *,
    executor: RunbookExecutor,
    run_as: str,
    node_scope: tuple[str, ...] = ("source",),
    command_type: RunbookCommandType | None = None,
    risk: RunbookRiskLevel = RunbookRiskLevel.LOW,
    expected: tuple[str, ...] = (),
    artifact_ref: str | None = None,
    target_path: str | None = None,
    notes: tuple[str, ...] = (),
) -> RunbookCommand:
    type_map = {
        RunbookExecutor.SQLPLUS: RunbookCommandType.SQLPLUS,
        RunbookExecutor.RMAN: RunbookCommandType.RMAN,
        RunbookExecutor.DGMGRL: RunbookCommandType.DGMGRL,
        RunbookExecutor.MANUAL: RunbookCommandType.MANUAL,
    }
    return RunbookCommand(
        command_id=command_id,
        command_type=command_type or type_map.get(
            executor, RunbookCommandType.SHELL
        ),
        title=title,
        content=content.strip(),
        notes=notes,
        executor=executor,
        run_as=run_as,
        node_scope=node_scope,
        expected_result=expected,
        artifact_ref=artifact_ref,
        risk_level=risk,
        target_path=target_path,
    )


def blocked_step(
    step_id: str,
    title: str,
    objective: str,
    missing_keys: tuple[str, ...],
) -> RunbookStep:
    return RunbookStep(
        step_id=step_id,
        title=title,
        applicability=RunbookApplicability.BLOCKED,
        rationale=objective,
        commands=(command(
            f"{step_id}.facts",
            "登记并重新核验实施事实",
            "在当前 Target 的部署拓扑、主机采集或策略配置中补齐本步骤列出的事实，然后重新生成实施文档。",
            executor=RunbookExecutor.MANUAL,
            run_as="DBA",
            risk=RunbookRiskLevel.LOW,
        ),),
        required_inputs=missing_keys,
        risks=("事实未核验前不得把示例值代入生产命令。",),
    )


def normal_step(
    phase_id: str,
    title: str,
    objective: str,
    commands: tuple[RunbookCommand, ...],
    *,
    risks: tuple[str, ...] = (),
) -> RunbookStep:
    effective_risks = risks
    if not effective_risks and any(
        item.risk_level in {RunbookRiskLevel.HIGH, RunbookRiskLevel.CRITICAL}
        for item in commands
    ):
        effective_risks = (
            "本步骤属于高风险操作，执行前必须完成实时复核和审批，执行后立即按本阶段验收项验证。",
        )
    return RunbookStep(
        step_id=f"{phase_id}.execute",
        title=title,
        applicability=RunbookApplicability.REQUIRED,
        rationale=objective,
        commands=commands,
        risks=effective_risks,
    )


def compile_profile(
    *,
    spec: ProfileSpec,
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any],
    commands_by_phase: dict[str, tuple[RunbookCommand, ...]],
    artifacts: tuple[GeneratedRunbookArtifact, ...] = (),
    derived_parameters: dict[str, str] | None = None,
) -> ImplementationRunbook:
    identity, identity_ref = first_row(evidence, "db.instance.identity")
    facts, fact_ref = first_row(evidence, spec.fact_tool_id)
    merged_facts = {**identity, **facts}
    missing: list[RunbookMissingFact] = []
    resolved: list[RunbookResolvedParameter] = []
    resolved_values = dict(derived_parameters or {})
    for key, reason, source, blocking_steps in spec.required_facts:
        item = value(context, merged_facts, key)
        if not item:
            missing.append(RunbookMissingFact(
                fact_key=key,
                resolution_source=source,
                reason=reason,
                blocking_steps=blocking_steps,
            ))
            continue
        resolved_values[key] = item
        resolved.append(RunbookResolvedParameter(
            key=key,
            label=key.replace("_", " ").title(),
            value=item,
            status=RunbookParameterStatus.VERIFIED,
            source=source,
        ))
    for key, item in sorted(resolved_values.items()):
        if any(parameter.key == key for parameter in resolved):
            continue
        resolved.append(RunbookResolvedParameter(
            key=key,
            label=key.replace("_", " ").title(),
            value=item,
            status=RunbookParameterStatus.DERIVED,
            source=f"策略模板 {spec.policy_id}",
        ))

    missing_by_step: dict[str, list[str]] = {}
    for item in missing:
        for step_id in item.blocking_steps:
            missing_by_step.setdefault(step_id, []).append(item.fact_key)
    phases: list[RunbookPhase] = []
    for phase_id, title, phase_steps in spec.phases:
        steps: list[RunbookStep] = []
        for step_id in phase_steps:
            if step_id in missing_by_step:
                steps.append(blocked_step(
                    step_id,
                    title,
                    "本阶段依赖尚未核验的基础设施或业务事实。",
                    tuple(missing_by_step[step_id]),
                ))
                continue
            phase_commands = commands_by_phase.get(step_id, ())
            if not phase_commands:
                phase_commands = (command(
                    f"{step_id}.checklist",
                    "按本阶段检查表实施并留存证据",
                    "逐项执行文档列出的检查、变更、验证和回退准备；所有实际值以参数附录固化事实为准。",
                    executor=RunbookExecutor.MANUAL,
                    run_as="DBA",
                    risk=RunbookRiskLevel.MEDIUM,
                ),)
            steps.append(normal_step(
                step_id,
                title,
                "完成该阶段的实施、验证和证据留存。",
                phase_commands,
            ))
        phases.append(RunbookPhase(
            phase_id=phase_id,
            title=title,
            objective=f"完成{title}并确认停止条件未触发。",
            steps=tuple(steps),
        ))
    status = RunbookStatus.READY
    if missing:
        status = RunbookStatus.BLOCKED_BY_REQUIRED_FACTS
    elif fact_ref is None or identity_ref is None:
        status = RunbookStatus.PARTIAL_EVIDENCE
    current_state = tuple(
        {"label": key, "value": str(item), "status": "VERIFIED"}
        for key, item in sorted(merged_facts.items())
        if item not in (None, "")
    )
    runbook = ImplementationRunbook(
        profile=spec.profile,
        title=spec.title,
        status=status,
        execution_policy=(
            "本产物是确定性生成的实施操作文档，不自动执行变更。"
            + (
                "该档案只允许人工执行并须经过业务确认和变更审批。"
                if spec.manual_only else
                "高风险步骤必须在实时复核、审批和维护窗口内人工执行。"
            )
        ),
        current_state=current_state,
        resolved_parameters=tuple(resolved),
        missing_facts=tuple(missing),
        phases=tuple(phases),
        stop_conditions=spec.stop_conditions,
        evidence_refs=tuple(
            item for item in (identity_ref, fact_ref) if item is not None
        ),
    )
    return attach_generated_artifacts(runbook, artifacts)
