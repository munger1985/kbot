"""数据库实施档案的公共确定性编译能力。"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Iterable

from aiops_agent.application.implementation.artifacts import (
    GeneratedRunbookArtifact,
    attach_generated_artifacts,
)
from aiops_agent.contracts.implementation import (
    ImplementationRunbook,
    RunbookApplicability,
    RunbookArtifactDescriptor,
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
        commands=(),
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


def _artifact_deployment_phase(
    artifacts: tuple[RunbookArtifactDescriptor, ...],
) -> RunbookPhase | None:
    """生成下载、解压、权限设置和摘要校验的统一前置阶段。"""
    package_roots: set[str] = set()
    target_roots: set[str] = set()
    for artifact in artifacts:
        relative_parts = artifact.relative_path.strip("/").split("/", 1)
        if len(relative_parts) != 2:
            return None
        package_root, artifact_path = relative_parts
        suffix = "/" + artifact_path
        if not artifact.target_path.endswith(suffix):
            return None
        package_roots.add(package_root)
        target_roots.add(artifact.target_path[: -len(suffix)] or "/")
    if len(package_roots) != 1 or len(target_roots) != 1:
        return None

    package_root = next(iter(package_roots))
    target_root = next(iter(target_roots))
    target_parent = str(PurePosixPath(target_root).parent)
    if PurePosixPath(target_root).name != package_root:
        return None
    archive_path = f"/var/tmp/{package_root}.zip"
    deployment_lines = [
        "command -v unzip >/dev/null",
        f"test -f {shlex.quote(archive_path)}",
        f"install -d -m 0755 {shlex.quote(target_parent)}",
        (
            f"unzip -oq {shlex.quote(archive_path)} "
            f"-d {shlex.quote(target_parent)}"
        ),
    ]
    verification_lines = [f"test -d {shlex.quote(target_root)}"]
    for artifact in artifacts:
        target_path = shlex.quote(artifact.target_path)
        owner = artifact.run_as if artifact.run_as in {"oracle", "grid", "root"} else ""
        if owner:
            deployment_lines.append(
                f"chown {shlex.quote(owner)} {target_path}"
            )
        deployment_lines.append(
            f"chmod {shlex.quote(artifact.file_mode)} {target_path}"
        )
        access_test = "-x" if int(artifact.file_mode, 8) & 0o111 else "-r"
        verification_lines.extend((
            f"test {access_test} {target_path}",
            (
                "printf '%s  %s\\n' "
                f"{shlex.quote(artifact.sha256)} {target_path} "
                "| sha256sum --check -"
            ),
        ))

    manual = command(
        "artifact.package.transfer",
        "下载并传输本轮脚本 ZIP",
        (
            "在当前 Agent 回答中点击“下载脚本 ZIP”。将下载文件传输到需要执行本包文件的"
            f"目标主机，并统一保存为 {archive_path}。下载动作只取得文件，不会自动解压或"
            f"部署；后续命令要求包内 {package_root}/ 最终位于 {target_root}/。"
        ),
        executor=RunbookExecutor.MANUAL,
        command_type=RunbookCommandType.MANUAL,
        run_as="DBA",
        node_scope=("artifact-targets",),
    )
    deploy = command(
        "artifact.package.deploy",
        "解压配套文件并设置清单权限",
        "\n".join(deployment_lines),
        executor=RunbookExecutor.BASH,
        run_as="root",
        node_scope=("artifact-targets",),
        expected=(f"包内文件已部署到 {target_root}。",),
    )
    verify = command(
        "artifact.package.verify",
        "核对配套文件是否存在且内容未变化",
        "\n".join(verification_lines),
        executor=RunbookExecutor.BASH,
        run_as="root",
        node_scope=("artifact-targets",),
        expected=("全部文件权限检查和 SHA-256 校验通过。",),
    )
    return RunbookPhase(
        phase_id="artifact_package",
        title="配套 ZIP 制品部署",
        objective="在任何引用脚本路径的命令执行前，先完成下载、传输、解压和完整性校验。",
        steps=(RunbookStep(
            step_id="artifact_package.deploy",
            title="下载并部署本轮配套文件",
            applicability=RunbookApplicability.REQUIRED,
            rationale=(
                "后续 SQL、RMAN、Shell 或配置命令引用本阶段部署的固定路径；"
                "未完成本阶段时这些路径不存在，不得继续执行。"
            ),
            commands=(manual, deploy),
            verification_commands=(verify,),
            risks=("必须使用当前 Turn 下载的 ZIP，禁止复用其他数据库或历史 Turn 的脚本包。",),
        ),),
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
    runbook = attach_generated_artifacts(runbook, artifacts)
    referenced_artifacts = {
        item.artifact_ref
        for phase_commands in commands_by_phase.values()
        for item in phase_commands
        if item.artifact_ref
    }
    if referenced_artifacts:
        deployment_phase = _artifact_deployment_phase(runbook.artifacts)
        if deployment_phase is not None:
            runbook.phases = (deployment_phase, *runbook.phases)
    return runbook
