"""数据库实施 Runbook 的确定性 Markdown 投影。"""

from __future__ import annotations

import re
from html import escape
from typing import Any


def _items(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _text(value: object, default: str = "-") -> str:
    rendered = str(value or "").strip()
    return escape(rendered or default, quote=False)


def _table_cell(value: object) -> str:
    return _text(value).replace("|", "\\|").replace("\n", "<br>")


def _fenced_code(content: object, language: str) -> list[str]:
    """生成不会与命令正文冲突的 fenced code block。"""
    source = str(content or "").strip()
    longest = max((len(run) for run in re.findall(r"`+", source)), default=0)
    fence = "`" * max(3, longest + 1)
    return [f"{fence}{language}", source, fence]


def _command_language(command_type: object) -> str:
    return {
        "SQLPLUS": "sql",
        "RMAN": "rman",
        "DGMGRL": "dgmgrl",
        "SHELL": "bash",
        "CONFIG": "ini",
        "MANUAL": "text",
    }.get(str(command_type or "").upper(), "text")


def _append_commands(
    lines: list[str],
    commands: object,
    *,
    heading: str,
    manual: bool = False,
) -> None:
    values = _items(commands)
    if not values:
        return
    lines.extend((f"### {heading}", ""))
    for command_value in values:
        command = _mapping(command_value)
        command_type = _text(command.get("command_type"), "COMMAND")
        title = _text(command.get("title"), "命令")
        lines.extend((f"#### {title}" if manual else f"#### {command_type} · {title}", ""))
        metadata = [] if manual else [
            f"执行器 `{_text(command.get('executor'), command_type)}`",
        ]
        metadata.append(f"身份 `{_text(command.get('run_as'))}`")
        if _items(command.get("node_scope")):
            metadata.append(
                "节点 " + "、".join(
                    f"`{_text(item)}`"
                    for item in _items(command.get("node_scope"))
                )
            )
        if command.get("risk_level"):
            metadata.append(f"风险 `{_text(command.get('risk_level'))}`")
        lines.extend(("；".join(metadata), ""))
        if manual:
            lines.extend((_text(command.get("content")), ""))
        else:
            lines.extend(_fenced_code(
                command.get("content"),
                _command_language(command.get("command_type")),
            ))
            lines.append("")
        notes = _items(command.get("notes"))
        expected = _items(command.get("expected_result"))
        if expected:
            lines.extend(("预期结果：", ""))
            lines.extend(f"- {_text(item)}" for item in expected)
            lines.append("")
        if notes:
            lines.append("说明：")
            lines.append("")
            lines.extend(f"- {_text(note)}" for note in notes)
            lines.append("")


def render_implementation_runbook_markdown(payload: dict[str, Any]) -> str:
    """从持久化结构化 Runbook 生成可下载、可复现的 Markdown。"""
    title = _text(payload.get("title"), "数据库实施操作文档")
    lines = [
        f"# {title}",
        "",
        "> KBot 智能运维 · 数据库实施操作文档",
        "",
        "| 项目 | 值 |",
        "| --- | --- |",
        f"| 实施档案 | {_table_cell(payload.get('profile'))} |",
        f"| 文档状态 | {_table_cell(payload.get('status'))} |",
        f"| 契约版本 | {_table_cell(payload.get('schema_version'))} |",
        "",
        "## 执行边界",
        "",
        _text(payload.get("execution_policy")),
        "",
    ]

    phases = _items(payload.get("phases"))
    if phases:
        lines.extend(("## 目录", ""))
        for phase_index, phase_value in enumerate(phases, start=1):
            phase = _mapping(phase_value)
            phase_title = _text(
                phase.get("title") or phase.get("phase_id"), "实施阶段"
            )
            lines.append(
                f"{phase_index}. [{phase_title}](#phase-{phase_index})"
            )
            for step_index, step_value in enumerate(
                _items(phase.get("steps")), start=1
            ):
                step = _mapping(step_value)
                step_title = _text(
                    step.get("title") or step.get("step_id"), "操作步骤"
                )
                lines.append(
                    "   - "
                    f"[{phase_index}.{step_index} {step_title}]"
                    f"(#phase-{phase_index}-step-{step_index})"
                )
        lines.append("")

    group_labels = (
        ("commands", "实施命令"),
        ("verification_commands", "验证命令"),
        ("rollback", "回退命令"),
    )
    for phase_index, phase_value in enumerate(phases, start=1):
        phase = _mapping(phase_value)
        phase_title = _text(
            phase.get("title") or phase.get("phase_id"), "实施阶段"
        )
        lines.extend((
            f'<a id="phase-{phase_index}"></a>',
            "",
            f"## {phase_index}. {phase_title}",
            "",
            _text(phase.get("objective")),
            "",
        ))
        for step_index, step_value in enumerate(
            _items(phase.get("steps")), start=1
        ):
            step = _mapping(step_value)
            step_title = _text(
                step.get("title") or step.get("step_id"), "操作步骤"
            )
            applicability = _text(step.get("applicability"), "UNKNOWN")
            lines.extend((
                f'<a id="phase-{phase_index}-step-{step_index}"></a>',
                "",
                f"### {phase_index}.{step_index} {step_title}",
                "",
                f"**适用性：** `{applicability}`",
                "",
                _text(step.get("rationale")),
                "",
            ))
            required_inputs = _items(step.get("required_inputs"))
            if required_inputs:
                lines.extend((
                    "**所需输入：** "
                    + "、".join(f"`{_text(item)}`" for item in required_inputs),
                    "",
                ))
            for key, label in group_labels:
                values = _items(step.get(key))
                if key == "commands":
                    executable = [
                        item for item in values
                        if "MANUAL" not in {
                            str(_mapping(item).get("command_type") or "").upper(),
                            str(_mapping(item).get("executor") or "").upper(),
                        }
                    ]
                    manual = [
                        item for item in values
                        if "MANUAL" in {
                            str(_mapping(item).get("command_type") or "").upper(),
                            str(_mapping(item).get("executor") or "").upper(),
                        }
                    ]
                    _append_commands(lines, executable, heading=label)
                    _append_commands(
                        lines,
                        manual,
                        heading="人工确认项",
                        manual=True,
                    )
                    continue
                _append_commands(lines, values, heading=label)
            risks = _items(step.get("risks"))
            if risks:
                lines.extend(("### 风险与注意事项", ""))
                lines.extend(f"- {_text(risk)}" for risk in risks)
                lines.append("")

    stop_conditions = _items(payload.get("stop_conditions"))
    if stop_conditions:
        lines.extend(("## 停止条件", ""))
        lines.extend(f"- {_text(item)}" for item in stop_conditions)
        lines.append("")

    state = _items(payload.get("current_state"))
    parameters = _items(payload.get("resolved_parameters"))
    required_inputs = _items(payload.get("required_inputs"))
    missing_facts = _items(payload.get("missing_facts"))
    artifacts = _items(payload.get("artifacts"))
    if state or parameters or required_inputs or missing_facts or artifacts:
        lines.extend(("## 附录：当前状态与实施参数", ""))
    if state:
        lines.extend((
            "### 当前环境摘要",
            "",
            "| 项目 | 当前值 | 状态 |",
            "| --- | --- | --- |",
        ))
        for item_value in state:
            item = _mapping(item_value)
            lines.append(
                f"| {_table_cell(item.get('label'))} | "
                f"{_table_cell(item.get('value'))} | "
                f"{_table_cell(item.get('status'))} |"
            )
        lines.append("")
    if parameters:
        lines.extend((
            "### 已解析实施参数",
            "",
            "| 参数 | 值 | 来源 | 状态 |",
            "| --- | --- | --- | --- |",
        ))
        for item_value in parameters:
            item = _mapping(item_value)
            lines.append(
                f"| {_table_cell(item.get('label') or item.get('key'))} | "
                f"{_table_cell(item.get('value'))} | "
                f"{_table_cell(item.get('source'))} | "
                f"{_table_cell(item.get('status'))} |"
            )
        lines.append("")
    if required_inputs:
        lines.extend((
            "### 实施前必须确认的外部输入",
            "",
            "| 输入 | 说明 | 示例 |",
            "| --- | --- | --- |",
        ))
        for item_value in required_inputs:
            item = _mapping(item_value)
            lines.append(
                f"| {_table_cell(item.get('label') or item.get('key'))} | "
                f"{_table_cell(item.get('description'))} | "
                f"{_table_cell(item.get('placeholder'))} |"
            )
        lines.append("")
    if missing_facts:
        lines.extend((
            "### 缺失的必要事实",
            "",
            "| 事实 | 补齐位置 | 原因 | 阻断步骤 |",
            "| --- | --- | --- | --- |",
        ))
        for item_value in missing_facts:
            item = _mapping(item_value)
            lines.append(
                f"| {_table_cell(item.get('fact_key'))} | "
                f"{_table_cell(item.get('resolution_source'))} | "
                f"{_table_cell(item.get('reason'))} | "
                f"{_table_cell('、'.join(map(str, _items(item.get('blocking_steps')))))} |"
            )
        lines.append("")
    if artifacts:
        lines.extend((
            "### 脚本与配置清单",
            "",
            "| 文件 | 目标路径 | 权限/身份 | SHA256 |",
            "| --- | --- | --- | --- |",
        ))
        for item_value in artifacts:
            item = _mapping(item_value)
            lines.append(
                f"| {_table_cell(item.get('file_name'))} | "
                f"{_table_cell(item.get('target_path'))} | "
                f"{_table_cell(str(item.get('file_mode') or '') + '/' + str(item.get('run_as') or ''))} | "
                f"`{_text(item.get('sha256'))}` |"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
