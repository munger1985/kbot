"""Action 模板与渲染命令的离线安全校验。"""

from __future__ import annotations

import re

from .contracts import ActionTemplateDefinition


_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")
_FORBIDDEN = re.compile(
    r"(;|--|/\*|\*/|\b(?:begin|declare|call|exec|execute|"
    r"insert|update|delete|merge|drop|truncate|grant|revoke)\b)",
    re.IGNORECASE,
)


def validate_action_template(
    command: str, definition: ActionTemplateDefinition
) -> None:
    text = command.strip()
    if not text or len(text) > 4000 or "\n\n" in text:
        raise ValueError("Action 命令模板长度或结构无效")
    if _FORBIDDEN.search(text):
        raise ValueError("Action 命令模板包含禁止结构")
    placeholders = _PLACEHOLDER.findall(text)
    expected = [item.name for item in definition.parameters]
    if sorted(placeholders) != sorted(expected):
        raise ValueError("Action 命令占位符与参数定义不一致")
    if definition.action_template_id != "db.session.terminate":
        raise ValueError("首期 Action Catalog 仅允许单会话终止")
    if definition.db_type == "ORACLE":
        allowed = re.fullmatch(
            r"ALTER SYSTEM DISCONNECT SESSION "
            r"'\{\{session_id\}\},\{\{serial_number\}\},"
            r"@\{\{instance_id\}\}' IMMEDIATE",
            text,
            re.IGNORECASE,
        )
    else:
        allowed = re.fullmatch(
            r"KILL CONNECTION \{\{session_id\}\}",
            text,
            re.IGNORECASE,
        )
    if allowed is None:
        raise ValueError("Action 命令不在首期精确模板 Allowlist")


def validate_rendered_action(command: str, *, db_type: str) -> None:
    if _PLACEHOLDER.search(command) or _FORBIDDEN.search(command):
        raise ValueError("渲染后的 Action 命令结构无效")
    if db_type == "ORACLE":
        pattern = (
            r"ALTER SYSTEM DISCONNECT SESSION "
            r"'[1-9][0-9]*,[1-9][0-9]*,@[1-9][0-9]*' IMMEDIATE"
        )
    else:
        pattern = r"KILL CONNECTION [1-9][0-9]*"
    if re.fullmatch(pattern, command, re.IGNORECASE) is None:
        raise ValueError("渲染后的 Action 命令越过精确 Allowlist")
