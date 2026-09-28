"""数据库实施 Runbook 的统一安全和完整性校验。"""

from __future__ import annotations

import hashlib
import re

from aiops_agent.contracts.implementation import ImplementationRunbook


_PLACEHOLDER_PATTERNS = (
    re.compile(r"\$\{[^}]+\}"),
    re.compile(r"<[A-Z][A-Z0-9_.-]{2,63}>"),
    re.compile(r"\b(?:TODO|CHANGEME)\b", re.IGNORECASE),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bpassword\s*=\s*[^\s'\"]+", re.IGNORECASE),
    re.compile(
        r"(?<![A-Za-z0-9_./-])(?:sys|system)/[^@\s]+@",
        re.IGNORECASE,
    ),
)


def validate_implementation_runbook(runbook: ImplementationRunbook) -> None:
    """拒绝伪可执行命令、秘密泄露和悬空脚本引用。"""
    descriptors = {item.artifact_id: item for item in runbook.artifacts}
    payloads = runbook._artifact_payloads
    texts: list[tuple[str, str]] = []
    for phase in runbook.phases:
        for step in phase.steps:
            has_high_risk = False
            for command in (
                *step.commands,
                *step.verification_commands,
                *step.rollback,
            ):
                texts.append((command.command_id, command.content))
                if command.artifact_ref and command.artifact_ref not in descriptors:
                    raise ValueError(
                        f"命令引用了不存在的 Artifact：{command.artifact_ref}"
                    )
                if command.executor is None or not command.run_as or not command.node_scope:
                    raise ValueError(f"命令缺少执行元数据：{command.command_id}")
                if command.risk_level.value in {"HIGH", "CRITICAL"}:
                    has_high_risk = True
            if has_high_risk and not (
                step.verification_commands or step.rollback or step.risks
            ):
                raise ValueError(f"高风险步骤缺少验证、回退或风险边界：{step.step_id}")
    texts.extend((key, value) for key, value in payloads.items())
    for source, text in texts:
        for pattern in _PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"实施文档包含未解析占位符：{source}")
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"实施文档包含疑似秘密：{source}")
    for artifact_id, descriptor in descriptors.items():
        content = payloads.get(artifact_id)
        if content is None:
            raise ValueError(f"Artifact 正文缺失：{artifact_id}")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != descriptor.sha256:
            raise ValueError(f"Artifact 摘要不一致：{artifact_id}")
