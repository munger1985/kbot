"""功能入口的固定目录准入。"""

from __future__ import annotations

from collections.abc import Iterable
import re

from aiops_agent.application.errors import AIOpsApplicationError


class StarterAdmissionError(AIOpsApplicationError):
    """明确的功能准入缺口，不重试，也不展开原始异常链。"""

    def __init__(self, *, code: str, message: str, safe_detail: str):
        super().__init__(code=code, message=message, status_code=422, retryable=False)
        self.stage = "CHECK_STARTER_ADMISSION"
        self.cause_type = type(self).__name__
        self.safe_detail = safe_detail
        self.location = "starter-admission"


def _safe_identifiers(values: Iterable[str]) -> tuple[str, ...]:
    """诊断字段只允许目录标识，不把任意输入拼入日志。"""
    return tuple(sorted({
        str(value) if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", str(value))
        else "<invalid-identifier>"
        for value in values
    }))


def require_starter_candidates(
    *,
    tool_ids: tuple[str, ...],
    tools: tuple[dict, ...],
    playbook_ids: tuple[str, ...],
    playbooks: tuple[dict, ...],
) -> None:
    """固定入口的缺失工具是准入缺口，不作为模型虚构目录或内部异常重试。"""
    missing_tools = _safe_identifiers(
        set(tool_ids) - {str(item["tool_id"]) for item in tools},
    )
    missing_playbooks = _safe_identifiers(
        set(playbook_ids) - {str(item["playbook_id"]) for item in playbooks},
    )
    if not missing_tools and not missing_playbooks:
        return
    details = []
    if missing_tools:
        details.append("missing-tools:" + ",".join(missing_tools))
    if missing_playbooks:
        details.append("missing-playbooks:" + ",".join(missing_playbooks))
    raise StarterAdmissionError(
        code="AIOPS_STARTER_TOOLS_UNAVAILABLE",
        message=(
            "当前目标的可用能力目录不能执行所选功能。"
            "请管理员核对数据库版本、只读连接和诊断工具目录。"
        ),
        safe_detail=";".join(details),
    )
