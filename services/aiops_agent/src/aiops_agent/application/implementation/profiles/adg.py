"""现有整库 ADG 与 26ai DGPDB 确定性编译器入口。"""

from __future__ import annotations

from typing import Any

from aiops_agent.application.implementation.runbooks import compile_adg_runbook
from aiops_agent.contracts.implementation import ImplementationRunbook
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import ImplementationProfile


def compile_adg(
    evidence: tuple[TurnEvidenceFact, ...], context: dict[str, Any]
) -> ImplementationRunbook:
    """保留已验收 ADG/DGPDB 行为，通过 Registry 的唯一入口调用。"""
    return compile_adg_runbook(
        profile=ImplementationProfile.ORACLE_ADG_BUILD,
        evidence=evidence,
        context=context,
    )
