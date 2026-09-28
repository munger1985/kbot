"""数据库实施档案 Registry，是唯一编译分发入口。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aiops_agent.application.implementation.profiles.adg import compile_adg
from aiops_agent.application.implementation.profiles.adg_drill import compile_adg_drill
from aiops_agent.application.implementation.profiles.clone import compile_clone
from aiops_agent.application.implementation.profiles.datapump import compile_datapump
from aiops_agent.application.implementation.profiles.migration import compile_migration
from aiops_agent.application.implementation.profiles.patch import compile_patch
from aiops_agent.application.implementation.profiles.rac import compile_rac
from aiops_agent.application.implementation.profiles.rman_backup import compile_rman_backup
from aiops_agent.application.implementation.profiles.rman_recovery import compile_rman_recovery
from aiops_agent.application.implementation.profiles.upgrade import compile_upgrade
from aiops_agent.application.implementation.validation import validate_implementation_runbook
from aiops_agent.contracts.implementation import ImplementationRunbook
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import ImplementationProfile


Compiler = Callable[
    [tuple[TurnEvidenceFact, ...], dict[str, Any]], ImplementationRunbook
]
_REGISTRY: dict[ImplementationProfile, Compiler] = {
    ImplementationProfile.ORACLE_ADG_BUILD: compile_adg,
    ImplementationProfile.ORACLE_RAC_BUILD: compile_rac,
    ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD: compile_rman_backup,
    ImplementationProfile.ORACLE_RMAN_RECOVERY: compile_rman_recovery,
    ImplementationProfile.ORACLE_RU_PATCH: compile_patch,
    ImplementationProfile.ORACLE_DATABASE_UPGRADE: compile_upgrade,
    ImplementationProfile.ORACLE_DATABASE_MIGRATION: compile_migration,
    ImplementationProfile.ORACLE_CLONE_REFRESH: compile_clone,
    ImplementationProfile.ORACLE_DATAPUMP_MIGRATION: compile_datapump,
    ImplementationProfile.ORACLE_ADG_DRILL: compile_adg_drill,
}


def registered_implementation_profiles() -> tuple[ImplementationProfile, ...]:
    return tuple(_REGISTRY)


def compile_implementation_runbook(
    *,
    profile: ImplementationProfile,
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any] | None = None,
) -> ImplementationRunbook:
    """选择固定编译器并在固化前执行统一校验。"""
    compiler = _REGISTRY.get(profile)
    if compiler is None:
        raise ValueError(f"不支持的实施方案档案：{profile}")
    runbook = compiler(evidence, dict(context or {}))
    validate_implementation_runbook(runbook)
    return runbook
