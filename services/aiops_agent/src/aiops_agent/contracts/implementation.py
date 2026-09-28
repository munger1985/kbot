"""数据库实施方案 Runbook 的结构化契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from platform_core.contracts.aiops import ImplementationProfile


class RunbookStatus(StrEnum):
    READY = "READY"
    READY_WITH_REQUIRED_INPUTS = "READY_WITH_REQUIRED_INPUTS"
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"


class RunbookApplicability(StrEnum):
    REQUIRED = "REQUIRED"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    CONDITIONAL = "CONDITIONAL"


class RunbookCommandType(StrEnum):
    SQLPLUS = "SQLPLUS"
    RMAN = "RMAN"
    DGMGRL = "DGMGRL"
    SHELL = "SHELL"
    CONFIG = "CONFIG"
    MANUAL = "MANUAL"


class RunbookRequiredInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    label: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    placeholder: str = Field(min_length=4, max_length=128)
    required: bool = True


class RunbookCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    command_type: RunbookCommandType
    title: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=16000)
    notes: tuple[str, ...] = ()


class RunbookStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    title: str = Field(min_length=1, max_length=256)
    applicability: RunbookApplicability
    rationale: str = Field(min_length=1, max_length=2000)
    commands: tuple[RunbookCommand, ...] = ()
    verification_commands: tuple[RunbookCommand, ...] = ()
    rollback: tuple[RunbookCommand, ...] = ()
    risks: tuple[str, ...] = ()
    required_inputs: tuple[str, ...] = ()


class RunbookPhase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    title: str = Field(min_length=1, max_length=256)
    objective: str = Field(min_length=1, max_length=1000)
    steps: tuple[RunbookStep, ...]


class ImplementationRunbook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "AIOPS_IMPLEMENTATION_RUNBOOK.v1"
    profile: ImplementationProfile
    title: str = Field(min_length=1, max_length=256)
    status: RunbookStatus
    execution_policy: str = Field(min_length=1, max_length=1000)
    current_state: tuple[dict[str, Any], ...] = ()
    required_inputs: tuple[RunbookRequiredInput, ...] = ()
    phases: tuple[RunbookPhase, ...]
    stop_conditions: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
