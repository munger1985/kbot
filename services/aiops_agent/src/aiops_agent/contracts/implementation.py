"""数据库实施方案 Runbook 的结构化契约。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from platform_core.contracts.aiops import ImplementationProfile


class RunbookStatus(StrEnum):
    READY = "READY"
    BLOCKED_BY_REQUIRED_INPUTS = "BLOCKED_BY_REQUIRED_INPUTS"
    BLOCKED_BY_REQUIRED_FACTS = "BLOCKED_BY_REQUIRED_FACTS"
    PARTIAL_EVIDENCE = "PARTIAL_EVIDENCE"


class RunbookApplicability(StrEnum):
    REQUIRED = "REQUIRED"
    ALREADY_SATISFIED = "ALREADY_SATISFIED"
    CONDITIONAL = "CONDITIONAL"
    BLOCKED = "BLOCKED"


class RunbookParameterStatus(StrEnum):
    USER_SUPPLIED = "USER_SUPPLIED"
    VERIFIED = "VERIFIED"
    DERIVED = "DERIVED"


class RunbookCommandType(StrEnum):
    SQLPLUS = "SQLPLUS"
    RMAN = "RMAN"
    DGMGRL = "DGMGRL"
    SHELL = "SHELL"
    CONFIG = "CONFIG"
    MANUAL = "MANUAL"


class RunbookExecutor(StrEnum):
    SQLPLUS = "SQLPLUS"
    RMAN = "RMAN"
    DGMGRL = "DGMGRL"
    BASH = "BASH"
    CRSCTL = "CRSCTL"
    SRVCTL = "SRVCTL"
    ASMCMD = "ASMCMD"
    DBCA = "DBCA"
    OPATCH = "OPATCH"
    DATAPUMP = "DATAPUMP"
    MANUAL = "MANUAL"


class RunbookRiskLevel(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RunbookFactSource(StrEnum):
    TARGET_FACT = "TARGET_FACT"
    DEPLOYMENT_TOPOLOGY = "DEPLOYMENT_TOPOLOGY"
    HOST_COLLECTOR = "HOST_COLLECTOR"
    POLICY_TEMPLATE = "POLICY_TEMPLATE"
    EXPLICIT_USER_DECISION = "EXPLICIT_USER_DECISION"


class RunbookRequiredInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    label: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=1000)
    placeholder: str = Field(min_length=4, max_length=128)
    required: bool = True


class RunbookResolvedParameter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")
    label: str = Field(min_length=1, max_length=128)
    value: str = Field(min_length=1, max_length=2048)
    status: RunbookParameterStatus
    source: str = Field(min_length=1, max_length=256)


class RunbookGenerationContext(BaseModel):
    """固化文档生成入口及用户显式覆盖值，供页面重新生成。"""

    model_config = ConfigDict(extra="forbid")

    starter_id: str | None = Field(default=None, max_length=128)
    catalog_version: str | None = Field(default=None, max_length=32)
    supplied_parameters: dict[str, Any] = Field(default_factory=dict)


class RunbookCommand(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    command_type: RunbookCommandType
    title: str = Field(min_length=1, max_length=256)
    content: str = Field(min_length=1, max_length=16000)
    notes: tuple[str, ...] = ()
    executor: RunbookExecutor | None = None
    run_as: str | None = Field(default=None, max_length=64)
    node_scope: tuple[str, ...] = ()
    container_name: str | None = Field(default=None, max_length=128)
    working_directory: str | None = Field(default=None, max_length=1024)
    target_path: str | None = Field(default=None, max_length=2048)
    expected_result: tuple[str, ...] = ()
    artifact_ref: str | None = Field(default=None, max_length=256)
    risk_level: RunbookRiskLevel = RunbookRiskLevel.LOW

    @model_validator(mode="after")
    def fill_execution_metadata(self) -> "RunbookCommand":
        """为既有 ADG 编译器补齐 v3 执行元数据。"""
        executor_map = {
            RunbookCommandType.SQLPLUS: RunbookExecutor.SQLPLUS,
            RunbookCommandType.RMAN: RunbookExecutor.RMAN,
            RunbookCommandType.DGMGRL: RunbookExecutor.DGMGRL,
            RunbookCommandType.SHELL: RunbookExecutor.BASH,
            RunbookCommandType.CONFIG: RunbookExecutor.MANUAL,
            RunbookCommandType.MANUAL: RunbookExecutor.MANUAL,
        }
        if self.executor is None:
            self.executor = executor_map[self.command_type]
        if self.run_as is None:
            self.run_as = (
                "SYSDBA"
                if self.executor == RunbookExecutor.SQLPLUS
                else "oracle"
            )
        if not self.node_scope:
            self.node_scope = ("source",)
        return self


class RunbookMissingFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_key: str = Field(pattern=r"^[A-Z][A-Z0-9_.-]{1,127}$")
    resolution_source: RunbookFactSource
    reason: str = Field(min_length=1, max_length=1000)
    blocking_steps: tuple[str, ...] = ()


class RunbookArtifactDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{1,127}$")
    file_name: str = Field(min_length=1, max_length=256)
    relative_path: str = Field(min_length=1, max_length=1024)
    media_type: str = Field(min_length=1, max_length=128)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_mode: str = Field(pattern=r"^0[0-7]{3}$")
    run_as: str = Field(min_length=1, max_length=64)
    target_path: str = Field(min_length=1, max_length=2048)
    description: str = Field(min_length=1, max_length=1000)
    contains_secret: bool = False


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

    schema_version: str = "AIOPS_IMPLEMENTATION_RUNBOOK.v3"
    profile: ImplementationProfile
    title: str = Field(min_length=1, max_length=256)
    status: RunbookStatus
    execution_policy: str = Field(min_length=1, max_length=1000)
    generation: RunbookGenerationContext | None = None
    current_state: tuple[dict[str, Any], ...] = ()
    resolved_parameters: tuple[RunbookResolvedParameter, ...] = ()
    required_inputs: tuple[RunbookRequiredInput, ...] = ()
    missing_facts: tuple[RunbookMissingFact, ...] = ()
    artifacts: tuple[RunbookArtifactDescriptor, ...] = ()
    phases: tuple[RunbookPhase, ...]
    stop_conditions: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    _artifact_payloads: dict[str, str] = PrivateAttr(default_factory=dict)
