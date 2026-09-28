"""AIOps AI DBA 输入理解、调查计划与工具调用契约。"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from .types import (
    AIOpsContract,
    JsonObject,
    MeasurementSemantics,
    SufficiencyStatus,
)


INPUT_ENVELOPE_SCHEMA_VERSION = "aiops.input-envelope.v1"
TASK_FRAME_SCHEMA_VERSION = "aiops.task-frame.v1"
INVESTIGATION_PLAN_SCHEMA_VERSION = "aiops.investigation-plan.v1"
INVESTIGATION_ASSESSMENT_SCHEMA_VERSION = "aiops.investigation-assessment.v1"
COMPACT_PLANNING_SCHEMA_VERSION = "aiops.compact-planning.v3"


class InputContentType(StrEnum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    FILE = "FILE"
    SQL_OUTPUT = "SQL_OUTPUT"
    COMMAND_OUTPUT = "COMMAND_OUTPUT"
    LOG = "LOG"


class InputContent(AIOpsContract):
    content_type: InputContentType
    text: str | None = Field(default=None, max_length=128_000)
    upload_id: str | None = Field(default=None, max_length=64)
    media_type: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_content(self) -> "InputContent":
        if (self.text is None) == (self.upload_id is None):
            raise ValueError("输入内容必须且只能提供文字或上传文件")
        if self.content_type == InputContentType.TEXT and self.text is None:
            raise ValueError("TEXT 输入必须提供文字")
        if self.content_type in {
            InputContentType.IMAGE,
            InputContentType.FILE,
        } and self.upload_id is None:
            raise ValueError("IMAGE 和 FILE 输入必须提供上传文件引用")
        if self.content_type not in {
            InputContentType.IMAGE,
            InputContentType.FILE,
        } and self.text is None:
            raise ValueError("粘贴材料必须提供文字正文")
        return self


CONVERSATION_UPLOAD_MAX_BYTES = 20 * 1024 * 1024


class ConversationUploadReceipt(AIOpsContract):
    upload_id: str = Field(min_length=1, max_length=64)
    file_name: str = Field(min_length=1, max_length=256)
    media_type: str = Field(min_length=1, max_length=128)
    byte_size: int = Field(ge=1)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: datetime


class MaterialKind(StrEnum):
    QUESTION = "QUESTION"
    ORACLE_ALERT_LOG = "ORACLE_ALERT_LOG"
    DATABASE_LOG = "DATABASE_LOG"
    SQL_RESULT = "SQL_RESULT"
    COMMAND_RESULT = "COMMAND_RESULT"
    METRIC_SNAPSHOT = "METRIC_SNAPSHOT"
    CONFIGURATION = "CONFIGURATION"
    SCREENSHOT = "SCREENSHOT"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class InputMaterial(AIOpsContract):
    item_no: int = Field(ge=1)
    material_kind: MaterialKind
    summary: str = Field(min_length=1, max_length=2000)
    key_facts: tuple[str, ...] = ()
    confidence: float = Field(ge=0, le=1)
    contains_user_evidence: bool = False


class TurnInputEnvelope(AIOpsContract):
    schema_version: str = INPUT_ENVELOPE_SCHEMA_VERSION
    materials: tuple[InputMaterial, ...]
    explicit_question: str | None = Field(default=None, max_length=4000)
    inferred_question: str | None = Field(default=None, max_length=4000)
    supplied_evidence_summary: tuple[str, ...] = ()
    ambiguities: tuple[str, ...] = ()


class TaskObjective(StrEnum):
    UNDERSTAND = "UNDERSTAND"
    DIAGNOSE = "DIAGNOSE"
    EXPLAIN = "EXPLAIN"
    ASSESS = "ASSESS"
    COMPARE = "COMPARE"
    PLAN = "PLAN"
    CHANGE = "CHANGE"
    VERIFY = "VERIFY"


class ActionIntent(StrEnum):
    """用户对登记动作的真实诉求，不与执行权限混为一谈。"""

    NONE = "NONE"
    ADVISORY = "ADVISORY"
    EXECUTE = "EXECUTE"


class DiagnosticProfile(StrEnum):
    """由语义路由选择、由服务端展开的确定性诊断能力档案。"""

    GENERAL = "GENERAL"
    SINGLE_SQL_PERFORMANCE = "SINGLE_SQL_PERFORMANCE"


class ImplementationProfile(StrEnum):
    """由语义路由选择、由服务端编译的实施方案档案。"""

    NONE = "NONE"
    ORACLE_ADG_BUILD = "ORACLE_ADG_BUILD"


class EvidenceSourceStrategy(StrEnum):
    """同类事实在监控与数据库之间的确定性取证顺序。"""

    MONITORING_FIRST = "MONITORING_FIRST"
    DATABASE_FIRST = "DATABASE_FIRST"
    COMBINED = "COMBINED"


class TemporalAnalysisMode(StrEnum):
    """区分当前快照、历史分析以及基于历史的未来预测。"""

    CURRENT = "CURRENT"
    HISTORICAL = "HISTORICAL"
    HISTORICAL_AND_FORECAST = "HISTORICAL_AND_FORECAST"


class CompletionRequirement(AIOpsContract):
    """用户明确要求交付、且必须由真实证据满足的一项完成义务。"""

    requirement_id: str = Field(pattern=r"^r[0-9]+$")
    description: str = Field(min_length=1, max_length=1000)
    accepted_tool_ids: tuple[str, ...] = Field(default=(), max_length=8)
    accepted_evidence_kinds: tuple[str, ...] = Field(
        default=(), max_length=8
    )
    minimum_successful_results: int = Field(default=1, ge=1, le=8)

    @model_validator(mode="after")
    def validate_selectors(self) -> "CompletionRequirement":
        if not self.accepted_tool_ids and not self.accepted_evidence_kinds:
            raise ValueError("完成义务必须声明可验证的工具或证据类型")
        return self


class TaskFrame(AIOpsContract):
    schema_version: str = TASK_FRAME_SCHEMA_VERSION
    objectives: tuple[TaskObjective, ...] = Field(
        min_length=1, max_length=8
    )
    problem_statement: str = Field(min_length=1, max_length=4000)
    database_context: JsonObject = Field(default_factory=dict)
    time_scope: str | None = Field(default=None, max_length=512)
    requested_window_seconds: int | None = Field(
        default=None,
        ge=60,
        le=31_536_000,
        description=(
            "用户明确时间窗口对应的秒数；MONITORING_FIRST且用户未明确时"
            "使用Prometheus最大保留范围2592000秒，其他情况为空。"
        ),
    )
    temporal_analysis_mode: TemporalAnalysisMode = (
        TemporalAnalysisMode.CURRENT
    )
    forecast_scope: str | None = Field(default=None, max_length=512)
    forecast_horizon_seconds: int | None = Field(
        default=None,
        ge=60,
        le=31_536_000,
        description=(
            "未来预测窗口；HISTORICAL_AND_FORECAST且用户未明确时"
            "默认使用未来30天2592000秒。"
        ),
    )
    known_facts: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...]
    completion_requirements: tuple[CompletionRequirement, ...] = Field(
        default=(), max_length=8
    )
    action_intent: ActionIntent = Field(
        default=ActionIntent.NONE,
        description=(
            "NONE表示不需要动作；ADVISORY表示只生成或展示登记模板语句、"
            "不请求执行；EXECUTE表示请求系统在审批后执行。"
        ),
    )
    diagnostic_profile: DiagnosticProfile = DiagnosticProfile.GENERAL
    implementation_profile: ImplementationProfile = ImplementationProfile.NONE
    evidence_source_strategy: EvidenceSourceStrategy = (
        EvidenceSourceStrategy.DATABASE_FIRST
    )
    subject_ref: JsonObject = Field(default_factory=dict)
    requires_change: bool = False

    @model_validator(mode="before")
    @classmethod
    def normalize_action_intent(cls, value):
        """兼容旧 Artifact，并令执行标志只表达真实执行诉求。"""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "action_intent" not in normalized:
            normalized["action_intent"] = (
                ActionIntent.EXECUTE
                if bool(normalized.get("requires_change"))
                else ActionIntent.NONE
            )
        normalized["requires_change"] = (
            str(normalized["action_intent"]) == ActionIntent.EXECUTE
        )
        return normalized

    @model_validator(mode="after")
    def validate_objectives(self) -> "TaskFrame":
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("任务目标不能重复")
        requirement_ids = tuple(
            item.requirement_id for item in self.completion_requirements
        )
        if len(set(requirement_ids)) != len(requirement_ids):
            raise ValueError("完成义务ID不能重复")
        if self.implementation_profile != ImplementationProfile.NONE:
            if self.objectives != (TaskObjective.PLAN,):
                raise ValueError("实施方案任务只能使用 PLAN 目标")
            if self.action_intent != ActionIntent.NONE or self.requires_change:
                raise ValueError("实施方案生成阶段不能请求或标记执行")
        return self


class InvestigationHypothesis(AIOpsContract):
    hypothesis_id: str = Field(pattern=r"^h[0-9]+$")
    statement: str = Field(min_length=1, max_length=2000)
    rationale: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)


class InvestigationAction(AIOpsContract):
    action_id: str = Field(pattern=r"^a[0-9]+$")
    question: str = Field(min_length=1, max_length=2000)
    tool_id: str = Field(min_length=1, max_length=128)
    input: JsonObject = Field(default_factory=dict)
    expected_evidence_kind: str = Field(min_length=1, max_length=64)
    measurement_semantics: MeasurementSemantics
    depends_on: tuple[str, ...] = ()
    optional: bool = False
    deferred: bool = False


class InvestigationPlan(AIOpsContract):
    schema_version: str = INVESTIGATION_PLAN_SCHEMA_VERSION
    revision_no: int = Field(ge=1)
    hypotheses: tuple[InvestigationHypothesis, ...] = ()
    actions: tuple[InvestigationAction, ...] = Field(max_length=12)
    answer_if_no_more_evidence: bool = False
    stop_reason: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_action_graph(self) -> "InvestigationPlan":
        action_ids = tuple(action.action_id for action in self.actions)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("调查动作ID不能重复")
        known = set(action_ids)
        for action in self.actions:
            unknown = set(action.depends_on) - known
            if unknown:
                raise ValueError(
                    f"调查动作 {action.action_id} 引用了未知依赖："
                    f"{', '.join(sorted(unknown))}"
                )
            if action.action_id in action.depends_on:
                raise ValueError("调查动作不能依赖自身")
        graph = {
            action.action_id: tuple(action.depends_on)
            for action in self.actions
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(action_id: str) -> None:
            if action_id in visiting:
                raise ValueError("调查动作依赖图不能包含环")
            if action_id in visited:
                return
            visiting.add(action_id)
            for dependency in graph[action_id]:
                visit(dependency)
            visiting.remove(action_id)
            visited.add(action_id)

        for action_id in action_ids:
            visit(action_id)
        return self


class InvestigationPlanningOutput(AIOpsContract):
    """模型对本轮输入的一次完整理解结果，不直接执行工具。"""

    input_envelope: TurnInputEnvelope
    task_frame: TaskFrame
    plan: InvestigationPlan
    suggested_playbook_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_investigation_path(self) -> "InvestigationPlanningOutput":
        supplied = any(
            material.contains_user_evidence
            for material in self.input_envelope.materials
        )
        requires_observation = bool(
            set(self.task_frame.objectives)
            & {
                TaskObjective.DIAGNOSE,
                TaskObjective.ASSESS,
                TaskObjective.COMPARE,
                TaskObjective.VERIFY,
            }
        )
        if requires_observation and not supplied and not self.plan.actions:
            raise ValueError("诊断或评估任务在没有用户证据时必须安排取证动作")
        if (
            TaskObjective.COMPARE in self.task_frame.objectives
            and not self.task_frame.completion_requirements
        ):
            raise ValueError("对比任务必须声明结构化完成义务")
        if self.plan.revision_no == 1:
            for requirement in self.task_frame.completion_requirements:
                matching = sum(
                    1
                    for action in self.plan.actions
                    if not action.optional
                    and (
                        action.tool_id in requirement.accepted_tool_ids
                        or action.expected_evidence_kind
                        in requirement.accepted_evidence_kinds
                    )
                )
                if matching < requirement.minimum_successful_results:
                    raise ValueError(
                        "首轮调查计划未覆盖完成义务："
                        f"{requirement.requirement_id}"
                    )
        return self


class CompactPlanningMode(StrEnum):
    READ_ONLY_LOOKUP = "READ_ONLY_LOOKUP"
    CONTROLLED_ACTION = "CONTROLLED_ACTION"
    IMPLEMENTATION_RUNBOOK = "IMPLEMENTATION_RUNBOOK"
    FULL_INVESTIGATION = "FULL_INVESTIGATION"


class CompactPlanningOutput(AIOpsContract):
    """生成精简查询、受控动作前置核验或完整规划路由结果。

    精简路由允许暂时不返回动作，也不在数据契约层判断路由与动作选择的
    交叉字段一致性。应用层会统一协调候选工具或携带完整对话上下文进入
    Investigation Planner，避免把模型可恢复的语义缺项误报为内部错误。
    """

    schema_version: str = COMPACT_PLANNING_SCHEMA_VERSION
    planning_mode: CompactPlanningMode
    objectives: tuple[TaskObjective, ...] = Field(min_length=1, max_length=8)
    action_intent: ActionIntent = Field(
        description=(
            "NONE表示只读回答；ADVISORY表示只生成或展示登记动作模板的语句、"
            "不执行；EXECUTE表示用户明确要求审批后执行。"
        )
    )
    diagnostic_profile: DiagnosticProfile = Field(
        description=(
            "SINGLE_SQL_PERFORMANCE表示围绕一个明确SQL_ID执行完整SQL性能基线；"
            "其他问题使用GENERAL。"
        )
    )
    implementation_profile: ImplementationProfile = Field(
        description=(
            "实施方案档案；ORACLE_ADG_BUILD 表示生成完整 ADG 建设 Runbook，"
            "NONE 表示普通调查。"
        ),
    )
    evidence_source_strategy: EvidenceSourceStrategy = (
        EvidenceSourceStrategy.DATABASE_FIRST
    )
    subject_ref: JsonObject = Field(
        description=(
            "结构化调查对象；单SQL性能分析必须提供sql_id，其他问题为空对象。"
        )
    )
    problem_statement: str = Field(min_length=1, max_length=2000)
    time_scope: str | None = Field(default=None, max_length=512)
    requested_window_seconds: int | None = Field(
        default=None,
        ge=60,
        le=31_536_000,
        description=(
            "用户明确时间窗口对应的秒数；MONITORING_FIRST且用户未明确时"
            "使用Prometheus最大保留范围2592000秒，其他情况为空。"
        ),
    )
    temporal_analysis_mode: TemporalAnalysisMode = (
        TemporalAnalysisMode.CURRENT
    )
    forecast_scope: str | None = Field(default=None, max_length=512)
    forecast_horizon_seconds: int | None = Field(
        default=None,
        ge=60,
        le=31_536_000,
        description=(
            "未来预测窗口；HISTORICAL_AND_FORECAST且用户未明确时"
            "默认使用未来30天2592000秒。"
        ),
    )
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=4)
    completion_requirements: tuple[CompletionRequirement, ...] = Field(
        default=(), max_length=8
    )
    selected_tool_ids: tuple[str, ...] = Field(default=(), max_length=5)
    selected_playbook_ids: tuple[str, ...] = Field(default=(), max_length=3)
    actions: tuple[InvestigationAction, ...] = Field(default=(), max_length=4)
    public_reasoning_summary: str = Field(min_length=1, max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def normalize_implementation_contract(cls, value):
        """归一化实施方案固定语义，避免模型默认字段遗漏中断规划。"""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        mode = str(normalized.get("planning_mode") or "")
        profile = str(normalized.get("implementation_profile") or "")
        implementation_route = (
            mode == CompactPlanningMode.IMPLEMENTATION_RUNBOOK
            or profile == ImplementationProfile.ORACLE_ADG_BUILD
        )
        if implementation_route:
            if profile in {"", ImplementationProfile.NONE}:
                # 当前实施方案目录只登记 ADG；模型选中实施方案模式后，
                # 其安全语义由服务端固定，不依赖模型重复填写关联字段。
                normalized["implementation_profile"] = (
                    ImplementationProfile.ORACLE_ADG_BUILD
                )
            normalized["planning_mode"] = (
                CompactPlanningMode.IMPLEMENTATION_RUNBOOK
            )
            normalized["objectives"] = (TaskObjective.PLAN,)
            normalized["action_intent"] = ActionIntent.NONE
            normalized["diagnostic_profile"] = DiagnosticProfile.GENERAL
        elif "implementation_profile" not in normalized:
            normalized["implementation_profile"] = ImplementationProfile.NONE
        return normalized

    @model_validator(mode="after")
    def validate_completion_contract(self) -> "CompactPlanningOutput":
        if len(set(self.objectives)) != len(self.objectives):
            raise ValueError("任务目标不能重复")
        if (
            TaskObjective.COMPARE in self.objectives
            and not self.completion_requirements
        ):
            raise ValueError("对比任务必须声明结构化完成义务")
        implementation_mode = (
            self.planning_mode == CompactPlanningMode.IMPLEMENTATION_RUNBOOK
        )
        has_implementation_profile = (
            self.implementation_profile != ImplementationProfile.NONE
        )
        if implementation_mode != has_implementation_profile:
            raise ValueError("实施方案模式必须与 implementation_profile 同时出现")
        if implementation_mode and (
            self.objectives != (TaskObjective.PLAN,)
            or self.action_intent != ActionIntent.NONE
            or self.diagnostic_profile != DiagnosticProfile.GENERAL
        ):
            raise ValueError("实施方案必须使用 PLAN、NONE 和 GENERAL")
        return self

class InvestigationAssessment(AIOpsContract):
    schema_version: str = INVESTIGATION_ASSESSMENT_SCHEMA_VERSION
    round_no: int = Field(ge=1)
    sufficiency_status: SufficiencyStatus
    verified_facts: tuple[str, ...] = ()
    remaining_unknowns: tuple[str, ...] = ()
    hypothesis_updates: JsonObject = Field(default_factory=dict)
    evidence_gaps: tuple[str, ...] = ()
    next_action: str = Field(
        pattern=r"^(ANSWER|REPLAN|ASK_USER|STOP_UNSAFE)$"
    )
    progress_made: bool
    reason: str = Field(min_length=1, max_length=2000)
    clarification_question: str | None = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def validate_clarification(self) -> "InvestigationAssessment":
        """确保要求用户补充时能交付一条可展示的具体问题。"""
        needs_clarification = (
            self.sufficiency_status == SufficiencyStatus.NEEDS_CLARIFICATION
        )
        asks_user = self.next_action == "ASK_USER"
        if needs_clarification != asks_user:
            raise ValueError(
                "NEEDS_CLARIFICATION 必须与 ASK_USER 一起使用"
            )
        if needs_clarification and not self.clarification_question:
            raise ValueError("NEEDS_CLARIFICATION 必须包含澄清问题")
        return self


class ToolDefinition(AIOpsContract):
    tool_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    tool_class: str = Field(min_length=1, max_length=32)
    description: str = Field(min_length=1, max_length=2000)
    input_schema: JsonObject
    output_schema: JsonObject
    readonly: bool = True
    required_capabilities: tuple[str, ...] = ()


class PlaybookDefinition(AIOpsContract):
    playbook_id: str = Field(pattern=r"^[a-z][a-z0-9_.-]{2,127}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    description: str = Field(min_length=1, max_length=2000)
    applicability: tuple[str, ...] = ()
    recommended_tools: tuple[str, ...] = ()
    reasoning_guidance: tuple[str, ...] = ()
