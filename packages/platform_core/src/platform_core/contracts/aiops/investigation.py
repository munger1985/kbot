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
COMPACT_PLANNING_SCHEMA_VERSION = "aiops.compact-planning.v2"


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


class TaskFrame(AIOpsContract):
    schema_version: str = TASK_FRAME_SCHEMA_VERSION
    objectives: tuple[TaskObjective, ...] = Field(
        min_length=1, max_length=8
    )
    problem_statement: str = Field(min_length=1, max_length=4000)
    database_context: JsonObject = Field(default_factory=dict)
    time_scope: str | None = Field(default=None, max_length=512)
    known_facts: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    success_criteria: tuple[str, ...]
    action_intent: ActionIntent = Field(
        default=ActionIntent.NONE,
        description=(
            "NONE表示不需要动作；ADVISORY表示只生成或展示登记模板语句、"
            "不请求执行；EXECUTE表示请求系统在审批后执行。"
        ),
    )
    diagnostic_profile: DiagnosticProfile = DiagnosticProfile.GENERAL
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
        return self


class CompactPlanningMode(StrEnum):
    READ_ONLY_LOOKUP = "READ_ONLY_LOOKUP"
    CONTROLLED_ACTION = "CONTROLLED_ACTION"
    FULL_INVESTIGATION = "FULL_INVESTIGATION"


class CompactPlanningOutput(AIOpsContract):
    """生成精简查询、受控动作前置核验或完整规划路由结果。

    精简路由允许暂时不返回动作，也不在数据契约层判断路由与动作选择的
    交叉字段一致性。应用层会统一协调候选工具或携带完整对话上下文进入
    Investigation Planner，避免把模型可恢复的语义缺项误报为内部错误。
    """

    schema_version: str = COMPACT_PLANNING_SCHEMA_VERSION
    planning_mode: CompactPlanningMode
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
    subject_ref: JsonObject = Field(
        description=(
            "结构化调查对象；单SQL性能分析必须提供sql_id，其他问题为空对象。"
        )
    )
    problem_statement: str = Field(min_length=1, max_length=2000)
    success_criteria: tuple[str, ...] = Field(min_length=1, max_length=4)
    selected_tool_ids: tuple[str, ...] = Field(default=(), max_length=5)
    selected_playbook_ids: tuple[str, ...] = Field(default=(), max_length=3)
    actions: tuple[InvestigationAction, ...] = Field(default=(), max_length=4)
    public_reasoning_summary: str = Field(min_length=1, max_length=1000)

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
