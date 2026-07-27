"""结构化 Agent 计划与确定性校验。"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExecutionKind(StrEnum):
    LOCAL_SKILL = "LOCAL_SKILL"
    DELEGATION = "DELEGATION"


class ExecutionMode(StrEnum):
    READ_ONLY = "read_only"
    MUTATION = "mutation"
    DELEGATED = "delegated"


class CompletionRequirement(StrEnum):
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"


class TaskSpec(BaseModel):
    """Planner 允许生成的单个类型化任务。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    task_type: str = Field(min_length=1, max_length=64)
    execution_kind: ExecutionKind
    specialist: str | None = Field(default=None, max_length=64)
    skill_id: str | None = Field(
        default=None, pattern=r"^[a-z][a-z0-9-]{0,127}$",
    )
    skill_version: str | None = Field(default=None, max_length=64)
    delegate_service: str | None = Field(default=None, max_length=128)
    delegate_capability: str | None = Field(default=None, max_length=128)
    depends_on: tuple[str, ...] = ()
    input_refs: tuple[str, ...] = ()
    expected_outputs: tuple[str, ...] = ()
    required_scopes: tuple[str, ...] = ()
    timeout_seconds: int = Field(ge=1, le=3600)
    max_retries: int = Field(default=0, ge=0, le=10)
    completion_requirement: CompletionRequirement = (
        CompletionRequirement.REQUIRED
    )
    execution_mode: ExecutionMode

    @model_validator(mode="after")
    def validate_executor(self) -> "TaskSpec":
        if self.execution_kind == ExecutionKind.LOCAL_SKILL:
            if not self.skill_id or not self.skill_version:
                raise ValueError("LOCAL_SKILL 必须声明 skill_id 和 skill_version")
            if self.delegate_service or self.delegate_capability:
                raise ValueError("LOCAL_SKILL 不能声明委派目标")
            if self.execution_mode == ExecutionMode.DELEGATED:
                raise ValueError("LOCAL_SKILL 不能使用 delegated 模式")
        else:
            if not self.delegate_service or not self.delegate_capability:
                raise ValueError(
                    "DELEGATION 必须声明 delegate_service 和 capability"
                )
            if self.skill_id or self.skill_version:
                raise ValueError("DELEGATION 不能声明本地 Skill")
            if self.execution_mode != ExecutionMode.DELEGATED:
                raise ValueError("DELEGATION 必须使用 delegated 模式")
        return self


class PlanDraft(BaseModel):
    """Planner 输出的不可执行计划草案。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan_version: str = Field(min_length=1, max_length=64)
    objective: str = Field(min_length=1, max_length=4000)
    tasks: tuple[TaskSpec, ...]
    final_task_key: str = Field(min_length=1, max_length=64)
    expires_at: datetime


class PlanLimits(BaseModel):
    """由服务端策略冻结的计划上限。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_tasks: int = Field(default=16, ge=1, le=128)
    max_parallel_tasks: int = Field(default=4, ge=1, le=32)
    max_total_retries: int = Field(default=16, ge=0, le=128)
    max_task_timeout_seconds: int = Field(default=600, ge=1, le=3600)


class PlanValidationError(ValueError):
    """返回稳定错误码的计划拒绝。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class PlanValidator:
    """不依赖 LLM 的 DAG、预算和终点校验。"""

    def __init__(
        self,
        *,
        skill_exists,
        capability_exists,
        public_artifact_types: set[str],
    ):
        self._skill_exists = skill_exists
        self._capability_exists = capability_exists
        self._public_artifact_types = frozenset(public_artifact_types)

    def validate(self, plan: PlanDraft, limits: PlanLimits) -> None:
        if not plan.tasks or len(plan.tasks) > limits.max_tasks:
            raise PlanValidationError(
                "BUDGET_EXCEEDED", "计划任务数量超过上限",
            )
        tasks = {task.task_key: task for task in plan.tasks}
        if len(tasks) != len(plan.tasks):
            raise PlanValidationError("PLAN_DUPLICATE_TASK", "task_key 必须唯一")
        if plan.final_task_key not in tasks:
            raise PlanValidationError(
                "PLAN_FINAL_TASK_MISSING", "最终任务不存在",
            )
        if sum(task.max_retries for task in plan.tasks) > limits.max_total_retries:
            raise PlanValidationError(
                "BUDGET_EXCEEDED", "计划总重试次数超过上限",
            )
        for task in plan.tasks:
            if task.timeout_seconds > limits.max_task_timeout_seconds:
                raise PlanValidationError(
                    "BUDGET_EXCEEDED",
                    f"任务 {task.task_key} 超过超时上限",
                )
            missing = set(task.depends_on) - tasks.keys()
            if missing:
                raise PlanValidationError(
                    "PLAN_DEPENDENCY_MISSING",
                    f"任务 {task.task_key} 依赖不存在：{sorted(missing)}",
                )
            if task.execution_kind == ExecutionKind.LOCAL_SKILL:
                if not self._skill_exists(task.skill_id, task.skill_version):
                    raise PlanValidationError(
                        "SKILL_NOT_FOUND",
                        f"Skill 未注册：{task.skill_id}@{task.skill_version}",
                    )
            elif not self._capability_exists(
                task.delegate_service, task.delegate_capability,
            ):
                raise PlanValidationError(
                    "CAPABILITY_NOT_FOUND",
                    f"委派能力未注册：{task.delegate_service}/"
                    f"{task.delegate_capability}",
                )
        self._validate_acyclic(tasks)
        final_task = tasks[plan.final_task_key]
        if (
            final_task.completion_requirement
            != CompletionRequirement.REQUIRED
        ):
            raise PlanValidationError(
                "PLAN_FINAL_TASK_OPTIONAL",
                "最终任务必须是 REQUIRED",
            )
        reachable = self._dependency_closure(
            plan.final_task_key, tasks
        )
        required_keys = {
            task.task_key
            for task in plan.tasks
            if task.completion_requirement
            == CompletionRequirement.REQUIRED
        }
        if not required_keys.issubset(reachable):
            raise PlanValidationError(
                "PLAN_FINAL_TASK_INCOMPLETE",
                "最终任务必须依赖全部 REQUIRED 任务",
            )
        final_outputs = set(final_task.expected_outputs)
        if not final_outputs.intersection(self._public_artifact_types):
            raise PlanValidationError(
                "SCHEMA_MISMATCH", "最终任务没有可公开的 Artifact 输出",
            )

    @staticmethod
    def _validate_acyclic(tasks: dict[str, TaskSpec]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_key: str) -> None:
            if task_key in visiting:
                raise PlanValidationError("PLAN_CYCLE", "计划存在循环依赖")
            if task_key in visited:
                return
            visiting.add(task_key)
            for dependency in tasks[task_key].depends_on:
                visit(dependency)
            visiting.remove(task_key)
            visited.add(task_key)

        for task_key in tasks:
            visit(task_key)

    @staticmethod
    def _dependency_closure(
        task_key: str, tasks: dict[str, TaskSpec]
    ) -> set[str]:
        reachable: set[str] = set()

        def collect(current: str) -> None:
            if current in reachable:
                return
            reachable.add(current)
            for dependency in tasks[current].depends_on:
                collect(dependency)

        collect(task_key)
        return reachable
