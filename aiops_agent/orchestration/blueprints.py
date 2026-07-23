"""版本化、不可变的 AIOps Blueprint Registry。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    task_key: str
    task_type: str
    handler_id: str
    handler_version: str
    input_schema_version: str
    output_schema_version: str
    depends_on: tuple[str, ...] = ()
    input_artifact_keys: tuple[str, ...] = ()
    timeout_seconds: int = 30
    max_attempts: int = 3
    priority: int = 100


@dataclass(frozen=True)
class Blueprint:
    blueprint_id: str
    version: str
    tasks: tuple[TaskSpec, ...]
    final_task_key: str


class BlueprintValidationError(ValueError):
    """Blueprint 违反运行内核的确定性约束。"""


class BlueprintRegistry:
    def __init__(self, blueprints: tuple[Blueprint, ...]):
        self._items = {
            (item.blueprint_id, item.version): item for item in blueprints
        }
        if len(self._items) != len(blueprints):
            raise BlueprintValidationError("Blueprint ID 与版本不能重复")

    def resolve(self, blueprint_id: str, version: str) -> Blueprint:
        try:
            return self._items[(blueprint_id, version)]
        except KeyError as exc:
            raise BlueprintValidationError(
                f"Blueprint 不存在：{blueprint_id}@{version}"
            ) from exc

    @staticmethod
    def validate(blueprint: Blueprint, *, max_tasks: int) -> None:
        if not blueprint.tasks or len(blueprint.tasks) > max_tasks:
            raise BlueprintValidationError("Blueprint Task 数量超出限制")
        task_map = {task.task_key: task for task in blueprint.tasks}
        if len(task_map) != len(blueprint.tasks):
            raise BlueprintValidationError("Task Key 不能重复")
        if blueprint.final_task_key not in task_map:
            raise BlueprintValidationError("最终 Task 不存在")
        for task in blueprint.tasks:
            if task.task_key in task.depends_on:
                raise BlueprintValidationError("Task 不能依赖自身")
            if any(key not in task_map for key in task.depends_on):
                raise BlueprintValidationError("Task 依赖不存在")
            if task.timeout_seconds < 1 or task.max_attempts < 1:
                raise BlueprintValidationError("Task 执行参数无效")
            for dependency in task.depends_on:
                upstream = task_map[dependency]
                if (
                    task.input_schema_version
                    != upstream.output_schema_version
                    and task.task_type != "REPORT"
                ):
                    raise BlueprintValidationError(
                        f"Task Schema 不匹配：{dependency} -> {task.task_key}"
                    )
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise BlueprintValidationError("Blueprint 依赖存在环")
            if key in visited:
                return
            visiting.add(key)
            for dependency in task_map[key].depends_on:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in task_map:
            visit(key)


KERNEL_BLUEPRINT = Blueprint(
    blueprint_id="kernel.observe-report",
    version="1",
    final_task_key="report",
    tasks=(
        TaskSpec(
            task_key="scope",
            task_type="SCOPE",
            handler_id="kernel.scope",
            handler_version="1",
            input_schema_version="RUN_INPUT.v1",
            output_schema_version="SCOPE_RESULT.v1",
            timeout_seconds=30,
        ),
        TaskSpec(
            task_key="observe",
            task_type="OBSERVE",
            handler_id="kernel.observe",
            handler_version="1",
            input_schema_version="SCOPE_RESULT.v1",
            output_schema_version="OBSERVATION_SET.v1",
            depends_on=("scope",),
            input_artifact_keys=("scope",),
            timeout_seconds=30,
        ),
        TaskSpec(
            task_key="report",
            task_type="REPORT",
            handler_id="kernel.report",
            handler_version="1",
            input_schema_version="KERNEL_REPORT_INPUT.v1",
            output_schema_version="KERNEL_TEST_REPORT.v1",
            depends_on=("scope", "observe"),
            input_artifact_keys=("scope", "observe"),
            timeout_seconds=30,
        ),
    ),
)


def create_kernel_blueprint_registry() -> BlueprintRegistry:
    registry = BlueprintRegistry((KERNEL_BLUEPRINT,))
    registry.validate(KERNEL_BLUEPRINT, max_tasks=64)
    return registry
