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
                    and not task.input_schema_version.endswith("_INPUT.v1")
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


def build_monitor_observe_blueprint(
    binding_ids: tuple[str, ...],
) -> Blueprint:
    """每个冻结的监控绑定一个 Observe Task，报告等待全部来源。"""
    observe_tasks = tuple(
        TaskSpec(
            task_key=f"observe:{binding_id}",
            task_type="OBSERVE",
            handler_id="monitor.observe",
            handler_version="1",
            input_schema_version="MONITOR_SCOPE_RESULT.v1",
            output_schema_version="OBSERVATION_SET.v1",
            depends_on=("scope",),
            input_artifact_keys=("scope",),
            timeout_seconds=120,
            max_attempts=3,
            priority=50,
        )
        for binding_id in sorted(binding_ids)
    )
    dependencies = ("scope",) + tuple(
        item.task_key for item in observe_tasks
    )
    return Blueprint(
        blueprint_id="monitor.observe-report",
        version="1",
        final_task_key="report",
        tasks=(
            TaskSpec(
                task_key="scope",
                task_type="SCOPE",
                handler_id="monitor.scope",
                handler_version="1",
                input_schema_version="RUN_INPUT.v1",
                output_schema_version="MONITOR_SCOPE_RESULT.v1",
                timeout_seconds=30,
            ),
            *observe_tasks,
            TaskSpec(
                task_key="report",
                task_type="REPORT",
                handler_id="monitor.report",
                handler_version="1",
                input_schema_version="OBSERVE_REPORT_INPUT.v1",
                output_schema_version="OBSERVE_REPORT.v1",
                depends_on=dependencies,
                input_artifact_keys=dependencies,
                timeout_seconds=30,
            ),
        ),
    )


def build_database_diagnostic_blueprint(
    tool_ids: tuple[str, ...],
) -> Blueprint:
    """身份探测先行，其余只读工具在固定小并发下由 Worker 领取。"""
    ordered = tuple(dict.fromkeys(tool_ids))
    if ordered and ordered[0] != "db.instance.identity":
        raise BlueprintValidationError("数据库诊断必须先执行实例身份工具")
    diagnostic_tasks = []
    for index, tool_id in enumerate(ordered):
        task_key = f"diagnostic:{tool_id}"
        dependency = (
            ("scope",)
            if index == 0
            else ("diagnostic:db.instance.identity",)
        )
        diagnostic_tasks.append(
            TaskSpec(
                task_key=task_key,
                task_type="DIAGNOSE",
                handler_id="database.diagnostic",
                handler_version="1",
                input_schema_version=(
                    "DATABASE_SCOPE_RESULT.v1"
                    if index == 0
                    else "DATABASE_DIAGNOSTIC_RESULT.v1"
                ),
                output_schema_version="DATABASE_DIAGNOSTIC_RESULT.v1",
                depends_on=dependency,
                input_artifact_keys=dependency,
                timeout_seconds=90,
                max_attempts=2,
                priority=40 + index,
            )
        )
    diagnostic_keys = tuple(item.task_key for item in diagnostic_tasks)
    aggregate_dependencies = ("scope",) + diagnostic_keys
    return Blueprint(
        blueprint_id="database.diagnostic-baseline",
        version="1",
        final_task_key="report",
        tasks=(
            TaskSpec(
                task_key="scope",
                task_type="SCOPE",
                handler_id="database.scope",
                handler_version="1",
                input_schema_version="RUN_INPUT.v1",
                output_schema_version="DATABASE_SCOPE_RESULT.v1",
                timeout_seconds=30,
            ),
            *diagnostic_tasks,
            TaskSpec(
                task_key="aggregate",
                task_type="DIAGNOSE",
                handler_id="database.aggregate",
                handler_version="1",
                input_schema_version="DATABASE_AGGREGATE_INPUT.v1",
                output_schema_version="DATABASE_OBSERVATION_AGGREGATE.v1",
                depends_on=aggregate_dependencies,
                input_artifact_keys=aggregate_dependencies,
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="report",
                task_type="REPORT",
                handler_id="database.report",
                handler_version="1",
                input_schema_version="DATABASE_OBSERVATION_AGGREGATE.v1",
                output_schema_version="DB_DIAGNOSTIC_REPORT.v1",
                depends_on=("aggregate",),
                input_artifact_keys=("aggregate",),
                timeout_seconds=30,
            ),
        ),
    )
