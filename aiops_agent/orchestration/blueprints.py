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


def build_advisory_verification_blueprint(
    tool_ids: tuple[str, ...],
) -> Blueprint:
    """使用全新只读证据验证人工执行效果，不再生成变更建议。"""
    ordered = tuple(dict.fromkeys(tool_ids))
    if ordered and ordered[0] != "db.instance.identity":
        raise BlueprintValidationError("效果验证必须先执行实例身份工具")
    diagnostic_tasks = []
    for index, tool_id in enumerate(ordered):
        dependency = (
            ("scope",)
            if index == 0
            else ("diagnostic:db.instance.identity",)
        )
        diagnostic_tasks.append(
            TaskSpec(
                task_key=f"diagnostic:{tool_id}",
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
    dependencies = ("scope",) + tuple(
        item.task_key for item in diagnostic_tasks
    )
    return Blueprint(
        blueprint_id="change.advisory-verify",
        version="1",
        final_task_key="verify",
        tasks=(
            TaskSpec(
                task_key="scope",
                task_type="SCOPE",
                handler_id="change.verification-scope",
                handler_version="1",
                input_schema_version="RUN_INPUT.v1",
                output_schema_version="ADVISORY_VERIFICATION_SCOPE.v1",
                timeout_seconds=30,
            ),
            *diagnostic_tasks,
            TaskSpec(
                task_key="verify",
                task_type="VERIFY",
                handler_id="change.verify",
                handler_version="1",
                input_schema_version="ADVISORY_VERIFY_INPUT.v1",
                output_schema_version="ACTION_VERIFICATION.v1",
                depends_on=dependencies,
                input_artifact_keys=dependencies,
                timeout_seconds=30,
                max_attempts=1,
            ),
        ),
    )


def build_diagnosis_blueprint(
    *,
    binding_ids: tuple[str, ...],
    tool_ids: tuple[str, ...],
) -> Blueprint:
    """组合监控、数据库基线和受约束模型诊断的一轮可恢复 DAG。"""
    observe_tasks = tuple(
        TaskSpec(
            task_key=f"observe:{binding_id}",
            task_type="OBSERVE",
            handler_id="monitor.observe",
            handler_version="1",
            input_schema_version="DIAGNOSIS_SCOPE.v1",
            output_schema_version="OBSERVATION_SET.v1",
            depends_on=("scope",),
            input_artifact_keys=("scope",),
            timeout_seconds=120,
            max_attempts=3,
            priority=40,
        )
        for binding_id in sorted(binding_ids)
    )
    ordered_tools = tuple(dict.fromkeys(tool_ids))
    diagnostic_tasks = []
    for index, tool_id in enumerate(ordered_tools):
        dependency = (
            ("scope",)
            if index == 0
            else ("diagnostic:db.instance.identity",)
        )
        diagnostic_tasks.append(
            TaskSpec(
                task_key=f"diagnostic:{tool_id}",
                task_type="DIAGNOSE",
                handler_id="database.diagnostic",
                handler_version="1",
                input_schema_version=(
                    "DIAGNOSIS_SCOPE.v1"
                    if index == 0
                    else "DATABASE_DIAGNOSTIC_RESULT.v1"
                ),
                output_schema_version="DATABASE_DIAGNOSTIC_RESULT.v1",
                depends_on=dependency,
                input_artifact_keys=dependency,
                timeout_seconds=90,
                max_attempts=2,
                priority=50 + index,
            )
        )
    source_keys = tuple(
        item.task_key for item in (*observe_tasks, *diagnostic_tasks)
    )
    evidence_dependencies = ("scope", *source_keys)
    return Blueprint(
        blueprint_id="diagnosis.root-cause",
        version="1",
        final_task_key="diagnosis:report",
        tasks=(
            TaskSpec(
                task_key="scope",
                task_type="SCOPE",
                handler_id="diagnosis.scope",
                handler_version="1",
                input_schema_version="RUN_INPUT.v1",
                output_schema_version="DIAGNOSIS_SCOPE.v1",
                timeout_seconds=30,
            ),
            *observe_tasks,
            *diagnostic_tasks,
            TaskSpec(
                task_key="diagnosis:evidence:r0",
                task_type="DIAGNOSE",
                handler_id="diagnosis.evidence-index",
                handler_version="1",
                input_schema_version="EVIDENCE_BUILD_INPUT.v1",
                output_schema_version="EVIDENCE_INDEX.v1",
                depends_on=evidence_dependencies,
                input_artifact_keys=evidence_dependencies,
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:r1:draft",
                task_type="DIAGNOSE",
                handler_id="diagnosis.round-draft",
                handler_version="1",
                input_schema_version="DIAGNOSIS_DRAFT_INPUT.v1",
                output_schema_version="DIAGNOSIS_ROUND_DRAFT.v1",
                depends_on=("scope", "diagnosis:evidence:r0"),
                input_artifact_keys=("scope", "diagnosis:evidence:r0"),
                timeout_seconds=180,
                max_attempts=2,
            ),
            TaskSpec(
                task_key="diagnosis:r1:validate",
                task_type="DIAGNOSE",
                handler_id="diagnosis.request-validator",
                handler_version="1",
                input_schema_version="DIAGNOSIS_VALIDATE_INPUT.v1",
                output_schema_version="VALIDATED_EVIDENCE_PLAN.v1",
                depends_on=(
                    "diagnosis:evidence:r0",
                    "diagnosis:r1:draft",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:r0",
                    "diagnosis:r1:draft",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:r1:collect",
                task_type="DIAGNOSE",
                handler_id="diagnosis.evidence-collect",
                handler_version="1",
                input_schema_version="DIAGNOSIS_COLLECT_INPUT.v1",
                output_schema_version="DIAGNOSIS_EVIDENCE_COLLECTION.v1",
                depends_on=(
                    "diagnosis:r1:validate",
                    *(
                        ("diagnostic:db.instance.identity",)
                        if "db.instance.identity" in ordered_tools
                        else ()
                    ),
                ),
                input_artifact_keys=(
                    "diagnosis:r1:validate",
                    *(
                        ("diagnostic:db.instance.identity",)
                        if "db.instance.identity" in ordered_tools
                        else ()
                    ),
                ),
                timeout_seconds=180,
                max_attempts=2,
            ),
            TaskSpec(
                task_key="diagnosis:evidence:r1",
                task_type="DIAGNOSE",
                handler_id="diagnosis.evidence-index",
                handler_version="1",
                input_schema_version="EVIDENCE_BUILD_INPUT.v1",
                output_schema_version="EVIDENCE_INDEX.v1",
                depends_on=(
                    *source_keys,
                    "diagnosis:r1:collect",
                ),
                input_artifact_keys=(
                    *source_keys,
                    "diagnosis:r1:collect",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:r1:assess",
                task_type="DIAGNOSE",
                handler_id="diagnosis.round-assess",
                handler_version="1",
                input_schema_version="DIAGNOSIS_ASSESS_INPUT.v1",
                output_schema_version="DIAGNOSIS_ROUND_ASSESSMENT.v1",
                depends_on=(
                    "diagnosis:evidence:r1",
                    "diagnosis:r1:draft",
                    "diagnosis:r1:validate",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:r1",
                    "diagnosis:r1:draft",
                    "diagnosis:r1:validate",
                ),
                timeout_seconds=180,
                max_attempts=2,
            ),
            TaskSpec(
                task_key="diagnosis:root-cause",
                task_type="DIAGNOSE",
                handler_id="diagnosis.root-cause",
                handler_version="1",
                input_schema_version="ROOT_CAUSE_INPUT.v1",
                output_schema_version="ROOT_CAUSE_ASSESSMENT.v1",
                depends_on=(
                    "diagnosis:evidence:r1",
                    "diagnosis:r1:assess",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:r1",
                    "diagnosis:r1:assess",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:verify",
                task_type="DIAGNOSE",
                handler_id="diagnosis.grounding",
                handler_version="1",
                input_schema_version="GROUNDING_INPUT.v1",
                output_schema_version="GROUNDING_VERIFICATION.v1",
                depends_on=(
                    "diagnosis:evidence:r1",
                    "diagnosis:root-cause",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:r1",
                    "diagnosis:root-cause",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:solution",
                task_type="DIAGNOSE",
                handler_id="diagnosis.solution",
                handler_version="1",
                input_schema_version="SOLUTION_INPUT.v1",
                output_schema_version="SOLUTION_DRAFT.v1",
                depends_on=("diagnosis:root-cause",),
                input_artifact_keys=("diagnosis:root-cause",),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="change:action-plan",
                task_type="PROPOSE",
                handler_id="change.action-plan",
                handler_version="1",
                input_schema_version="ACTION_PLAN_INPUT.v1",
                output_schema_version="ACTION_PLAN.v1",
                depends_on=(
                    "diagnosis:evidence:r1",
                    "diagnosis:root-cause",
                    "diagnosis:solution",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:r1",
                    "diagnosis:root-cause",
                    "diagnosis:solution",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="change:proposal",
                task_type="PROPOSE",
                handler_id="change.proposal",
                handler_version="1",
                input_schema_version="PROPOSAL_INPUT.v1",
                output_schema_version="PROPOSAL_OUTCOME.v1",
                depends_on=("change:action-plan",),
                input_artifact_keys=("change:action-plan",),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:report",
                task_type="REPORT",
                handler_id="diagnosis.report",
                handler_version="1",
                input_schema_version="DIAGNOSIS_REPORT_INPUT.v1",
                output_schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                depends_on=(
                    "diagnosis:evidence:r1",
                    "diagnosis:r1:assess",
                    "diagnosis:root-cause",
                    "diagnosis:verify",
                    "diagnosis:solution",
                    "change:action-plan",
                    "change:proposal",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:r1",
                    "diagnosis:r1:assess",
                    "diagnosis:root-cause",
                    "diagnosis:verify",
                    "diagnosis:solution",
                    "change:action-plan",
                    "change:proposal",
                ),
                timeout_seconds=30,
            ),
        ),
    )


def build_multi_round_diagnosis_blueprint(
    *,
    binding_ids: tuple[str, ...],
    tool_ids: tuple[str, ...],
    max_rounds: int,
) -> Blueprint:
    """创建最多三轮的固定上限 DAG，未使用轮次由 Handler 安全短路。"""
    if not 1 <= max_rounds <= 3:
        raise BlueprintValidationError("诊断轮次上限必须位于 1 到 3")
    first = build_diagnosis_blueprint(
        binding_ids=binding_ids, tool_ids=tool_ids
    )
    baseline = []
    for task in first.tasks:
        baseline.append(task)
        if task.task_key == "diagnosis:evidence:r0":
            break
    baseline_evidence = baseline.pop()
    knowledge_task = TaskSpec(
        task_key="diagnosis:knowledge",
        task_type="DIAGNOSE",
        handler_id="diagnosis.knowledge-citation",
        handler_version="1",
        input_schema_version="DIAGNOSIS_SCOPE.v1",
        output_schema_version="KNOWLEDGE_CITATION_PACK.v1",
        depends_on=("scope",),
        input_artifact_keys=("scope",),
        timeout_seconds=120,
        max_attempts=2,
    )
    baseline.append(knowledge_task)
    baseline.append(
        TaskSpec(
            task_key=baseline_evidence.task_key,
            task_type=baseline_evidence.task_type,
            handler_id=baseline_evidence.handler_id,
            handler_version=baseline_evidence.handler_version,
            input_schema_version=baseline_evidence.input_schema_version,
            output_schema_version=baseline_evidence.output_schema_version,
            depends_on=(
                *baseline_evidence.depends_on,
                knowledge_task.task_key,
            ),
            input_artifact_keys=(
                *baseline_evidence.input_artifact_keys,
                knowledge_task.task_key,
            ),
            timeout_seconds=baseline_evidence.timeout_seconds,
            max_attempts=baseline_evidence.max_attempts,
            priority=baseline_evidence.priority,
        )
    )
    source_keys = tuple(
        task.task_key
        for task in baseline
        if task.output_schema_version
        in {"OBSERVATION_SET.v1", "DATABASE_DIAGNOSTIC_RESULT.v1"}
    )
    identity_key = (
        ("diagnostic:db.instance.identity",)
        if "db.instance.identity" in tool_ids
        else ()
    )
    tasks = list(baseline)
    prior_evidence = "diagnosis:evidence:r0"
    prior_assessment: str | None = None
    prior_plans: list[str] = []
    for round_no in range(1, max_rounds + 1):
        prefix = f"diagnosis:r{round_no}"
        draft_dependencies = ["scope", prior_evidence]
        if prior_assessment:
            draft_dependencies.append(prior_assessment)
        tasks.append(
            TaskSpec(
                task_key=f"{prefix}:draft",
                task_type="DIAGNOSE",
                handler_id="diagnosis.round-draft",
                handler_version="1",
                input_schema_version="DIAGNOSIS_DRAFT_INPUT.v1",
                output_schema_version="DIAGNOSIS_ROUND_DRAFT.v1",
                depends_on=tuple(draft_dependencies),
                input_artifact_keys=tuple(draft_dependencies),
                timeout_seconds=180,
                max_attempts=2,
            )
        )
        validate_inputs = (
            prior_evidence,
            f"{prefix}:draft",
            *prior_plans,
        )
        tasks.append(
            TaskSpec(
                task_key=f"{prefix}:validate",
                task_type="DIAGNOSE",
                handler_id="diagnosis.request-validator",
                handler_version="1",
                input_schema_version="DIAGNOSIS_VALIDATE_INPUT.v1",
                output_schema_version="VALIDATED_EVIDENCE_PLAN.v1",
                depends_on=validate_inputs,
                input_artifact_keys=validate_inputs,
                timeout_seconds=30,
            )
        )
        tasks.append(
            TaskSpec(
                task_key=f"{prefix}:collect",
                task_type="DIAGNOSE",
                handler_id="diagnosis.evidence-collect",
                handler_version="1",
                input_schema_version="DIAGNOSIS_COLLECT_INPUT.v1",
                output_schema_version="DIAGNOSIS_EVIDENCE_COLLECTION.v1",
                depends_on=(f"{prefix}:validate", *identity_key),
                input_artifact_keys=(f"{prefix}:validate", *identity_key),
                timeout_seconds=180,
                max_attempts=2,
            )
        )
        evidence_key = f"diagnosis:evidence:r{round_no}"
        tasks.append(
            TaskSpec(
                task_key=evidence_key,
                task_type="DIAGNOSE",
                handler_id="diagnosis.evidence-index",
                handler_version="1",
                input_schema_version="EVIDENCE_BUILD_INPUT.v1",
                output_schema_version="EVIDENCE_INDEX.v1",
                depends_on=(
                    prior_evidence,
                    f"{prefix}:collect",
                ),
                input_artifact_keys=(
                    prior_evidence,
                    f"{prefix}:collect",
                ),
                timeout_seconds=30,
            )
        )
        assess_inputs = [
            prior_evidence,
            evidence_key,
            f"{prefix}:draft",
            f"{prefix}:validate",
        ]
        if prior_assessment:
            assess_inputs.append(prior_assessment)
        assessment_key = f"{prefix}:assess"
        tasks.append(
            TaskSpec(
                task_key=assessment_key,
                task_type="DIAGNOSE",
                handler_id="diagnosis.round-assess",
                handler_version="1",
                input_schema_version="DIAGNOSIS_ASSESS_INPUT.v1",
                output_schema_version="DIAGNOSIS_ROUND_ASSESSMENT.v1",
                depends_on=tuple(assess_inputs),
                input_artifact_keys=tuple(assess_inputs),
                timeout_seconds=180,
                max_attempts=2,
            )
        )
        prior_evidence = evidence_key
        prior_assessment = assessment_key
        prior_plans.append(f"{prefix}:validate")
    assert prior_assessment is not None
    tasks.extend(
        (
            TaskSpec(
                task_key="diagnosis:interactive",
                task_type="DIAGNOSE",
                handler_id="diagnosis.interactive",
                handler_version="1",
                input_schema_version="INTERACTIVE_DIAGNOSIS_INPUT.v1",
                output_schema_version="HITL_OUTCOME.v1",
                depends_on=(prior_evidence, prior_assessment),
                input_artifact_keys=(prior_evidence, prior_assessment),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:evidence:final",
                task_type="DIAGNOSE",
                handler_id="diagnosis.evidence-index",
                handler_version="1",
                input_schema_version="EVIDENCE_BUILD_INPUT.v1",
                output_schema_version="EVIDENCE_INDEX.v1",
                depends_on=(prior_evidence, "diagnosis:interactive"),
                input_artifact_keys=(
                    prior_evidence,
                    "diagnosis:interactive",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key=(
                    f"diagnosis:r{max_rounds + 1}:assess-manual"
                ),
                task_type="DIAGNOSE",
                handler_id="diagnosis.round-assess",
                handler_version="1",
                input_schema_version="DIAGNOSIS_ASSESS_INPUT.v1",
                output_schema_version=(
                    "DIAGNOSIS_ROUND_ASSESSMENT.v1"
                ),
                depends_on=(
                    prior_evidence,
                    "diagnosis:evidence:final",
                    f"diagnosis:r{max_rounds}:draft",
                    f"diagnosis:r{max_rounds}:validate",
                    prior_assessment,
                ),
                input_artifact_keys=(
                    prior_evidence,
                    "diagnosis:evidence:final",
                    f"diagnosis:r{max_rounds}:draft",
                    f"diagnosis:r{max_rounds}:validate",
                    prior_assessment,
                ),
                timeout_seconds=180,
                max_attempts=2,
            ),
            TaskSpec(
                task_key="diagnosis:root-cause",
                task_type="DIAGNOSE",
                handler_id="diagnosis.root-cause",
                handler_version="1",
                input_schema_version="ROOT_CAUSE_INPUT.v1",
                output_schema_version="ROOT_CAUSE_ASSESSMENT.v1",
                depends_on=(
                    "diagnosis:evidence:final",
                    f"diagnosis:r{max_rounds + 1}:assess-manual",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:final",
                    f"diagnosis:r{max_rounds + 1}:assess-manual",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:verify",
                task_type="DIAGNOSE",
                handler_id="diagnosis.grounding",
                handler_version="1",
                input_schema_version="GROUNDING_INPUT.v1",
                output_schema_version="GROUNDING_VERIFICATION.v1",
                depends_on=(
                    "diagnosis:evidence:final",
                    "diagnosis:root-cause",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:final",
                    "diagnosis:root-cause",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:solution",
                task_type="DIAGNOSE",
                handler_id="diagnosis.solution",
                handler_version="1",
                input_schema_version="SOLUTION_INPUT.v1",
                output_schema_version="SOLUTION_DRAFT.v1",
                depends_on=("diagnosis:root-cause",),
                input_artifact_keys=("diagnosis:root-cause",),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="change:action-plan",
                task_type="PROPOSE",
                handler_id="change.action-plan",
                handler_version="1",
                input_schema_version="ACTION_PLAN_INPUT.v1",
                output_schema_version="ACTION_PLAN.v1",
                depends_on=(
                    "diagnosis:evidence:final",
                    "diagnosis:root-cause",
                    "diagnosis:solution",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:final",
                    "diagnosis:root-cause",
                    "diagnosis:solution",
                ),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="change:proposal",
                task_type="PROPOSE",
                handler_id="change.proposal",
                handler_version="1",
                input_schema_version="PROPOSAL_INPUT.v1",
                output_schema_version="PROPOSAL_OUTCOME.v1",
                depends_on=("change:action-plan",),
                input_artifact_keys=("change:action-plan",),
                timeout_seconds=30,
            ),
            TaskSpec(
                task_key="diagnosis:report",
                task_type="REPORT",
                handler_id="diagnosis.report",
                handler_version="1",
                input_schema_version="DIAGNOSIS_REPORT_INPUT.v1",
                output_schema_version="DIAGNOSIS_REPORT_DRAFT.v1",
                depends_on=(
                    "diagnosis:evidence:final",
                    f"diagnosis:r{max_rounds + 1}:assess-manual",
                    "diagnosis:root-cause",
                    "diagnosis:verify",
                    "diagnosis:solution",
                    "change:action-plan",
                    "change:proposal",
                ),
                input_artifact_keys=(
                    "diagnosis:evidence:final",
                    f"diagnosis:r{max_rounds + 1}:assess-manual",
                    "diagnosis:root-cause",
                    "diagnosis:verify",
                    "diagnosis:solution",
                    "change:action-plan",
                    "change:proposal",
                ),
                timeout_seconds=30,
            ),
        )
    )
    return Blueprint(
        blueprint_id="diagnosis.root-cause",
        version="1",
        tasks=tuple(tasks),
        final_task_key="diagnosis:report",
    )
