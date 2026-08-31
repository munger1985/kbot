"""把已验证的调查计划编译为通用 Task DAG。"""

from __future__ import annotations

from dataclasses import dataclass

from aiops_agent.orchestration.blueprints import TaskSpec
from platform_core.contracts.aiops.playbooks import DbaPlaybookPlan

from aiops_agent.playbooks import PlaybookRegistry


class InvestigationCatalogChangedError(ValueError):
    code = "AIOPS_INVESTIGATION_CATALOG_CHANGED"


@dataclass(frozen=True, slots=True)
class CompiledInvestigationPlan:
    tasks: tuple[TaskSpec, ...]
    invocation_task_keys: tuple[str, ...]
    diagnostic_task_keys: tuple[str, ...]
    monitoring_task_keys: tuple[str, ...]
    log_task_keys: tuple[str, ...]
    dynamic_task_keys: tuple[str, ...]
    assessment_task_key: str
    action_plan_task_key: str | None
    proposal_task_key: str | None
    answer_task_key: str | None


class InvestigationTaskCompiler:
    """将验证后的调查计划编译为通用运行内核 Task DAG。"""

    def __init__(
        self,
        registry: PlaybookRegistry,
        *,
        model_timeout_seconds: int = 300,
    ) -> None:
        if model_timeout_seconds < 1:
            raise ValueError("模型调用超时必须大于零")
        self._registry = registry
        self._model_timeout_seconds = int(model_timeout_seconds)

    def compile(
        self,
        plan: DbaPlaybookPlan,
        *,
        monitoring_binding_ids: tuple[str, ...] = (),
        log_binding_ids: tuple[str, ...] = (),
        user_evidence_artifact_keys: tuple[str, ...] = (),
        revision_no: int = 1,
        include_answer: bool = True,
        include_change: bool = False,
        investigation_actions: tuple[object, ...] = (),
    ) -> CompiledInvestigationPlan:
        if plan.catalog_hash != self._registry.catalog_hash:
            raise InvestigationCatalogChangedError(
                "Playbook Plan 的目录 Hash 已失效"
            )
        task_keys: dict[int, str] = {}
        tasks: list[TaskSpec] = []
        suffix = "" if revision_no == 1 else f":r{revision_no}"
        for item in plan.items:
            manifest = self._registry.resolve(
                item.playbook_id, item.playbook_version
            )
            if item.manifest_hash != self._registry.manifest_hash(
                item.playbook_id, item.playbook_version
            ):
                raise InvestigationCatalogChangedError(
                    f"Playbook Manifest Hash 已失效：{item.playbook_id}"
                )
            task_key = (
                f"playbook:{item.ordinal}:{item.playbook_id}{suffix}"
            )
            task_keys[item.ordinal] = task_key
            tasks.append(
                TaskSpec(
                    task_key=task_key,
                    task_type="PLAYBOOK_INVOKE",
                    handler_id="dba.playbook.invoke",
                    handler_version="1",
                    input_schema_version=manifest.input_schema,
                    output_schema_version="DBA_TOOL_RESULT.v1",
                    depends_on=tuple(task_keys[value] for value in item.depends_on),
                    timeout_seconds=manifest.limits.timeout_seconds,
                    max_attempts=manifest.limits.max_attempts,
                    priority=50 + item.ordinal,
                )
            )
        invocation_keys = tuple(task.task_key for task in tasks)
        monitoring_keys: list[str] = []
        for binding_id in dict.fromkeys(monitoring_binding_ids):
            task_key = f"observe:{binding_id}{suffix}"
            monitoring_keys.append(task_key)
            tasks.append(
                TaskSpec(
                    task_key=task_key,
                    task_type="TOOL_INVOKE",
                    handler_id="monitor.observe",
                    handler_version="1",
                    input_schema_version="MONITOR_OBSERVE_INPUT.v1",
                    output_schema_version="OBSERVATION_SET.v1",
                    timeout_seconds=120,
                    max_attempts=3,
                    priority=45,
                )
            )
        log_keys: list[str] = []
        for binding_id in dict.fromkeys(log_binding_ids):
            task_key = f"log:{binding_id}{suffix}"
            log_keys.append(task_key)
            tasks.append(
                TaskSpec(
                    task_key=task_key,
                    task_type="TOOL_INVOKE",
                    handler_id="evidence.log-query",
                    handler_version="1",
                    input_schema_version="DIAGNOSIS_SCOPE.v1",
                    output_schema_version="LOG_EVIDENCE_SET.v1",
                    timeout_seconds=120,
                    max_attempts=3,
                    priority=46,
                )
            )
        direct_actions = tuple(
            action
            for action in investigation_actions
            if action.tool_id
            not in {
                "monitor.query_range",
                "loki.query_range",
                "db.oracle.readonly_query",
            }
        )
        direct_key_by_action = {
            action.action_id: f"diagnostic:{action.action_id}{suffix}"
            for action in direct_actions
        }
        dynamic_actions = tuple(
            action
            for action in investigation_actions
            if action.tool_id == "db.oracle.readonly_query"
        )
        dynamic_key_by_action = {
            action.action_id: f"dynamic:{action.action_id}{suffix}"
            for action in dynamic_actions
        }
        action_task_keys: dict[str, tuple[str, ...]] = {
            str(item.action_id): (task_keys[item.ordinal],)
            for item in plan.items
            if item.action_id is not None
        }
        for action in investigation_actions:
            if action.tool_id == "monitor.query_range":
                action_task_keys[action.action_id] = tuple(monitoring_keys)
            elif action.tool_id == "loki.query_range":
                action_task_keys[action.action_id] = tuple(log_keys)
        action_task_keys.update(
            {
                action_id: (task_key,)
                for action_id, task_key in direct_key_by_action.items()
            }
        )
        action_task_keys.update(
            {
                action_id: (task_key,)
                for action_id, task_key in dynamic_key_by_action.items()
            }
        )
        identity_task_keys = tuple(
            direct_key_by_action[action.action_id]
            for action in direct_actions
            if action.tool_id == "db.instance.identity"
        )

        def dependencies_for(action, *, require_identity: bool):
            model_dependencies = (
                task
                for action_id in action.depends_on
                for task in action_task_keys.get(action_id, ())
            )
            enforced_dependencies = (
                identity_task_keys
                if require_identity
                and action.tool_id != "db.instance.identity"
                else ()
            )
            return tuple(
                dict.fromkeys((*enforced_dependencies, *model_dependencies))
            )

        diagnostic_keys: list[str] = []
        for action in direct_actions:
            task_key = direct_key_by_action[action.action_id]
            diagnostic_keys.append(task_key)
            dependencies = dependencies_for(
                action,
                require_identity=action.tool_id.startswith("db."),
            )
            tasks.append(
                TaskSpec(
                    task_key=task_key,
                    task_type="TOOL_INVOKE",
                    handler_id="dba.diagnostic.invoke",
                    handler_version="1",
                    input_schema_version="DIAGNOSTIC_TOOL_INPUT.v1",
                    output_schema_version="DBA_TOOL_RESULT.v1",
                    depends_on=dependencies,
                    input_artifact_keys=dependencies,
                    timeout_seconds=45,
                    max_attempts=2,
                    priority=47,
                )
            )
        dynamic_keys: list[str] = []
        for action in dynamic_actions:
            task_key = dynamic_key_by_action[action.action_id]
            dynamic_keys.append(task_key)
            dependencies = dependencies_for(action, require_identity=True)
            tasks.append(
                TaskSpec(
                    task_key=task_key,
                    task_type="TOOL_INVOKE",
                    handler_id="dba.dynamic-query.invoke",
                    handler_version="1",
                    input_schema_version="DYNAMIC_QUERY_INPUT.v1",
                    output_schema_version="DBA_TOOL_RESULT.v1",
                    depends_on=dependencies,
                    input_artifact_keys=dependencies,
                    timeout_seconds=45,
                    max_attempts=3,
                    priority=48,
                )
            )
        evidence_task_keys = (
            *invocation_keys,
            *diagnostic_keys,
            *monitoring_keys,
            *log_keys,
            *dynamic_keys,
        )
        evidence_artifact_keys = (
            *user_evidence_artifact_keys,
            *evidence_task_keys,
        )
        assessment_task_key = f"evidence:assess{suffix}"
        tasks.append(
            TaskSpec(
                task_key=assessment_task_key,
                task_type="EVIDENCE_ASSESS",
                handler_id="dba.evidence.assess",
                handler_version="1",
                input_schema_version="DBA_EVIDENCE_ASSESS_INPUT.v1",
                output_schema_version="DBA_SUFFICIENCY.v1",
                depends_on=evidence_task_keys,
                input_artifact_keys=evidence_artifact_keys,
                timeout_seconds=self._model_timeout_seconds + 15,
                max_attempts=2,
                priority=90,
            )
        )
        action_plan_task_key = None
        proposal_task_key = None
        answer_dependency = assessment_task_key
        answer_inputs = (assessment_task_key,)
        if include_change:
            action_plan_task_key = "change:action-plan"
            proposal_task_key = "change:proposal"
            tasks.extend(
                (
                    TaskSpec(
                        task_key=action_plan_task_key,
                        task_type="ACTION_PLAN",
                        handler_id="change.chat-action-plan",
                        handler_version="1",
                        input_schema_version="CHAT_ACTION_PLAN_INPUT.v1",
                        output_schema_version="ACTION_PLAN.v1",
                        depends_on=(assessment_task_key,),
                        input_artifact_keys=(assessment_task_key,),
                        timeout_seconds=30,
                        max_attempts=2,
                        priority=95,
                    ),
                    TaskSpec(
                        task_key=proposal_task_key,
                        task_type="PROPOSAL",
                        handler_id="change.proposal",
                        handler_version="1",
                        input_schema_version="ACTION_PLAN.v1",
                        output_schema_version="PROPOSAL_OUTCOME.v1",
                        depends_on=(action_plan_task_key,),
                        input_artifact_keys=(action_plan_task_key,),
                        timeout_seconds=30,
                        max_attempts=2,
                        priority=98,
                    ),
                )
            )
            answer_dependency = proposal_task_key
            answer_inputs = (assessment_task_key, proposal_task_key)
        answer_task_key = "answer:compose" if include_answer else None
        if answer_task_key is not None:
            tasks.append(
                TaskSpec(
                    task_key=answer_task_key,
                    task_type="ANSWER",
                    handler_id="dba.answer.compose",
                    handler_version="1",
                    input_schema_version="DBA_ANSWER_INPUT.v1",
                    output_schema_version="AIOPS_TURN_RESULT.v1",
                    depends_on=(answer_dependency,),
                    input_artifact_keys=answer_inputs,
                    timeout_seconds=(
                        self._model_timeout_seconds * 2 + 30
                    ),
                    max_attempts=2,
                    priority=100,
                )
            )
        return CompiledInvestigationPlan(
            tasks=tuple(tasks),
            invocation_task_keys=invocation_keys,
            diagnostic_task_keys=tuple(diagnostic_keys),
            monitoring_task_keys=tuple(monitoring_keys),
            log_task_keys=tuple(log_keys),
            dynamic_task_keys=tuple(dynamic_keys),
            assessment_task_key=assessment_task_key,
            action_plan_task_key=action_plan_task_key,
            proposal_task_key=proposal_task_key,
            answer_task_key=answer_task_key,
        )
