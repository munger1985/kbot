"""从可信诊断 Evidence 生成 Action Plan 与 Proposal Snapshot。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from aiops_agent.actions import (
    ActionCompilerRegistry,
    ActionRegistry,
    ActionRenderer,
)
from aiops_agent.contracts.change import (
    ActionPlan,
    ActionPlanItem,
    ActionVerification,
    AdvisoryVerificationScope,
    ProposalOutcome,
)
from aiops_agent.contracts.artifacts import DatabaseDiagnosticResult
from aiops_agent.contracts.diagnosis import (
    EvidenceIndex,
    RootCauseAssessment,
    SolutionDraft,
)
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment
from aiops_agent.application.changes.proposal_snapshot import (
    build_proposal_snapshot,
)

from .handlers import TaskExecutionContext


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _artifact(context: TaskExecutionContext, schema: str) -> dict[str, Any]:
    return next(
        item["payload"]
        for item in reversed(context.input_artifacts)
        if item["schema_version"] == schema
    )


_SYSTEM_SCHEMAS = {"SYS", "SYSTEM", "OUTLN", "DBSNMP", "XDB", "AUDSYS"}


def _object_in_scope(parameters: dict[str, Any], policy: dict[str, Any]) -> bool:
    object_refs = tuple(
        value
        for value in parameters.values()
        if isinstance(value, dict)
        and {"schema", "object_type", "object_name"}.issubset(value)
    )
    scopes = dict(policy.get("object_scopes") or {})
    allowed = {str(item).upper() for item in scopes.get("schemas", ())}
    for object_ref in object_refs:
        schema = str(object_ref.get("schema", "")).upper()
        if allowed and schema not in allowed:
            return False
        if (
            scopes.get("exclude_system_objects", True)
            and schema in _SYSTEM_SCHEMAS
        ):
            return False
    if "parameter_name" in parameters:
        allowed_parameters = {
            str(item.get("name", "")).lower(): {
                str(value).upper()
                for value in item.get("allowed_values", ())
            }
            for item in scopes.get("dynamic_parameters", ())
            if isinstance(item, dict)
        }
        name = str(parameters["parameter_name"]).lower()
        value = str(parameters.get("parameter_value", "")).upper()
        if value not in allowed_parameters.get(name, set()):
            return False
    if "resource_plan_name" in parameters:
        plans = {
            str(item).upper()
            for item in scopes.get("resource_manager_plans", ())
        }
        if str(parameters["resource_plan_name"]).upper() not in plans:
            return False
    if "grantee_name" in parameters:
        grantees = {
            str(item).upper()
            for item in scopes.get("privilege_grantees", ())
        }
        if str(parameters["grantee_name"]).upper() not in grantees:
            return False
        privilege_key = (
            "object_privileges" if "object_ref" in parameters else "system_privileges"
        )
        privileges = {
            str(item).upper() for item in scopes.get(privilege_key, ())
        }
        if str(parameters.get("privilege", "")).upper() not in privileges:
            return False
    return True


def _canonical_object_ref(parameters: dict[str, Any]) -> dict[str, str] | None:
    return next(
        (
            dict(value)
            for value in parameters.values()
            if isinstance(value, dict)
            and {"schema", "object_type", "object_name"}.issubset(value)
        ),
        None,
    )


class ActionPlanHandler:
    def __init__(
        self,
        *,
        registry: ActionRegistry,
        execution_enabled: bool,
    ):
        self._registry = registry
        self._renderer = ActionRenderer()
        self._execution_enabled = execution_enabled

    async def execute(self, context: TaskExecutionContext) -> ActionPlan:
        root = RootCauseAssessment.model_validate(
            _artifact(context, "ROOT_CAUSE_ASSESSMENT.v1")
        )
        evidence = EvidenceIndex.model_validate(
            _artifact(context, "EVIDENCE_INDEX.v1")
        )
        SolutionDraft.model_validate(
            _artifact(context, "SOLUTION_DRAFT.v1")
        )
        policy_basis = {
            "target": context.plan_snapshot["target"],
            "binding": context.plan_snapshot["binding"],
            "policy": context.policy_snapshot,
            "trigger_type": context.trigger_type,
            "root_cause_level": root.effective_level,
            "execution_enabled": self._execution_enabled,
        }
        policy_hash = _hash(policy_basis)
        if root.effective_level not in {"CONFIRMED", "PROBABLE"}:
            return self._empty(
                context,
                root.effective_level,
                policy_hash,
                "ROOT_CAUSE_INSUFFICIENT",
            )
        candidate = self._session_candidate(
            evidence=evidence,
            db_type=context.plan_snapshot["target"]["db_type"],
        )
        if candidate is None:
            return self._empty(
                context,
                root.effective_level,
                policy_hash,
                "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE",
            )
        parameters, fact_refs = candidate
        target = context.plan_snapshot["target"]
        binding = context.plan_snapshot["binding"]
        policy = context.policy_snapshot.get("rules", {})
        allowed_actions = set(binding.get("allowed_actions", ()))
        if "db.session.terminate" not in allowed_actions:
            return self._empty(
                context,
                root.effective_level,
                policy_hash,
                "ACTION_NOT_ALLOWED_BY_BINDING",
            )
        capabilities = {
            name
            for name, enabled in target.get("capabilities", {}).items()
            if enabled is True
        }
        entitlements = set(policy.get("entitlements", ()))
        try:
            template = self._registry.resolve(
                action_template_id="db.session.terminate",
                version="1.0.0",
                db_type=target["db_type"],
                db_version=target.get("version_code") or "UNKNOWN",
                capabilities=capabilities,
                entitlements=entitlements,
                environment=target["environment"],
            )
            rendered = self._renderer.render(template, parameters)
        except (LookupError, ValueError):
            return self._empty(
                context,
                root.effective_level,
                policy_hash,
                "ACTION_TEMPLATE_UNAVAILABLE",
            )
        can_execute = (
            self._execution_enabled
            and context.trigger_type == "CHAT"
            and binding.get("allow_mutation") is True
            and target.get("controlled_change_enabled") is True
            and bool(target.get("execution_secret_configured"))
            and rendered.execution_mode
            == "EXECUTABLE_AFTER_APPROVAL"
        )
        mode = "AGENT_EXECUTE" if can_execute else "ADVISORY"
        action = ActionPlanItem(
            ordinal=1,
            action_template_id=rendered.action_template_id,
            action_template_version=rendered.action_template_version,
            variant=rendered.variant,
            mode=mode,
            action_family=rendered.action_family,
            effect_class=rendered.effect_class,
            execution_mode=rendered.execution_mode,
            executor_kind=rendered.executor_kind,
            canonical_object_ref=_canonical_object_ref(
                rendered.typed_parameters
            ),
            canonical_parameters=rendered.typed_parameters,
            parameter_fact_refs=fact_refs,
            rationale="阻塞会话参数来自当前 Target 的可信数据库事实",
            expected_effects=rendered.expected_effects,
            precondition_tool_refs=rendered.precondition_tool_refs,
            verification_tool_refs=rendered.verification_tool_refs,
            rollback_description=rendered.rollback_description,
            lock_impact=rendered.lock_impact,
            estimated_duration_seconds=rendered.estimated_duration_seconds,
            rendered_action=rendered.model_dump(mode="json"),
        )
        return ActionPlan(
            solution_group_key=f"diagnosis:{context.run_id}:solution",
            target_id=context.target_id,
            root_cause_level=root.effective_level,
            actions=(action,),
            decision=mode,
            decision_reasons=(
                "MUTATION_POLICY_ALLOWED"
                if can_execute
                else "MUTATION_EXECUTION_UNAVAILABLE",
            ),
            policy_decision_hash=policy_hash,
            action_catalog_hash=self._registry.catalog_hash,
        )

    def _empty(
        self,
        context: TaskExecutionContext,
        root_level: str,
        policy_hash: str,
        reason: str,
    ) -> ActionPlan:
        return ActionPlan(
            solution_group_key=f"diagnosis:{context.run_id}:solution",
            target_id=context.target_id,
            root_cause_level=root_level,
            decision="NO_ACTION",
            decision_reasons=(reason,),
            policy_decision_hash=policy_hash,
            action_catalog_hash=self._registry.catalog_hash,
        )

    @staticmethod
    def _session_candidate(
        *, evidence: EvidenceIndex, db_type: str
    ) -> tuple[dict[str, int], dict[str, str]] | None:
        blocking = next(
            (
                item
                for item in evidence.facts
                if item.trust_level == "SOURCE_VERIFIED"
                and item.metric_or_fact_type
                == "db.session.blocking_chain"
                and isinstance(item.value, dict)
                and item.value.get("blocking_session_id")
            ),
            None,
        )
        if blocking is None:
            return None
        session_id = int(blocking.value["blocking_session_id"])
        if db_type == "MYSQL":
            return (
                {"session_id": session_id},
                {"session_id": blocking.fact_id},
            )
        active = next(
            (
                item
                for item in evidence.facts
                if item.trust_level == "SOURCE_VERIFIED"
                and item.metric_or_fact_type == "db.session.active"
                and isinstance(item.value, dict)
                and int(item.value.get("session_id", -1)) == session_id
                and item.value.get("serial_number")
                and item.value.get("instance_id")
            ),
            None,
        )
        if active is None:
            return None
        return (
            {
                "session_id": session_id,
                "serial_number": int(active.value["serial_number"]),
                "instance_id": int(active.value["instance_id"]),
            },
            {
                "session_id": blocking.fact_id,
                "serial_number": active.fact_id,
                "instance_id": active.fact_id,
            },
        )


class ChatActionPlanHandler:
    """只把当前 Turn 的可信结构化事实编译成目录内动作。"""

    def __init__(
        self,
        *,
        registry: ActionRegistry,
        execution_enabled: bool,
    ) -> None:
        self._registry = registry
        self._renderer = ActionRenderer()
        self._compilers = ActionCompilerRegistry()
        self._execution_enabled = execution_enabled

    async def execute(self, context: TaskExecutionContext) -> ActionPlan:
        assessment = DbaSufficiencyAssessment.model_validate(
            _artifact(context, "DBA_SUFFICIENCY.v1")
        )
        change_context = dict(context.plan_snapshot.get("change_context", {}))
        target = dict(change_context.get("target", {}))
        policy = dict(change_context.get("policy", {}))
        rules = dict(policy.get("rules", {}))
        action_policy = dict(
            change_context.get("controlled_action_execution", {})
        )
        policy_hash = _hash(
            {
                "change_context": change_context,
                "trigger_type": context.trigger_type,
                "execution_enabled": self._execution_enabled,
            }
        )
        task_frame = dict(
            dict(context.plan_snapshot.get("answer_context", {})).get(
                "task_frame", {}
            )
        )
        if not bool(task_frame.get("requires_change")):
            return self._empty(context, policy_hash, "CHANGE_NOT_REQUESTED")
        capabilities = set(
            dict(context.plan_snapshot.get("capability_snapshot", {})).get(
                "target_capabilities", ()
            )
        )
        allowed_actions = set(action_policy.get("allowed_action_ids", ()))
        templates = self._registry.compatible(
            db_type=str(target.get("db_type")),
            db_version=str(target.get("version_code") or "UNKNOWN"),
            capabilities=capabilities,
            entitlements=set(rules.get("entitlements", ())),
            environment=str(target.get("environment")),
            include_planned=False,
        )
        actions = []
        for template in templates:
            definition = template.definition
            if definition.action_template_id not in allowed_actions:
                continue
            compiled = self._compilers.compile_turn(
                compiler_id=definition.compiler_id,
                assessment=assessment,
                db_type=definition.db_type,
            )
            if compiled is None:
                continue
            if not _object_in_scope(compiled.parameters, action_policy):
                continue
            try:
                rendered = self._renderer.render(template, compiled.parameters)
            except ValueError:
                continue
            can_execute = (
                self._execution_enabled
                and context.trigger_type == "CHAT"
                and action_policy.get("enabled") is True
                and target.get("status") == "ENABLED"
                and target.get("connectivity_status")
                in {"CONNECTED", "DEGRADED"}
                and bool(target.get("execution_secret_configured"))
                and rendered.execution_mode
                == "EXECUTABLE_AFTER_APPROVAL"
            )
            mode = "AGENT_EXECUTE" if can_execute else "ADVISORY"
            if (
                rendered.execution_mode == "EXECUTABLE_AFTER_APPROVAL"
                and not can_execute
            ):
                continue
            actions.append(
                ActionPlanItem(
                    ordinal=len(actions) + 1,
                    action_template_id=rendered.action_template_id,
                    action_template_version=rendered.action_template_version,
                    variant=rendered.variant,
                    mode=mode,
                    action_family=rendered.action_family,
                    effect_class=rendered.effect_class,
                    execution_mode=rendered.execution_mode,
                    executor_kind=rendered.executor_kind,
                    canonical_object_ref=_canonical_object_ref(
                        rendered.typed_parameters
                    ),
                    canonical_parameters=rendered.typed_parameters,
                    parameter_fact_refs=compiled.fact_refs,
                    rationale=compiled.rationale,
                    expected_effects=rendered.expected_effects,
                    precondition_tool_refs=rendered.precondition_tool_refs,
                    verification_tool_refs=rendered.verification_tool_refs,
                    rollback_description=rendered.rollback_description,
                    lock_impact=rendered.lock_impact,
                    estimated_duration_seconds=(
                        rendered.estimated_duration_seconds
                    ),
                    rendered_action=rendered.model_dump(mode="json"),
                )
            )
        object_family_keys = set()
        for action in actions:
            object_ref = action.canonical_object_ref
            if object_ref is None:
                continue
            key = (
                action.action_family,
                str(object_ref.get("schema", "")).upper(),
                str(object_ref.get("object_type", "")).upper(),
                str(object_ref.get("object_name", "")).upper(),
            )
            if key in object_family_keys:
                return self._empty(
                    context,
                    policy_hash,
                    "AMBIGUOUS_ACTION_INTENT",
                )
            object_family_keys.add(key)
        if not actions:
            return self._empty(
                context,
                policy_hash,
                (
                    "AGENT_EXECUTION_NOT_ALLOWED"
                    if not allowed_actions
                    else "VERIFIED_ACTION_PARAMETERS_UNAVAILABLE"
                ),
            )
        return ActionPlan(
            solution_group_key=f"turn:{context.run_id}:change",
            target_id=context.target_id,
            root_cause_level="EVIDENCE_VERIFIED",
            actions=tuple(actions),
            decision=(
                "AGENT_EXECUTE"
                if any(item.mode == "AGENT_EXECUTE" for item in actions)
                else "ADVISORY"
            ),
            decision_reasons=("MUTATION_POLICY_ALLOWED",),
            policy_decision_hash=policy_hash,
            action_catalog_hash=self._registry.catalog_hash,
        )

    def _empty(
        self, context: TaskExecutionContext, policy_hash: str, reason: str
    ) -> ActionPlan:
        return ActionPlan(
            solution_group_key=f"turn:{context.run_id}:change",
            target_id=context.target_id,
            root_cause_level="INCONCLUSIVE",
            decision="NO_ACTION",
            decision_reasons=(reason,),
            policy_decision_hash=policy_hash,
            action_catalog_hash=self._registry.catalog_hash,
        )

    @classmethod
    def _session_candidate(
        cls,
        assessment: DbaSufficiencyAssessment,
        db_type: str,
    ) -> tuple[dict[str, int], dict[str, str]] | None:
        blocking_rows = cls._verified_rows(
            assessment, "db.session.blocking_chain"
        )
        active_rows = cls._verified_rows(assessment, "db.session.active")
        for blocking, blocking_ref in blocking_rows:
            session_id = blocking.get("blocking_session_id")
            if session_id is None:
                continue
            if db_type == "MYSQL":
                return (
                    {"session_id": int(session_id)},
                    {"session_id": blocking_ref},
                )
            for active, active_ref in active_rows:
                if int(active.get("session_id", -1)) != int(session_id):
                    continue
                serial_number = active.get("serial_number")
                instance_id = active.get("instance_id")
                if serial_number is None or instance_id is None:
                    continue
                return (
                    {
                        "session_id": int(session_id),
                        "serial_number": int(serial_number),
                        "instance_id": int(instance_id),
                    },
                    {
                        "session_id": blocking_ref,
                        "serial_number": active_ref,
                        "instance_id": active_ref,
                    },
                )
        return None

    @staticmethod
    def _verified_rows(
        assessment: DbaSufficiencyAssessment, tool_id: str
    ) -> tuple[tuple[dict[str, Any], str], ...]:
        rows: list[tuple[dict[str, Any], str]] = []
        for fact in assessment.evidence:
            if (
                fact.trust_level != "SOURCE_VERIFIED"
                or fact.tool_id != tool_id
            ):
                continue
            names = [str(item.get("name", "")).lower() for item in fact.columns]
            for values in fact.rows:
                rows.append(
                    (
                        dict(zip(names, values, strict=True)),
                        fact.evidence_ref,
                    )
                )
        return tuple(rows)


class ProposalSnapshotHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> ProposalOutcome:
        plan = ActionPlan.model_validate(
            _artifact(context, "ACTION_PLAN.v1")
        )
        if not plan.actions:
            return ProposalOutcome(
                status="NOT_REQUIRED",
                reason=(
                    plan.decision_reasons[0]
                    if plan.decision_reasons
                    else "NO_ACTION"
                ),
            )
        action = plan.actions[0]
        now = datetime.now(UTC)
        target_snapshot = dict(
            context.plan_snapshot.get("target")
            or dict(context.plan_snapshot.get("change_context", {})).get(
                "target", {}
            )
        )
        return ProposalOutcome(
            status="CREATED",
            proposal=build_proposal_snapshot(
                plan=plan,
                action=action,
                run_id=context.run_id,
                task_id=context.task_id,
                target_id=context.target_id,
                target_version=int(target_snapshot["row_version"]),
                now=now,
            ),
        )


class AdvisoryVerificationScopeHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> AdvisoryVerificationScope:
        snapshot = context.plan_snapshot["advisory_verification"]
        return AdvisoryVerificationScope.model_validate(snapshot)


class ActionVerificationHandler:
    """仅根据回填后的全新只读数据库观测判断动作效果。"""

    async def execute(
        self, context: TaskExecutionContext
    ) -> ActionVerification:
        scope = AdvisoryVerificationScope.model_validate(
            _artifact(context, "ADVISORY_VERIFICATION_SCOPE.v1")
        )
        results = {
            result.tool_id: result
            for result in (
                DatabaseDiagnosticResult.model_validate(item["payload"])
                for item in context.input_artifacts
                if item["schema_version"]
                == "DATABASE_DIAGNOSTIC_RESULT.v1"
            )
        }
        required = set(scope.verification_tool_refs)
        gap_codes = set(scope.initial_gap_codes)
        for tool_id in required:
            result = results.get(tool_id)
            if (
                result is not None
                and result.status == "SUCCEEDED"
                and result.observation is not None
            ):
                continue
            gap_codes.add(
                result.gap.code
                if result is not None and result.gap is not None
                else "VERIFICATION_EVIDENCE_MISSING"
            )
        gaps = tuple(sorted(gap_codes))
        successful = {
            tool_id: results[tool_id].observation
            for tool_id in required
            if tool_id in results
            and results[tool_id].status == "SUCCEEDED"
            and results[tool_id].observation is not None
        }
        hashes = tuple(
            sorted(
                observation.result_sha256
                for observation in successful.values()
            )
        )
        if gaps:
            return self._result(
                scope,
                status="INCONCLUSIVE",
                summary="只读验证证据不完整，不能确认人工动作是否生效",
                gap_codes=gaps,
                evidence_hashes=hashes,
            )
        parameters = scope.canonical_parameters
        if scope.action_template_id == "db.session.cancel_sql":
            observation = successful.get("db.session.current_sql")
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少会话当前 SQL 状态，不能确认取消效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            session_present = self._contains_session_identity(
                observation, parameters
            )
            if not session_present:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标会话在取消 SQL 后不可见，需复核是否被意外断开",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            still_running = self._contains_current_sql(observation, parameters)
            return self._result(
                scope,
                status="NOT_ACHIEVED" if still_running else "VERIFIED",
                summary=(
                    "指定 SQL 仍由目标会话执行，取消未达到预期效果"
                    if still_running
                    else "指定 SQL 已不再由目标会话执行，且未要求断开会话"
                ),
                effect_achieved=not still_running,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        if scope.action_template_id == "db.object.compile":
            observation = successful.get("db.object.status")
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少对象状态，不能确认编译效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            ref = dict(parameters["object_ref"])
            matched = self._matching_object_rows(
                observation,
                ref,
                object_type=str(parameters["object_type"]),
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标对象在编译后不可见，需 DBA 复核对象状态",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            valid = all(
                str(row.get("status", "")).upper() == "VALID"
                for row in matched
            )
            return self._result(
                scope,
                status="VERIFIED" if valid else "NOT_ACHIEVED",
                summary=(
                    "目标对象状态为 VALID，编译效果已验证"
                    if valid
                    else "目标对象仍为 INVALID，需检查编译错误和依赖对象"
                ),
                effect_achieved=valid,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        if scope.action_template_id == "db.statistics.gather":
            observation = successful.get("db.table.statistics")
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少表统计信息状态，不能确认收集效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            matched = self._matching_table_rows(
                observation, dict(parameters["table_ref"])
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标表在统计信息收集后不可见，需 DBA 复核对象状态",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            current = matched[0]
            gathered = (
                current.get("last_analyzed") is not None
                and str(current.get("stale_stats") or "").upper() != "YES"
                and not str(current.get("stattype_locked") or "").strip()
            )
            return self._result(
                scope,
                status="VERIFIED" if gathered else "NOT_ACHIEVED",
                summary=(
                    "目标表统计信息已刷新且不再过期"
                    if gathered
                    else "目标表统计信息仍缺失、过期或被锁定"
                ),
                effect_achieved=gathered,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        statistics_state_tools = {
            "db.statistics.lock": "db.table.statistics.lock_candidate",
            "db.statistics.unlock": "db.table.statistics.unlock_candidate",
        }
        if scope.action_template_id in statistics_state_tools:
            observation = successful.get(
                statistics_state_tools[scope.action_template_id]
            )
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少表统计锁状态，不能确认状态变更效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            matched = self._matching_table_rows(
                observation, dict(parameters["table_ref"])
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标表在统计锁状态变更后不可见",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            locked = bool(
                str(matched[0].get("stattype_locked") or "").strip()
            )
            achieved = (
                locked
                if scope.action_template_id == "db.statistics.lock"
                else not locked
            )
            return self._result(
                scope,
                status="VERIFIED" if achieved else "NOT_ACHIEVED",
                summary=(
                    "表统计锁状态变更已验证"
                    if achieved
                    else "表统计锁状态尚未达到目标状态"
                ),
                effect_achieved=achieved,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        if scope.action_template_id == "db.scheduler.job.run":
            observation = successful.get("db.scheduler.job.status")
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少 Scheduler Job 状态，不能确认启动效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            matched = self._matching_scheduler_job_rows(
                observation, dict(parameters["job_ref"])
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标 Scheduler Job 在启动后不可见，需 DBA 复核对象状态",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            current = matched[0]
            state = str(current.get("state") or "").upper()
            run_count = int(current.get("run_count") or 0)
            failure_count = int(current.get("failure_count") or 0)
            started = state == "RUNNING" or (
                run_count > int(parameters["previous_run_count"])
                and failure_count
                == int(parameters["previous_failure_count"])
            )
            failed = failure_count > int(parameters["previous_failure_count"])
            return self._result(
                scope,
                status="VERIFIED" if started else "NOT_ACHIEVED",
                summary=(
                    "目标 Scheduler Job 已开始运行或完成一次无新增失败的运行"
                    if started
                    else (
                        "目标 Scheduler Job 本次运行新增失败记录"
                        if failed
                        else "目标 Scheduler Job 尚未开始运行"
                    )
                ),
                effect_achieved=started,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        scheduler_state_tools = {
            "db.scheduler.job.enable": "db.scheduler.job.enable_candidate",
            "db.scheduler.job.disable": "db.scheduler.job.disable_candidate",
            "db.scheduler.job.stop": "db.scheduler.job.stop_candidate",
        }
        if scope.action_template_id in scheduler_state_tools:
            observation = successful.get(
                scheduler_state_tools[scope.action_template_id]
            )
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少 Scheduler Job 状态，不能确认状态变更效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            matched = self._matching_scheduler_job_rows(
                observation, dict(parameters["job_ref"])
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标 Scheduler Job 在状态变更后不可见",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            current = matched[0]
            enabled = str(current.get("enabled") or "").upper()
            state = str(current.get("state") or "").upper()
            achieved = {
                "db.scheduler.job.enable": (
                    enabled == "TRUE" and state != "DISABLED"
                ),
                "db.scheduler.job.disable": (
                    enabled == "FALSE" and state == "DISABLED"
                ),
                "db.scheduler.job.stop": state != "RUNNING",
            }[scope.action_template_id]
            return self._result(
                scope,
                status="VERIFIED" if achieved else "NOT_ACHIEVED",
                summary=(
                    "Scheduler Job 状态变更已验证"
                    if achieved
                    else "Scheduler Job 尚未达到目标状态"
                ),
                effect_achieved=achieved,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        user_state_tools = {
            "db.user.lock": "db.user.lock_candidate",
            "db.user.unlock": "db.user.unlock_candidate",
            "db.user.password.expire": "db.user.password_expire_candidate",
        }
        if scope.action_template_id in user_state_tools:
            observation = successful.get(
                user_state_tools[scope.action_template_id]
            )
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少用户账号状态，不能确认状态变更效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            matched = self._matching_user_rows(
                observation, dict(parameters["user_ref"])
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标用户在账号状态变更后不可见",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            account_status = str(
                matched[0].get("account_status") or ""
            ).upper()
            locked = "LOCKED" in account_status
            expired = "EXPIRED" in account_status
            achieved = {
                "db.user.lock": locked,
                "db.user.unlock": not locked,
                "db.user.password.expire": expired,
            }[scope.action_template_id]
            return self._result(
                scope,
                status="VERIFIED" if achieved else "NOT_ACHIEVED",
                summary=(
                    "用户账号锁状态变更已验证"
                    if achieved
                    else "用户账号锁状态尚未达到目标状态"
                ),
                effect_achieved=achieved,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        if scope.action_template_id in {
            "db.parameter.set",
            "db.resource_manager.plan.switch",
            "db.user.privilege.grant",
            "db.user.privilege.revoke",
        }:
            if scope.action_template_id == "db.parameter.set":
                tool_id = "db.parameter.dynamic_state"
            elif scope.action_template_id == "db.resource_manager.plan.switch":
                tool_id = "db.resource_manager.plan_state"
            else:
                tool_id = (
                    "db.user.object_privilege_state"
                    if "object_ref" in parameters
                    else "db.user.system_privilege_state"
                )
            observation = successful.get(tool_id)
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少配置或权限状态，不能确认动作效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            names = [
                str(column.name).lower() for column in observation.columns
            ]
            rows = [dict(zip(names, row)) for row in observation.rows]
            if scope.action_template_id == "db.parameter.set":
                matched = [
                    row
                    for row in rows
                    if str(row.get("parameter_name", "")).lower()
                    == str(parameters["parameter_name"]).lower()
                ]
                achieved = bool(matched) and str(
                    matched[0].get("current_value") or ""
                ).upper() == str(parameters["parameter_value"]).upper()
            elif scope.action_template_id == "db.resource_manager.plan.switch":
                matched = [
                    row
                    for row in rows
                    if str(row.get("resource_plan_name", "")).upper()
                    == str(parameters["resource_plan_name"]).upper()
                ]
                achieved = bool(matched) and str(
                    matched[0].get("current_plan_name") or ""
                ).upper() == str(parameters["resource_plan_name"]).upper()
            else:
                matched = self._matching_privilege_rows(rows, parameters)
                expected = scope.action_template_id.endswith(".grant")
                achieved = bool(matched) and (
                    str(matched[0].get("is_granted") or "").upper() == "YES"
                ) == expected
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标配置、对象或用户在执行后不可见，需 DBA 立即复核",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            return self._result(
                scope,
                status="VERIFIED" if achieved else "NOT_ACHIEVED",
                summary=(
                    "配置或权限状态已达到批准目标"
                    if achieved
                    else "配置或权限状态尚未达到批准目标"
                ),
                effect_achieved=achieved,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        if scope.action_template_id in {
            "db.storage.datafile.resize",
            "db.storage.tempfile.resize",
            "db.storage.datafile.autoextend",
            "db.storage.tempfile.autoextend",
        }:
            datafile = ".datafile." in scope.action_template_id
            resize = scope.action_template_id.endswith(".resize")
            tool_id = (
                "db.storage.datafile.action_state"
                if datafile
                else "db.storage.tempfile.action_state"
            )
            observation = successful.get(tool_id)
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少文件状态，不能确认存储变更效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            matched = self._matching_file_rows(
                observation, str(parameters["file_name"])
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标数据库文件在执行后不可见，需 DBA 立即复核",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            row = matched[0]
            if resize:
                achieved = int(row.get("current_size_mb") or 0) >= int(
                    parameters["new_size_mb"]
                )
            else:
                achieved = (
                    str(row.get("autoextensible") or "").upper() == "YES"
                    and int(row.get("current_next_mb") or 0)
                    == int(parameters["next_mb"])
                    and int(row.get("current_max_size_mb") or 0)
                    == int(parameters["max_size_mb"])
                )
            return self._result(
                scope,
                status="VERIFIED" if achieved else "NOT_ACHIEVED",
                summary=(
                    "数据库文件已达到批准的存储目标"
                    if achieved
                    else "数据库文件尚未达到批准的存储目标"
                ),
                effect_achieved=achieved,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        if scope.action_template_id in {
            "db.index.rebuild",
            "db.index.partition.rebuild",
            "db.index.coalesce",
        }:
            partition_action = (
                scope.action_template_id == "db.index.partition.rebuild"
            )
            coalesce_action = scope.action_template_id == "db.index.coalesce"
            tool_id = (
                "db.index.partition.health"
                if partition_action
                else (
                    "db.index.coalesce_candidate"
                    if coalesce_action
                    else "db.index.health"
                )
            )
            observation = successful.get(tool_id)
            if observation is None:
                return self._result(
                    scope,
                    status="INCONCLUSIVE",
                    summary="缺少索引或分区健康状态，不能确认重建效果",
                    gap_codes=("VERIFICATION_EVIDENCE_MISSING",),
                    evidence_hashes=hashes,
                )
            ref = dict(parameters["index_ref"])
            matched = self._matching_index_rows(
                observation,
                ref,
                partition_name=(
                    str(parameters["partition_name"])
                    if partition_action
                    else None
                ),
            )
            if not matched:
                return self._result(
                    scope,
                    status="ADVERSE",
                    summary="目标索引在执行后不可见，需 DBA 立即复核对象状态",
                    effect_achieved=False,
                    adverse_effect=True,
                    evidence_hashes=hashes,
                )
            expected_status = "USABLE" if partition_action else "VALID"
            subject = "目标索引分区" if partition_action else "目标索引"
            valid = all(
                str(row.get("status", "")).upper() == expected_status
                for row in matched
            )
            if valid:
                summary = f"{subject}状态为 {expected_status}，" + (
                    "合并后状态已通过只读观测验证"
                    if coalesce_action
                    else "重建效果已通过只读观测验证"
                )
            else:
                summary = f"{subject}仍不是 {expected_status}，" + (
                    "合并后状态异常"
                    if coalesce_action
                    else "重建未达到预期效果"
                )
            return self._result(
                scope,
                status="VERIFIED" if valid else "NOT_ACHIEVED",
                summary=summary,
                effect_achieved=valid,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        if scope.action_template_id != "db.session.terminate":
            return self._result(
                scope,
                status="INCONCLUSIVE",
                summary="当前动作尚未登记专用验证器",
                gap_codes=("ACTION_VERIFIER_UNAVAILABLE",),
                evidence_hashes=hashes,
            )
        active = self._contains_session(
            successful["db.session.active"],
            parameters,
            blocking=False,
        )
        blocking = self._contains_session(
            successful["db.session.blocking_chain"],
            parameters,
            blocking=True,
        )
        if not active and not blocking:
            return self._result(
                scope,
                status="VERIFIED",
                summary="目标会话已不在活动会话和阻塞链中，人工动作效果已验证",
                target_still_present=False,
                blocking_still_present=False,
                effect_achieved=True,
                adverse_effect=False,
                evidence_hashes=hashes,
            )
        return self._result(
            scope,
            status="NOT_ACHIEVED",
            summary="目标会话或其阻塞关系仍然存在，人工动作未达到预期效果",
            target_still_present=active,
            blocking_still_present=blocking,
            effect_achieved=False,
            adverse_effect=False,
            evidence_hashes=hashes,
        )

    @staticmethod
    def _contains_session(observation, parameters, *, blocking: bool) -> bool:
        names = [
            str(column.name).lower() for column in observation.columns
        ]
        rows = [dict(zip(names, row)) for row in observation.rows]
        session_key = (
            "blocking_session_id" if blocking else "session_id"
        )
        expected_session = int(parameters["session_id"])
        expected_instance = parameters.get("instance_id")
        for row in rows:
            value = row.get(session_key)
            if value is None or int(value) != expected_session:
                continue
            if expected_instance is None:
                return True
            instance_key = (
                "blocking_instance_id" if blocking else "instance_id"
            )
            if int(row.get(instance_key, -1)) == int(expected_instance):
                return True
        return False

    @staticmethod
    def _contains_current_sql(observation, parameters) -> bool:
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        return any(
            int(row.get("instance_id", -1))
            == int(parameters["instance_id"])
            and int(row.get("session_id", -1))
            == int(parameters["session_id"])
            and int(row.get("serial_number", -1))
            == int(parameters["serial_number"])
            and str(row.get("sql_id", "")).lower()
            == str(parameters["sql_id"]).lower()
            for row in rows
        )

    @staticmethod
    def _contains_session_identity(observation, parameters) -> bool:
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        return any(
            int(row.get("instance_id", -1))
            == int(parameters["instance_id"])
            and int(row.get("session_id", -1))
            == int(parameters["session_id"])
            and int(row.get("serial_number", -1))
            == int(parameters["serial_number"])
            for row in rows
        )

    @staticmethod
    def _matching_index_rows(
        observation, object_ref, *, partition_name: str | None = None
    ):
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        matched = tuple(
            row
            for row in rows
            if str(row.get("owner", "")).upper()
            == str(object_ref["schema"]).upper()
            and str(row.get("index_name", "")).upper()
            == str(object_ref["object_name"]).upper()
            and (
                partition_name is None
                or str(row.get("partition_name", "")).upper()
                == partition_name.upper()
            )
        )
        return matched

    @staticmethod
    def _matching_object_rows(observation, object_ref, *, object_type: str):
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        return tuple(
            row
            for row in rows
            if str(row.get("owner", "")).upper()
            == str(object_ref["schema"]).upper()
            and str(row.get("object_name", "")).upper()
            == str(object_ref["object_name"]).upper()
            and str(row.get("object_type", "")).upper()
            == object_type.upper()
        )

    @staticmethod
    def _matching_table_rows(observation, table_ref):
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        return tuple(
            row
            for row in rows
            if str(row.get("owner", "")).upper()
            == str(table_ref["schema"]).upper()
            and str(row.get("table_name", "")).upper()
            == str(table_ref["object_name"]).upper()
        )

    @staticmethod
    def _matching_scheduler_job_rows(observation, job_ref):
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        return tuple(
            row
            for row in rows
            if str(row.get("owner", "")).upper()
            == str(job_ref["schema"]).upper()
            and str(row.get("job_name", "")).upper()
            == str(job_ref["object_name"]).upper()
        )

    @staticmethod
    def _matching_user_rows(observation, user_ref):
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        return tuple(
            row
            for row in rows
            if str(row.get("username", "")).upper()
            == str(user_ref["object_name"]).upper()
            and str(row.get("oracle_maintained", "")).upper() == "N"
            and str(row.get("common", "")).upper() == "NO"
        )

    @staticmethod
    def _matching_file_rows(observation, file_name: str):
        names = [str(column.name).lower() for column in observation.columns]
        rows = [dict(zip(names, row)) for row in observation.rows]
        return tuple(
            row
            for row in rows
            if str(row.get("file_name", "")) == file_name
            and str(row.get("status", "")).upper() == "AVAILABLE"
            and str(row.get("online_status", "")).upper() == "ONLINE"
        )

    @staticmethod
    def _matching_privilege_rows(rows, parameters):
        object_ref = dict(parameters.get("object_ref") or {})
        return tuple(
            row
            for row in rows
            if str(row.get("grantee_name", "")).upper()
            == str(parameters["grantee_name"]).upper()
            and str(row.get("privilege", "")).upper()
            == str(parameters["privilege"]).upper()
            and (
                not object_ref
                or (
                    str(row.get("owner", "")).upper()
                    == str(object_ref["schema"]).upper()
                    and str(row.get("object_name", "")).upper()
                    == str(object_ref["object_name"]).upper()
                    and str(row.get("object_type", "")).upper()
                    == str(object_ref["object_type"]).upper()
                )
            )
        )

    @staticmethod
    def _result(
        scope: AdvisoryVerificationScope,
        *,
        status: str,
        summary: str,
        target_still_present: bool | None = None,
        blocking_still_present: bool | None = None,
        effect_achieved: bool | None = None,
        adverse_effect: bool | None = None,
        gap_codes: tuple[str, ...] = (),
        evidence_hashes: tuple[str, ...] = (),
    ) -> ActionVerification:
        return ActionVerification(
            proposal_id=scope.proposal_id,
            source_run_id=scope.source_run_id,
            result_artifact_id=scope.result_artifact_id,
            status=status,
            summary=summary,
            target_still_present=target_still_present,
            blocking_still_present=blocking_still_present,
            effect_achieved=effect_achieved,
            adverse_effect=adverse_effect,
            checked_tool_refs=scope.verification_tool_refs,
            gap_codes=gap_codes,
            evidence_hashes=evidence_hashes,
        )
