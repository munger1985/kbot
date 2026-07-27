"""从可信诊断 Evidence 生成 Action Plan 与 Proposal Snapshot。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from aiops_agent.actions import ActionRegistry, ActionRenderer
from aiops_agent.contracts.change import (
    ActionPlan,
    ActionPlanItem,
    ActionVerification,
    AdvisoryVerificationScope,
    ChangeProposalSnapshot,
    ProposalOutcome,
)
from aiops_agent.contracts.artifacts import DatabaseDiagnosticResult
from aiops_agent.contracts.diagnosis import (
    EvidenceIndex,
    RootCauseAssessment,
    SolutionDraft,
)
from platform_core.identity import uuid7

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
            and target["execution_mode"] == "AGENT_EXECUTE"
            and binding.get("access_mode") == "EXECUTE"
            and policy.get("allow_agent_execution") is True
            and rendered.execution_capability
            == "EXECUTABLE_AFTER_APPROVAL"
        )
        mode = "AGENT_EXECUTE" if can_execute else "ADVISORY"
        action = ActionPlanItem(
            ordinal=1,
            action_template_id=rendered.action_template_id,
            action_template_version=rendered.action_template_version,
            variant=rendered.variant,
            mode=mode,
            canonical_parameters=rendered.typed_parameters,
            parameter_fact_refs=fact_refs,
            rationale="阻塞会话参数来自当前 Target 的可信数据库事实",
            expected_effects=rendered.expected_effects,
            precondition_tool_refs=rendered.precondition_tool_refs,
            verification_tool_refs=rendered.verification_tool_refs,
            rollback_description=rendered.rollback_description,
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
                else "ADVISORY_SAFETY_DOWNGRADE",
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
        rendered = action.rendered_action
        now = datetime.now(UTC)
        proposal_id = uuid7()
        body = {
            "proposal_id": str(proposal_id),
            "run_id": context.run_id,
            "task_id": context.task_id,
            "target_id": context.target_id,
            "target_version": int(
                context.plan_snapshot["target"]["row_version"]
            ),
            "solution_group_key": plan.solution_group_key,
            "command_ordinal": action.ordinal,
            "proposal_version": 1,
            "mode": action.mode,
            "action_template_id": action.action_template_id,
            "action_template_version": action.action_template_version,
            "action_template_variant": action.variant,
            "action_template_hash": rendered["template_hash"],
            "renderer_version": rendered["renderer_version"],
            "canonical_parameters": action.canonical_parameters,
            "parameter_fact_refs": action.parameter_fact_refs,
            "parameters_hash": rendered["parameters_hash"],
            "rendered_command": rendered["command_text"],
            "command_hash": rendered["command_hash"],
            "risk_level": rendered["risk_level"],
            "impact": "终止一个已由当前数据库事实确认的会话",
            "rationale": action.rationale,
            "preconditions": action.precondition_tool_refs,
            "rollback_plan": (
                action.rollback_description or "该动作不可逆，需重新建立会话"
            ),
            "verification_plan": action.verification_tool_refs,
            "evidence_refs": tuple(
                sorted(set(action.parameter_fact_refs.values()))
            ),
            "policy_decision_hash": plan.policy_decision_hash,
            "expires_at": now + timedelta(minutes=15),
        }
        return ProposalOutcome(
            status="CREATED",
            proposal=ChangeProposalSnapshot(
                **body,
                proposal_hash=_hash(body),
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
        required = {
            "db.session.active",
            "db.session.blocking_chain",
        }
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
                evidence_hashes=hashes,
            )
        return self._result(
            scope,
            status="NOT_ACHIEVED",
            summary="目标会话或其阻塞关系仍然存在，人工动作未达到预期效果",
            target_still_present=active,
            blocking_still_present=blocking,
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
    def _result(
        scope: AdvisoryVerificationScope,
        *,
        status: str,
        summary: str,
        target_still_present: bool | None = None,
        blocking_still_present: bool | None = None,
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
            checked_tool_refs=scope.verification_tool_refs,
            gap_codes=gap_codes,
            evidence_hashes=evidence_hashes,
        )
