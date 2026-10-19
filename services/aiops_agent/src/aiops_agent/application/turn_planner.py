"""统一 AIOps Turn 路由、范围解析与诊断内核桥接。"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Protocol
from uuid import UUID

from aiops_agent.application.errors import (
    AIOpsApplicationError,
    dependency_unavailable,
    resource_not_found,
    state_conflict,
    validation_failed,
)
from aiops_agent.entities import (
    OpsAnswerBlockEntity,
    OpsConversationMessageEntity,
    OpsRunEntity,
    OpsTurnEventEntity,
    OpsTurnRunEntity,
)
from aiops_agent.application.turn_capabilities import CapabilityResult
from platform_core.contracts.aiops import (
    AIOpsCapability,
    BusinessSystemRefV1,
    ClarificationCandidateV1,
    ClarificationRequestV1,
    ResourceRefV1,
    ScopeResolutionStatus,
    ScopeResolutionV1,
    ScopeType,
    TopologyRelationRefV1,
    TurnPlanV1,
    CreateOpsRunCommand,
)
from platform_core.identity import uuid7


_RESOURCE_CAPABILITIES = frozenset(
    capability
    for capability in AIOpsCapability
    if capability != AIOpsCapability.KNOWLEDGE_QA
)


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _json(model) -> dict[str, Any]:
    return model.model_dump(mode="json")


class CapabilityAdapter(Protocol):
    async def execute(
        self,
        *,
        capability: AIOpsCapability,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> CapabilityResult: ...


class ResourceCatalogAdapter:
    """只投影已授权并冻结的资源图，不执行自由问数。"""

    async def execute(
        self,
        *,
        capability: AIOpsCapability,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> CapabilityResult:
        del plan
        lines = ["### 业务系统资源"]
        details = {
            item["resource_id"]: item
            for item in internal_scope.get("resource_catalog", [])
        }
        by_system: dict[UUID, list[ResourceRefV1]] = {}
        for resource in scope.resources:
            by_system.setdefault(resource.business_system_id, []).append(resource)
        for system in scope.business_systems:
            environment = f"（{system.environment}）" if system.environment else ""
            lines.append(f"- {system.display_name}{environment}")
            for resource in by_system.get(system.business_system_id, []):
                item = details.get(str(resource.resource_id), {})
                facts = []
                if item.get("resource_code"):
                    facts.append(f"编码 {item['resource_code']}")
                if item.get("location_name"):
                    facts.append(f"位置 {item['location_name']}")
                facts.extend(
                    f"{key}={value}"
                    for key, value in dict(item.get("attributes") or {}).items()
                )
                if resource.topology_roles:
                    facts.append(f"角色 {', '.join(resource.topology_roles)}")
                suffix = f"；{'；'.join(facts)}" if facts else ""
                lines.append(
                    f"  - {resource.display_name} [{resource.resource_type_key}]{suffix}"
                )
        if len(lines) == 1:
            lines.append("- 当前授权范围内没有可展示的有效资源。")
        return CapabilityResult(capability, True, "\n".join(lines))


class UnsupportedCapabilityAdapter:
    async def execute(
        self,
        *,
        capability: AIOpsCapability,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> CapabilityResult:
        del plan, scope, internal_scope
        return CapabilityResult(
            capability,
            False,
            f"- {capability.value} 专业能力尚未接入统一 Turn Router。",
            f"{capability.value}_ADAPTER_UNAVAILABLE",
        )


class TurnPlannerService:
    """先规划并解析范围，再按能力委派；begin 本身绝不创建 Run。"""

    def __init__(
        self,
        *,
        uow_factory,
        structured_model=None,
        prompt_registry=None,
        model_resolver: Callable[[UUID], Awaitable[dict[str, str]]] | None = None,
        runtime_service=None,
        capability_adapters: dict[AIOpsCapability, CapabilityAdapter] | None = None,
    ):
        self._uow_factory = uow_factory
        self._structured_model = structured_model
        self._prompt_registry = prompt_registry
        self._model_resolver = model_resolver
        self._runtime_service = runtime_service
        self._capability_adapters = {
            AIOpsCapability.RESOURCE_CATALOG_QUERY: ResourceCatalogAdapter(),
            **dict(capability_adapters or {}),
        }
        self._unsupported_adapter = UnsupportedCapabilityAdapter()

    async def begin(self, payload: dict) -> dict:
        context = await self._load_context(payload)
        if context["status"] != "ACCEPTED":
            return self._result(context["turn_id"], context["status"])
        plan, receipt = await self._plan(context)
        scope, internal_scope, clarification = self._resolve_scope(context, plan)
        internal_scope = self._attach_runtime_context(
            internal_scope, context=context, payload=payload
        )
        return await self._persist_route(
            payload, context, plan, receipt, scope, internal_scope, clarification
        )

    async def resume(self, payload: dict) -> dict:
        """用冻结候选恢复同一 Turn；自由文字重新进入结构化规划。"""
        response = dict(payload.get("response") or {})
        if not response.get("candidate_id"):
            return await self.begin(payload)
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id, turn_id=turn_id, lock=True
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.status != "ACCEPTED":
                return self._result(turn_id, turn.status)
            internal = dict(turn.orchestration_internal_json or {})
            candidate = dict(internal.get("clarification_candidates", {})).get(
                str(response["candidate_id"])
            )
            if candidate is None:
                raise validation_failed("澄清候选不存在或已经失效")
            plan = TurnPlanV1.model_validate(turn.capability_plan_json)
            scope = ScopeResolutionV1.model_validate(candidate["scope"])
            selected = dict(candidate["internal_scope"])
            turn.scope_resolution_json = _json(scope)
            turn.clarification_json = None
            turn.orchestration_internal_json = {
                **internal,
                "selected_scope": selected,
                "clarification_response": {
                    "clarification_id": response.get("clarification_id"),
                    "candidate_id": str(response["candidate_id"]),
                    "confirmed_at": datetime.now(UTC).isoformat(),
                },
            }
            result = await self._route_resolved(uow, turn, plan, scope, selected)
            await uow.commit()
            return result

    async def start_active_diagnosis(self, payload: dict) -> dict:
        """单资源沿用 Primary Run，多资源启动有界的独立诊断子 Run。"""
        multi_context = await self._prepare_multi_resource_diagnosis(payload)
        if multi_context is not None:
            return await self._launch_multi_resource_diagnosis(
                payload, multi_context
            )
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id, turn_id=turn_id, lock=True
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            existing = await uow.turns.get_run_link(
                turn_id=turn_id, purpose="PRIMARY"
            )
            if existing is not None:
                return {
                    **payload,
                    "ops_run_id": str(existing.ops_run_id),
                    "target_id": str(turn.focused_target_id),
                    "status": turn.status,
                }
            if turn.status != "UNDERSTANDING":
                raise state_conflict("只有已完成统一路由的 Turn 可以启动诊断")
            plan = TurnPlanV1.model_validate(turn.capability_plan_json)
            if AIOpsCapability.ACTIVE_DIAGNOSIS not in plan.requested_capabilities:
                raise state_conflict("当前 Turn 未规划主动诊断能力")
            internal = dict(turn.orchestration_internal_json or {}).get(
                "selected_scope", {}
            )
            targets = [
                item
                for item in internal.get("execution_targets", [])
                if item.get("readiness") == "READY"
            ]
            if len(targets) != 1:
                raise state_conflict("现有诊断内核只接受一个已解析执行资源")
            target_id = UUID(str(targets[0]["target_id"]))
            facts = await uow.targets.list_target_facts(
                target_id=target_id, domain_id=domain_id, active_only=True
            )
            messages = await uow.turns.list_messages(turn_id=turn_id)
            question = next(
                str(row.payload_json.get("text") or "")
                for row in messages
                if row.message_type == "USER_MESSAGE"
            )
            execution = dict(payload.get("execution_context") or {})
            now = datetime.now(UTC)
            run_id = uuid7()
            await uow.runs.add_run(
                OpsRunEntity(
                    ops_run_id=run_id,
                    domain_id=domain_id,
                    target_id=target_id,
                    agent_id=UUID(str(payload["agent_id"])),
                    agent_version_id=UUID(str(payload["agent_version_id"])),
                    trigger_type=str(execution.get("trigger_type") or "CHAT"),
                    interaction_mode=str(
                        execution.get("interaction_mode") or "INTERACTIVE"
                    ),
                    workflow_kind=str(
                        execution.get("workflow_kind") or "CHAT_TURN"
                    ),
                    trigger_signal_event_id=self._optional_uuid(
                        execution.get("trigger_signal_event_id")
                    ),
                    situation_id=self._optional_uuid(
                        payload.get("source_situation_id")
                    ),
                    inspection_fire_id=self._optional_uuid(
                        execution.get("inspection_fire_id")
                    ),
                    actor_id=turn.created_by,
                    original_request=question,
                    idempotency_key=f"turn:{turn_id}:primary",
                    status="RUNNING",
                    plan_snapshot_json={
                        "schema_version": "aiops.turn-planning-context.v2",
                        "conversation_id": str(turn.conversation_id),
                        "turn_id": str(turn_id),
                        "capability_plan": dict(turn.capability_plan_json or {}),
                        "scope_resolution": dict(turn.scope_resolution_json or {}),
                        "source_situation_id": payload.get("source_situation_id"),
                        "source_run_id": payload.get("source_run_id"),
                        "observation_start": execution.get("observation_start"),
                        "observation_end": execution.get("observation_end"),
                        "client_metadata": {
                            "inspection": dict(execution.get("inspection") or {}),
                            "conversation_starter": dict(
                                execution.get("conversation_starter") or {}
                            ),
                        },
                        "target_facts": [
                            {
                                "fact_id": str(row.target_fact_id),
                                "fact_type": row.fact_type,
                                "fact_key": row.fact_key,
                                "fact_value": dict(row.fact_value or {}),
                                "source": row.source,
                                "status": row.status,
                            }
                            for row in facts
                        ],
                    },
                    policy_snapshot_json={
                        "agent_version_id": str(payload["agent_version_id"]),
                        "turn_policy_snapshot_hash": plan.policy_snapshot_hash,
                    },
                    trace_id=str(payload["trace_id"]),
                    deadline_at=(
                        datetime.fromisoformat(str(execution["deadline_at"]))
                        if execution.get("deadline_at")
                        else None
                    ),
                    started_at=now,
                )
            )
            await uow.turns.add_run(
                OpsTurnRunEntity(
                    turn_run_id=uuid7(),
                    turn_id=turn_id,
                    ops_run_id=run_id,
                    purpose="PRIMARY",
                    sequence_no=1,
                )
            )
            turn.focused_target_id = target_id
            await self._append_event(
                uow,
                turn,
                "diagnosis.delegated",
                {
                    "public_summary": "已解析业务资源，正在启动主动诊断",
                    "resource_id": str(targets[0]["resource_id"]),
                },
            )
            await uow.commit()
            return {
                **payload,
                "ops_run_id": str(run_id),
                "target_id": str(target_id),
                "status": turn.status,
            }

    async def _prepare_multi_resource_diagnosis(
        self, payload: dict
    ) -> dict[str, Any] | None:
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id, turn_id=turn_id, lock=True
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            internal = dict(turn.orchestration_internal_json or {})
            selected_scope = dict(internal.get("selected_scope") or {})
            targets = [
                dict(item)
                for item in selected_scope.get("execution_targets", ())
                if item.get("readiness") == "READY"
            ]
            if len(targets) <= 1:
                return None
            if turn.status in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}:
                return {"terminal_status": turn.status}
            if turn.status not in {"UNDERSTANDING", "COLLECTING"}:
                raise state_conflict("只有已完成统一路由的 Turn 可以启动诊断")
            if self._runtime_service is None:
                raise dependency_unavailable("多资源诊断运行服务尚未配置")
            plan = TurnPlanV1.model_validate(turn.capability_plan_json)
            scope = ScopeResolutionV1.model_validate(turn.scope_resolution_json)
            resources = {
                str(item.resource_id): item for item in scope.resources
            }
            budget = int(plan.budget.max_diagnosis_runs)
            selected = targets[:budget]
            omitted = targets[budget:]
            coordination = dict(internal.get("multi_resource_diagnosis") or {})
            if not coordination:
                resource_entries = []
                for sequence, target in enumerate(selected, start=1):
                    public = resources.get(str(target["resource_id"]))
                    resource_entries.append(
                        {
                            "sequence_no": sequence,
                            "resource_id": str(target["resource_id"]),
                            "target_id": str(target["target_id"]),
                            "display_name": (
                                public.display_name if public is not None
                                else f"资源 {sequence}"
                            ),
                            "resource_type_key": (
                                public.resource_type_key if public is not None
                                else "UNKNOWN"
                            ),
                            "topology_roles": list(
                                public.topology_roles if public is not None else ()
                            ),
                        }
                    )
                coordination = {
                    "schema_version": "AIOPS_MULTI_RESOURCE_DIAGNOSIS.v1",
                    "launch_complete": False,
                    "budget": budget,
                    "resources": resource_entries,
                    "omitted_resource_ids": [
                        str(item["resource_id"]) for item in omitted
                    ],
                    "launch_failures": [],
                }
                internal["multi_resource_diagnosis"] = coordination
                turn.orchestration_internal_json = internal
                turn.status = "COLLECTING"
                turn.focused_target_id = None
                await self._append_event(
                    uow,
                    turn,
                    "diagnosis.delegated",
                    {
                        "public_summary": (
                            f"已启动 {len(resource_entries)} 个资源的并行诊断"
                        ),
                        "resource_count": len(resource_entries),
                        "budget_omitted_count": len(omitted),
                    },
                )
                await uow.commit()
            messages = await uow.turns.list_messages(turn_id=turn_id)
            question = next(
                str(row.payload_json.get("text") or "")
                for row in messages
                if row.message_type == "USER_MESSAGE"
            )
            return {
                "domain_id": domain_id,
                "turn_id": str(turn_id),
                "conversation_id": str(turn.conversation_id),
                "actor_id": str(turn.created_by),
                "question": question,
                "resources": list(coordination["resources"]),
            }

    async def _launch_multi_resource_diagnosis(
        self, payload: dict, context: dict[str, Any]
    ) -> dict:
        if context.get("terminal_status"):
            return self._result(payload["turn_id"], context["terminal_status"])
        execution = dict(payload.get("execution_context") or {})
        failures: list[dict[str, str]] = []
        run_ids: list[str] = []
        for resource in context["resources"]:
            try:
                receipt = await self._runtime_service.create_run(
                    CreateOpsRunCommand(
                        command_id=uuid7(),
                        idempotency_key=(
                            f"turn:{context['turn_id']}:resource:"
                            f"{resource['resource_id']}"
                        ),
                        domain_id=context["domain_id"],
                        actor_id=context["actor_id"],
                        agent_id=UUID(str(payload["agent_id"])),
                        expected_agent_version_id=UUID(
                            str(payload["agent_version_id"])
                        ),
                        target_id=UUID(str(resource["target_id"])),
                        trigger_type=str(
                            execution.get("trigger_type") or "CHAT"
                        ),
                        input=context["question"],
                        session_id=context["conversation_id"],
                        trigger_signal_event_id=self._optional_uuid(
                            execution.get("trigger_signal_event_id")
                        ),
                        situation_id=self._optional_uuid(
                            payload.get("source_situation_id")
                        ),
                        inspection_fire_id=self._optional_uuid(
                            execution.get("inspection_fire_id")
                        ),
                        deadline=(
                            datetime.fromisoformat(str(execution["deadline_at"]))
                            if execution.get("deadline_at") else None
                        ),
                        blueprint_id="diagnosis.root-cause",
                        observation_start=(
                            datetime.fromisoformat(
                                str(execution["observation_start"])
                            ) if execution.get("observation_start") else None
                        ),
                        observation_end=(
                            datetime.fromisoformat(
                                str(execution["observation_end"])
                            ) if execution.get("observation_end") else None
                        ),
                        client_metadata={
                            "trace_id": str(payload["trace_id"]),
                            "multi_resource_child": True,
                            "parent_turn_id": context["turn_id"],
                            "resource_sequence": resource["sequence_no"],
                            "resource_id": resource["resource_id"],
                            "resource_display_name": resource["display_name"],
                        },
                    )
                )
                run_ids.append(str(receipt.ops_run_id))
            except AIOpsApplicationError as exc:
                failures.append(
                    {
                        "resource_id": resource["resource_id"],
                        "code": str(
                            getattr(exc, "code", "CHILD_RUN_CREATE_FAILED")
                        ),
                    }
                )
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=context["domain_id"],
                turn_id=UUID(context["turn_id"]),
                lock=True,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            internal = dict(turn.orchestration_internal_json or {})
            coordination = dict(internal.get("multi_resource_diagnosis") or {})
            coordination["launch_complete"] = True
            coordination["launch_failures"] = failures
            coordination["ops_run_ids"] = run_ids
            internal["multi_resource_diagnosis"] = coordination
            turn.orchestration_internal_json = internal
            await uow.commit()
        await self.aggregate_multi_resource_diagnosis(
            {
                "domain_id": context["domain_id"],
                "turn_id": context["turn_id"],
            }
        )
        return {
            **payload,
            "ops_run_ids": run_ids,
            "status": "COLLECTING",
            "multi_resource": True,
        }

    async def aggregate_multi_resource_diagnosis(self, payload: dict) -> dict:
        domain_id = payload.get("domain_id")
        turn_id = payload.get("turn_id")
        if turn_id is None:
            ops_run_id = UUID(str(payload["ops_run_id"]))
            async with self._uow_factory() as lookup:
                link = await lookup.turns.get_run_link_by_ops_run_id(
                    ops_run_id=ops_run_id
                )
                run = await lookup.runs.get_run(ops_run_id=ops_run_id)
                if link is None or run is None or not link.purpose.startswith(
                    "RESOURCE_"
                ):
                    return {"status": "IGNORED"}
                turn_id = link.turn_id
                domain_id = int(run.domain_id)
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=int(domain_id), turn_id=UUID(str(turn_id)), lock=True
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.status in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}:
                return self._result(turn.turn_id, turn.status)
            internal = dict(turn.orchestration_internal_json or {})
            coordination = dict(internal.get("multi_resource_diagnosis") or {})
            if not coordination or not coordination.get("launch_complete"):
                return self._result(turn.turn_id, turn.status)
            links = [
                item for item in await uow.turns.list_run_links(turn_id=turn.turn_id)
                if item.purpose.startswith("RESOURCE_")
            ]
            runs = []
            for link in links:
                run = await uow.runs.get_run(ops_run_id=link.ops_run_id)
                if run is not None:
                    runs.append((link, run))
            terminal = {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED"}
            if any(run.status not in terminal for _, run in runs):
                return self._result(turn.turn_id, turn.status)
            resources = {
                int(item["sequence_no"]): dict(item)
                for item in coordination.get("resources", ())
            }
            sections = ["### 多资源诊断结果", ""]
            evidence: list[dict[str, Any]] = []
            successful = 0
            gaps: list[str] = []
            for link, run in runs:
                resource = resources.get(int(link.sequence_no), {})
                name = str(resource.get("display_name") or f"资源 {link.sequence_no}")
                sections.append(f"#### {name}")
                artifact = (
                    await uow.runs.get_artifact(artifact_id=run.final_artifact_id)
                    if run.final_artifact_id is not None else None
                )
                if run.status in {"COMPLETED", "PARTIAL"} and artifact is not None:
                    successful += 1
                    summary = self._diagnosis_artifact_markdown(artifact)
                    target_id = str(resource.get("target_id") or "")
                    sections.append(summary.replace(target_id, name) if target_id else summary)
                    evidence.append(
                        {
                            "resource_id": resource.get("resource_id"),
                            "resource_name": name,
                            "ops_run_id": str(run.ops_run_id),
                            "artifact_id": str(artifact.artifact_id),
                            "schema_version": artifact.schema_version,
                        }
                    )
                    if run.status == "PARTIAL":
                        gaps.append("MULTI_RESOURCE_CHILD_PARTIAL")
                else:
                    sections.append(f"诊断未完成（{run.status}）。")
                    gaps.append("MULTI_RESOURCE_CHILD_FAILED")
                sections.append("")
            if coordination.get("launch_failures"):
                gaps.append("MULTI_RESOURCE_CHILD_LAUNCH_FAILED")
            omitted = coordination.get("omitted_resource_ids") or []
            if omitted:
                gaps.append("MULTI_RESOURCE_BUDGET_EXCEEDED")
                sections.append(
                    f"> 受本轮预算限制，另有 {len(omitted)} 个资源未执行主动诊断。"
                )
            planned = len(resources)
            failed = planned - successful
            sections.insert(
                1, f"已完成 {successful}/{planned} 个预算内资源的诊断。"
            )
            status = (
                "FAILED" if successful == 0 else
                "PARTIAL" if failed or gaps else "COMPLETED"
            )
            unique_gaps = tuple(dict.fromkeys(gaps))
            result = await self._complete_without_run(
                uow,
                turn,
                "\n".join(sections).strip(),
                status,
                unique_gaps,
                (AIOpsCapability.ACTIVE_DIAGNOSIS.value,),
                tuple(evidence),
            )
            await uow.commit()
            return result

    @staticmethod
    def _diagnosis_artifact_markdown(artifact) -> str:
        payload = dict(artifact.payload_json or {})
        if artifact.schema_version == "AIOPS_TURN_RESULT.v1":
            blocks = payload.get("blocks") or ()
            rendered = [
                str(dict(item).get("markdown") or dict(item).get("text") or "")
                for item in blocks if isinstance(item, dict)
            ]
            if any(rendered):
                return "\n\n".join(item for item in rendered if item)
        lines = []
        for key, label in (
            ("summary", "摘要"),
            ("root_cause", "根因"),
            ("diagnosis_rationale", "诊断依据"),
            ("findings", "发现"),
            ("gaps", "证据缺口"),
        ):
            value = payload.get(key)
            if value not in (None, "", [], {}):
                text = json.dumps(value, ensure_ascii=False, default=str) \
                    if isinstance(value, (dict, list)) else str(value)
                lines.append(f"- **{label}**：{text}")
        return "\n".join(lines) or "诊断已完成，未生成可展示的摘要。"

    async def cancel(self, payload: dict) -> dict:
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=domain_id, turn_id=turn_id, lock=True
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.status in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}:
                return self._result(turn_id, turn.status)
            now = datetime.now(UTC)
            links = await uow.turns.list_run_links(turn_id=turn_id)
            for link in links:
                run = await uow.runs.get_run(ops_run_id=link.ops_run_id, lock=True)
                if run is not None and run.status not in {
                    "COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED"
                }:
                    run.cancel_requested_at = now
                    run.cancel_requested_by = str(payload["requested_by"])
                    run.status = "CANCELLED"
                    run.completed_at = now
            turn.status = "CANCELLED"
            turn.completed_at = now
            await self._append_event(
                uow,
                turn,
                "turn.status",
                {"status": "CANCELLED", "public_summary": "处理已取消"},
            )
            await uow.commit()
            return self._result(turn_id, turn.status)

    async def fail_terminal(
        self, payload: dict, *, error_code: str, error_message: str
    ) -> dict:
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=int(payload["domain_id"]),
                turn_id=UUID(str(payload["turn_id"])),
                lock=True,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.status in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED"}:
                return self._result(turn.turn_id, turn.status)
            turn.status = "FAILED"
            turn.error_domain = "ORCHESTRATION"
            turn.error_code = error_code
            turn.error_message = error_message[:2000]
            turn.completed_at = datetime.now(UTC)
            await self._append_event(
                uow,
                turn,
                "turn.status",
                {
                    "status": "FAILED",
                    "error_code": error_code,
                    "public_summary": "统一助手规划失败",
                },
            )
            await uow.commit()
            return self._result(turn.turn_id, turn.status)

    async def _load_context(self, payload: dict) -> dict[str, Any]:
        domain_id = int(payload["domain_id"])
        turn_id = UUID(str(payload["turn_id"]))
        agent_id = UUID(str(payload["agent_id"]))
        version_id = UUID(str(payload["agent_version_id"]))
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(domain_id=domain_id, turn_id=turn_id)
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.status != "ACCEPTED":
                return {"turn_id": str(turn_id), "status": turn.status}
            conversation = await uow.conversations.get_conversation(
                domain_id=domain_id, conversation_id=turn.conversation_id
            )
            version = await uow.agents.version(
                agent_id=agent_id, agent_version_id=version_id
            )
            if conversation is None or version is None:
                raise resource_not_found("Conversation Agent Version")
            allowed_ids = tuple(
                await uow.agents.version_business_system_ids(
                    agent_version_id=version_id
                )
            )
            systems = []
            for system_id in allowed_ids:
                system = await uow.business_systems.get_scoped(
                    business_system_id=system_id, domain_id=domain_id
                )
                if system is None or system.status != "ENABLED":
                    continue
                rows = await uow.business_systems.list_system_resource_rows(
                    system_id
                )
                relations = await uow.business_systems.list_relations(system_id)
                participants = await uow.business_systems.list_participants(
                    [row.relation_id for row in relations]
                )
                roles: dict[UUID, list[str]] = {}
                for participant in participants:
                    roles.setdefault(participant.system_resource_id, []).append(
                        participant.participant_role
                    )
                resources = []
                for link, resource, database, target in rows:
                    if link.status != "ACTIVE" or resource.lifecycle_status != "ACTIVE":
                        continue
                    resources.append(
                        {
                            "system_resource_id": str(link.system_resource_id),
                            "resource_id": str(resource.resource_id),
                            "resource_code": resource.resource_code,
                            "display_name": resource.display_name,
                            "resource_type_key": resource.resource_type_key,
                            "location_name": resource.location_name,
                            "attributes": self._public_attributes(
                                dict(resource.attributes_json or {})
                            ),
                            "topology_roles": sorted(
                                set(roles.get(link.system_resource_id, []))
                            ),
                            "target_id": (
                                str(database.target_id) if database is not None else None
                            ),
                            "target_ready": bool(
                                database is not None
                                and target is not None
                                and target.status == "ENABLED"
                                and bool(
                                    getattr(target, "readonly_connection_enabled", True)
                                )
                            ),
                        }
                    )
                systems.append(
                    {
                        "business_system_id": str(system.business_system_id),
                        "system_code": system.system_code,
                        "display_name": system.display_name,
                        "environment": system.environment,
                        "resources": resources,
                        "relations": [
                            {
                                "relation_id": str(row.relation_id),
                                "relation_type_key": row.relation_type_key,
                                "display_name": row.display_name,
                            }
                            for row in relations
                            if row.confirmation_status == "CONFIRMED"
                        ],
                    }
                )
            messages = await uow.turns.list_messages(turn_id=turn_id)
            question = "\n".join(
                str(row.payload_json.get("text") or row.payload_json.get("free_text") or "")
                for row in messages
                if row.message_type in {"USER_MESSAGE", "CLARIFICATION_RESPONSE"}
            )
            policy = {
                "agent_id": str(agent_id),
                "agent_version_id": str(version_id),
                "capabilities": list(version.capabilities_json or ()),
                "allowed_business_system_ids": [str(item) for item in allowed_ids],
                "knowledge_policy": dict(version.knowledge_policy_json or {}),
            }
            return {
                "turn_id": str(turn_id),
                "status": turn.status,
                "question": question,
                "capabilities": tuple(version.capabilities_json or ()),
                "planner_model_id": dict(version.models_json or {}).get(
                    "planner_llm"
                ),
                "models": dict(version.models_json or {}),
                "knowledge_policy": dict(version.knowledge_policy_json or {}),
                "agent_instruction": getattr(version, "instruction", None),
                "agent_id": str(agent_id),
                "agent_version_id": str(version_id),
                "domain_id": domain_id,
                "actor_id": turn.created_by,
                "policy_snapshot_hash": _hash(policy),
                "systems": systems,
                "context_hint": dict(conversation.context_hint_json or {}),
                "plan_revision": int(turn.current_plan_revision or 0) + 1,
            }

    async def _plan(self, context: dict[str, Any]):
        if not all(
            (self._structured_model, self._prompt_registry, self._model_resolver)
        ):
            raise dependency_unavailable("统一 Turn Router 的结构化规划器未配置")
        model_id = context.get("planner_model_id")
        if not model_id:
            raise validation_failed("AIOps Agent 必须配置 models.planner_llm")
        prompt = await self._prompt_registry.resolve("turn_router")
        result = await self._structured_model.generate_structured(
            purpose="aiops_turn_router",
            output_model=TurnPlanV1,
            model_snapshot=await self._model_resolver(UUID(str(model_id))),
            prompt_ref={**prompt.ref(), "content": prompt.content},
            input_payload={
                "turn_id": context["turn_id"],
                "revision_no": context["plan_revision"],
                "policy_snapshot_hash": context["policy_snapshot_hash"],
                "question": context["question"],
                "allowed_capabilities": list(context["capabilities"]),
                "trusted_context_hint": context["context_hint"],
                "authorized_business_systems": [
                    {
                        key: item[key]
                        for key in (
                            "business_system_id",
                            "system_code",
                            "display_name",
                            "environment",
                        )
                    }
                    for item in context["systems"]
                ],
            },
            deadline=None,
            idempotency_key=(
                f"turn-router:{context['turn_id']}:{context['plan_revision']}"
            ),
        )
        plan = (
            result.output
            if isinstance(result.output, TurnPlanV1)
            else TurnPlanV1.model_validate(result.output)
        )
        allowed = {AIOpsCapability(item) for item in context["capabilities"]}
        if not set(plan.requested_capabilities) <= allowed:
            raise validation_failed("规划器选择了 Agent 未授权能力")
        if str(plan.turn_id) != context["turn_id"]:
            raise validation_failed("规划器返回了错误的 Turn 身份")
        if plan.revision_no != context["plan_revision"]:
            raise validation_failed("规划器返回了错误的计划版本")
        if plan.policy_snapshot_hash != context["policy_snapshot_hash"]:
            raise validation_failed("规划器返回的策略快照不一致")
        return plan, result.receipt

    def _resolve_scope(self, context: dict[str, Any], plan: TurnPlanV1):
        requested = {
            AIOpsCapability(item) for item in plan.requested_capabilities
        }
        authorization_hash = context["policy_snapshot_hash"]
        if not requested & _RESOURCE_CAPABILITIES:
            return (
                ScopeResolutionV1(
                    status=ScopeResolutionStatus.NOT_REQUIRED,
                    scope_type=ScopeType.NONE,
                    authorization_snapshot_hash=authorization_hash,
                ),
                {"execution_targets": []},
                None,
            )
        systems = list(context["systems"])
        if not systems:
            return (
                ScopeResolutionV1(
                    status=ScopeResolutionStatus.DENIED,
                    scope_type=ScopeType.SYSTEM,
                    resolution_basis=("NO_AUTHORIZED_ACTIVE_SYSTEM",),
                    authorization_snapshot_hash=authorization_hash,
                ),
                {"execution_targets": []},
                None,
            )
        text = context["question"].casefold()
        mentioned = [
            item
            for item in systems
            if any(
                token and token.casefold() in text
                for token in (item["system_code"], item["display_name"])
            )
        ]
        basis = ["EXPLICIT_SYSTEM_MENTION"] if mentioned else []
        if not mentioned:
            hinted = str(
                context.get("context_hint", {}).get("business_system_id") or ""
            )
            mentioned = [
                item for item in systems if item["business_system_id"] == hinted
            ]
            if mentioned:
                basis.append("TRUSTED_CONTEXT_HINT")
        active = AIOpsCapability.ACTIVE_DIAGNOSIS in requested
        if not mentioned and (not active or len(systems) == 1):
            mentioned = systems
            basis.append("AUTHORIZED_SCOPE")
        if active and len(mentioned) != 1:
            return self._clarification_scope(
                context, mentioned or systems, authorization_hash
            )
        scope, internal = self._resolved_scope(
            mentioned or systems,
            text,
            authorization_hash,
            tuple(basis),
            active,
        )
        return scope, internal, None

    def _clarification_scope(
        self,
        context: dict[str, Any],
        systems: list[dict[str, Any]],
        authorization_hash: str,
    ):
        clarification_id = uuid7()
        candidates = []
        candidate_map = {}
        for system in systems[:20]:
            candidate_id = hashlib.sha256(
                f"{clarification_id}:{system['business_system_id']}".encode()
            ).hexdigest()
            scope, internal = self._resolved_scope(
                [system], "", authorization_hash, ("USER_CONFIRMED",), True
            )
            candidates.append(
                ClarificationCandidateV1(
                    candidate_id=candidate_id,
                    label=system["display_name"],
                    description=(
                        f"{system['environment']} · {len(system['resources'])} 个有效资源"
                    ),
                    scope_type=ScopeType.SYSTEM,
                )
            )
            candidate_map[candidate_id] = {
                "scope": _json(scope),
                "internal_scope": internal,
            }
        if len(candidates) < 2:
            raise state_conflict("主动诊断缺少唯一可用业务系统")
        clarification = ClarificationRequestV1(
            clarification_id=clarification_id,
            turn_id=UUID(context["turn_id"]),
            question="请确认本轮需要处理的业务系统。",
            reason_code="BUSINESS_SYSTEM_AMBIGUOUS",
            candidates=tuple(candidates),
            allow_free_text=True,
        )
        scope = ScopeResolutionV1(
            status=ScopeResolutionStatus.NEEDS_CLARIFICATION,
            scope_type=ScopeType.SYSTEM,
            business_systems=tuple(
                BusinessSystemRefV1(
                    business_system_id=UUID(item["business_system_id"]),
                    display_name=item["display_name"],
                    environment=item["environment"],
                )
                for item in systems[:20]
            ),
            resolution_basis=("AMBIGUOUS_AUTHORIZED_SYSTEMS",),
            authorization_snapshot_hash=authorization_hash,
        )
        return scope, {"clarification_candidates": candidate_map}, clarification

    def _resolved_scope(
        self,
        systems: list[dict[str, Any]],
        text: str,
        authorization_hash: str,
        basis: tuple[str, ...],
        active_diagnosis: bool,
    ) -> tuple[ScopeResolutionV1, dict[str, Any]]:
        all_resources = [
            (system, resource)
            for system in systems
            for resource in system["resources"]
        ]
        matched = [
            (system, resource)
            for system, resource in all_resources
            if text
            and any(
                token and str(token).casefold() in text
                for token in (
                    resource["resource_code"],
                    resource["display_name"],
                    *resource["topology_roles"],
                    *(
                        str(value)
                        for value in dict(resource["attributes"]).values()
                    ),
                )
            )
        ]
        selected = matched or all_resources
        if len(matched) == 1:
            scope_type = ScopeType.RESOURCE
            resolution_basis = (*basis, "EXPLICIT_RESOURCE_MENTION")
        elif matched:
            scope_type = ScopeType.RESOURCE_SET
            resolution_basis = (*basis, "EXPLICIT_RESOURCE_SET")
        else:
            scope_type = ScopeType.SYSTEM
            resolution_basis = (*basis, "SYSTEM_TRIAGE")
        execution_targets = [
            {
                "business_system_id": system["business_system_id"],
                "system_resource_id": resource["system_resource_id"],
                "resource_id": resource["resource_id"],
                "target_id": resource["target_id"],
                "readiness": (
                    "READY" if resource["target_ready"] else "UNAVAILABLE"
                ),
            }
            for system, resource in selected
            if resource["target_id"]
        ]
        if active_diagnosis and not any(
            item["readiness"] == "READY" for item in execution_targets
        ):
            return (
                ScopeResolutionV1(
                    status=ScopeResolutionStatus.DENIED,
                    scope_type=scope_type,
                    resolution_basis=("NO_DIAGNOSIS_READY_RESOURCE",),
                    authorization_snapshot_hash=authorization_hash,
                ),
                {"execution_targets": execution_targets},
            )
        public_systems = tuple(
            BusinessSystemRefV1(
                business_system_id=UUID(item["business_system_id"]),
                display_name=item["display_name"],
                environment=item["environment"],
            )
            for item in systems
        )
        public_resources = tuple(
            ResourceRefV1(
                resource_id=UUID(resource["resource_id"]),
                system_resource_id=UUID(resource["system_resource_id"]),
                business_system_id=UUID(system["business_system_id"]),
                display_name=resource["display_name"],
                resource_type_key=resource["resource_type_key"],
                topology_roles=tuple(resource["topology_roles"]),
            )
            for system, resource in selected
        )
        public_relations = tuple(
            TopologyRelationRefV1(
                relation_id=UUID(relation["relation_id"]),
                business_system_id=UUID(system["business_system_id"]),
                relation_type_key=relation["relation_type_key"],
                display_name=relation["display_name"],
            )
            for system in systems
            for relation in system["relations"]
        )
        graph_hash = _hash(
            {
                "systems": [_json(item) for item in public_systems],
                "resources": [_json(item) for item in public_resources],
                "relations": [_json(item) for item in public_relations],
            }
        )
        return (
            ScopeResolutionV1(
                status=ScopeResolutionStatus.RESOLVED,
                scope_type=scope_type,
                business_systems=public_systems,
                resources=public_resources,
                relations=public_relations,
                resolution_basis=resolution_basis,
                confidence=1.0 if matched else 0.8,
                authorization_snapshot_hash=authorization_hash,
                resource_graph_snapshot_hash=graph_hash,
            ),
            {
                "execution_targets": execution_targets,
                "resource_catalog": [
                    {
                        "business_system_id": system["business_system_id"],
                        **resource,
                    }
                    for system, resource in selected
                ],
            },
        )

    async def _persist_route(
        self,
        payload,
        context,
        plan,
        receipt,
        scope,
        internal_scope,
        clarification,
    ) -> dict:
        async with self._uow_factory() as uow:
            turn = await uow.turns.get_turn(
                domain_id=int(payload["domain_id"]),
                turn_id=UUID(context["turn_id"]),
                lock=True,
            )
            if turn is None:
                raise resource_not_found("Conversation Turn")
            if turn.status != "ACCEPTED":
                return self._result(turn.turn_id, turn.status)
            turn.current_plan_revision = plan.revision_no
            turn.capability_plan_json = _json(plan)
            turn.scope_resolution_json = _json(scope)
            turn.clarification_json = _json(clarification) if clarification else None
            turn.orchestration_internal_json = {
                "schema_version": "aiops.orchestration-internal.v1",
                "planner_receipt": receipt.model_dump(mode="json"),
                "selected_scope": internal_scope if clarification is None else None,
                **internal_scope,
            }
            turn.started_at = turn.started_at or datetime.now(UTC)
            await self._append_event(
                uow,
                turn,
                "planning.completed",
                {
                    "requested_capabilities": [
                        str(item) for item in plan.requested_capabilities
                    ],
                    "public_summary": "已完成问题理解、能力规划和授权范围解析",
                },
            )
            if clarification is not None:
                turn.status = "WAITING_USER"
                await self._append_event(
                    uow, turn, "clarification.requested", _json(clarification)
                )
                await self._append_event(
                    uow,
                    turn,
                    "turn.status",
                    {
                        "status": "WAITING_USER",
                        "public_summary": clarification.question,
                    },
                )
                await uow.commit()
                return self._result(turn.turn_id, turn.status)
            result = await self._route_resolved(
                uow, turn, plan, scope, internal_scope
            )
            await uow.commit()
            return result

    async def _route_resolved(
        self,
        uow,
        turn,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> dict:
        if scope.status == ScopeResolutionStatus.DENIED:
            return await self._complete_without_run(
                uow,
                turn,
                "当前 Agent 授权范围内没有满足本轮能力要求的有效资源。",
                "PARTIAL",
                ("NO_ELIGIBLE_RESOURCE",),
            )
        capabilities = tuple(
            AIOpsCapability(item) for item in plan.requested_capabilities
        )
        if AIOpsCapability.ACTIVE_DIAGNOSIS in capabilities:
            ready = [
                item
                for item in internal_scope.get("execution_targets", [])
                if item.get("readiness") == "READY"
            ]
            if ready:
                turn.status = "UNDERSTANDING"
                await self._append_event(
                    uow,
                    turn,
                    "turn.status",
                    {
                        "status": "UNDERSTANDING",
                        "public_summary": "已确定业务资源，准备进入主动诊断",
                    },
                )
                return self._result(
                    turn.turn_id, turn.status, dispatch="ACTIVE_DIAGNOSIS"
                )
            return await self._complete_without_run(
                uow, turn, "当前范围内没有可执行的诊断资源。", "PARTIAL",
                ("NO_DIAGNOSIS_READY_RESOURCE",),
            )
        results = []
        for capability in capabilities:
            adapter = self._capability_adapters.get(
                capability, self._unsupported_adapter
            )
            results.append(
                await adapter.execute(
                    capability=capability,
                    plan=plan,
                    scope=scope,
                    internal_scope=internal_scope,
                )
            )
        unsupported = [item for item in results if item.gap_code]
        evidence = tuple(
            item
            for result in results
            for item in result.evidence
        )
        return await self._complete_without_run(
            uow,
            turn,
            "\n\n".join(item.markdown for item in results),
            "PARTIAL" if unsupported else "COMPLETED",
            tuple(item.gap_code for item in unsupported if item.gap_code),
            tuple(
                item.capability.value
                for item in results
                if item.supported or item.evidence
            ),
            evidence,
        )

    async def _complete_without_run(
        self,
        uow,
        turn,
        markdown: str,
        status: str,
        gaps: tuple[str, ...],
        capabilities_used: tuple[str, ...] = (),
        evidence: tuple[dict[str, Any], ...] = (),
    ) -> dict:
        conversation = await uow.conversations.get_conversation(
            domain_id=int(turn.domain_id),
            conversation_id=turn.conversation_id,
            lock=True,
        )
        if conversation is None:
            raise resource_not_found("Conversation")
        conversation.last_message_no = int(conversation.last_message_no) + 1
        conversation.updated_by = turn.created_by
        conversation.updated_at = datetime.now(UTC)
        message_id = uuid7()
        await uow.turns.add_message(
            OpsConversationMessageEntity(
                message_id=message_id,
                conversation_id=turn.conversation_id,
                turn_id=turn.turn_id,
                sequence_no=conversation.last_message_no,
                role="AGENT",
                message_type="ASSISTANT_MESSAGE",
                payload_schema="AIOPS_ASSISTANT_MESSAGE.v2",
                payload_json={
                    "text": markdown,
                    "capabilities_used": list(capabilities_used),
                    "gaps": list(gaps),
                    "citations": list(evidence),
                },
                created_by="aiops.turn-router",
            )
        )
        await uow.turns.add_answer_block(
            OpsAnswerBlockEntity(
                answer_block_id=uuid7(),
                turn_id=turn.turn_id,
                message_id=message_id,
                block_no=1,
                block_type="MARKDOWN",
                schema_version="AIOPS_MARKDOWN_BLOCK.v1",
                payload_json={
                    "markdown": markdown,
                    "citations": list(evidence),
                },
                content_hash=hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
            )
        )
        turn.status = status
        turn.sufficiency_status = (
            "ANSWERABLE" if status == "COMPLETED" else "PARTIAL"
        )
        turn.sufficiency_json = {
            "gaps": [
                {
                    "source_id": "turn-router",
                    "code": code,
                    "detail": "专业能力尚未接入或当前资源范围不可执行",
                    "retryable": False,
                }
                for code in gaps
            ]
        }
        turn.completed_at = datetime.now(UTC)
        await self._append_event(
            uow, turn, "answer.delta", {"delta": markdown}
        )
        await self._append_event(
            uow,
            turn,
            "answer.completed",
            {"answer_block_count": 1, "status": status},
        )
        await self._append_event(
            uow, turn, "turn.status", {"status": status}
        )
        return self._result(turn.turn_id, status)

    @staticmethod
    def _attach_runtime_context(
        internal_scope: dict[str, Any], *, context: dict[str, Any], payload: dict
    ) -> dict[str, Any]:
        runtime_context = {
            "turn_id": context["turn_id"],
            "domain_id": int(context.get("domain_id") or payload["domain_id"]),
            "agent_id": str(context.get("agent_id") or payload["agent_id"]),
            "agent_version_id": str(
                context.get("agent_version_id") or payload["agent_version_id"]
            ),
            "actor_id": str(
                context.get("actor_id")
                or payload.get("actor_id")
                or payload.get("requested_by")
                or "aiops.turn-router"
            ),
            "question": context["question"],
            "models": dict(context.get("models") or {}),
            "knowledge_policy": dict(context.get("knowledge_policy") or {}),
            "agent_instruction": context.get("agent_instruction"),
            "caller_service": str(payload.get("caller_service") or "aiops-worker"),
            "request_id": str(payload.get("request_id") or context["turn_id"]),
            "trace_id": str(payload.get("trace_id") or context["turn_id"]),
        }
        result = {**internal_scope, "runtime_context": runtime_context}
        candidates = dict(result.get("clarification_candidates") or {})
        if candidates:
            result["clarification_candidates"] = {
                key: {
                    **value,
                    "internal_scope": {
                        **dict(value["internal_scope"]),
                        "runtime_context": runtime_context,
                    },
                }
                for key, value in candidates.items()
            }
        return result

    @staticmethod
    def _public_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
        denied = ("password", "secret", "token", "credential", "dsn", "connect")
        return {
            str(key): value
            for key, value in attributes.items()
            if not any(marker in str(key).casefold() for marker in denied)
            and isinstance(value, (str, int, float, bool, type(None)))
        }

    @staticmethod
    def _optional_uuid(value) -> UUID | None:
        return UUID(str(value)) if value else None

    @staticmethod
    def _result(turn_id, status, *, dispatch=None) -> dict[str, Any]:
        return {
            "turn_id": str(turn_id),
            "status": status,
            "dispatch": dispatch,
        }

    @staticmethod
    async def _append_event(uow, turn, event_type: str, payload: dict) -> None:
        turn.event_cursor = int(turn.event_cursor) + 1
        await uow.turns.add_event(
            OpsTurnEventEntity(
                turn_id=turn.turn_id,
                sequence_no=turn.event_cursor,
                event_type=event_type,
                event_key=f"{event_type}:{turn.turn_id}:{turn.event_cursor}",
                visibility="USER",
                payload_json=payload,
            )
        )
