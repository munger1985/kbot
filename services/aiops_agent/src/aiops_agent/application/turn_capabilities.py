"""统一 Turn Router 的只读专业能力适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.contracts.knowledge import KnowledgeSearchRequest
from platform_core.contracts.aiops import (
    AIOpsCapability,
    AgentKnowledgePolicy,
    ScopeResolutionV1,
    TurnPlanV1,
)
from platform_core.security import create_service_auth_context


@dataclass(frozen=True)
class CapabilityResult:
    capability: AIOpsCapability
    supported: bool
    markdown: str
    gap_code: str | None = None
    evidence: tuple[dict[str, Any], ...] = ()


class KnowledgeAnswerDraft(BaseModel):
    """知识回答只允许引用本次检索提供的稳定标签。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_markdown: str = Field(min_length=1, max_length=16000)
    citation_labels: tuple[str, ...] = Field(max_length=20)


def _runtime(internal_scope: dict[str, Any]) -> dict[str, Any]:
    return dict(internal_scope.get("runtime_context") or {})


def _scope(internal_scope: dict[str, Any]) -> ConfigurationScope:
    runtime = _runtime(internal_scope)
    return ConfigurationScope(
        domain_id=int(runtime["domain_id"]),
        principal_id=f"SERVICE:{runtime['caller_service']}",
        actor_id=str(runtime["actor_id"]),
        request_id=str(runtime["request_id"]),
        trace_id=str(runtime["trace_id"]),
    )


def _target_ids(internal_scope: dict[str, Any]) -> tuple[UUID, ...]:
    return tuple(
        dict.fromkeys(
            UUID(str(item["target_id"]))
            for item in internal_scope.get("execution_targets", ())
            if item.get("target_id")
        )
    )


def _knowledge_auth(runtime: dict[str, Any]):
    return create_service_auth_context(
        caller_service=str(runtime["caller_service"]),
        request_id=str(runtime["request_id"]),
        trace_id=str(runtime["trace_id"]),
    )


def _step_parameters(
    plan: TurnPlanV1, capability: AIOpsCapability
) -> dict[str, Any]:
    step = next(
        (
            item
            for item in getattr(plan, "steps", ())
            if item.capability == capability
        ),
        None,
    )
    return dict(step.parameters) if step is not None else {}


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def _bounded_text(value: Any, limit: int = 600) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[:limit]}…"


class KnowledgeQuestionAdapter:
    """检索已发布运维知识，并由当前 Agent 版本的模型组织有引回答。"""

    def __init__(
        self,
        *,
        knowledge_service,
        structured_model,
        prompt_registry,
        model_resolver: Callable[[UUID], Awaitable[dict[str, str]]],
        caller_service: str,
    ) -> None:
        self._knowledge = knowledge_service
        self._model = structured_model
        self._prompts = prompt_registry
        self._model_resolver = model_resolver
        self._caller = caller_service

    async def execute(
        self,
        *,
        capability: AIOpsCapability,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> CapabilityResult:
        del scope
        runtime = _runtime(internal_scope)
        parameters = _step_parameters(plan, capability)
        catalog_mode = parameters.get("query_mode") == "CATALOG"
        policy = AgentKnowledgePolicy.model_validate(
            runtime.get("knowledge_policy") or {}
        )
        if not policy.enabled:
            return CapabilityResult(
                capability,
                False,
                "当前 Agent 版本未启用运维知识问答。",
                "KNOWLEDGE_POLICY_DISABLED",
            )
        if catalog_mode and "MANUAL" not in policy.source_kinds:
            return CapabilityResult(
                capability,
                False,
                "当前 Agent 版本没有授权查询已发布运维手册。",
                "KNOWLEDGE_MANUAL_POLICY_DISABLED",
            )
        model_id = dict(runtime.get("models") or {}).get("composer_llm")
        if not catalog_mode and not model_id:
            return CapabilityResult(
                capability,
                False,
                "当前 Agent 版本未配置回答组织模型。",
                "KNOWLEDGE_COMPOSER_MODEL_REQUIRED",
            )
        auth = _knowledge_auth({**runtime, "caller_service": self._caller})
        targets = _target_ids(internal_scope)
        try:
            result = await self._knowledge.search(
                domain_id=int(runtime["domain_id"]),
                agent_id=str(runtime["agent_id"]),
                context=auth,
                target_id=targets[0] if len(targets) == 1 else None,
                request=KnowledgeSearchRequest(
                    query=str(runtime["question"])[:4000],
                    purpose="CATALOG" if catalog_mode else "EXPLAIN",
                    knowledge_base_ids=policy.knowledge_base_ids,
                    tag_ids=tuple(str(item) for item in policy.tag_ids),
                    source_kinds=("MANUAL",) if catalog_mode else policy.source_kinds,
                    max_results=policy.maximum_references,
                    max_security_level=policy.max_security_level,
                ),
            )
        except Exception:
            return CapabilityResult(
                capability,
                False,
                "运维知识检索暂时不可用，未生成无依据回答。",
                "OPERATIONS_KNOWLEDGE_UNAVAILABLE",
            )
        if catalog_mode:
            evidence = []
            lines = ["### 已发布运维知识"]
            for item in result.get("results", ()):
                topics = tuple(item.get("topics") or ())
                topic_text = "、".join(str(topic) for topic in topics)
                summary = (
                    f"可继续提问：{topic_text}"
                    if topic_text
                    else "可按资料标题继续询问适用范围和操作主题"
                )
                evidence.append(
                    {
                        "label": f"K{len(evidence) + 1}",
                        "source_kind": "KNOWLEDGE",
                        "source_id": str(item["asset_version_id"]),
                        "title": str(item["title"]),
                        "quote": summary,
                        "locator": {
                            "asset_id": str(item["asset_id"]),
                            "asset_version_id": str(item["asset_version_id"]),
                        },
                    }
                )
                lines.append(
                    f"- {item['title']}（版本 {item['version']}）："
                    f"{summary} [K{len(evidence)}]"
                )
            if not evidence:
                lines.append("- 当前 Agent 授权范围内没有已发布运维知识。")
            return CapabilityResult(
                capability=capability,
                supported=True,
                markdown="\n".join(lines),
                evidence=tuple(evidence),
            )
        raw_citations = [
            citation
            for item in result.get("results", ())
            for citation in item.get("citation_pack", ())
        ]
        if not raw_citations:
            return CapabilityResult(
                capability,
                False,
                "没有检索到可以直接支持回答的已发布运维知识。",
                "KNOWLEDGE_EVIDENCE_NOT_FOUND",
            )
        citations = tuple(
            {
                "label": f"K{index}",
                "source_kind": "KNOWLEDGE",
                "title": next(
                    (
                        str(item.get("title") or "运维知识")
                        for item in result.get("results", ())
                        if citation in item.get("citation_pack", ())
                    ),
                    "运维知识",
                ),
                "asset_version_id": next(
                    (
                        str(item.get("asset_version_id") or "")
                        for item in result.get("results", ())
                        if citation in item.get("citation_pack", ())
                    ),
                    "",
                ),
                "quote": _bounded_text(
                    citation.get("quote")
                    or citation.get("content")
                    or citation.get("text")
                ),
                "locator": dict(citation.get("locator") or {}),
                "bundle_revision_id": citation.get("bundle_revision_id"),
                "document_version_id": citation.get("document_version_id"),
            }
            for index, citation in enumerate(raw_citations, start=1)
        )
        try:
            prompt = await self._prompts.resolve("knowledge_answer_compose")
            generated = await self._model.generate_structured(
                purpose="aiops_knowledge_answer",
                output_model=KnowledgeAnswerDraft,
                model_snapshot=await self._model_resolver(UUID(str(model_id))),
                prompt_ref={**prompt.ref(), "content": prompt.content},
                input_payload={
                    "question": runtime["question"],
                    "agent_instruction": runtime.get("agent_instruction"),
                    "knowledge_policy": policy.model_dump(mode="json"),
                    "evidence": list(citations),
                },
                deadline=None,
                idempotency_key=f"knowledge-answer:{runtime['turn_id']}",
            )
            draft = KnowledgeAnswerDraft.model_validate(generated.output)
        except Exception:
            return CapabilityResult(
                capability,
                False,
                "已取得运维知识证据，但回答组织模型暂时不可用。",
                "KNOWLEDGE_COMPOSITION_UNAVAILABLE",
                citations,
            )
        available = {item["label"] for item in citations}
        selected = tuple(dict.fromkeys(draft.citation_labels))
        if not selected or not set(selected) <= available:
            return CapabilityResult(
                capability,
                False,
                "回答组织结果包含无效引用，已阻止输出未经证据支持的内容。",
                "KNOWLEDGE_CITATION_INVALID",
                citations,
            )
        selected_evidence = tuple(
            item for item in citations if item["label"] in selected
        )
        sources = "\n".join(
            f"- [{item['label']}] {item['title']}"
            + (f"：{item['quote']}" if item["quote"] else "")
            for item in selected_evidence
        )
        return CapabilityResult(
            capability,
            True,
            f"{draft.answer_markdown}\n\n### 引用\n{sources}",
            evidence=selected_evidence,
        )


class RuntimeQueryAdapter:
    """把告警、巡检与历史案例查询限制在已解析 Target 集合内。"""

    def __init__(
        self, *, runtime_service, knowledge_service=None, caller_service=None
    ) -> None:
        self._runtime = runtime_service
        self._knowledge = knowledge_service
        self._caller = caller_service

    async def execute(
        self,
        *,
        capability: AIOpsCapability,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> CapabilityResult:
        del scope
        targets = _target_ids(internal_scope)
        if not targets:
            return CapabilityResult(
                capability,
                False,
                "当前授权业务范围没有可用于该查询的数据库资源。",
                f"{capability.value}_TARGET_SCOPE_EMPTY",
            )
        query_scope = _scope(internal_scope)
        parameters = _step_parameters(plan, capability)
        try:
            if capability == AIOpsCapability.ALERT_QUERY:
                return await self._alerts(query_scope, targets, parameters)
            if capability == AIOpsCapability.INSPECTION_QUERY:
                return await self._inspections(query_scope, targets, parameters)
            if capability == AIOpsCapability.REPORT_CASE_QUERY:
                return await self._cases(
                    query_scope, targets, parameters, internal_scope
                )
        except Exception:
            return CapabilityResult(
                capability,
                False,
                "固定业务视图暂时不可用。",
                f"{capability.value}_VIEW_UNAVAILABLE",
            )
        return CapabilityResult(
            capability,
            False,
            "该固定业务视图尚未接入。",
            f"{capability.value}_VIEW_UNAVAILABLE",
        )

    async def _alerts(self, scope, targets, parameters) -> CapabilityResult:
        rows = []
        for target_id in targets:
            page = await self._runtime.list_situations(
                scope=scope,
                target_id=target_id,
                agent_id=None,
                status=parameters.get("status"),
                severity=parameters.get("severity"),
                cursor=None,
                limit=min(int(parameters.get("limit") or 20), 50),
            )
            rows.extend(page.items)
        rows.sort(key=lambda item: item.last_observed_at, reverse=True)
        evidence = tuple(
            {
                "source_kind": "ALERT",
                "source_id": str(item.situation_id),
                "target_id": str(item.target_id),
                "snapshot": _model_dump(item),
            }
            for item in rows[:50]
        )
        lines = ["### 告警情况"]
        lines.extend(
            f"- [{item.severity}] {item.title}（{item.status}，最近 {item.last_observed_at.isoformat()}）"
            for item in rows[:50]
        )
        if not rows:
            lines.append("- 当前范围内没有符合条件的告警情境。")
        return CapabilityResult(
            AIOpsCapability.ALERT_QUERY, True, "\n".join(lines), evidence=evidence
        )

    async def _inspections(self, scope, targets, parameters) -> CapabilityResult:
        page = await self._runtime.list_inspection_fires(
            scope=scope,
            plan_id=(
                UUID(str(parameters["plan_id"]))
                if parameters.get("plan_id")
                else None
            ),
            status=parameters.get("status"),
            cursor=None,
            limit=min(int(parameters.get("limit") or 20), 50),
        )
        allowed = set(targets)
        matched = []
        evidence = []
        for item in page.items:
            detail = await self._runtime.get_inspection_fire(
                inspection_fire_id=item.fire_id,
                domain_id=scope.domain_id,
            )
            scoped_runs = tuple(
                run for run in detail.runs if run.target_id in allowed
            )
            if not scoped_runs:
                continue
            matched.append((item, scoped_runs))
            evidence.append(
                {
                    "source_kind": "INSPECTION",
                    "source_id": str(item.fire_id),
                    "target_ids": [str(run.target_id) for run in scoped_runs],
                    "snapshot": {
                        **_model_dump(item),
                        "runs": [_model_dump(run) for run in scoped_runs],
                    },
                }
            )
        lines = ["### 巡检记录"]
        lines.extend(
            f"- {item.plan_display_name}（{item.status}，计划时间 {item.scheduled_at.isoformat()}，范围内 {len(runs)} 个执行结果）"
            for item, runs in matched
        )
        if not matched:
            lines.append("- 当前范围内没有符合条件的巡检记录。")
        return CapabilityResult(
            AIOpsCapability.INSPECTION_QUERY,
            True,
            "\n".join(lines),
            evidence=tuple(evidence),
        )

    async def _cases(
        self, scope, targets, parameters, internal_scope
    ) -> CapabilityResult:
        catalog_mode = parameters.get("query_mode") == "CATALOG"
        lines = ["### 已发布历史诊断案例"]
        case_gap = None
        case_count = 0
        runtime = _runtime(internal_scope)
        policy = AgentKnowledgePolicy.model_validate(
            runtime.get("knowledge_policy") or {}
        )
        case_evidence = []
        if (
            self._knowledge is not None
            and self._caller
            and policy.enabled
            and "DIAGNOSIS_CASE" in policy.source_kinds
        ):
            try:
                seen_cases: set[str] = set()
                for target_id in targets:
                    result = await self._knowledge.search(
                        domain_id=scope.domain_id,
                        agent_id=str(runtime["agent_id"]),
                        context=_knowledge_auth(
                            {**runtime, "caller_service": self._caller}
                        ),
                        target_id=target_id,
                        request=KnowledgeSearchRequest(
                            query=str(runtime["question"])[:4000],
                            purpose="CATALOG" if catalog_mode else "EXPLAIN",
                            source_kinds=("DIAGNOSIS_CASE",),
                            knowledge_base_ids=policy.knowledge_base_ids,
                            tag_ids=tuple(str(item) for item in policy.tag_ids),
                            max_results=policy.maximum_references,
                            max_security_level=policy.max_security_level,
                        ),
                    )
                    for item in result.get("results", ()):
                        asset_version_id = str(item.get("asset_version_id") or "")
                        if not asset_version_id or asset_version_id in seen_cases:
                            continue
                        seen_cases.add(asset_version_id)
                        case_count += 1
                        citations = tuple(item.get("citation_pack", ()))
                        topics = tuple(item.get("topics") or ())
                        topic_text = "、".join(str(topic) for topic in topics)
                        summary = (
                            f"可继续提问：{topic_text}"
                            if topic_text
                            else "可按案例标题继续询问问题特征、根因和验证结果"
                        )
                        lines.append(
                            f"- 历史案例：{item.get('title') or '未命名案例'}"
                            f"（{item.get('authority') or item.get('applicability') or 'UNKNOWN'}）"
                            f"{f'：{summary}' if catalog_mode else ''}"
                        )
                        case_evidence.append(
                            {
                                "source_kind": "DIAGNOSIS_CASE",
                                "source_id": asset_version_id,
                                "target_id": str(target_id),
                                "title": str(item.get("title") or "未命名案例"),
                                "summary": summary if catalog_mode else None,
                                "citations": [
                                    {
                                        "quote": _bounded_text(
                                            citation.get("quote")
                                            or citation.get("content")
                                            or citation.get("text")
                                        ),
                                        "locator": dict(
                                            citation.get("locator") or {}
                                        ),
                                        "bundle_revision_id": citation.get(
                                            "bundle_revision_id"
                                        ),
                                    }
                                    for citation in citations
                                ],
                            }
                        )
            except Exception:
                case_gap = "DIAGNOSIS_CASE_VIEW_UNAVAILABLE"
        elif "DIAGNOSIS_CASE" not in policy.source_kinds:
            case_gap = "DIAGNOSIS_CASE_POLICY_DISABLED"
        if not case_count:
            lines.append("- 当前范围内没有符合条件的已发布历史诊断案例。")
        return CapabilityResult(
            AIOpsCapability.REPORT_CASE_QUERY,
            case_gap is None,
            "\n".join(lines),
            case_gap,
            evidence=tuple(case_evidence),
        )


class MonitoringQueryAdapter:
    """只调用冻结 Monitoring Profile/View，不接受自由 PromQL。"""

    def __init__(self, *, monitoring_service) -> None:
        self._monitoring = monitoring_service

    async def execute(
        self,
        *,
        capability: AIOpsCapability,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> CapabilityResult:
        del scope
        targets = _target_ids(internal_scope)
        if not targets:
            return CapabilityResult(
                capability,
                False,
                "当前授权业务范围没有可展示固定监控视图的数据库资源。",
                "MONITORING_TARGET_SCOPE_EMPTY",
            )
        query_scope = _scope(internal_scope)
        parameters = _step_parameters(plan, capability)
        window = str(parameters.get("window") or "1h")
        if window not in {"15m", "1h", "6h", "24h"}:
            window = "1h"
        evidence = []
        lines = ["### 当前监控情况"]
        gaps = []
        try:
            sources = await self._monitoring.list_sources(scope=query_scope)
            for target_id in targets:
                profiles = await self._monitoring.list_target_profiles(
                    scope=query_scope, target_id=target_id
                )
                profile = next(
                    (item for item in profiles if item.is_default),
                    profiles[0] if profiles else None,
                )
                source = None
                for candidate in sources:
                    instances = await self._monitoring.list_instances(
                        scope=query_scope, source_id=candidate.source_id
                    )
                    if any(item.instance_id == target_id for item in instances):
                        source = candidate
                        break
                if profile is None or source is None:
                    gaps.append(str(target_id))
                    continue
                view = await self._monitoring.get_target_view(
                    scope=query_scope,
                    target_id=target_id,
                    source_id=source.source_id,
                    profile_id=profile.profile_id,
                    window=window,
                )
                evidence.append(
                    {
                        "source_kind": "MONITORING",
                        "source_id": (
                            f"{source.source_id}:{target_id}:{profile.profile_id}"
                        ),
                        "target_id": str(target_id),
                        "snapshot": _model_dump(view),
                    }
                )
                lines.append(
                    f"- {view.instances[0].display_name} · {view.profile.display_name}"
                    f"（{view.window.start.isoformat()} 至 {view.window.end.isoformat()}）"
                )
                for panel in view.panels:
                    lines.append(
                        f"  - {panel.title}：{panel.summary or '无汇总值'}（{panel.quality}）"
                    )
        except Exception:
            return CapabilityResult(
                capability,
                False,
                "固定监控视图暂时不可用；未执行任何自由 PromQL。",
                "MONITORING_FIXED_VIEW_UNAVAILABLE",
                tuple(evidence),
            )
        if gaps:
            lines.append(f"- {len(gaps)} 个资源没有可用的固定监控来源或 Profile。")
        if not evidence:
            return CapabilityResult(
                capability,
                False,
                "当前范围没有可用的固定监控视图；未执行任何自由 PromQL。",
                "MONITORING_FIXED_VIEW_NOT_CONFIGURED",
            )
        return CapabilityResult(
            capability,
            not gaps,
            "\n".join(lines),
            "MONITORING_SCOPE_PARTIAL" if gaps else None,
            tuple(evidence),
        )
