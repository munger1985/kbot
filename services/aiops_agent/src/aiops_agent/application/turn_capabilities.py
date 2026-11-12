"""统一 Turn Router 的只读专业能力适配器。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from loguru import logger

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.exadata_chat import ExadataChatReader, exadata_parameters
from aiops_agent.application.alert_query import (
    AlertQueryParameters, TERMINAL_SIGNAL_STATUSES, signal_status_label,
)
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
    context = create_service_auth_context(caller_service=str(runtime["caller_service"]),
        request_id=str(runtime["request_id"]), trace_id=str(runtime["trace_id"]))
    return context.model_copy(update={"app_id": "aiops", "domain_id": str(runtime["domain_id"]),

        "asserted_user_id": str(runtime["actor_id"]),
        "scopes": ("knowledge.scope.create", "knowledge.retrieve", "knowledge.content.read")})


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
        runtime_child=None,
    ) -> None:
        self._child = runtime_child
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
        if catalog_mode and not policy.source_kinds:
            return CapabilityResult(
                capability,
                False,
                "当前 Agent 版本没有授权查询运维资料或历史案例。",
                "KNOWLEDGE_SOURCE_POLICY_DISABLED",
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
        if not catalog_mode:
            try:
                if self._child is None:
                    raise ValueError("知识子执行尚未装配")
                markdown, evidence = await self._child.answer(context=auth,
                    turn_id=UUID(str(runtime["turn_id"])), agent_id=UUID(str(runtime["agent_id"])),
                    agent_version_id=UUID(str(runtime["agent_version_id"])),
                    target_id=targets[0] if len(targets) == 1 else None,
                    request=KnowledgeSearchRequest(query=str(runtime["question"])[:4000], purpose="EXPLAIN",
                        knowledge_base_ids=policy.knowledge_base_ids,
                        tag_ids=tuple(str(item) for item in policy.tag_ids), source_kinds=policy.source_kinds,
                        max_results=policy.maximum_references, max_security_level=policy.max_security_level))
                return CapabilityResult(capability, True, markdown, evidence=evidence)
            except Exception:
                return CapabilityResult(capability, False, "知识子执行暂时不可用，未生成无来源回答。",
                    "OPERATIONS_KNOWLEDGE_UNAVAILABLE")
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
                    source_kinds=policy.source_kinds,
                    max_results=policy.maximum_references,
                    max_security_level=policy.max_security_level,
                ),
            )
        except Exception:
            logger.exception("运维资料目录查询失败：agent_id={}", runtime.get("agent_id"))
            return CapabilityResult(
                capability,
                False,
                "资料目录暂时无法读取，请稍后重试；这不表示没有已发布资料。",
                "OPERATIONS_KNOWLEDGE_UNAVAILABLE",
            )
        if catalog_mode:
            if result.get("status") not in {"READY", "NO_APPLICABLE_KNOWLEDGE"}:
                return CapabilityResult(capability, False,
                    "当前知识授权或资料范围不可用，请刷新后重新查询目录。",
                    "KNOWLEDGE_CATALOG_SCOPE_UNAVAILABLE")
            evidence = []
            lines = ["### 已发布资料目录"]
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
                        "source_kind": "DIAGNOSIS_CASE" if item["asset_kind"] == "DIAGNOSIS_CASE" else "KNOWLEDGE",
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
                    f"- {item['title']}（{'历史诊断案例' if item['asset_kind'] == 'DIAGNOSIS_CASE' else '运维资料'}，版本 {item['version']}）："
                    f"{summary} [K{len(evidence)}]"
                )
            if not evidence:
                if result.get("coverage", {}).get("registry_scope_complete") is False:
                    lines.append("- 本轮已扫描范围内未列出可展示资料，可缩小知识库或资料范围后重试。")
                else:
                    lines.append("- 当前 Agent 授权范围内没有已发布运维资料或历史诊断案例。")
            lines.extend(str(warning) for warning in result.get("warnings", ()))
            return CapabilityResult(
                capability=capability,
                supported=True,
                markdown="\n".join(lines),
                evidence=tuple(evidence),
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
                return await self._alerts(query_scope, targets, parameters, plan.time_range)
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

    async def _alerts(self, scope, targets, parameters, time_range=None) -> CapabilityResult:
        try:
            parameters = AlertQueryParameters.model_validate(parameters)
        except ValidationError:
            return CapabilityResult(AIOpsCapability.ALERT_QUERY, False,
                "告警筛选条件无效，请指定当前或历史告警及有效的状态条件。", "ALERT_QUERY_INVALID_FILTER")
        start, end = parameters.time_bounds(time_range, now=datetime.now(UTC))
        excluded = TERMINAL_SIGNAL_STATUSES if parameters.mode == "CURRENT" else ()
        status_label = signal_status_label(parameters.status) if parameters.status else (
            "未恢复、未关闭" if excluded else "全部监控信号状态"
        )
        rows = []
        truncated = False
        for target_id in targets:
            page = await self._runtime.list_situations(
                scope=scope,
                target_id=target_id,
                agent_id=None,
                status=parameters.status,
                severity=parameters.severity,
                cursor=None,
                limit=parameters.limit,
                exclude_statuses=excluded,
                observed_since=start,
                observed_until=end,
            )
            truncated = truncated or bool(getattr(page, "has_more", False))
            rows.extend(page.items)
        rows.sort(key=lambda item: item.last_observed_at, reverse=True)
        evidence = tuple(
            {
                "source_kind": "ALERT",
                "source_id": str(item.situation_id),
                "target_id": str(item.target_id),
                "snapshot": _model_dump(item),
                "signal_status_label": signal_status_label(item.status),
                "query_mode": parameters.mode,
                "observed_since": start.isoformat(), "observed_until": end.isoformat(),
            }
            for item in rows[:50]
        )
        lines = ["### 当前告警" if excluded else "### 历史告警",
            f"最近观测筛选：{start.isoformat()} 至 {end.isoformat()}（不含结束时刻）；状态：{status_label}。",
            "状态来自最近入库的监控信号；告警恢复不代表人工处置完成。"]
        if parameters.severity:
            lines.append(f"严重程度筛选：{parameters.severity}。")
        lines.extend(
            f"- [{item.severity}] {item.title}（{signal_status_label(item.status)}，最近观测 {item.last_observed_at.isoformat()}）"
            for item in rows[:50]
        )
        if not rows:
            lines.append("- 当前授权资源在上述时间与状态条件下未查到匹配的告警记录。")
        if truncated or len(rows) > 50:
            lines.append("- 仍有匹配记录未展示，可缩小筛选范围继续查询。")
        return CapabilityResult(
            AIOpsCapability.ALERT_QUERY, True, "\n".join(lines), evidence=evidence,
            gap_code="ALERT_QUERY_TRUNCATED" if truncated or len(rows) > 50 else None,
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
                    if catalog_mode and result.get("status") not in {"READY", "NO_APPLICABLE_KNOWLEDGE"}:
                        case_gap = "KNOWLEDGE_CATALOG_SCOPE_UNAVAILABLE"
                        lines.append("- 当前知识授权或资料范围已变化，请重新查询历史案例目录。")
                        continue
                    lines.extend(str(warning) for warning in result.get("warnings", ()))
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
                                        "document_version_id": citation.get("document_version_id"),
                                        "parse_run_id": citation.get("parse_run_id"),
                                        "index_generation_id": citation.get("index_generation_id"),
                                    }
                                    for citation in citations
                                ],
                            }
                        )
            except Exception:
                logger.exception("历史案例查询失败：agent_id={}", runtime.get("agent_id"))
                case_gap = "DIAGNOSIS_CASE_VIEW_UNAVAILABLE"
        elif "DIAGNOSIS_CASE" not in policy.source_kinds:
            case_gap = "DIAGNOSIS_CASE_POLICY_DISABLED"
        if case_gap:
            lines.append("- 历史案例查询未完成，请检查当前知识授权或稍后重试；这不表示没有已发布案例。")
        elif not case_count:
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

    def __init__(self, *, monitoring_service, exadata_service=None) -> None:
        self._monitoring = monitoring_service
        self._exadata = ExadataChatReader(monitoring_service=monitoring_service, exadata_service=exadata_service)

    async def execute(
        self,
        *,
        capability: AIOpsCapability,
        plan: TurnPlanV1,
        scope: ScopeResolutionV1,
        internal_scope: dict[str, Any],
    ) -> CapabilityResult:
        parameters = exadata_parameters(plan)
        resources = internal_scope.get("resource_catalog", ())
        if parameters is not None or resources and all(item.get("resource_type_key") == "EXADATA" for item in resources):
            # 与公开解析范围求交集，不能仅信任内部传入的资源列表。
            allowed = {(str(item.business_system_id), str(item.resource_id)) for item in scope.resources}
            selected = [item for item in resources
                        if (str(item["business_system_id"]), str(item["resource_id"])) in allowed]
            result = await self._exadata.read(parameters=parameters or {"resource_type_key": "EXADATA", "query_kind": "UNRESOLVED"},
                plan=plan, resources=selected, scope=_scope(internal_scope),
                analyze=capability == AIOpsCapability.ACTIVE_DIAGNOSIS)
            evidence = tuple({"source_kind": "MONITORING", "label": f"C{index}", **item}
                             for index, item in enumerate(result.evidence, 1))
            return CapabilityResult(capability, result.gap_code is None, result.markdown, result.gap_code, evidence)
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
