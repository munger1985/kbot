"""聊天中的 Exadata 指标选择、授权筛选和有界只读取数。"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
import json
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

from platform_core.contracts.aiops import AIOpsCapability
from platform_core.contracts.aiops.exadata import ExadataQuery, ExadataBaselineQuery
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.application.exadata_analysis import analyze_view, baseline_observations


class ExadataChatQuery(BaseModel):
    """选择真实目录编码，不接受模型生成的查询或内部 Target 身份。"""

    model_config = ConfigDict(extra="forbid")

    resource_type_key: Literal["EXADATA"]
    query_kind: Literal["METRIC", "OVERVIEW", "UNRESOLVED"] = "METRIC"
    metric_codes: tuple[str, ...] = Field(default=(), max_length=12)
    system_names: tuple[str, ...] = Field(default=(), max_length=8)
    resource_names: tuple[str, ...] = Field(default=(), max_length=8)
    component_names: tuple[str, ...] = Field(default=(), max_length=8)
    source_name: str | None = Field(default=None, max_length=256)
    mode: Literal["LATEST_SNAPSHOT", "HISTORY"] = "LATEST_SNAPSHOT"
    baseline_period: Literal["1w", "1m"] | None = None

    @model_validator(mode="after")
    def validate_metrics(self):
        if self.query_kind == "METRIC" and not self.metric_codes:
            raise ValueError("具体指标查询必须选择真实目录编码；无法映射时使用 UNRESOLVED")
        if self.query_kind != "METRIC" and self.metric_codes:
            raise ValueError("只有具体指标查询可以携带指标编码")
        if len(set(self.metric_codes)) != len(self.metric_codes):
            raise ValueError("指标编码不能重复")
        metrics = load_catalog()[2]
        if any(code not in metrics for code in self.metric_codes):
            raise ValueError("指标不在真实 OEM Exadata 审核目录中")
        return self


def exadata_parameters(plan) -> dict[str, Any] | None:
    """只识别路由器的结构化类型，不检查用户文本关键词。"""
    steps = [step for step in plan.steps if step.capability in {
        AIOpsCapability.MONITORING_QUERY, AIOpsCapability.ACTIVE_DIAGNOSIS}]
    queries = [dict(step.parameters) for step in steps if step.parameters.get("resource_type_key") == "EXADATA"]
    if not queries:
        return None
    if len(steps) != 1:
        # 一项能力只委派一次，不能静默忽略其他监控步骤或退回普通数据库路径。
        return {"resource_type_key": "EXADATA", "query_kind": "UNRESOLVED"}
    return queries[0]


def exadata_parameter_contract(*, diagnosis=False) -> dict[str, Any]:
    version, digest, metrics, _, _ = load_catalog()
    return {
        "parameters_schema": {"oneOf": [
            ({"title": "现有非 Exadata 诊断参数", "type": "object", "not": {"required": ["resource_type_key"],
                "properties": {"resource_type_key": {"const": "EXADATA"}}}} if diagnosis else
             {"title": "现有非 Exadata 监控", "type": "object", "properties": {
                "window": {"type": "string", "enum": ["15m", "1h", "6h", "24h"]}}, "additionalProperties": False}),
            ExadataChatQuery.model_json_schema(),
        ]},
        "policy": (
            "首先按用户实际所问资源选择 oneOf 分支，不能因输入包含 Exadata 目录就选择 Exadata。"
            "普通数据库、主机或非 Exadata 概览保持现有参数：返回空对象 {} 或仅提供 window，"
            "不得设置 resource_type_key=EXADATA。只有所问资源确实是 EXADATA 时使用专用分支。"
            "用户询问 Exadata 具体指标时必须设置 resource_type_key=EXADATA、query_kind=METRIC，"
            "从 metric_catalog 按语义选择准确的 metric_codes，不得替换成数据库总览。"
            "同一问题所问的多个 Exadata 指标放入同一个步骤的 metric_codes，不拆成多个同能力步骤。"
            "只询问指标读数使用 MONITORING_QUERY；要求分析、诊断、定位异常或解释变化使用 ACTIVE_DIAGNOSIS，"
            "Exadata 分支仅基于 OEM 指标只读分析，不运行数据库诊断，不创建变更动作。"
            "可用 baseline_period=1w/1m 选择7天/30天均值比较；主动分析默认7天，明确不比较则设null。"
            "基线窗口与 HISTORY 取样窗口独立，询问均值比较不把time_range扩为7/30天历史。"
            "区分当前读数、上限、下限、累计量和速率；未询问的阈值不附带查询。"
            "找不到对应指标使用 UNRESOLVED；只有明确要求概览才用 OVERVIEW。"
            "即使 authorized_business_systems 没有 EXADATA，也保留用户的 EXADATA 类型意图，"
            "由服务端拒绝不匹配范围，不改查其他资源。"
            "system_names/resource_names 只能用授权目录中的准确名称或编码；用户指定的对象"
            "不在目录时保留原名，不能改成其他资源或清空筛选。"
            "component_names 保留用户明确指定的完整组件名，不猜 OEM GUID；未指定则逐组件列出。"
            "当前值使用 LATEST_SNAPSHOT；用户明确要求历史才用 HISTORY 和 TurnPlan.time_range。"
            "历史仅支持最近一小时内的相对窗口，超范围必须保留原时间意图并说明不支持，不能缩短窗口。"
            "近期追问应保留上轮明确的资源和指标意图，但重新检查当轮授权。"
        ),
        "catalog_version": version,
        "catalog_sha256": digest,
        "metric_catalog": [{key: item[key] for key in
                            ("code", "display_name", "target_type", "scene", "semantics", "baseline_supported")}
                           for item in metrics.values()],
    }


def select_exadata_systems(systems, parameters):
    """先与 Agent 的系统授权求交集；任何显式名称不匹配都不能扩大范围。"""
    query = ExadataChatQuery.model_validate(parameters)
    selected = []
    matched_systems, matched_resources = set(), set()
    for system in systems:
        aliases = {str(system.get(key) or "").casefold()
                   for key in ("system_code", "display_name")}
        system_matches = {name for name in query.system_names if name.casefold() in aliases}
        if query.system_names and not system_matches:
            continue
        matched_systems.update(system_matches)
        resources = []
        for resource in system["resources"]:
            if resource["resource_type_key"] != "EXADATA":
                continue
            aliases = {str(resource.get(key) or "").casefold()
                       for key in ("resource_code", "display_name")}
            resource_matches = {name for name in query.resource_names if name.casefold() in aliases}
            if query.resource_names and not resource_matches:
                continue
            matched_resources.update(resource_matches)
            resources.append(resource)
        if resources:
            selected.append({**system, "resources": resources, "relations": []})
    if set(query.system_names) != matched_systems or set(query.resource_names) != matched_resources:
        return []
    return selected


@dataclass(frozen=True)
class ExadataChatResult:
    markdown: str
    gap_code: str | None = None
    evidence: tuple[dict[str, Any], ...] = ()


def _text(value):
    """来源原文只用于显示；完整未转义原值保存在证据快照中。"""
    value = str(value).replace("\n", " ").replace("\r", " ")
    for token in ("\\", "`", "*", "_", "[", "]", "<", ">", "|"):
        value = value.replace(token, "\\" + token)
    return value


def _time(value: datetime):
    return value.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S（北京时间）")


class ExadataChatReader:
    """复用专用采集服务；只读指定编码，不调用数据库 Profile 或执行器。"""

    def __init__(self, *, monitoring_service, exadata_service):
        self._monitoring = monitoring_service
        self._exadata = exadata_service

    async def read(self, *, parameters, plan, resources, scope, analyze=False):
        try:
            query = ExadataChatQuery.model_validate(parameters)
        except ValueError:
            return ExadataChatResult("Exadata 指标或筛选条件未能准确解析，请明确指标和资源；没有改查数据库总览。",
                                     "EXADATA_QUERY_INVALID")
        if query.query_kind == "UNRESOLVED":
            return ExadataChatResult("真实 OEM Exadata 目录中尚未准确匹配所问指标，请明确指标名称；没有改查其他指标。",
                                     "EXADATA_METRIC_UNRESOLVED")
        if not resources or any(item["resource_type_key"] != "EXADATA" for item in resources):
            return ExadataChatResult("当前 Agent 授权范围内没有匹配的 Exadata 资源，请选择已授权该资源的 Agent；没有查询其他系统。",
                                     "EXADATA_RESOURCE_SCOPE_EMPTY")
        if self._exadata is None:
            return ExadataChatResult("Exadata 专用采集尚未装配，没有回退到数据库监控。", "EXADATA_READER_UNAVAILABLE")
        time_range = getattr(plan, "time_range", None)
        seconds = getattr(time_range, "relative_window_seconds", None) or 3600
        if time_range is not None and (getattr(time_range, "start_at", None) is not None or seconds > 3600):
            return ExadataChatResult("Exadata 聊天指标目前仅支持最新快照或最近一小时内的相对历史窗口；未改变所问时间范围。",
                                     "EXADATA_TIME_RANGE_UNSUPPORTED")
        if seconds % 60:
            return ExadataChatResult("Exadata 历史窗口需为整分钟，未擅自扩大或缩短查询范围。", "EXADATA_TIME_RANGE_UNSUPPORTED")
        remaining = min(plan.budget.max_live_queries, 24)
        if remaining < 1:
            return ExadataChatResult("本轮没有可用的实时指标查询预算。", "EXADATA_QUERY_BUDGET_EXCEEDED")
        _, _, metrics, profiles, _ = load_catalog()
        codes = query.metric_codes or tuple(next(item for item in profiles if item["id"] == "overview")["metric_codes"])
        groups = {}
        for code in codes:
            groups.setdefault(metrics[code]["scene"], []).append(code)
        evidence, lines, gaps = [], ["### Exadata 指标分析" if analyze else "### Exadata 指标查询"], []
        deadline = asyncio.get_running_loop().time() + min(plan.budget.timeout_seconds, 55)

        def remaining_seconds():
            return max(0.01, deadline - asyncio.get_running_loop().time())

        try:
            catalog = await asyncio.wait_for(self._monitoring.list_business_systems(scope=scope), remaining_seconds())
            for resource in resources:
                current = next((item for system in catalog
                                if str(system.business_system_id) == str(resource["business_system_id"])
                                for item in system.resources
                                if str(item.resource_id) == str(resource["resource_id"])
                                and item.resource_type_key == "EXADATA"), None)
                sources = {str(source.source_id): source for source in (current.sources if current else ())
                           if source.source_type == "OEM" and
                           (query.source_name is None or source.display_name.casefold() == query.source_name.casefold())}
                if len(sources) != 1:
                    lines.append(f"- {_text(resource['display_name'])}：{'没有匹配的 OEM 绑定' if not sources else '存在多个 OEM 绑定，请明确来源'}。")
                    gaps.append("EXADATA_SOURCE_AMBIGUOUS" if sources else "EXADATA_SOURCE_UNAVAILABLE")
                    continue
                source = next(iter(sources.values()))
                if source.monitoring_readiness != "READY":
                    lines.append(f"- {_text(resource['display_name'])}：OEM 来源未就绪，未执行取数。")
                    gaps.append("EXADATA_SOURCE_UNAVAILABLE")
                    continue
                for scene, selected in groups.items():
                    if not remaining:
                        gaps.append("EXADATA_QUERY_BUDGET_EXCEEDED")
                        continue
                    view = await asyncio.wait_for(self._exadata.collect(
                        scope=scope, system_id=UUID(str(resource["business_system_id"])),
                        resource_id=UUID(str(resource["resource_id"])), source_id=source.source_id,
                        query=ExadataQuery(scene=scene, mode=query.mode, window_minutes=seconds // 60, limit=remaining),
                        _metric_codes=tuple(selected), _component_names=query.component_names), remaining_seconds())
                    remaining -= min(view.total_queries, remaining)
                    index = len(evidence) + 1
                    summary = self._render(view, selected)
                    lines.append(f"- {_text(resource['display_name'])} · {_text(source.display_name)} [C{index}]\n{summary}")
                    evidence.append({"resource_id": str(resource["resource_id"]),
                                     "business_system_id": str(resource["business_system_id"]),
                                     "source_id": str(source.source_id), "summary": summary,
                                     "snapshot": view.model_dump(mode="json")})
                    if analyze:
                        findings = analyze_view(view)
                        evidence[-1]["snapshot"]["analysis_findings"] = [item.model_dump(mode="json") for item in findings]
                        for finding in findings:
                            lines.append(f"  - 分析 [C{index}]：{_text(finding.detail)}")
                        if any(item.severity == "WARNING" for item in findings):
                            lines.append("  - 已定位上述组件状态异常；尚不足以确定整机故障或数据库根因。")
                    bad_gaps = [gap for gap in view.gaps if gap.code not in {
                        "THRESHOLDS_UNVERIFIED", "HISTORY_UNAVAILABLE", "MEMBERSHIP_SCOPE",
                        "HISTORY_RESOLUTION_PROVIDER_MANAGED"}]
                    if bad_gaps or any(sample.quality != "OBSERVED" for sample in view.samples) or not view.samples:
                        gaps.append("EXADATA_METRIC_DATA_PARTIAL")
                    member_names = {member.target_id: member.display_name for member in view.members}
                    for gap in bad_gaps:
                        component = f"{_text(member_names.get(gap.component_id, gap.component_id))} · " if gap.component_id else ""
                        metric = f"`{gap.metric_code}` · " if gap.metric_code else ""
                        lines.append(f"  - 采集缺口：{component}{metric}{_text(gap.code)}；{_text(gap.detail)}")
            period = query.baseline_period
            if analyze and "baseline_period" not in query.model_fields_set:
                period = "1w"
            # 先采集所有已选当前指标，再以剩余预算比较基线，避免历史读取挤占具体指标。
            if period:
                for index, item in enumerate(evidence, 1):
                    snapshot = item["snapshot"]
                    pairs = list(dict.fromkeys((sample["component_id"], sample["metric_code"])
                        for sample in snapshot["samples"] if metrics[sample["metric_code"]].get("baseline_supported")))
                    snapshot["analysis_baselines"] = []
                    for component_id, code in pairs:
                        if remaining < 2:
                            gaps.append("EXADATA_QUERY_BUDGET_EXCEEDED")
                            break
                        remaining -= 2
                        try:
                            result = await asyncio.wait_for(self._exadata.baseline(scope=scope,
                                system_id=UUID(item["business_system_id"]), resource_id=UUID(item["resource_id"]),
                                source_id=UUID(item["source_id"]), query=ExadataBaselineQuery(
                                    component_id=component_id, metric_code=code, period=period)), remaining_seconds())
                            snapshot["analysis_baselines"].append(result.model_dump(mode="json"))
                            name = next(member["display_name"] for member in snapshot["members"] if member["target_id"] == component_id)
                            lines.append(f"  - {_text(name)} · {_text(metrics[code]['display_name'])} · {'7天' if period == '1w' else '30天'}窗口 [C{index}]：")
                            lines.extend(f"    - {_text(line)}" for line in baseline_observations(result))
                            if result.baseline.unavailable_reason or not result.comparisons:
                                gaps.append("EXADATA_BASELINE_UNAVAILABLE")
                        except (AIOpsApplicationError, DiagnosticSourceAdapterError, TimeoutError):
                            lines.append(f"  - `{code}` 历史均值比较未完成；当前采样证据保留 [C{index}]。")
                            gaps.append("EXADATA_BASELINE_UNAVAILABLE")
        except (AIOpsApplicationError, DiagnosticSourceAdapterError, TimeoutError) as exc:
            message = exc.message if isinstance(exc, AIOpsApplicationError) else "OEM 读取或成员复核失败，请稍后重试"
            lines.append(f"- Exadata 专用查询未完成：{_text(message)}；未回退数据库总览。")
            gaps.append("EXADATA_QUERY_UNAVAILABLE")
        if analyze:
            scenes = {metrics[code]["scene"] for code in codes}
            suggestions = {"overview": "检查异常设备的 OEM 事件及服务日志，区分管理可达性、硬件健康和数据库可用性",
                "cell": "在相同时段核对 CPU、内存、运行队列与 IO 负载，确认变化是否伴随资源压力",
                "disk": "同一 Cell/磁盘类型核对 IOPS、吞吐、利用率与单请求延迟，避免将负载增加直接判断为磁盘故障",
                "flash": "核对 Flash 命中率、Flash/磁盘吞吐与读写延迟，确认缓存命中变化是否伴随 IO 等待",
                "iorm": "按数据库、PDB 或消费组核对等待与吞吐，确认限流策略和资源竞争，不跨消费组平均",
                "network": "按接口核对吞吐、丢弃/重传与端口状态，并检查同时间原厂事件",
                "capacity": "按磁盘组或设备核对容量与使用比例，结合增长历史评估余量，不将容量值等同故障"}
            lines.append("下一步只读排查：")
            lines.extend(f"- {suggestions[scene]}。" for scene in sorted(scenes) if scene in suggestions)
        lines.append("单位采用 OEM 原始单位；状态按已审核字典解释，未知编码不判定可用性。")
        if query.baseline_period or analyze:
            lines.append("历史均值为窗口内实际留存采样的算术平均，覆盖率未知；不跨组件/维度合并，不补造历史。")
        if query.mode == "HISTORY":
            lines.append("历史分辨率由 OEM 留存策略决定；各组件/维度分别列示末次采样和采样数，不合并求均值。")
        if "EXADATA_QUERY_BUDGET_EXCEEDED" in gaps:
            lines.append("部分指标/组件超出本轮查询预算，请缩小范围后继续。")
        return ExadataChatResult("\n".join(lines), gaps[0] if gaps else None, tuple(evidence))

    @staticmethod
    def _render(view, codes):
        members = {member.target_id: member.display_name for member in view.members}
        definitions = {item.code: item for item in view.definitions}
        lines, groups = [], {}
        for sample in view.samples:
            if sample.metric_code in codes:
                key = (sample.component_id, sample.metric_code, sample.unit, sample.native_value_type,
                       tuple(sorted(sample.dimensions.items())))
                groups.setdefault(key, []).append(sample)
        for (component, code, unit, _, dimensions), samples in groups.items():
            # 同一时间的末条为空时也不得从前条回填。
            latest = max(reversed(samples), key=lambda sample: sample.observed_at)
            value = "无读数" if latest.value in (None, "") else _text(latest.value)
            status = definitions[code].status
            if status and latest.value not in (None, ""):
                entry = next((item for raw, item in status.values.items()
                              if raw.casefold() == latest.value.strip().casefold()), None)
                value = entry.label if entry and latest.quality == "OBSERVED" else "未知"
            unit_text = f" {_text(unit)}" if unit and unit.strip() else "（OEM 未返回单位）"
            if status:
                unit_text = ""
            dimension_text = f"；维度：{_text(json.dumps(dict(dimensions), ensure_ascii=False))}" if dimensions else ""
            quality = {"OBSERVED": "已采集", "EMPTY": "末次采样为空", "STALE": "旧采样，非当前有效值", "TRUNCATED": "来源数据截断"}[latest.quality]
            lines.append(f"  - {_text(members.get(component, component))} · {_text(definitions[code].display_name)}"
                         f"（`{code}`）：{value}{unit_text}；采集时间 {_time(latest.observed_at)}；{quality}{dimension_text}"
                         + (f"；窗口采样数 {len(samples)}" if view.query_mode == "HISTORY" else ""))
        for code in codes:
            if not any(sample.metric_code == code for sample in view.samples):
                lines.append(f"  - `{code}`：OEM 未返回采样，不能解释为零或硬件正常。")
        return "\n".join(lines)
