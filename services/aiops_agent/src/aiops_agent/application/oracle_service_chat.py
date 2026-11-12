"""基础服务的结构化选指标、当轮授权和只读证据分析。"""

import asyncio
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from platform_core.contracts.aiops import AIOpsCapability
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeQuery
from platform_core.contracts.aiops.oracle_services import ORACLE_SERVICE_TYPES
from aiops_agent.application.errors import AIOpsApplicationError
from .oracle_service_catalog import service_metrics
from .exadata_chat import ExadataChatResult, _text


class OracleServiceChatQuery(BaseModel):
    model_config = ConfigDict(extra='forbid')
    resource_type_key: Literal['ORACLE_ASM', 'ORACLE_RAC', 'ORACLE_LISTENER']
    query_kind: Literal['METRIC', 'OVERVIEW', 'UNRESOLVED'] = 'METRIC'
    metric_codes: tuple[str, ...] = Field(default=(), max_length=12)
    system_names: tuple[str, ...] = Field(default=(), max_length=8)
    resource_names: tuple[str, ...] = Field(default=(), max_length=8)
    source_name: str | None = Field(default=None, max_length=256)
    baseline_period: Literal['1w', '1m'] | None = None
    window: Literal['15m', '1h', '6h', '24h'] = '1h'

    @model_validator(mode='after')
    def validate_metrics(self):
        metrics = service_metrics()
        if (self.query_kind == 'METRIC') != bool(self.metric_codes) or len(set(self.metric_codes)) != len(self.metric_codes):
            raise ValueError('具体指标需明确目录编码，概览或无法解析时不携带指标编码')
        if any(code not in metrics or metrics[code]['resource_type'] != self.resource_type_key for code in self.metric_codes):
            raise ValueError('所选指标与服务资源类型不匹配')
        return self


def oracle_service_parameters(plan):
    steps = [step for step in plan.steps if step.capability in {AIOpsCapability.MONITORING_QUERY, AIOpsCapability.ACTIVE_DIAGNOSIS}]
    queries = [dict(step.parameters) for step in steps if step.parameters.get('resource_type_key') in ORACLE_SERVICE_TYPES]
    if not queries:
        return None
    if len(steps) != 1:
        return {'resource_type_key': queries[0]['resource_type_key'], 'query_kind': 'UNRESOLVED'}
    return queries[0]


def select_oracle_service_systems(systems, parameters):
    query = OracleServiceChatQuery.model_validate(parameters)
    result, matched_systems, matched_resources = [], set(), set()
    for system in systems:
        names = {str(system.get(key) or '').casefold() for key in ('display_name', 'system_code')}
        matches = {name for name in query.system_names if name.casefold() in names}
        if query.system_names and not matches:
            continue
        matched_systems.update(matches)
        resources = []
        for resource in system['resources']:
            if resource['resource_type_key'] != query.resource_type_key:
                continue
            names = {str(resource.get(key) or '').casefold() for key in ('display_name', 'resource_code')}
            matches = {name for name in query.resource_names if name.casefold() in names}
            if query.resource_names and not matches:
                continue
            matched_resources.update(matches)
            resources.append(resource)
        if resources:
            result.append({**system, 'resources': resources, 'relations': []})
    return result if set(query.system_names) == matched_systems and set(query.resource_names) == matched_resources else []


def analyze_service_samples(samples):
    """解释状态和同维度偏离；不凭采集成功判健康，不推断因果或自动阈值。"""
    metric, lines = samples.metric, []
    latest = {}
    for point in samples.points:
        key = tuple(sorted(point.dimensions.items()))
        if key not in latest or point.observed_at > latest[key].observed_at:
            latest[key] = point
    if not latest:
        return ['来源未返回采样，不能判断服务状态。']
    for key, point in latest.items():
        dimensions = ' · '.join(name + '=' + value for name, value in key) or '无维度'
        prefix = dimensions + '：'
        if not point.current_eligible:
            lines.append(prefix + '缺值、过期或截断；不作为当前诊断证据。')
            continue
        raw = point.value
        if metric.semantics == 'STATE':
            from platform_core.contracts.aiops.oracle_services import state_key
            state = next((value for token, value in metric.state_values.items()
                if state_key(token, metric.native_value_type) == state_key(raw, metric.native_value_type)), None)
            lines.append(prefix + (state.get('label', '未知状态') if state else '未知状态') + '；不代表其他设备或服务层健康。')
        elif metric.semantics == 'COUNTER':
            lines.append(prefix + '这是累计原值；未审核重置与完整时间覆盖，不推导增量、IOPS或单次延迟。')
        elif metric.semantics == 'PROPERTY':
            lines.append(prefix + '配置或成员列表证据，不将静态配置等同动态服务注册或运行实例健康。')
        else:
            historical = next((entry for entry in samples.baseline.series if tuple(sorted(entry.dimensions.items())) == key), None) if samples.baseline else None
            try:
                current = Decimal(raw)
                if metric.name == 'asm.usable_file_mb' and current < 0:
                    lines.append(prefix + '考虑冗余后的可用容量为负，存在冗余预留空间不足证据；需结合故障组与磁盘状态核查。')
                if historical:
                    average = Decimal(historical.average)
                    delta = current-average
                    deviation = f'{delta/abs(average)*100:.2f}%' if average else '零均值，不定义相对偏离'
                    lines.append(prefix + f'当前 {current:.2f}，基线 {average:.2f} {metric.unit}，差值 {delta:.2f}，偏离 {deviation}（{historical.sample_count} 条）；偏离不是故障阈值。')
                else:
                    lines.append(prefix + '没有可用的同来源同维度均值，不能判断相对基线的变化。')
            except (TypeError, InvalidOperation):
                lines.append(prefix + '原值非有效数值，保留来源证据，不生成数值结论。')
    return lines


class OracleServiceChatReader:
    def __init__(self, monitoring_service):
        self.monitoring = monitoring_service

    async def read(self, *, parameters, plan, resources, scope, analyze=False):
        try:
            query = OracleServiceChatQuery.model_validate(parameters)
        except ValueError:
            return ExadataChatResult('服务指标条件无效，未改查数据库或 Exadata 总览。', 'ORACLE_SERVICE_QUERY_INVALID')
        if query.query_kind == 'UNRESOLVED':
            return ExadataChatResult('尚未准确匹配服务指标，请明确 ASM、RAC 或 Listener 的具体指标。', 'ORACLE_SERVICE_METRIC_UNRESOLVED')
        if not resources or any(item['resource_type_key'] != query.resource_type_key for item in resources):
            return ExadataChatResult('当前 Agent 授权范围内没有匹配的服务资源；未查询其他资源。', 'ORACLE_SERVICE_SCOPE_EMPTY')
        time_range = getattr(plan, 'time_range', None)
        relative = getattr(time_range, 'relative_window_seconds', None) if time_range else None
        if time_range and (getattr(time_range, 'start_at', None) is not None or relative and relative not in {900, 3600, 21600, 86400}):
            return ExadataChatResult('服务监控只支持15分钟、1小时、6小时、24小时相对观测窗口；没有擅自缩短时间。', 'ORACLE_SERVICE_TIME_UNSUPPORTED')
        window = {900: '15m', 3600: '1h', 21600: '6h', 86400: '24h'}.get(relative, query.window)
        remaining = min(plan.budget.max_live_queries, 24)
        lines, evidence, gaps = ['### Oracle 服务指标分析' if analyze else '### Oracle 服务监控'], [], []
        try:
            async with asyncio.timeout(min(plan.budget.timeout_seconds, 55)):
                systems = await self.monitoring.list_business_systems(scope=scope)
                for resource in resources:
                    selected = next((item for system in systems if str(system.business_system_id) == str(resource['business_system_id'])
                        for item in system.resources if str(item.resource_id) == str(resource['resource_id'])), None)
                    sources = [source for source in selected.sources if source.monitoring_readiness == 'READY'
                        and (query.source_name is None or source.display_name.casefold() == query.source_name.casefold())] if selected else []
                    if len(sources) != 1:
                        gaps.append('ORACLE_SERVICE_SOURCE_AMBIGUOUS')
                        lines.append('- ' + _text(resource['display_name']) + '：没有唯一就绪的所选来源，请明确来源；不自动合并。')
                        continue
                    source = sources[0]
                    codes = query.metric_codes or tuple(code for code, definition in service_metrics().items() if definition['resource_type'] == query.resource_type_key)
                    for code in codes:
                        definition = service_metrics()[code]
                        baseline = query.baseline_period if definition['semantics'] == 'GAUGE' else None
                        if analyze and 'baseline_period' not in query.model_fields_set and definition['semantics'] == 'GAUGE':
                            baseline = '1w'
                        # 预算按绑定中的实际来源映射数核算，不只按语义指标数。
                        frozen = await self.monitoring.native._binding(scope, source.source_id, selected.instance_id)
                        mappings = self.monitoring.native.oracle.definitions(frozen)[code][1]
                        cost = max(1, len(mappings)) * (2 if baseline else 1)
                        if remaining < cost:
                            gaps.append('ORACLE_SERVICE_BUDGET_EXCEEDED')
                            lines.append('- 剩余查询预算不足，未查询的指标不视为无数据。')
                            break
                        remaining -= cost
                        try:
                            samples = await self.monitoring.native.samples(scope=scope, source_id=source.source_id,
                                instance_id=selected.instance_id, query=MonitoringNativeQuery(metric_name=code, window=window,
                                    mode='LATEST' if definition['semantics'] in {'STATE', 'PROPERTY'} else 'HISTORY', baseline=baseline))
                        except AIOpsApplicationError as exc:
                            gaps.append(exc.code)
                            lines.append('- ' + _text(definition['title']) + '：无法读取（' + exc.code + '），未以其他指标替代。')
                            continue
                        lines.append('- ' + _text(resource['display_name']) + ' / ' + _text(definition['title']) + ' / ' + _text(source.display_name))
                        for point in samples.points[-12:]:
                            lines.append('  - ' + _text(str(point.dimensions)) + '：' + _text(point.value) + ' ' + _text(samples.metric.unit)
                                + ' · ' + point.observed_at.isoformat() + ' · ' + point.quality)
                        if analyze:
                            lines.extend('  - ' + _text(line) for line in analyze_service_samples(samples))
                        evidence.append({'resource_id': str(resource['resource_id']), 'business_system_id': str(resource['business_system_id']),
                            'source_id': str(source.source_id), 'summary': definition['title'], 'snapshot': samples.model_dump(mode='json')})
        except (TimeoutError, AIOpsApplicationError) as exc:
            gaps.append(getattr(exc, 'code', 'ORACLE_SERVICE_TIMEOUT'))
            lines.append('- 查询未全部完成，已读取的证据保留；未完成部分不作判断。')
        return ExadataChatResult('\n'.join(lines), gaps[0] if gaps else None, tuple(evidence))
