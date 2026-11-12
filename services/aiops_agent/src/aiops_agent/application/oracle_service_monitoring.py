"""ASM/RAC/Listener 专用监控，不改变普通数据库和主机的既有路径。"""

import asyncio
from datetime import UTC, datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import json

from aiops_agent.application.errors import validation_failed, resource_not_found
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.ports.diagnostic_source import NativeMetricQueryRequest
from aiops_agent.monitoring import resolve_monitoring_window, resolve_monitoring_baseline_window
from platform_core.contracts.aiops.monitoring_native import (
    MonitoringNativeCatalog, MonitoringNativeMetric, MonitoringNativePanel,
    MonitoringNativeSamples, MonitoringNativeBaseline, MonitoringNativeBaselineSeries,
)
from .oracle_service_catalog import mappings_for, service_metrics


class OracleServiceMonitoring:
    def __init__(self, native):
        self.native = native

    @staticmethod
    def definitions(frozen):
        try:
            mappings = mappings_for(frozen['scope'], frozen['source']['source_type'], frozen['target_type'], frozen['mapping_overrides'])
        except ValueError as exc:
            raise validation_failed(str(exc)) from exc
        return {code: (service_metrics()[code], values) for code, values in mappings.items()}

    @staticmethod
    def metric(code, definition, mappings):
        first = mappings[0]
        if any(item.native_value_type != first.native_value_type or item.state_values != first.state_values for item in mappings):
            raise validation_failed('同一语义指标的各个 Item 必须使用相同原生类型和状态字典')
        return MonitoringNativeMetric(name=code, display_name=definition['title'], group=definition['resource_type'],
            native_value_type=first.native_value_type, unit=definition['unit'],
            dimensions=tuple(definition['dimensions']), query_supported=first.native_value_type == 'NUMBER',
            latest_supported=True, current_max_age_seconds=definition['max_age_seconds'], semantics=definition['semantics'],
            state_values={key: value.model_dump() for key, value in first.state_values.items()})

    async def catalog(self, *, scope, source_id, instance_id, frozen, search='', offset=0, limit=100):
        definitions = self.definitions(frozen)
        adapter = await self.native._adapter(frozen, 'metric.query_range')
        now = datetime.now(UTC)
        request = self.request(frozen, instance_id, scope, now-timedelta(hours=1), now)
        async with asyncio.timeout(30):
            await adapter.verify_oracle_service(request, resource_type=frozen['scope'], locator=frozen['source_locator'])
        entries, panels = [], []
        for code, (definition, mappings) in definitions.items():
            metric = self.metric(code, definition, mappings) if mappings else None
            if metric:
                entries.append(metric)
            panels.append(MonitoringNativePanel(panel_id=code, title=definition['title'], description=definition['description'],
                semantics=definition['semantics'],
                metric=metric, mode='LATEST' if definition['semantics'] in {'STATE', 'PROPERTY'} else 'HISTORY',
                baseline_supported=bool(metric and definition['semantics'] == 'GAUGE'),
                unavailable_reason=None if mappings else '此来源尚未配置或不提供该服务指标；不以其他状态代替。'))
        filtered = [item for item in entries if search.strip().casefold() in (item.name + item.display_name).casefold()]
        await self.native._unchanged(scope, source_id, instance_id, frozen)
        revision = hashlib.sha256(json.dumps({'binding': str(frozen['binding_id']), 'version': int(frozen['version']),
            'definitions': {code: [definition, [item.model_dump(mode='json') for item in mappings]]
                for code, (definition, mappings) in definitions.items()}}, sort_keys=True).encode()).hexdigest()
        return self.native._response(MonitoringNativeCatalog(source_id=source_id, instance_id=instance_id,
            entries=tuple(filtered[offset:offset+limit]), discovered_count=len(filtered), truncated=False, revision=revision,
            default_metric=next((item for item in entries if item.semantics == 'GAUGE'), None),
            next_offset=offset+limit if offset+limit < len(filtered) else None, overview_panels=tuple(panels),
            warnings=('目录为审核的服务语义与显式映射；映射存在不等于来源已有采样。',
                '所有曲线、基线和状态只使用所选来源，不跨设备或维度合并。')), frozen)

    @staticmethod
    def request(frozen, instance_id, scope, start, end, *, budget=None):
        return NativeMetricQueryRequest(action_id='a0', target_id=str(instance_id), binding_id=str(frozen['binding_id']),
            source_locator_key=frozen['locator'], window_start=start, window_end=end, trace_id=scope.trace_id,
            max_response_bytes=budget or frozen['max_response_bytes'])

    async def read(self, adapter, request, frozen, mappings, *, latest):
        budget = request.max_response_bytes // len(mappings)
        if budget < 2048:
            raise validation_failed('来源映射数超出单次读取预算')
        points, dimensions, hashes, seen, truncated = [], [], [], set(), False
        captured = datetime.now(UTC)
        for mapping in mappings:
            value = await adapter.query_oracle_service_metric(request.model_copy(update={'max_response_bytes': budget}),
                resource_type=frozen['scope'], locator=frozen['source_locator'], mapping=mapping, latest=latest)
            for point, labels in zip(value['points'], value['dimensions'], strict=True):
                identity = (point[0], tuple(sorted(labels.items())))
                if identity in seen:
                    raise DiagnosticSourceAdapterError('SOURCE_ITEM_AMBIGUOUS', '多个映射返回了同一维度同一时间的采样')
                seen.add(identity)
                points.append(point)
                dimensions.append(labels)
            hashes.append(value['response_sha256'])
            truncated = truncated or value['truncated']
            captured = max(captured, value['captured_at'])
        if len(points) > 1000:
            truncated = True
            points, dimensions = points[:1000], dimensions[:1000]
        return points, dimensions, captured, truncated, hashlib.sha256(':'.join(hashes).encode()).hexdigest()

    async def samples(self, *, scope, source_id, instance_id, frozen, query):
        from .monitoring_native import public_points
        definitions = self.definitions(frozen)
        if query.metric_name not in definitions:
            raise resource_not_found('Oracle Service Metric')
        definition, mappings = definitions[query.metric_name]
        if not mappings:
            raise validation_failed('该来源未映射此服务指标')
        metric = self.metric(query.metric_name, definition, mappings)
        if query.mode == 'HISTORY' and not metric.query_supported:
            raise validation_failed('文本属性和状态不伪造数值历史')
        if query.baseline and definition['semantics'] != 'GAUGE':
            raise validation_failed('状态、属性与累计量不计算历史均值')
        adapter = await self.native._adapter(frozen, 'metric.query_range')
        now = datetime.now(UTC)
        window = resolve_monitoring_window(query.window, now=now)
        # 读取当前和基线分别占用响应预算；任何一部分截断都不参与均值比较。
        request = self.request(frozen, instance_id, scope, window.start, window.end,
            budget=frozen['max_response_bytes']//(2 if query.baseline else 1))
        baseline = None
        async with asyncio.timeout(55):
            points, dimensions, captured, truncated, digest = await self.read(adapter, request, frozen, mappings, latest=query.mode == 'LATEST')
            if query.baseline:
                baseline_window = resolve_monitoring_baseline_window(query.baseline, now=now)
                try:
                    summaries, incomplete, baseline_digest = await self.read_baseline(adapter,
                        request.model_copy(update={'window_start': baseline_window.start, 'window_end': baseline_window.end}),
                        frozen, mappings)
                    totals = {}
                    with localcontext() as context:
                        context.prec = 80
                        for summary in summaries:
                            if not summary['sample_count']:
                                continue
                            identity = tuple(sorted(summary['dimensions'].items()))
                            total, count = totals.get(identity, (Decimal(0), 0))
                            totals[identity] = total + Decimal(summary['total']), count + summary['sample_count']
                        series = tuple(MonitoringNativeBaselineSeries(dimensions=dict(key), average=str(total/count), sample_count=count)
                            for key, (total, count) in totals.items()) if not incomplete else ()
                    baseline = MonitoringNativeBaseline(period=query.baseline, start=baseline_window.start, end=baseline_window.end,
                        series=series, truncated=incomplete, response_sha256=baseline_digest,
                        unavailable_reason='历史已截断，不生成不完整均值。' if incomplete else None if series else '窗口内无有效历史采样。',
                        note='同来源、相同完整维度的实际返回样本均值；留存覆盖率未知，不作故障阈值。')
                except DiagnosticSourceAdapterError as exc:
                    baseline = MonitoringNativeBaseline(period=query.baseline, start=baseline_window.start, end=baseline_window.end,
                        unavailable_reason='历史均值不可用：' + exc.code,
                        note='未用当前值或其他来源补出历史基线。')
        await self.native._unchanged(scope, source_id, instance_id, frozen)
        return self.native._response(MonitoringNativeSamples(source_id=source_id, instance_id=instance_id, metric=metric,
            window_start=window.start if query.mode == 'HISTORY' else None, window_end=window.end if query.mode == 'HISTORY' else None,
            captured_at=captured, points=public_points(points, dimensions, captured_at=captured,
                max_age=definition['max_age_seconds'], truncated=truncated, latest=query.mode == 'LATEST'),
            query_mode=query.mode, history_resolution='LATEST_SNAPSHOT' if query.mode == 'LATEST' else 'PROVIDER_MANAGED',
            truncated=truncated, response_sha256=digest, baseline=baseline,
            warnings=('原始状态和当前采样均不构成整个服务健康结论；未知字典值保留未知。',)), frozen)

    async def read_baseline(self, adapter, request, frozen, mappings):
        budget = request.max_response_bytes // len(mappings)
        if budget < 2048:
            raise validation_failed('来源映射数超出单次基线预算')
        summaries, hashes, owners, truncated = [], [], {}, False
        for position, mapping in enumerate(mappings):
            result = await adapter.summarize_oracle_service_metric(request.model_copy(update={'max_response_bytes': budget}),
                resource_type=frozen['scope'], locator=frozen['source_locator'], mapping=mapping)
            for summary in result['series']:
                identity = tuple(sorted(summary['dimensions'].items()))
                if identity in owners and owners[identity] != position:
                    raise DiagnosticSourceAdapterError('SOURCE_ITEM_AMBIGUOUS', '多个基线映射返回了同一完整维度')
                owners[identity] = position
                summaries.append(summary)
            truncated = truncated or result['truncated']
            hashes.append(result['response_sha256'])
        return summaries, truncated, hashlib.sha256(':'.join(hashes).encode()).hexdigest()
