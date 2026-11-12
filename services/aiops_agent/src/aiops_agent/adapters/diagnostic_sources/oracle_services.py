"""三种来源的服务指标精确读取；不执行用户查询语言，不跨来源补数。"""

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import re

from .base import DiagnosticSourceAdapterError


def fail(code):
    raise DiagnosticSourceAdapterError(code, 'Oracle 服务指标身份、单位或完整维度无法核验')


def validate_locator(source_type, resource_type, key, locator):
    from platform_core.contracts.aiops.oracle_services import OEM_SERVICE_TYPES
    if not isinstance(locator, dict) or locator.get('resource_scope') != resource_type:
        fail('SOURCE_TARGET_MISMATCH')
    if source_type == 'OEM':
        if (set(locator) - {'resource_scope', 'target_id', 'target_type', 'target_name'}
                or not re.fullmatch(r'[0-9a-f]{32}', key) or locator.get('target_id') != key
                or locator.get('target_type') not in OEM_SERVICE_TYPES[resource_type]):
            fail('SOURCE_TARGET_MISMATCH')
    elif source_type == 'PROMETHEUS':
        labels = locator.get('labels')
        if (set(locator) != {'resource_scope', 'labels'} or not isinstance(labels, dict) or not 1 <= len(labels) <= 16
                or not any(labels.get(name) == key for name in ('target_key', 'instance'))):
            fail('SOURCE_TARGET_MISMATCH')
        if any(not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]{0,63}', name) or not isinstance(value, str)
               or not value or len(value) > 256 or any(ord(c) < 32 for c in value) for name, value in labels.items()):
            fail('SOURCE_TARGET_MISMATCH')
    elif source_type == 'ZABBIX':
        if set(locator) != {'resource_scope', 'host'} or locator.get('host') != key:
            fail('SOURCE_TARGET_MISMATCH')
    else:
        fail('SOURCE_QUERY_UNSUPPORTED')


class OracleServiceMetrics:
    async def _service_prometheus_summary(self, request, selector, labels, mapping):
        """由原生 sum/count 逐完整标签汇总，不传输或截断高频历史后计算均值。"""
        span = (request.window_end-request.window_start).total_seconds()
        if span != int(span) or span <= 0:
            fail('SOURCE_QUERY_INVALID')
        # Prometheus 采样时间精度为毫秒；向前一微秒使范围与 [start,end) 一致。
        endpoint = (request.window_end-timedelta(microseconds=1)).timestamp()
        summaries, hashes = [], []
        for operation in ('sum_over_time', 'count_over_time'):
            query = operation + '(' + selector + '[' + str(int(span)) + 's])'
            async with self._session.get(self._endpoint().rstrip('/') + '/api/v1/query', headers=self._headers(),
                params={'query': query, 'time': endpoint}, timeout=self._timeout) as response:
                payload, digest = await self._response_json(response, max_bytes=request.max_response_bytes//2)
            if (not isinstance(payload, dict) or payload.get('status') != 'success'
                    or not isinstance(payload.get('data'), dict) or payload['data'].get('resultType') != 'vector'
                    or not isinstance(payload['data'].get('result'), list)):
                fail('SOURCE_RESPONSE_INVALID')
            if payload.get('warnings') or len(payload['data']['result']) > 256:
                fail('SOURCE_RESULT_TRUNCATED')
            values = {}
            for row in payload['data']['result']:
                raw = row.get('metric') if isinstance(row, dict) else None
                if (not isinstance(raw, dict) or set(raw) != set(labels) | set(mapping.dimensions)
                        or any(raw.get(key) != value for key, value in labels.items())
                        or any(not isinstance(value, str) for value in raw.values())):
                    fail('SOURCE_TARGET_MISMATCH')
                key = tuple(sorted((name, raw[name]) for name in mapping.dimensions))
                if key in values:
                    fail('SOURCE_ITEM_AMBIGUOUS')
                try:
                    if not isinstance(row['value'], list) or len(row['value']) != 2:
                        raise ValueError
                    value = Decimal(str(row['value'][1]))
                    if not value.is_finite() or abs(value.as_tuple().exponent) > 1024:
                        raise ValueError
                    if operation == 'count_over_time' and (value <= 0 or value != int(value)):
                        raise ValueError
                except (KeyError, TypeError, ValueError, InvalidOperation):
                    fail('SOURCE_RESPONSE_INVALID')
                values[key] = value
            summaries.append(values)
            hashes.append(digest)
        if set(summaries[0]) != set(summaries[1]):
            fail('SOURCE_PAGE_CHANGED')
        from .oem_native import _hash
        return {'series': tuple({'dimensions': dict(key), 'total': str(total),
                'sample_count': int(summaries[1][key])} for key, total in summaries[0].items()),
            'unit': mapping.native_unit, 'truncated': False, 'response_sha256': _hash(*hashes)}

    async def summarize_oracle_service_metric(self, request, *, resource_type, locator, mapping):
        """基线专用有界汇总；任何页或字节预算不足均不发布部分均值。"""
        from .oem_native import _hash
        await self.verify_oracle_service(request.model_copy(update={'max_response_bytes': request.max_response_bytes//2}),
            resource_type=resource_type, locator=locator)
        request = request.model_copy(update={'max_response_bytes': request.max_response_bytes//2})
        self._remaining_response_bytes = request.max_response_bytes
        source_type = self.context.source_type
        if mapping.native_value_type != 'NUMBER':
            fail('SOURCE_METRIC_TYPE_MISMATCH')
        if source_type == 'PROMETHEUS':
            from .prometheus import _escape_prometheus_label
            if not re.fullmatch(r'[a-zA-Z_:][a-zA-Z0-9_:]{0,255}', mapping.metric_name):
                fail('SOURCE_QUERY_INVALID')
            selector = mapping.metric_name + '{' + ','.join(name + '="' + _escape_prometheus_label(value) + '"'
                for name, value in sorted(locator['labels'].items())) + '}'
            result = await self._service_prometheus_summary(request, selector, locator['labels'], mapping)
        elif source_type == 'ZABBIX':
            result = await self.query_native_metric(request, metric_name=mapping.metric_name, limit=1000,
                _service_summary_only=True)
        else:
            lower, summaries = request.window_start, []
            while lower < request.window_end:
                upper = min(lower+timedelta(days=6), request.window_end)
                summaries.append(await self.query_native_metric(request.model_copy(update={'window_start': lower, 'window_end': upper}),
                    metric_name=mapping.metric_name, limit=1000, _expected_target_type=locator['target_type'],
                    _expected_keys=tuple(mapping.dimensions), _paginate=True, _summary_only=True, _shared_budget=True))
                lower = upper
            if any(item['unit'] != summaries[0]['unit'] for item in summaries):
                fail('SOURCE_METADATA_CHANGED')
            result = {'series': tuple(series for item in summaries for series in item['series']),
                'unit': summaries[0]['unit'], 'truncated': any(item['truncated'] for item in summaries),
                'response_sha256': _hash(*(item['response_sha256'] for item in summaries))}
        if result['unit'] != mapping.native_unit:
            fail('SOURCE_METADATA_CHANGED')
        totals = {}
        with localcontext() as context:
            context.prec = 4096
            for series in result['series']:
                raw = series['dimensions']
                if set(raw) != set(mapping.dimensions):
                    fail('SOURCE_DIMENSIONS_MISMATCH')
                labels = {mapping.dimensions[key]: value for key, value in raw.items()} | dict(mapping.fixed_dimensions)
                identity = tuple(sorted(labels.items()))
                total, count = totals.get(identity, (Decimal(0), 0))
                totals[identity] = (total + Decimal(series['total'])*Decimal(mapping.scale), count+series['sample_count'])
        return {**result, 'series': tuple({'dimensions': dict(key), 'total': str(total), 'sample_count': count}
            for key, (total, count) in totals.items()), 'response_sha256': _hash(result['response_sha256'],
                hashlib.sha256(json.dumps(mapping.model_dump(mode='json'), sort_keys=True).encode()).hexdigest())}

    async def verify_oracle_service(self, request, *, resource_type, locator):
        validate_locator(self.context.source_type, resource_type, request.source_locator_key, locator)
        self._remaining_response_bytes = request.max_response_bytes
        if self.context.source_type == 'OEM':
            target, _ = await self._native_target(request)
            if target['targetType'] != locator['target_type'] or locator.get('target_name', target['targetName']) != target['targetName']:
                fail('SOURCE_TARGET_MISMATCH')
        elif self.context.source_type == 'ZABBIX':
            await self._metric_host(request, max_bytes=request.max_response_bytes)
        else:
            from .prometheus import _escape_prometheus_label
            selector = '{' + ','.join(name + '="' + _escape_prometheus_label(value) + '"'
                for name, value in sorted(locator['labels'].items())) + '}'
            async with self._session.get(self._endpoint().rstrip('/') + '/api/v1/series',
                headers=self._headers(), params={'match[]': selector, 'start': request.window_start.timestamp(),
                    'end': request.window_end.timestamp(), 'limit': 1001}, timeout=self._timeout) as response:
                payload, _ = await self._response_json(response, max_bytes=request.max_response_bytes)
            if (not isinstance(payload, dict) or payload.get('status') != 'success'
                    or not isinstance(payload.get('data'), list)):
                fail('SOURCE_RESPONSE_INVALID')
            if payload.get('warnings') or len(payload['data']) > 1000:
                fail('SOURCE_RESULT_TRUNCATED')
            if not payload['data']:
                fail('SOURCE_NO_DATA')
            if any(not isinstance(labels, dict) or any(labels.get(key) != value
                    for key, value in locator['labels'].items()) for labels in payload['data']):
                fail('SOURCE_TARGET_MISMATCH')

    async def _service_prometheus_samples(self, request, selector):
        """范围向量返回原始采样时间，避免 query_range 评估时间掩盖停采。"""
        from math import ceil
        seconds = ceil((request.window_end-request.window_start).total_seconds()) + 1
        async with self._session.get(self._endpoint().rstrip('/') + '/api/v1/query', headers=self._headers(),
            params={'query': selector + '[' + str(seconds) + 's]', 'time': request.window_end.timestamp()},
            timeout=self._timeout) as response:
            payload, digest = await self._response_json(response, max_bytes=request.max_response_bytes)
        if (not isinstance(payload, dict) or payload.get('status') != 'success'
                or not isinstance(payload.get('data'), dict) or payload['data'].get('resultType') != 'matrix'
                or not isinstance(payload['data'].get('result'), list)):
            fail('SOURCE_RESPONSE_INVALID')
        if payload.get('warnings'):
            fail('SOURCE_RESULT_TRUNCATED')
        result = []
        for item in payload['data']['result']:
            if not isinstance(item, dict) or not isinstance(item.get('values'), list):
                fail('SOURCE_RESPONSE_INVALID')
            try:
                points = [(datetime.fromtimestamp(float(when), tz=UTC), value) for when, value in item['values']]
            except (TypeError, ValueError, OverflowError):
                fail('SOURCE_RESPONSE_INVALID')
            result.append((item.get('metric'), points))
        return result, digest

    async def query_oracle_service_metric(self, request, *, resource_type, locator, mapping, latest=False):
        """NUMBER/STRING 原值保留，数值单位换算由审核映射明确给出。"""
        from .oem_native import _hash, parse_metric_selector
        from .oem_latest import query_latest
        from .prometheus import _escape_prometheus_label
        await self.verify_oracle_service(request.model_copy(update={'max_response_bytes': request.max_response_bytes//2}),
            resource_type=resource_type, locator=locator)
        # 身份复核与采样读取分别最多占一半，底层重新初始化预算也不会扩大总额。
        request = request.model_copy(update={'max_response_bytes': request.max_response_bytes//2})
        self._remaining_response_bytes = request.max_response_bytes
        source_type = self.context.source_type
        if source_type == 'OEM':
            parse_metric_selector(mapping.metric_name)
            if latest:
                result = await query_latest(self, request, metric_name=mapping.metric_name,
                    expected_target_type=locator['target_type'], expected_keys=tuple(mapping.dimensions),
                    expected_value_type=mapping.native_value_type, allow_empty_number=True)
            else:
                native = await self.query_native_metric(request, metric_name=mapping.metric_name, limit=1000,
                    _expected_target_type=locator['target_type'], _paginate=True)
                result = {'points': native.points, 'dimensions': native.sample_dimensions, 'unit': native.unit,
                    'native_value_type': native.native_value_type, 'captured_at': native.captured_at,
                    'truncated': native.truncated, 'response_sha256': native.response_sha256}
        elif source_type == 'ZABBIX':
            native = await self.query_native_metric(request, metric_name=mapping.metric_name, limit=1000,
                allow_text=mapping.native_value_type == 'STRING')
            numeric = native.native_value_type in {'FLOAT', 'UNSIGNED', '0', '3'}
            result = {'points': native.points, 'dimensions': tuple({} for _ in native.points), 'unit': native.unit,
                'native_value_type': 'NUMBER' if numeric else 'STRING', 'captured_at': native.captured_at,
                'truncated': native.truncated, 'response_sha256': native.response_sha256}
        else:
            if mapping.native_value_type != 'NUMBER' or not re.fullmatch(r'[a-zA-Z_:][a-zA-Z0-9_:]{0,255}', mapping.metric_name):
                fail('SOURCE_QUERY_INVALID')
            labels = locator['labels']
            selector = mapping.metric_name + '{' + ','.join(name + '="' + _escape_prometheus_label(value) + '"'
                for name, value in sorted(labels.items())) + '}'
            # 仅由精确指标名和绑定标签构造；不允许表达式、正则或回退查询。
            series, digest = await self._service_prometheus_samples(request, selector)
            points, dimensions = [], []
            for native_labels, values in series:
                if (not isinstance(native_labels, dict) or native_labels.get('__name__') != mapping.metric_name
                        or any(native_labels.get(key) != value for key, value in labels.items())
                        or set(native_labels) - {'__name__'} - set(labels) != set(mapping.dimensions)):
                    fail('SOURCE_TARGET_MISMATCH')
                for when, value in values:
                    if request.window_start <= when < request.window_end:
                        points.append((when, str(value)))
                        dimensions.append({key: native_labels[key] for key in mapping.dimensions})
            if len(points) > 1000 and not latest:
                fail('SOURCE_RESULT_TRUNCATED')
            # Prometheus 无原生单位字段；单位只由管理员审核映射声明，不冒充在线元数据。
            result = {'points': tuple(points), 'dimensions': tuple(dimensions), 'unit': mapping.native_unit,
                'native_value_type': 'NUMBER', 'captured_at': datetime.now(UTC), 'truncated': False,
                'response_sha256': digest}
        if result['unit'] != mapping.native_unit or result['native_value_type'] != mapping.native_value_type:
            fail('SOURCE_METADATA_CHANGED')
        projected, dimensions, identities = [], [], set()
        for (when, value), raw in zip(result['points'], result['dimensions'], strict=True):
            # 各来源原生端点的包含端点规则不同；服务历史统一使用冻结半开窗口。
            # OEM 最新快照可保留窗口外旧值，由新鲜度规则明确标为过期。
            if not (latest and source_type == 'OEM') and not request.window_start <= when < request.window_end:
                continue
            if set(raw) != set(mapping.dimensions):
                fail('SOURCE_DIMENSIONS_MISMATCH')
            labels = {mapping.dimensions[key]: val for key, val in raw.items()} | dict(mapping.fixed_dimensions)
            identity = (when, tuple(sorted(labels.items())))
            if identity in identities:
                fail('SOURCE_ITEM_AMBIGUOUS')
            identities.add(identity)
            if value not in (None, '') and mapping.native_value_type == 'NUMBER':
                try:
                    with localcontext() as context:
                        context.prec = 80
                        numeric = Decimal(value)
                        if not numeric.is_finite() or abs(numeric.as_tuple().exponent) > 1024:
                            raise ValueError
                        value = str(numeric * Decimal(mapping.scale))
                except (ValueError, InvalidOperation):
                    fail('SOURCE_RESPONSE_INVALID')
            projected.append((when, value))
            dimensions.append(labels)
        if latest:
            latest_points = {}
            for point, labels in zip(projected, dimensions, strict=True):
                identity = tuple(sorted(labels.items()))
                if identity not in latest_points or point[0] > latest_points[identity][0][0]:
                    latest_points[identity] = (point, labels)
            projected = [value[0] for value in latest_points.values()]
            dimensions = [value[1] for value in latest_points.values()]
        return {**result, 'points': tuple(projected), 'dimensions': tuple(dimensions),
            'response_sha256': _hash(result['response_sha256'], hashlib.sha256(json.dumps(
                mapping.model_dump(mode='json'), sort_keys=True).encode()).hexdigest())}
