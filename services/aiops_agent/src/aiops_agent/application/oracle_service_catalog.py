"""跨来源 Oracle 基础服务语义目录；只有审核映射才能执行。"""

from functools import lru_cache
from importlib.resources import files
import json
import re

from platform_core.contracts.aiops.oracle_services import OracleServiceMetricMapping


@lru_cache(maxsize=1)
def service_metrics():
    document = json.loads(files('aiops_agent.resources').joinpath('monitoring/oracle_services.v1.json').read_text())
    return {item['code']: item for item in document['metrics']}


def mappings_for(resource_type, source_type, target_type, overrides):
    """允许明确清空某项，不把缺映射改成其他指标或跨源回退。"""
    if not isinstance(overrides, dict) or set(overrides) - {'oracle_service_metrics'}:
        raise ValueError('服务指标映射只接受 oracle_service_metrics')
    configured = overrides.get('oracle_service_metrics', {})
    metrics = {code: item for code, item in service_metrics().items() if item['resource_type'] == resource_type}
    if not isinstance(configured, dict) or set(configured) - set(metrics):
        raise ValueError('服务指标映射引用了未知或其他类型指标')
    result = {}
    for code, definition in metrics.items():
        raw = configured.get(code)
        if raw is None:
            native = definition['oem']
            raw = [native['mapping']] if source_type == 'OEM' and native and native['target_type'] == target_type else []
        if not isinstance(raw, list) or len(raw) > 16:
            raise ValueError('每项指标最多允许16个精确来源映射')
        mappings = tuple(OracleServiceMetricMapping.model_validate(value) for value in raw)
        for mapping in mappings:
            if source_type == 'PROMETHEUS' and (mapping.native_value_type != 'NUMBER'
                    or not re.fullmatch(r'[a-zA-Z_:][a-zA-Z0-9_:]{0,255}', mapping.metric_name)
                    or any(not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_]{0,63}', name) for name in mapping.dimensions)):
                raise ValueError('Prometheus 服务映射只接受精确指标名、数字类型和合法标签，不接受表达式')
            if source_type == 'OEM':
                from aiops_agent.adapters.diagnostic_sources.oem_native import parse_metric_selector
                from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
                try:
                    parse_metric_selector(mapping.metric_name)
                except DiagnosticSourceAdapterError:
                    raise ValueError('OEM 服务映射必须为规范编码的 metric group/column') from None
            if set(mapping.dimensions.values()) | set(mapping.fixed_dimensions) != set(definition['dimensions']):
                raise ValueError('完整维度与服务语义不匹配')
            if definition['semantics'] == 'STATE' and (not mapping.state_values or mapping.scale != '1'):
                raise ValueError('状态指标必须显式声明原始值字典且不可换算')
            if definition['semantics'] != 'STATE' and mapping.state_values:
                raise ValueError('非状态指标不可套用状态字典')
            if definition['semantics'] in {'GAUGE', 'COUNTER'} and mapping.native_value_type != 'NUMBER':
                raise ValueError('数值语义不可绑定文本指标')
            if source_type == 'ZABBIX' and mapping.dimensions:
                raise ValueError('Zabbix 精确 Item 的维度必须显式填写 fixed_dimensions')
        result[code] = mappings
    if sum(len(values) for values in result.values()) > 64:
        raise ValueError('服务映射总数不可超过64项')
    return result
