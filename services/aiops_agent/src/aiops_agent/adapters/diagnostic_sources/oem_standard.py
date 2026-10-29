"""OEM标准指标的显式原生映射；不猜测组名、单位或留存粒度。"""

from decimal import Decimal, localcontext
import re

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.domain.evidence.observations import numeric_wire
from aiops_agent.playbooks import canonical_hash
from aiops_agent.ports.diagnostic_source import MetricsEvidenceResult, NativeMetricQueryRequest
from .base import DiagnosticSourceAdapterError
from .oem_native import metric_selector


class OEMMetricMapping(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    target_type: str = Field(min_length=1, max_length=64)
    metric_group: str = Field(min_length=1, max_length=64)
    metric_name: str = Field(min_length=1, max_length=64)
    native_unit: str = Field(min_length=1, max_length=64)
    unit: str = Field(min_length=1, max_length=32)
    value_kind: str = Field(pattern=r"^(GAUGE|STATE)$")
    scale: str = Field(default="1", pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
    dimensions: dict[str, str] = Field(default_factory=dict, max_length=16)
    filters: dict[str, str] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def validate_mapping(self):
        metric_selector(self.metric_group, self.metric_name)
        if any(any(ord(char) < 32 or ord(char) == 127 for char in text)
               for text in (self.target_type, self.native_unit, self.unit, *self.filters.values())):
            raise ValueError("OEM映射包含控制字符")
        if (len(self.scale) > 64 or not Decimal("1e-18") <= Decimal(self.scale) <= Decimal("1e18")
            or self.native_unit == self.unit and Decimal(self.scale) != 1):
            raise ValueError("OEM单位换算必须显式、有界；同单位不允许缩放")
        if (len(set(self.dimensions.values())) != len(self.dimensions)
            or any(not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", key)
                   for key in (*self.dimensions, *self.dimensions.values(), *self.filters))
            or any(not value or len(value) > 256 for value in self.filters.values())):
            raise ValueError("OEM维度映射必须有界、唯一且保留原生键")
        if self.value_kind == "STATE" and Decimal(self.scale) != 1:
            raise ValueError("OEM状态值不得作为普通数值缩放")
        return self


def normalize_oem_config(config):
    if set(config) - {"standard_metric_mappings"}:
        raise ValueError("OEM只接受standard_metric_mappings配置")
    mappings = config.get("standard_metric_mappings", {})
    if not isinstance(mappings, dict) or len(mappings) > 128:
        raise ValueError("OEM标准指标映射最多128项")
    normalized = {}
    for code, raw in mappings.items():
        if not isinstance(code, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{2,127}", code):
            raise ValueError("OEM标准指标编码格式无效")
        normalized[code] = OEMMetricMapping.model_validate(raw).model_dump(mode="json")
    return {"standard_metric_mappings": normalized} if "standard_metric_mappings" in config else {}


class OEMStandardMetrics:
    async def query_metrics(self, request):
        observations, gaps = [], []
        self._remaining_response_bytes = request.max_response_bytes
        try:
            config = normalize_oem_config(self.context.config)
        except (ValueError, DiagnosticSourceAdapterError):
            return MetricsEvidenceResult(gaps=(self._gap(request, metric_code=None,
                code="SOURCE_CONFIG_INVALID", detail="OEM标准指标映射无法核验"),))
        mappings = config.get("standard_metric_mappings", {})
        for definition in request.metric_definitions:
            provider = definition.providers.get("OEM")
            raw = mappings.get(definition.metric_code)
            if raw is None or provider is None:
                gaps.append(self._gap(request, metric_code=definition.metric_code,
                    code="SOURCE_MAPPING_REQUIRED",
                    detail="该OEM标准指标尚未配置精确原生组、指标及单位映射；可使用已授权原生目录发现"))
                continue
            try:
                mapping = OEMMetricMapping.model_validate(raw)
                if (mapping.unit != definition.unit or mapping.value_kind != definition.value_kind
                    or provider.target_type != mapping.target_type
                    or set(mapping.dimensions.values()) - set(definition.expected_dimensions)):
                    raise DiagnosticSourceAdapterError("SOURCE_MAPPING_MISMATCH", "OEM映射与标准指标语义不一致")
                native = await self.query_native_metric(NativeMetricQueryRequest(
                    action_id="a0", target_id=request.target_id, binding_id=request.binding_id,
                    source_locator_key=request.source_locator_key, window_start=request.window_start,
                    window_end=request.window_end, max_response_bytes=request.max_response_bytes,
                    trace_id=request.trace_id),
                    metric_name=metric_selector(mapping.metric_group, mapping.metric_name),
                    limit=min(1000, definition.max_points * definition.max_series),
                    _expected_target_type=mapping.target_type, _shared_budget=True)
                if native.unit != mapping.native_unit:
                    raise DiagnosticSourceAdapterError("SOURCE_UNIT_MISMATCH", "OEM原生单位与审核映射不一致")
                grouped = {}
                with localcontext() as context:
                    context.prec = 2200
                    for (when, value), dimensions in zip(native.points, native.sample_dimensions, strict=True):
                        if set(dimensions) != set(mapping.dimensions) | set(mapping.filters):
                            raise DiagnosticSourceAdapterError("SOURCE_DIMENSIONS_MISMATCH", "OEM原生维度未被完整保留或精确筛选")
                        if any(dimensions[key] != expected for key, expected in mapping.filters.items()):
                            continue
                        labels = tuple(sorted((target, dimensions[key]) for key, target in mapping.dimensions.items()))
                        grouped.setdefault(labels, []).append((when, numeric_wire(Decimal(value) * Decimal(mapping.scale))))
                if not grouped:
                    gaps.append(self._gap(request, metric_code=definition.metric_code,
                        code="SOURCE_NO_DATA", detail="OEM精确映射在请求窗口内没有返回匹配采样"))
                    continue
                observation = self._observation(request=request, definition=definition,
                    raw_series=[(dict(labels), points) for labels, points in sorted(grouped.items())],
                    provider_response_hash=native.response_sha256,
                    effective_step=max(request.requested_step_seconds, definition.min_step_seconds),
                    truncated=native.truncated or len(grouped) > definition.max_series
                        or any(len(points) > definition.max_points for points in grouped.values()),
                    warnings=("OEM按留存自动返回原始、小时或日聚合；采样覆盖率未知，不证明原始完整窗口、窗口增量或执行延迟分布。",
                        "原生名称及单位已与元数据核对；业务语义及换算依据为管理员配置，不等同于供应商语义认证。"),
                    provenance={"history_resolution": "PROVIDER_MANAGED", "coverage_state": "UNKNOWN",
                        "mapping_sha256": canonical_hash(mapping.model_dump(mode="json")),
                        "metric_selector": native.metric_name, "native_unit": mapping.native_unit,
                        "scale": mapping.scale, "mapping_authority": "OPERATOR_CONFIGURED"})
                # 返回点数不能证明OEM留存层的原始时间覆盖，不以请求step倒算完整率。
                observations.append(observation.model_copy(update={"expected_points": 0, "coverage_ratio": 0.0}))
            except (DiagnosticSourceAdapterError, ValueError, TypeError, ArithmeticError, aiohttp.ClientError, TimeoutError) as exc:
                gaps.append(self._gap(request, metric_code=definition.metric_code,
                    code=exc.code if isinstance(exc, DiagnosticSourceAdapterError) else "SOURCE_RESPONSE_INVALID",
                    detail="OEM标准指标的精确来源、单位、维度或响应无法核验",
                    retryable=exc.retryable if isinstance(exc, DiagnosticSourceAdapterError) else isinstance(exc, (aiohttp.ClientError, TimeoutError))))
        return MetricsEvidenceResult(observations=tuple(observations), gaps=tuple(gaps))
