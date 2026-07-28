"""步骤 5 的只观测 Handler；不访问目标数据库或 LLM。"""

from __future__ import annotations

from datetime import UTC, datetime

from aiops_agent.contracts.artifacts import (
    MonitorScopeResult,
    ObserveReport,
)
from aiops_agent.contracts.monitoring import (
    MetricDefinition,
    ObservationGap,
    ObservationSet,
)
from aiops_agent.ports.monitor import (
    AlertQueryRequest,
    MetricQueryRequest,
    MonitorProviderContext,
)

from .handlers import TaskExecutionContext


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        UTC
    )


def _metric_definitions(snapshot: dict) -> tuple[MetricDefinition, ...]:
    """把目标绑定的受控 Provider 查询覆盖应用到冻结指标定义。"""
    overrides = dict(snapshot.get("mapping_overrides") or {})
    prometheus_queries = overrides.get("prometheus_queries") or {}
    if not isinstance(prometheus_queries, dict):
        raise ValueError("prometheus_queries 必须是对象")
    definitions = []
    for item in snapshot["metrics"]:
        definition = MetricDefinition.model_validate(item)
        query = prometheus_queries.get(definition.metric_code)
        if query is not None:
            if (
                not isinstance(query, str)
                or not query.strip()
                or len(query) > 2000
                or "${" in query.replace("${external_target}", "")
            ):
                raise ValueError("Prometheus 指标查询覆盖格式无效")
            provider = definition.providers.get("PROMETHEUS")
            if provider is None:
                raise ValueError("指标不支持 Prometheus 查询覆盖")
            definition = definition.model_copy(
                update={
                    "providers": {
                        **definition.providers,
                        "PROMETHEUS": provider.model_copy(
                            update={
                                "template_id": (
                                    f"binding.{definition.metric_code}"
                                ),
                                "template_version": str(
                                    snapshot["binding_version"]
                                ),
                                "query_template": query.strip(),
                            }
                        ),
                    }
                }
            )
        definitions.append(definition)
    return tuple(definitions)


class MonitorScopeHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> MonitorScopeResult:
        monitoring = context.plan_snapshot["monitoring"]
        window = monitoring["window"]
        return MonitorScopeResult(
            target_id=context.target_id,
            agent_id=context.agent_id,
            trigger_type=context.trigger_type,
            window_start=_parse_time(window["start"]),
            window_end=_parse_time(window["end"]),
            catalog_version=monitoring["catalog_version"],
            catalog_hash=monitoring["catalog_hash"],
            binding_count=len(monitoring["bindings"]),
        )


class MonitorObserveHandler:
    def __init__(self, *, provider_registry, secret_store):
        self._providers = provider_registry
        self._secrets = secret_store

    async def execute(
        self, context: TaskExecutionContext
    ) -> ObservationSet:
        monitoring = context.plan_snapshot["monitoring"]
        binding_id = context.task_key.removeprefix("observe:")
        snapshot = next(
            item
            for item in monitoring["bindings"]
            if item["binding_id"] == binding_id
        )
        source = snapshot["source"]
        gaps = [
            ObservationGap(
                metric_code=code,
                source_id=source["source_id"],
                binding_id=binding_id,
                code="MONITOR_QUERY_UNSUPPORTED",
                detail="当前 Provider 不支持该标准指标",
            )
            for code in snapshot["unsupported_metrics"]
        ]
        credentials: dict[str, str] = {}
        observations = ()
        active_alerts = ()
        try:
            if source.get("secret_ref"):
                secret = await self._secrets.resolve(
                    source["secret_ref"]
                )
                credentials = secret.values
                if "value" in credentials:
                    credentials = {
                        **credentials,
                        "token": credentials["value"],
                    }
            if not source.get("endpoint"):
                raise ValueError("监控源未配置查询地址")
            adapter = self._providers.create(
                MonitorProviderContext(
                    source_id=source["source_id"],
                    source_type=source["source_type"],
                    source_version=source["source_version"],
                    endpoint=source["endpoint"],
                    credentials=credentials,
                    capabilities=source["capabilities"],
                )
            )
            window = monitoring["window"]
        except Exception:
            gaps.append(
                ObservationGap(
                    source_id=source["source_id"],
                    binding_id=binding_id,
                    code="MONITOR_UNREACHABLE",
                    detail="监控源配置或凭据本次不可用",
                    retryable=True,
                )
            )
            return ObservationSet(
                target_id=context.target_id,
                binding_id=binding_id,
                source_id=source["source_id"],
                gaps=tuple(gaps),
                collected_at=datetime.now(UTC),
            )
        try:
            result = await adapter.query_metrics(
                MetricQueryRequest(
                    target_id=context.target_id,
                    binding_id=binding_id,
                    external_target_key=snapshot[
                        "external_target_key"
                    ],
                    metric_definitions=_metric_definitions(snapshot),
                    window_start=_parse_time(window["start"]),
                    window_end=_parse_time(window["end"]),
                    requested_step_seconds=60,
                    max_response_bytes=monitoring[
                        "max_response_bytes"
                    ],
                    trace_id=context.trace_id,
                )
            )
            gaps.extend(result.gaps)
            observations = result.observations
        except Exception:
            gaps.append(
                ObservationGap(
                    source_id=source["source_id"],
                    binding_id=binding_id,
                    code="MONITOR_UNREACHABLE",
                    detail="监控指标本次采集不可用",
                    retryable=True,
                )
            )
        try:
            alert_result = await adapter.query_alerts(
                AlertQueryRequest(
                    target_id=context.target_id,
                    binding_id=binding_id,
                    external_target_key=snapshot[
                        "external_target_key"
                    ],
                    window_start=_parse_time(window["start"]),
                    window_end=_parse_time(window["end"]),
                    max_alerts=100,
                    trace_id=context.trace_id,
                )
            )
            gaps.extend(alert_result.gaps)
            active_alerts = alert_result.alerts
        except Exception:
            gaps.append(
                ObservationGap(
                    source_id=source["source_id"],
                    binding_id=binding_id,
                    code="MONITOR_UNREACHABLE",
                    detail="活动告警本次采集不可用",
                    retryable=True,
                )
            )
        return ObservationSet(
            target_id=context.target_id,
            binding_id=binding_id,
            source_id=source["source_id"],
            observations=observations,
            active_alerts=active_alerts,
            gaps=tuple(gaps),
            collected_at=datetime.now(UTC),
        )


class MonitorReportHandler:
    async def execute(
        self, context: TaskExecutionContext
    ) -> ObserveReport:
        monitoring = context.plan_snapshot["monitoring"]
        observation_artifacts = [
            item
            for item in context.input_artifacts
            if item["schema_version"] == "OBSERVATION_SET.v1"
        ]
        observations = [
            observation
            for artifact in observation_artifacts
            for observation in artifact["payload"].get(
                "observations", []
            )
        ]
        alerts = [
            alert
            for artifact in observation_artifacts
            for alert in artifact["payload"].get("active_alerts", [])
        ]
        gaps = [
            gap
            for artifact in observation_artifacts
            for gap in artifact["payload"].get("gaps", [])
        ]
        gaps.extend(monitoring.get("initial_gaps", []))
        metric_summaries = tuple(
            {
                "metric_code": item["metric_code"],
                "unit": item["unit"],
                "source_id": item["source_id"],
                "binding_id": item["binding_id"],
                "summary": item["summary"],
                "coverage_ratio": item["coverage_ratio"],
                "truncated": item["truncated"],
            }
            for item in observations
        )
        availability = tuple(
            item
            for item in metric_summaries
            if item["metric_code"] == "db.availability"
        )
        return ObserveReport(
            target_id=context.target_id,
            status="PARTIAL" if gaps else "READY",
            window_start=_parse_time(monitoring["window"]["start"]),
            window_end=_parse_time(monitoring["window"]["end"]),
            source_count=len(observation_artifacts),
            metric_count=len(observations),
            alert_count=len(alerts),
            gap_count=len(gaps),
            availability=availability,
            metric_summaries=metric_summaries,
            active_alerts=tuple(alerts),
            gaps=tuple(gaps),
            evidence_artifact_ids=tuple(
                item["artifact_id"] for item in observation_artifacts
            ),
            provenance={
                "catalog_version": monitoring["catalog_version"],
                "catalog_hash": monitoring["catalog_hash"],
                "report_builder": "monitor.report@1",
            },
        )
