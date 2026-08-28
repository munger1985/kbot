"""步骤 5 的只观测 Handler；不访问目标数据库或 LLM。"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from aiops_agent.contracts.artifacts import (
    MonitorScopeResult,
    ObserveReport,
)
from aiops_agent.contracts.evidence import (
    MetricDefinition,
    LogEvidenceSet,
    ObservationGap,
    ObservationSet,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_QUERY,
    CAPABILITY_LOG_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
    DiagnosticSourceContext,
    EventEvidenceRequest,
    LogEvidenceRequest,
    LogSourceLocator,
    MetricsEvidenceRequest,
)
from aiops_agent.monitoring import (
    LogQueryPolicy,
    LogQueryPolicySnapshot,
    MonitoringQueryRejected,
    PromQueryPolicy,
    PromQueryPolicySnapshot,
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
                or "${" in query.replace("${external_target}", "").replace(
                    "${host_target}", ""
                )
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
        provider = definition.providers.get("PROMETHEUS")
        if provider is not None and provider.query_template is not None:
            checked = PromQueryPolicy(PromQueryPolicySnapshot()).validate(
                provider.query_template,
                window_seconds=min(
                    definition.default_window_seconds,
                    PromQueryPolicySnapshot().max_window_seconds,
                ),
            )
            definition = definition.model_copy(
                update={
                    "providers": {
                        **definition.providers,
                        "PROMETHEUS": provider.model_copy(
                            update={
                                "query_template": checked.normalized_query
                            }
                        ),
                    }
                }
            )
        definitions.append(definition)
    return tuple(definitions)


def _ad_hoc_metric_definition(item: dict) -> MetricDefinition:
    """把已冻结 PromQL 转为只供本次调用使用的临时指标定义。"""
    validated = dict(item["validated_query"])
    return MetricDefinition.model_validate(
        {
            "metric_code": f"adhoc.{item['action_id']}",
            "semantic_version": "1.0.0",
            "name": item["question"][:128],
            "description": "本轮调查规划生成的受控 PromQL 结果",
            "unit": "value",
            "value_kind": "GAUGE",
            "expected_dimensions": [
                "instance",
                "target_key",
                "job",
                "tablespace",
                "device",
                "mountpoint",
                "event",
                "status",
                "type",
                "resource_name",
            ],
            "supported_db_types": ["ORACLE", "MYSQL", "POSTGRESQL"],
            "allowed_aggregations": ["AVG", "MAX", "LAST"],
            "default_window_seconds": validated["window_seconds"],
            "min_step_seconds": validated["step_seconds"],
            "max_points": max(
                2,
                validated["window_seconds"] // validated["step_seconds"]
                + 1,
            ),
            "max_series": validated["max_series"],
            "providers": {
                "PROMETHEUS": {
                    "template_id": f"adhoc.{item['action_id']}",
                    "template_version": "1.0.0",
                    "query_template": validated["normalized_query"],
                }
            },
        }
    )


class EvidenceScopeHandler:
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


class EvidenceObserveHandler:
    def __init__(self, *, diagnostic_source_registry, secret_store):
        self._diagnostic_sources = diagnostic_source_registry
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
                code="SOURCE_QUERY_UNSUPPORTED",
                detail="当前诊断源不支持该标准指标",
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
            source_context = DiagnosticSourceContext(
                source_id=source["source_id"],
                source_type=source["source_type"],
                adapter_id=source["adapter_id"],
                adapter_version=source["adapter_version"],
                config_version=source["config_version"],
                endpoint=source["endpoint"],
                credentials=credentials,
                declared_capabilities=source["declared_capabilities"],
                config=source["config"],
            )
            window = monitoring["window"]
        except Exception:
            gaps.append(
                ObservationGap(
                    source_id=source["source_id"],
                    binding_id=binding_id,
                    code="SOURCE_UNREACHABLE",
                    detail="诊断源配置或凭据本次不可用",
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
        if CAPABILITY_METRIC_QUERY_RANGE in snapshot[
            "effective_capabilities"
        ]:
            try:
                adapter = self._diagnostic_sources.create(
                    source_context,
                    capability=CAPABILITY_METRIC_QUERY_RANGE,
                )
                result = await adapter.query_metrics(
                    MetricsEvidenceRequest(
                        target_id=context.target_id,
                        binding_id=binding_id,
                        source_locator_key=snapshot[
                            "source_locator_key"
                        ],
                        source_locator=dict(snapshot["source_locator"]),
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
                if source["source_type"] == "PROMETHEUS":
                    ad_hoc_observations = []
                    ad_hoc_queries = tuple(
                        monitoring.get("ad_hoc_prometheus_queries", ())
                    )
                    ad_hoc_response_bytes = max(
                        1024,
                        monitoring["max_response_bytes"]
                        // max(1, len(ad_hoc_queries)),
                    )
                    for item in ad_hoc_queries:
                        try:
                            frozen = dict(item["validated_query"])
                            checked = PromQueryPolicy(
                                PromQueryPolicySnapshot()
                            ).validate(
                                frozen["normalized_query"],
                                window_seconds=frozen["window_seconds"],
                            )
                            if (
                                checked.query_sha256
                                != frozen["query_sha256"]
                                or checked.policy_sha256
                                != frozen["policy_sha256"]
                            ):
                                raise ValueError("冻结 PromQL Hash 不一致")
                        except (
                            KeyError,
                            TypeError,
                            ValueError,
                            MonitoringQueryRejected,
                        ):
                            gaps.append(
                                ObservationGap(
                                    metric_code=(
                                        f"adhoc.{item.get('action_id', 'unknown')}"
                                    ),
                                    source_id=source["source_id"],
                                    binding_id=binding_id,
                                    code="SOURCE_QUERY_INVALID",
                                    detail="冻结 PromQL 未通过执行端复核",
                                )
                            )
                            continue
                        if (
                            "HOST" in checked.target_scopes
                            and not snapshot["source_locator"].get(
                                "host_target_key"
                            )
                        ):
                            gaps.append(
                                ObservationGap(
                                    metric_code=f"adhoc.{item['action_id']}",
                                    source_id=source["source_id"],
                                    binding_id=binding_id,
                                    code="SOURCE_QUERY_UNSUPPORTED",
                                    detail="PromQL 需要主机 Target 映射",
                                )
                            )
                            continue
                        query_end = _parse_time(window["end"])
                        query_start = query_end - timedelta(
                            seconds=checked.window_seconds
                        )
                        ad_hoc_result = await adapter.query_metrics(
                            MetricsEvidenceRequest(
                                target_id=context.target_id,
                                binding_id=binding_id,
                                source_locator_key=snapshot[
                                    "source_locator_key"
                                ],
                                source_locator=dict(
                                    snapshot["source_locator"]
                                ),
                                metric_definitions=(
                                    _ad_hoc_metric_definition(item),
                                ),
                                window_start=query_start,
                                window_end=query_end,
                                requested_step_seconds=(
                                    checked.step_seconds
                                ),
                                max_response_bytes=ad_hoc_response_bytes,
                                trace_id=context.trace_id,
                            )
                        )
                        gaps.extend(ad_hoc_result.gaps)
                        ad_hoc_observations.extend(
                            observation.model_copy(
                                update={
                                    "provenance": {
                                        **observation.provenance,
                                        "query_sha256": (
                                            checked.query_sha256
                                        ),
                                        "policy_sha256": (
                                            checked.policy_sha256
                                        ),
                                        "query_policy": (
                                            "bounded-promql.v1"
                                        ),
                                    }
                                }
                            )
                            for observation in ad_hoc_result.observations
                        )
                    observations = (
                        *observations,
                        *tuple(ad_hoc_observations),
                    )
            except LookupError as exc:
                gaps.append(
                    ObservationGap(
                        source_id=source["source_id"],
                        binding_id=binding_id,
                        code="SOURCE_QUERY_UNSUPPORTED",
                        detail=str(exc),
                    )
                )
            except Exception:
                gaps.append(
                    ObservationGap(
                        source_id=source["source_id"],
                        binding_id=binding_id,
                        code="SOURCE_UNREACHABLE",
                        detail="指标证据本次采集不可用",
                        retryable=True,
                    )
                )
        if CAPABILITY_EVENT_QUERY in snapshot["effective_capabilities"]:
            try:
                adapter = self._diagnostic_sources.create(
                    source_context,
                    capability=CAPABILITY_EVENT_QUERY,
                )
                event_result = await adapter.query_events(
                    EventEvidenceRequest(
                        target_id=context.target_id,
                        binding_id=binding_id,
                        source_locator_key=snapshot[
                            "source_locator_key"
                        ],
                        window_start=_parse_time(window["start"]),
                        window_end=_parse_time(window["end"]),
                        max_events=100,
                        trace_id=context.trace_id,
                    )
                )
                gaps.extend(event_result.gaps)
                active_alerts = event_result.events
            except LookupError as exc:
                gaps.append(
                    ObservationGap(
                        source_id=source["source_id"],
                        binding_id=binding_id,
                        code="SOURCE_QUERY_UNSUPPORTED",
                        detail=str(exc),
                    )
                )
            except Exception:
                gaps.append(
                    ObservationGap(
                        source_id=source["source_id"],
                        binding_id=binding_id,
                        code="SOURCE_UNREACHABLE",
                        detail="活动事件本次采集不可用",
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


class LogEvidenceHandler:
    """按冻结 Binding 使用精确标签查询日志证据。"""

    def __init__(self, *, diagnostic_source_registry, secret_store):
        self._diagnostic_sources = diagnostic_source_registry
        self._secrets = secret_store

    async def execute(
        self, context: TaskExecutionContext
    ) -> LogEvidenceSet:
        monitoring = context.plan_snapshot["monitoring"]
        binding_id = context.task_key.removeprefix("log:")
        snapshot = next(
            item
            for item in monitoring["bindings"]
            if item["binding_id"] == binding_id
        )
        source = snapshot["source"]
        window = monitoring["window"]
        query_fingerprint = hashlib.sha256(
            (
                f"{binding_id}|{window['start']}|{window['end']}"
            ).encode("utf-8")
        ).hexdigest()
        try:
            locator = LogSourceLocator.model_validate(
                snapshot["source_locator"]
            )
            budget = dict(snapshot.get("query_budget") or {})
            max_entries = int(budget.get("max_log_entries", 200))
            max_response_bytes = int(
                budget.get(
                    "max_log_response_bytes",
                    monitoring["max_response_bytes"],
                )
            )
            if not 1 <= max_entries <= 5000:
                raise ValueError("max_log_entries 超出允许范围")
            if not 1024 <= max_response_bytes <= 20 * 1024 * 1024:
                raise ValueError("max_log_response_bytes 超出允许范围")
            credentials: dict[str, str] = {}
            if source.get("secret_ref"):
                secret = await self._secrets.resolve(source["secret_ref"])
                credentials = dict(secret.values)
                if "value" in credentials:
                    credentials["token"] = credentials["value"]
            adapter = self._diagnostic_sources.create(
                DiagnosticSourceContext(
                    source_id=source["source_id"],
                    source_type=source["source_type"],
                    adapter_id=source["adapter_id"],
                    adapter_version=source["adapter_version"],
                    config_version=source["config_version"],
                    endpoint=source["endpoint"],
                    credentials=credentials,
                    declared_capabilities=source[
                        "declared_capabilities"
                    ],
                    config=source["config"],
                ),
                capability=CAPABILITY_LOG_QUERY,
            )
            frozen_queries = tuple(
                monitoring.get("ad_hoc_log_queries", ())
            )
            if not frozen_queries:
                frozen_queries = (
                    {
                        "action_id": "baseline",
                        "validated_query": {
                            "normalized_query": "${binding_selector}",
                            "filters": (),
                            "window_seconds": int(
                                (
                                    _parse_time(window["end"])
                                    - _parse_time(window["start"])
                                ).total_seconds()
                            ),
                            "max_entries": max_entries,
                        },
                    },
                )
            entries_by_fingerprint = {}
            gaps = []
            query_hashes = []
            query_end = _parse_time(window["end"])
            per_query_entries = max(1, max_entries // len(frozen_queries))
            per_query_response_bytes = max(
                1024, max_response_bytes // len(frozen_queries)
            )
            window_starts = []
            for item in frozen_queries:
                frozen = dict(item["validated_query"])
                if item["action_id"] != "baseline":
                    try:
                        checked = LogQueryPolicy(
                            LogQueryPolicySnapshot()
                        ).validate(
                            frozen["normalized_query"],
                            window_seconds=frozen["window_seconds"],
                        )
                        if (
                            checked.query_sha256
                            != frozen["query_sha256"]
                            or checked.policy_sha256
                            != frozen["policy_sha256"]
                        ):
                            raise ValueError("冻结 LogQL Hash 不一致")
                    except (
                        KeyError,
                        TypeError,
                        ValueError,
                        MonitoringQueryRejected,
                    ):
                        gaps.append(
                            ObservationGap(
                                source_id=source["source_id"],
                                binding_id=binding_id,
                                code="SOURCE_QUERY_INVALID",
                                detail="冻结 LogQL 未通过执行端复核",
                            )
                        )
                        continue
                    filters = checked.filters
                    window_seconds = checked.window_seconds
                    query_hashes.append(checked.query_sha256)
                else:
                    filters = ()
                    window_seconds = frozen["window_seconds"]
                query_start = query_end - timedelta(
                    seconds=window_seconds
                )
                window_starts.append(query_start)
                result = await adapter.query_logs(
                    LogEvidenceRequest(
                        target_id=context.target_id,
                        binding_id=binding_id,
                        source_locator_key=snapshot["source_locator_key"],
                        selector_labels=locator.labels,
                        line_filters=filters,
                        window_start=query_start,
                        window_end=query_end,
                        max_entries=min(
                            per_query_entries,
                            int(frozen["max_entries"]),
                        ),
                        max_response_bytes=per_query_response_bytes,
                        trace_id=context.trace_id,
                    )
                )
                gaps.extend(result.gaps)
                for entry in result.entries:
                    entries_by_fingerprint[entry.entry_fingerprint] = entry
            if not window_starts:
                return LogEvidenceSet(
                    target_id=context.target_id,
                    binding_id=binding_id,
                    source_id=source["source_id"],
                    window_start=_parse_time(window["start"]),
                    window_end=query_end,
                    gaps=tuple(gaps),
                    collected_at=datetime.now(UTC),
                    query_fingerprint=query_fingerprint,
                    provenance={"query_policy": "bounded-logql.v1"},
                )
            entries = tuple(
                sorted(
                    entries_by_fingerprint.values(),
                    key=lambda item: (
                        item.observed_at,
                        item.entry_fingerprint,
                    ),
                )[-max_entries:]
            )
            combined_hash = hashlib.sha256(
                "|".join(sorted(query_hashes) or [query_fingerprint]).encode(
                    "utf-8"
                )
            ).hexdigest()
            return LogEvidenceSet(
                target_id=context.target_id,
                binding_id=binding_id,
                source_id=source["source_id"],
                window_start=min(window_starts),
                window_end=query_end,
                entries=entries,
                gaps=tuple(gaps),
                collected_at=datetime.now(UTC),
                truncated=len(entries_by_fingerprint) > len(entries),
                query_fingerprint=combined_hash,
                provenance={
                    "query_hashes": query_hashes,
                    "query_policy": "bounded-logql.v1",
                },
            )
        except LookupError as exc:
            code = "SOURCE_QUERY_UNSUPPORTED"
            detail = str(exc)
        except (KeyError, TypeError, ValueError):
            code = "SOURCE_CONFIGURATION_INVALID"
            detail = "日志定位或查询预算配置无效"
        except Exception:
            code = "SOURCE_UNREACHABLE"
            detail = "日志证据本次采集不可用"
        return LogEvidenceSet(
            target_id=context.target_id,
            binding_id=binding_id,
            source_id=source["source_id"],
            window_start=_parse_time(window["start"]),
            window_end=_parse_time(window["end"]),
            gaps=(
                ObservationGap(
                    source_id=source["source_id"],
                    binding_id=binding_id,
                    code=code,
                    detail=detail,
                    retryable=code == "SOURCE_UNREACHABLE",
                ),
            ),
            collected_at=datetime.now(UTC),
            query_fingerprint=query_fingerprint,
        )


class EvidenceReportHandler:
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
