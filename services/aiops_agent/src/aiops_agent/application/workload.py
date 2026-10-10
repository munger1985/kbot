"""数据库无关的工作负载快照持久化与确定性报告计算。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

from aiops_agent.application.errors import resource_not_found, validation_failed
from aiops_agent.application.postgresql_artifacts import (
    PostgreSQLArtifactStore,
    validate_pgbadger_artifact,
)
from aiops_agent.application.reporting import resolve_system_template
from aiops_agent.contracts.report import ReportContent
from aiops_agent.contracts.tool_execution import DbaToolResult
from aiops_agent.entities import (
    ActivitySampleEntity,
    OpsArtifactEntity,
    OpsRunEntity,
    OpsTaskEntity,
    ReportEntity,
    ReportSourceEntity,
    WorkloadMetricEntity,
    WorkloadSnapshotEntity,
    WorkloadStatementEntity,
)
from platform_core.contracts.aiops.workload import (
    ACTIVITY_REPORT_SCHEMA_VERSION,
    WORKLOAD_DIFF_REPORT_SCHEMA_VERSION,
    WORKLOAD_REPORT_SCHEMA_VERSION,
    ActivitySample,
    StatementIdentityType,
    WorkloadDatabaseType,
    WorkloadSnapshotCreate,
    ReportArtifactView,
    WorkloadReportType,
    ReportOrigin,
)
from platform_core.identity import uuid7


_STATEMENT_COUNTER_FIELDS = (
    "execution_count",
    "total_duration_microseconds",
    "rows_processed",
)
_SQL_LITERAL = re.compile(
    r"'(?:''|[^'])*'|\b\d+(?:\.\d+)?\b|0x[0-9a-f]+",
    re.IGNORECASE,
)


def sanitize_statement_text(value: str | None) -> str | None:
    """对服务器摘要文本做第二层字面量清理和有界化。"""

    if value is None:
        return None
    compact = " ".join(value.split())
    return _SQL_LITERAL.sub("?", compact)[:4000]


def counter_delta(
    previous: Mapping[str, int | float],
    current: Mapping[str, int | float],
    *,
    previous_instance_identity: Mapping[str, Any],
    current_instance_identity: Mapping[str, Any],
    previous_started_at,
    current_started_at,
) -> dict[str, Any]:
    """只在实例连续且所有公共计数器单调时计算差值。"""

    if (
        dict(previous_instance_identity) != dict(current_instance_identity)
        or previous_started_at != current_started_at
    ):
        return {
            "status": "DISCONTINUITY",
            "reason": "SERVER_RESTART_OR_IDENTITY_CHANGED",
            "deltas": {},
        }
    shared = sorted(set(previous) & set(current))
    resets = [
        key
        for key in shared
        if _number(current[key]) < _number(previous[key])
    ]
    if resets:
        return {
            "status": "DISCONTINUITY",
            "reason": "COUNTER_RESET",
            "reset_metrics": resets,
            "deltas": {},
        }
    return {
        "status": "CONTINUOUS",
        "deltas": {
            key: _number(current[key]) - _number(previous[key])
            for key in shared
        },
    }


def build_workload_report(
    *,
    target_id: UUID,
    snapshots: Sequence[object],
    statements: Sequence[object],
    metrics: Sequence[object] = (),
) -> dict[str, Any]:
    """从已排序快照生成可复算的单窗口工作负载报告。"""

    ordered = sorted(
        snapshots,
        key=lambda item: (item.collected_at, item.workload_snapshot_id),
    )
    if len(ordered) < 2:
        raise ValueError("工作负载报告至少需要两个快照")

    load_totals: dict[str, float] = defaultdict(float)
    instance_metric_gaps: Counter[str] = Counter()
    continuous_seconds = 0.0
    discontinuities: list[dict[str, Any]] = []
    continuous_pairs: set[tuple[UUID, UUID]] = set()
    for previous, current in zip(ordered, ordered[1:]):
        elapsed = (current.collected_at - previous.collected_at).total_seconds()
        same_instance = (
            dict(previous.instance_identity_json)
            == dict(current.instance_identity_json)
            and previous.instance_started_at == current.instance_started_at
        )
        if not same_instance or elapsed <= 0:
            discontinuities.append(
                {
                    "previous_snapshot_id": str(previous.workload_snapshot_id),
                    "current_snapshot_id": str(current.workload_snapshot_id),
                    "reason": (
                        "SERVER_RESTART_OR_IDENTITY_CHANGED"
                        if not same_instance
                        else "INVALID_INTERVAL"
                    ),
                    "reset_metrics": [],
                }
            )
            continue
        continuous_pairs.add(
            (previous.workload_snapshot_id, current.workload_snapshot_id)
        )
        continuous_seconds += elapsed
        before_metrics = dict(previous.instance_metrics_json or {})
        after_metrics = dict(current.instance_metrics_json or {})
        for key in sorted(set(before_metrics) | set(after_metrics)):
            if key not in before_metrics:
                instance_metric_gaps["NEW_METRIC"] += 1
                continue
            if key not in after_metrics:
                instance_metric_gaps["MISSING_METRIC"] += 1
                continue
            before_value = _number(before_metrics[key])
            after_value = _number(after_metrics[key])
            if after_value < before_value:
                discontinuities.append(
                    {
                        "previous_snapshot_id": str(
                            previous.workload_snapshot_id
                        ),
                        "current_snapshot_id": str(
                            current.workload_snapshot_id
                        ),
                        "reason": "COUNTER_RESET",
                        "reset_metrics": [key],
                    }
                )
                continue
            load_totals[key] += after_value - before_value

    metric_by_snapshot: dict[
        UUID, dict[tuple[str, str, str], object]
    ] = defaultdict(dict)
    for item in metrics:
        metric_by_snapshot[item.workload_snapshot_id][
            (
                str(item.metric_family),
                str(item.metric_code),
                str(item.dimension_key),
            )
        ] = item
    metric_totals: dict[str, float] = defaultdict(float)
    metric_discontinuities: list[dict[str, Any]] = []
    metric_gaps: Counter[str] = Counter()
    for previous, current in zip(ordered, ordered[1:]):
        pair = (previous.workload_snapshot_id, current.workload_snapshot_id)
        if pair not in continuous_pairs:
            continue
        left_metrics = metric_by_snapshot.get(previous.workload_snapshot_id, {})
        right_metrics = metric_by_snapshot.get(current.workload_snapshot_id, {})
        for key in sorted(set(left_metrics) | set(right_metrics)):
            if key not in left_metrics:
                metric_gaps["NEW_DIMENSION"] += 1
                continue
            if key not in right_metrics:
                metric_gaps["MISSING_DIMENSION"] += 1
                continue
            before_metric = left_metrics[key]
            after_metric = right_metrics[key]
            if (
                str(before_metric.quality_status) != "AVAILABLE"
                or str(after_metric.quality_status) != "AVAILABLE"
            ):
                metric_gaps["QUALITY_UNAVAILABLE"] += 1
                continue
            before_value = before_metric.counter_value
            after_value = after_metric.counter_value
            if before_value is None or after_value is None:
                continue
            before_number = _number(before_value)
            after_number = _number(after_value)
            if after_number < before_number:
                metric_discontinuities.append(
                    {
                        "metric_family": key[0],
                        "metric_code": key[1],
                        "dimension_key": key[2],
                        "previous_snapshot_id": str(previous.workload_snapshot_id),
                        "current_snapshot_id": str(current.workload_snapshot_id),
                        "reason": "COUNTER_RESET",
                    }
                )
                continue
            metric_totals[key[0]] += after_number - before_number

    statement_by_snapshot: dict[
        UUID, dict[tuple[str, str, str, str, int], object]
    ] = defaultdict(dict)
    for item in statements:
        statement_by_snapshot[item.workload_snapshot_id][
            (
                str(item.statement_identity_type),
                str(item.statement_identity_value),
                str(item.database_name or ""),
                str(item.user_identifier or ""),
                int(item.top_level),
            )
        ] = item
    statement_totals: dict[
        tuple[str, str, str, str, int], dict[str, Any]
    ] = {}
    statement_gaps: Counter[str] = Counter()
    for previous, current in zip(ordered, ordered[1:]):
        pair = (previous.workload_snapshot_id, current.workload_snapshot_id)
        if pair not in continuous_pairs:
            continue
        left = statement_by_snapshot.get(previous.workload_snapshot_id, {})
        right = statement_by_snapshot.get(current.workload_snapshot_id, {})
        for key in sorted(set(left) | set(right)):
            if key not in left:
                statement_gaps["NEW_ACTIVITY"] += 1
                continue
            if key not in right:
                statement_gaps["EVICTED_OR_INACTIVE"] += 1
                continue
            before = left[key]
            after = right[key]
            reset = any(
                int(getattr(after, field)) < int(getattr(before, field))
                for field in _STATEMENT_COUNTER_FIELDS
            )
            if reset:
                statement_gaps["COUNTER_RESET"] += 1
                continue
            aggregate = statement_totals.setdefault(
                key,
                {
                    "statement_identity_type": key[0],
                    "statement_identity_value": key[1],
                    "database_name": key[2],
                    "user_identifier": key[3],
                    "top_level": None if key[4] < 0 else bool(key[4]),
                    "normalized_statement": sanitize_statement_text(
                        getattr(after, "normalized_statement", None)
                    ),
                    **{field: 0 for field in _STATEMENT_COUNTER_FIELDS},
                },
            )
            for field in _STATEMENT_COUNTER_FIELDS:
                aggregate[field] += int(getattr(after, field)) - int(
                    getattr(before, field)
                )

    top_statements = sorted(
        statement_totals.values(),
        key=lambda item: (
            -int(item["total_duration_microseconds"]),
            str(item["statement_identity_value"]),
        ),
    )[:100]
    for index, item in enumerate(top_statements, start=1):
        item["rank_no"] = index
        executions = int(item["execution_count"])
        item["mean_duration_microseconds"] = (
            int(item["total_duration_microseconds"]) / executions
            if executions
            else None
        )

    coverage_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for snapshot in ordered:
        for family, quality in dict(snapshot.coverage_json or {}).items():
            coverage_counts[str(family)][str(quality)] += 1
    coverage = {
        family: {
            "available": counts.get("AVAILABLE", 0),
            "total": sum(counts.values()),
            "ratio": (
                counts.get("AVAILABLE", 0) / sum(counts.values())
                if counts
                else 0
            ),
            "statuses": dict(sorted(counts.items())),
        }
        for family, counts in sorted(coverage_counts.items())
    }
    rates = {
        key + "_per_second": value / continuous_seconds
        for key, value in sorted(load_totals.items())
        if continuous_seconds > 0
    }
    status = "READY" if continuous_seconds > 0 else "PARTIAL"
    database_type = str(ordered[0].database_type)
    return {
        "schema_version": WORKLOAD_REPORT_SCHEMA_VERSION,
        "report_type": f"{database_type}_WORKLOAD",
        "report_origin": str(ReportOrigin.AIOPS_GENERATED),
        "status": status,
        "target_id": str(target_id),
        "period_start": ordered[0].collected_at.isoformat(),
        "period_end": ordered[-1].collected_at.isoformat(),
        "snapshot_count": len(ordered),
        "database_versions": sorted(
            {
                str(getattr(item, "database_version", ""))
                for item in ordered
                if getattr(item, "database_version", None)
            }
        ),
        "catalog_versions": sorted(
            {
                str(getattr(item, "catalog_version", ""))
                for item in ordered
                if getattr(item, "catalog_version", None)
            }
        ),
        "capability_probe_versions": sorted(
            {
                str(getattr(item, "capability_probe_version", ""))
                for item in ordered
                if getattr(item, "capability_probe_version", None)
            }
        ),
        "continuous_seconds": continuous_seconds,
        "load_totals": dict(sorted(load_totals.items())),
        "load_profile": rates,
        "database_type": database_type,
        "top_statements": top_statements,
        "coverage": coverage,
        "discontinuities": discontinuities,
        "instance_metric_gaps": dict(sorted(instance_metric_gaps.items())),
        "statement_gaps": dict(sorted(statement_gaps.items())),
        "metric_totals_by_family": dict(sorted(metric_totals.items())),
        "metric_discontinuities": metric_discontinuities,
        "metric_gaps": dict(sorted(metric_gaps.items())),
    }


def build_workload_diff_report(
    *, baseline: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    """对两个已经确定性生成的 Workload Report 做差异计算。"""

    changes: dict[str, dict[str, Any]] = {}
    left = dict(baseline.get("load_profile") or {})
    right = dict(after.get("load_profile") or {})
    for key in sorted(set(left) | set(right)):
        before = float(left.get(key, 0) or 0)
        current = float(right.get(key, 0) or 0)
        if before == 0:
            changes[key] = {
                "baseline": before,
                "after": current,
                "delta": current,
                "change_status": (
                    "NEW_ACTIVITY" if current > 0 else "UNCHANGED"
                ),
                "change_percent": None,
            }
        else:
            changes[key] = {
                "baseline": before,
                "after": current,
                "delta": current - before,
                "change_status": "COMPARABLE",
                "change_percent": (current - before) / before * 100,
            }

    baseline_statements = {
        (
            item["statement_identity_type"],
            item["statement_identity_value"],
            item.get("database_name", ""),
            item.get("user_identifier", ""),
            item.get("top_level"),
        ): item
        for item in baseline.get("top_statements", [])
    }
    after_statements = {
        (
            item["statement_identity_type"],
            item["statement_identity_value"],
            item.get("database_name", ""),
            item.get("user_identifier", ""),
            item.get("top_level"),
        ): item
        for item in after.get("top_statements", [])
    }
    statement_changes = []
    for key in sorted(set(baseline_statements) | set(after_statements), key=str):
        before_item = baseline_statements.get(key)
        after_item = after_statements.get(key)
        before = int((before_item or {}).get("total_duration_microseconds", 0))
        current = int((after_item or {}).get("total_duration_microseconds", 0))
        statement_changes.append(
            {
                "statement_identity_type": key[0],
                "statement_identity_value": key[1],
                "database_name": key[2],
                "user_identifier": key[3],
                "top_level": key[4],
                "baseline_total_duration_microseconds": before,
                "after_total_duration_microseconds": current,
                "delta_total_duration_microseconds": current - before,
                "change_status": (
                    "NEW_ACTIVITY"
                    if before_item is None
                    else "NO_LONGER_TOP"
                    if after_item is None
                    else "COMPARABLE"
                ),
                "baseline_rank": (before_item or {}).get("rank_no"),
                "after_rank": (after_item or {}).get("rank_no"),
            }
        )
    statement_changes.sort(
        key=lambda item: (
            -int(item["delta_total_duration_microseconds"]),
            str(item["statement_identity_value"]),
        )
    )
    return {
        "schema_version": WORKLOAD_DIFF_REPORT_SCHEMA_VERSION,
        "report_type": f"{after.get('database_type')}_WORKLOAD_DIFF",
        "report_origin": str(ReportOrigin.AIOPS_GENERATED),
        "status": (
            "READY"
            if baseline.get("status") == after.get("status") == "READY"
            else "PARTIAL"
        ),
        "target_id": after.get("target_id"),
        "baseline_start": baseline.get("period_start"),
        "baseline_end": baseline.get("period_end"),
        "after_start": after.get("period_start"),
        "after_end": after.get("period_end"),
        "load_profile_changes": changes,
        "database_type": after.get("database_type"),
        "statement_changes": statement_changes[:100],
        "gaps": {
            "baseline_discontinuities": baseline.get(
                "discontinuities", []
            ),
            "after_discontinuities": after.get("discontinuities", []),
        },
    }


def build_activity_report(samples: Sequence[object]) -> dict[str, Any]:
    """按采样维度聚合 Activity Report，并保留覆盖率事实。"""

    if not samples:
        raise ValueError("Activity Report 至少需要一个样本")
    ordered = sorted(
        samples, key=lambda item: (item.sampled_at, item.activity_sample_id)
    )
    groups: dict[str, Counter[str]] = {
        "statement": Counter(),
        "schema": Counter(),
        "command": Counter(),
        "state": Counter(),
        "stage": Counter(),
        "wait": Counter(),
    }
    quality = Counter()
    total_weight = 0
    for item in ordered:
        weight = int(item.sample_weight)
        total_weight += weight
        quality[str(item.quality_status)] += 1
        values = {
            "statement": item.statement_identity_value,
            "schema": item.schema_name,
            "command": item.command_name,
            "state": item.session_state,
            "stage": item.stage_name,
            "wait": item.wait_name,
        }
        for dimension, value in values.items():
            if value:
                groups[dimension][str(value)] += weight
    return {
        "schema_version": ACTIVITY_REPORT_SCHEMA_VERSION,
        "report_type": f"{ordered[0].database_type}_ACTIVITY",
        "report_origin": str(ReportOrigin.AIOPS_GENERATED),
        "status": "READY",
        "period_start": ordered[0].sampled_at.isoformat(),
        "period_end": ordered[-1].sampled_at.isoformat(),
        "sample_count": len(ordered),
        "sample_weight": total_weight,
        "quality": dict(sorted(quality.items())),
        "top_activity": {
            dimension: [
                {
                    "name": name,
                    "sample_weight": count,
                    "sample_percent": (
                        count / total_weight * 100 if total_weight else 0
                    ),
                }
                for name, count in values.most_common(20)
            ]
            for dimension, values in groups.items()
        },
        "disclaimer": (
            "本报告由KBot对数据库当前活动进行周期采样后生成，"
            "不是数据库原生ASH，也不代表连续捕获的每一次执行。"
        ),
    }


class WorkloadService:
    """在Oracle UoW边界内保存快照并生成数据库工作负载报告。"""

    def __init__(
        self,
        *,
        uow_factory,
        postgresql_artifact_store: PostgreSQLArtifactStore | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._postgresql_artifact_store = postgresql_artifact_store

    async def import_pgbadger_artifact(
        self,
        *,
        domain_id: int,
        actor_id: str,
        target_id: UUID,
        period_start: datetime,
        period_end: datetime,
        file_name: str,
        content_type: str,
        body: bytes,
        idempotency_key: str,
        trace_id: str,
    ) -> ReportArtifactView:
        """校验、持久化并登记外部pgBadger静态报告。"""

        if period_start >= period_end:
            raise validation_failed("pgBadger报告起始时间必须早于结束时间")
        if self._postgresql_artifact_store is None:
            raise validation_failed("PostgreSQL外部报告存储未配置")
        validated = validate_pgbadger_artifact(
            file_name=file_name,
            content_type=content_type,
            body=body,
        )
        async with self._uow_factory() as uow:
            target = await self._require_target(
                uow=uow,
                domain_id=domain_id,
                target_id=target_id,
            )
            if str(target.db_type) != "POSTGRESQL":
                raise validation_failed("pgBadger报告只能绑定PostgreSQL Target")
            existing = await uow.runs.get_by_idempotency(
                target_id=target_id,
                trigger_type="MANUAL",
                actor_id=actor_id,
                idempotency_key=idempotency_key,
            )
            if existing is not None:
                artifact = await uow.runs.get_artifact_by_key(
                    ops_run_id=existing.ops_run_id,
                    artifact_key="postgresql-pgbadger:source",
                )
                if artifact is None:
                    raise validation_failed("幂等导入运行缺少pgBadger Artifact")
                if artifact.content_hash != validated.content_hash:
                    raise validation_failed("相同幂等键对应不同pgBadger正文")
                provenance = dict(artifact.provenance_json or {})
                if (
                    provenance.get("period_start") != period_start.isoformat()
                    or provenance.get("period_end") != period_end.isoformat()
                ):
                    raise validation_failed("相同幂等键对应不同pgBadger报告窗口")
                return _artifact_view(
                    artifact=artifact,
                    report_type=WorkloadReportType.POSTGRESQL_PGBADGER,
                    report_origin=ReportOrigin.EXTERNAL_IMPORTED,
                    title=str(
                        provenance.get("title")
                        or "PostgreSQL pgBadger Report"
                    ),
                    period_start=period_start,
                    period_end=period_end,
                )
            binding = await self._active_execution_binding(
                uow=uow,
                target=target,
                domain_id=domain_id,
            )
            if binding is None:
                raise validation_failed("Target缺少可用于登记报告的Active Agent绑定")
            now = datetime.now(UTC)
            run_id = uuid7()
            task_id = uuid7()
            artifact_id = uuid7()
            content_artifact_id = uuid7()
            report_id = uuid7()
            payload_uri = self._postgresql_artifact_store.write(
                artifact_id=artifact_id,
                artifact=validated,
            )
            title = f"{target.display_name} PostgreSQL pgBadger Report"
            safe_file_name = _safe_external_file_name(
                file_name=file_name,
                format_name=validated.format,
            )
            report_content = _pgbadger_report_content_payload(
                run_id=run_id,
                target_id=target_id,
                title=title,
                period_start=period_start,
                period_end=period_end,
                raw_artifact_id=artifact_id,
                raw_content_hash=validated.content_hash,
                format_name=validated.format,
                file_name=safe_file_name,
            )
            report_content_bytes = canonical_report_bytes(report_content)
            report_content_hash = hashlib.sha256(
                report_content_bytes
            ).hexdigest()
            run = OpsRunEntity(
                ops_run_id=run_id,
                domain_id=domain_id,
                target_id=target_id,
                agent_id=binding.agent_id,
                agent_version_id=binding.binding_id,
                trigger_type="MANUAL",
                interaction_mode="ASYNCHRONOUS",
                workflow_kind="EXTERNAL_REPORT_IMPORT",
                actor_id=actor_id,
                original_request=title,
                idempotency_key=idempotency_key,
                status="COMPLETED",
                plan_snapshot_json={
                    "report_type": "POSTGRESQL_PGBADGER",
                    "report_origin": "EXTERNAL_IMPORTED",
                },
                policy_snapshot_json={"active_content_forbidden": True},
                final_artifact_id=content_artifact_id,
                trace_id=trace_id,
                created_at=now,
                started_at=now,
                completed_at=now,
                updated_at=now,
            )
            task = _completed_report_task(
                task_id=task_id,
                run_id=run_id,
                output_artifact_id=content_artifact_id,
                task_key="postgresql-pgbadger:import",
                task_type="REPORT_IMPORT",
                handler_id="postgresql.pgbadger.import",
                input_schema_version="POSTGRESQL_PGBADGER_IMPORT.v1",
                now=now,
            )
            artifact = OpsArtifactEntity(
                artifact_id=artifact_id,
                ops_run_id=run_id,
                ops_task_id=task_id,
                artifact_key="postgresql-pgbadger:source",
                artifact_type="POSTGRESQL_PGBADGER_REPORT",
                schema_version="POSTGRESQL_PGBADGER_ARTIFACT.v1",
                payload_json=None,
                payload_uri=payload_uri,
                content_hash=validated.content_hash,
                byte_size=validated.byte_size,
                provenance_json={
                    "target_id": str(target_id),
                    "title": title,
                    "file_name": safe_file_name,
                    "content_type": validated.content_type,
                    "format": validated.format,
                    "period_start": period_start.isoformat(),
                    "period_end": period_end.isoformat(),
                    "report_origin": str(ReportOrigin.EXTERNAL_IMPORTED),
                    "validation_policy": "postgresql-pgbadger-artifact.v1",
                },
                trust_level="EXTERNAL_IMPORTED",
                security_level=1,
                created_at=now,
            )
            content_artifact = _report_content_artifact(
                artifact_id=content_artifact_id,
                run_id=run_id,
                task_id=task_id,
                artifact_key="report:postgresql-pgbadger:content:v1",
                payload=report_content,
                content_hash=report_content_hash,
                byte_size=len(report_content_bytes),
                source_artifact_id=artifact_id,
                producer="aiops.postgresql-pgbadger-import",
                producer_version="1",
                trust_level="EXTERNAL_IMPORTED",
                security_level=1,
                report_origin=ReportOrigin.EXTERNAL_IMPORTED,
                now=now,
            )
            formal_report = _formal_report(
                report_id=report_id,
                run_id=run_id,
                task_id=task_id,
                target_id=target_id,
                report_type=WorkloadReportType.POSTGRESQL_PGBADGER,
                title=title,
                report_content=report_content,
                content_artifact_id=content_artifact_id,
                content_hash=report_content_hash,
                security_level=1,
                period_start=period_start,
                period_end=period_end,
                result="READY",
                now=now,
            )
            try:
                await uow.runs.add_run(run)
                await uow.runs.add_task(task)
                await uow.runs.add_artifact(artifact)
                await uow.runs.add_artifact(content_artifact)
                await uow.inspections.add_report(formal_report)
                await uow.inspections.add_report_sources(
                    [
                        ReportSourceEntity(
                            report_id=report_id,
                            ops_run_id=run_id,
                            source_artifact_id=artifact_id,
                            source_kind="EXTERNAL_REPORT",
                            content_hash=validated.content_hash,
                            observed_at=period_end,
                            created_at=now,
                        )
                    ]
                )
                await uow.commit()
            except Exception:
                self._postgresql_artifact_store.delete(payload_uri)
                raise
            return _artifact_view(
                artifact=artifact,
                report_type=WorkloadReportType.POSTGRESQL_PGBADGER,
                report_origin=ReportOrigin.EXTERNAL_IMPORTED,
                title=title,
                period_start=period_start,
                period_end=period_end,
            )

    async def record_snapshot(
        self, *, domain_id: int, request: WorkloadSnapshotCreate
    ) -> UUID:
        async with self._uow_factory() as uow:
            target = await uow.targets.get_scoped(
                target_id=request.target_id,
                domain_id=domain_id,
                lock=True,
            )
            if target is None:
                raise resource_not_found("Target")
            if str(target.db_type) != str(request.database_type):
                raise validation_failed("快照数据库类型与Target不一致")
            existing = await uow.workloads.get_snapshot_by_schedule(
                domain_id=domain_id,
                target_id=request.target_id,
                scheduled_for=request.scheduled_for,
            )
            if existing is not None:
                if (
                    existing.schema_version != request.schema_version
                    or dict(existing.instance_identity_json)
                    != dict(request.instance_identity)
                    or existing.collected_at != request.collected_at
                ):
                    raise validation_failed(
                        "相同Target和计划时刻已存在不同的工作负载快照"
                    )
                return existing.workload_snapshot_id
            snapshot_id = uuid7()
            payload_size = len(
                json.dumps(
                    request.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            snapshot = WorkloadSnapshotEntity(
                workload_snapshot_id=snapshot_id,
                domain_id=domain_id,
                target_id=request.target_id,
                scheduled_for=request.scheduled_for,
                collected_at=request.collected_at,
                window_seconds=request.window_seconds,
                database_type=str(request.database_type),
                instance_identity_json=dict(request.instance_identity),
                instance_started_at=request.instance_started_at,
                database_version=request.database_version,
                capability_probe_version=request.capability_probe_version,
                catalog_version=request.catalog_version,
                schema_version=request.schema_version,
                continuity_json=dict(request.continuity),
                coverage_json={
                    key: str(value) for key, value in request.coverage.items()
                },
                instance_metrics_json=dict(request.instance_metrics),
                replication_metrics_json=dict(request.replication_metrics),
                status=request.status,
                error_code=request.error_code,
                byte_size=payload_size,
                expires_at=request.collected_at
                + timedelta(days=request.retention_days),
                created_at=request.collected_at,
            )
            statement_rows = [
                WorkloadStatementEntity(
                    workload_statement_id=uuid7(),
                    workload_snapshot_id=snapshot_id,
                    domain_id=domain_id,
                    target_id=request.target_id,
                    statement_identity_type=str(item.statement_identity_type),
                    statement_identity_value=item.statement_identity_value,
                    database_name=item.database_name or "__UNKNOWN__",
                    user_identifier=item.user_identifier or "__UNKNOWN__",
                    top_level=-1 if item.top_level is None else int(item.top_level),
                    normalized_statement=sanitize_statement_text(
                        item.normalized_statement
                    ),
                    execution_count=item.execution_count,
                    total_duration_microseconds=item.total_duration_microseconds,
                    mean_duration_microseconds=item.mean_duration_microseconds,
                    max_duration_microseconds=item.max_duration_microseconds,
                    rows_processed=item.rows_processed,
                    database_metrics_json=dict(item.database_metrics),
                    rank_dimension=item.rank_dimension,
                    rank_no=item.rank_no,
                    quality_status=str(item.quality_status),
                    created_at=request.collected_at,
                )
                for item in request.statements
            ]
            metric_rows = [
                WorkloadMetricEntity(
                    workload_metric_id=uuid7(),
                    workload_snapshot_id=snapshot_id,
                    domain_id=domain_id,
                    target_id=request.target_id,
                    metric_family=item.metric_family,
                    metric_code=item.metric_code,
                    dimension_key=item.dimension_key,
                    dimension_json=dict(item.dimensions),
                    counter_value=(
                        Decimal(str(item.counter_value))
                        if item.counter_value is not None
                        else None
                    ),
                    gauge_value=(
                        Decimal(str(item.gauge_value))
                        if item.gauge_value is not None
                        else None
                    ),
                    unit=item.unit,
                    quality_status=str(item.quality_status),
                    created_at=request.collected_at,
                )
                for item in request.metrics
            ]
            await uow.workloads.add_snapshot_bundle(
                snapshot=snapshot,
                statements=statement_rows,
                metrics=metric_rows,
            )
            runtime_values: dict[str, object] = {
                "workload_last_collected_at": request.collected_at,
                "workload_last_error_code": request.error_code,
            }
            if request.status == "FAILED":
                failures = int(target.workload_consecutive_failures) + 1
                policy = dict(target.workload_snapshot_policy_json or {})
                interval = int(policy.get("interval_minutes") or 5)
                retry_at = request.collected_at + timedelta(
                    minutes=min(60, interval * (2 ** min(failures, 6)))
                )
                runtime_values.update(
                    {
                        "workload_consecutive_failures": failures,
                        "workload_next_run_at": retry_at,
                    }
                )
            else:
                runtime_values["workload_consecutive_failures"] = 0
            await uow.targets.update_workload_runtime_state(
                target_id=target.target_id,
                values=runtime_values,
            )
            await uow.commit()
            return snapshot_id

    async def create_workload_report(
        self,
        *,
        domain_id: int,
        actor_id: str,
        target_id: UUID,
        period_start: datetime,
        period_end: datetime,
        idempotency_key: str,
        trace_id: str,
    ) -> ReportArtifactView:
        """从冻结快照创建正式工作负载报告。"""

        async with self._uow_factory() as uow:
            target = await self._require_target(
                uow=uow, domain_id=domain_id, target_id=target_id
            )
            snapshots = await uow.workloads.list_snapshots(
                domain_id=domain_id,
                target_id=target_id,
                period_start=period_start,
                period_end=period_end,
            )
            snapshot_ids = tuple(
                row.workload_snapshot_id for row in snapshots
            )
            statements = await uow.workloads.list_statements(
                domain_id=domain_id,
                target_id=target_id,
                workload_snapshot_ids=snapshot_ids,
            )
            metrics = await uow.workloads.list_metrics(
                domain_id=domain_id,
                target_id=target_id,
                workload_snapshot_ids=snapshot_ids,
            )
            payload = build_workload_report(
                target_id=target_id,
                snapshots=snapshots,
                statements=statements,
                metrics=metrics,
            )
            return await self._persist_report(
                uow=uow,
                domain_id=domain_id,
                actor_id=actor_id,
                target=target,
                payload=payload,
                report_type=_report_type(target.db_type, "WORKLOAD"),
                period_start=period_start,
                period_end=period_end,
                baseline_start=None,
                baseline_end=None,
                after_start=None,
                after_end=None,
                source_ids=snapshot_ids,
                idempotency_key=idempotency_key,
                trace_id=trace_id,
            )

    async def create_workload_diff_report(
        self,
        *,
        domain_id: int,
        actor_id: str,
        target_id: UUID,
        baseline_start: datetime,
        baseline_end: datetime,
        after_start: datetime,
        after_end: datetime,
        idempotency_key: str,
        trace_id: str,
    ) -> ReportArtifactView:
        """创建两个不重叠时间窗口的确定性差异报告。"""

        if not baseline_start < baseline_end <= after_start < after_end:
            raise validation_failed("对比窗口必须按时间先后且不重叠")
        async with self._uow_factory() as uow:
            target = await self._require_target(
                uow=uow, domain_id=domain_id, target_id=target_id
            )
            baseline_rows = await uow.workloads.list_snapshots(
                domain_id=domain_id,
                target_id=target_id,
                period_start=baseline_start,
                period_end=baseline_end,
            )
            after_rows = await uow.workloads.list_snapshots(
                domain_id=domain_id,
                target_id=target_id,
                period_start=after_start,
                period_end=after_end,
            )
            all_rows = (*baseline_rows, *after_rows)
            all_ids = tuple(row.workload_snapshot_id for row in all_rows)
            statements = await uow.workloads.list_statements(
                domain_id=domain_id,
                target_id=target_id,
                workload_snapshot_ids=all_ids,
            )
            metrics = await uow.workloads.list_metrics(
                domain_id=domain_id,
                target_id=target_id,
                workload_snapshot_ids=all_ids,
            )
            baseline_ids = {
                row.workload_snapshot_id for row in baseline_rows
            }
            after_ids = {row.workload_snapshot_id for row in after_rows}
            baseline = build_workload_report(
                target_id=target_id,
                snapshots=baseline_rows,
                statements=tuple(
                    row
                    for row in statements
                    if row.workload_snapshot_id in baseline_ids
                ),
                metrics=tuple(
                    row
                    for row in metrics
                    if row.workload_snapshot_id in baseline_ids
                ),
            )
            after = build_workload_report(
                target_id=target_id,
                snapshots=after_rows,
                statements=tuple(
                    row
                    for row in statements
                    if row.workload_snapshot_id in after_ids
                ),
                metrics=tuple(
                    row
                    for row in metrics
                    if row.workload_snapshot_id in after_ids
                ),
            )
            payload = build_workload_diff_report(
                baseline=baseline,
                after=after,
            )
            return await self._persist_report(
                uow=uow,
                domain_id=domain_id,
                actor_id=actor_id,
                target=target,
                payload=payload,
                report_type=_report_type(target.db_type, "WORKLOAD_DIFF"),
                period_start=baseline_start,
                period_end=after_end,
                baseline_start=baseline_start,
                baseline_end=baseline_end,
                after_start=after_start,
                after_end=after_end,
                source_ids=all_ids,
                idempotency_key=idempotency_key,
                trace_id=trace_id,
            )

    async def record_activity_samples(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        sampled_at: datetime,
        database_type: str,
        instance_identity: Mapping[str, Any],
        samples: Sequence[Mapping[str, Any]],
        retention_days: int = 7,
    ) -> tuple[UUID, ...]:
        if not 1 <= retention_days <= 30:
            raise validation_failed("活动样本保留天数必须在1到30天之间")
        if len(samples) > 1000:
            raise validation_failed("单次活动采样不能超过1000行")
        async with self._uow_factory() as uow:
            target = await self._require_target(
                uow=uow,
                domain_id=domain_id,
                target_id=target_id,
                lock=True,
            )
            normalized_database_type = WorkloadDatabaseType(database_type)
            if str(target.db_type) != str(normalized_database_type):
                raise validation_failed("活动样本数据库类型与Target不一致")
            expires_at = sampled_at + timedelta(days=retention_days)
            rows: list[ActivitySampleEntity] = []
            payload_bytes = 0
            for raw_item in samples:
                item = ActivitySample.model_validate(raw_item)
                item_payload = item.model_dump(mode="json")
                item_bytes = len(
                    json.dumps(
                        item_payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                )
                payload_bytes += item_bytes
                rows.append(
                    ActivitySampleEntity(
                        activity_sample_id=uuid7(),
                        domain_id=domain_id,
                        target_id=target_id,
                        database_type=str(normalized_database_type),
                        sampled_at=sampled_at,
                        instance_identity_json=dict(instance_identity),
                        session_identifier=_bounded_optional(item.session_identifier, 128),
                        worker_identifier=_bounded_optional(item.worker_identifier, 128),
                        statement_identity_type=_bounded_optional(
                            item.statement_identity_type, 32
                        ),
                        statement_identity_value=_bounded_optional(
                            item.statement_identity_value, 128
                        ),
                        database_name=_bounded_optional(item.database_name, 128),
                        schema_name=_bounded_optional(item.schema_name, 128),
                        username_hash=_hash_dimension(item.username),
                        client_hash=_hash_dimension(item.client),
                        command_name=_bounded_optional(item.command_name, 64),
                        session_state=_bounded_optional(item.session_state, 256),
                        stage_name=_bounded_optional(item.stage_name, 256),
                        wait_name=_bounded_optional(item.wait_name, 256),
                        database_sample_json=dict(item.database_sample),
                        transaction_active=int(item.transaction_active),
                        lock_waiting=int(item.lock_waiting),
                        sample_weight=max(1, min(int(item.sample_weight), 1000)),
                        quality_status=str(item.quality_status),
                        byte_size=item_bytes,
                        expires_at=expires_at,
                        created_at=sampled_at,
                    )
                )
            bucket_date = sampled_at.date()
            current_bytes = (
                int(target.activity_daily_bytes)
                if target.activity_daily_bucket is not None
                and target.activity_daily_bucket.date() == bucket_date
                else 0
            )
            daily_limit = int(
                dict(target.activity_sampler_policy_json or {}).get(
                    "max_daily_bytes", 100 * 1024 * 1024
                )
            )
            if current_bytes + payload_bytes > daily_limit:
                await uow.targets.update_workload_runtime_state(
                    target_id=target_id,
                    values={
                        "activity_sampler_status": "DEGRADED",
                        "activity_sampler_disabled_reason": "DAILY_BYTE_BUDGET_EXCEEDED",
                        "activity_next_sample_at": None,
                        "activity_consecutive_failures": int(
                            target.activity_consecutive_failures
                        )
                        + 1,
                    },
                )
                await uow.commit()
                raise validation_failed("活动采样超过Target每日字节预算")
            await uow.workloads.add_activity_samples(rows)
            bucket_start = sampled_at.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            await uow.targets.update_workload_runtime_state(
                target_id=target_id,
                values={
                    "activity_daily_bucket": bucket_start,
                    "activity_daily_bytes": current_bytes + payload_bytes,
                    "activity_last_sampled_at": sampled_at,
                    "activity_consecutive_failures": 0,
                    "activity_sampler_status": "READY",
                    "activity_sampler_disabled_reason": None,
                },
            )
            await uow.commit()
            return tuple(row.activity_sample_id for row in rows)

    async def record_activity_failure(
        self,
        *,
        domain_id: int,
        target_id: UUID,
        failed_at: datetime,
        error_code: str,
    ) -> None:
        async with self._uow_factory() as uow:
            target = await self._require_target(
                uow=uow,
                domain_id=domain_id,
                target_id=target_id,
                lock=True,
            )
            failures = int(target.activity_consecutive_failures) + 1
            interval = int(
                dict(target.activity_sampler_policy_json or {}).get(
                    "interval_seconds", 5
                )
            )
            await uow.targets.update_workload_runtime_state(
                target_id=target_id,
                values={
                    "activity_consecutive_failures": failures,
                    "activity_sampler_status": "DEGRADED",
                    "activity_sampler_disabled_reason": error_code[:128],
                    "activity_next_sample_at": (
                        None
                        if failures >= 3
                        else failed_at
                        + timedelta(seconds=min(300, interval * (2**failures)))
                    ),
                },
            )
            await uow.commit()

    async def create_activity_report(
        self,
        *,
        domain_id: int,
        actor_id: str,
        target_id: UUID,
        period_start: datetime,
        period_end: datetime,
        idempotency_key: str,
        trace_id: str,
    ) -> ReportArtifactView:
        """聚合受控活动样本并创建正式活动报告。"""

        async with self._uow_factory() as uow:
            target = await self._require_target(
                uow=uow, domain_id=domain_id, target_id=target_id
            )
            samples = await uow.workloads.list_activity_samples(
                domain_id=domain_id,
                target_id=target_id,
                period_start=period_start,
                period_end=period_end,
                limit=100_000,
            )
            payload = build_activity_report(samples)
            payload["target_id"] = str(target_id)
            return await self._persist_report(
                uow=uow,
                domain_id=domain_id,
                actor_id=actor_id,
                target=target,
                payload=payload,
                report_type=_report_type(target.db_type, "ACTIVITY"),
                period_start=period_start,
                period_end=period_end,
                baseline_start=None,
                baseline_end=None,
                after_start=None,
                after_end=None,
                source_ids=tuple(
                    row.activity_sample_id for row in samples
                ),
                idempotency_key=idempotency_key,
                trace_id=trace_id,
            )

    async def get_report_artifact_content(
        self,
        *,
        domain_id: int,
        artifact_id: UUID,
    ) -> tuple[bytes, str, str]:
        """按Domain边界读取不可变报告Artifact正文。"""

        async with self._uow_factory() as uow:
            artifact = await uow.runs.get_artifact_scoped(
                domain_id=domain_id,
                artifact_id=artifact_id,
            )
            if artifact is None:
                raise resource_not_found("Artifact")
            provenance = dict(artifact.provenance_json or {})
            if artifact.payload_json is None:
                if (
                    artifact.artifact_type == "POSTGRESQL_PGBADGER_REPORT"
                    and artifact.payload_uri is not None
                    and self._postgresql_artifact_store is not None
                ):
                    content = self._postgresql_artifact_store.read(
                        artifact.payload_uri,
                        expected_hash=artifact.content_hash,
                    )
                    return (
                        content,
                        str(
                            provenance.get("content_type")
                            or "application/octet-stream"
                        ),
                        _safe_external_file_name(
                            file_name=str(
                                provenance.get("file_name")
                                or f"pgbadger-{artifact.artifact_id}"
                            ),
                            format_name=str(
                                provenance.get("format") or "JSON"
                            ),
                        ),
                    )
                raise resource_not_found("Artifact content")
            if artifact.schema_version == "DBA_TOOL_RESULT.v1":
                result = DbaToolResult.model_validate(artifact.payload_json)
                reports: list[tuple[str, bytes]] = []
                for outcome in result.tool_outcomes:
                    observation = outcome.observation
                    if (
                        observation is None
                        or observation.truncated
                        or len(observation.columns) != 1
                        or observation.columns[0].name != "output"
                    ):
                        continue
                    body = "".join(
                        str(row[0])
                        for row in observation.rows
                        if row and row[0] is not None
                    ).encode("utf-8")
                    if body:
                        reports.append((outcome.tool_id, body))
                if (
                    len(reports) != 1
                    or not reports[0][1].lstrip().lower().startswith(
                        (b"<!doctype html", b"<html")
                    )
                ):
                    raise resource_not_found("Artifact content")
                tool_id, content = reports[0]
                file_name = _safe_file_name(
                    str(
                        provenance.get("file_name")
                        or f"{tool_id.replace('.', '-')}-{artifact.artifact_id}.html"
                    ),
                    suffix=".html",
                )
                return content, "text/html", file_name
            content = canonical_report_bytes(artifact.payload_json)
            file_name = _safe_file_name(
                str(
                    provenance.get("file_name")
                    or f"{artifact.artifact_type.lower()}-{artifact.artifact_id}.json"
                ),
                suffix=".json",
            )
            return (
                content,
                str(provenance.get("content_type") or "application/json"),
                file_name,
            )

    async def _persist_report(
        self,
        *,
        uow,
        domain_id: int,
        actor_id: str,
        target,
        payload: dict[str, Any],
        report_type: WorkloadReportType,
        period_start: datetime,
        period_end: datetime,
        baseline_start: datetime | None,
        baseline_end: datetime | None,
        after_start: datetime | None,
        after_end: datetime | None,
        source_ids: tuple[UUID, ...],
        idempotency_key: str,
        trace_id: str,
    ) -> ReportArtifactView:
        existing = await uow.runs.get_by_idempotency(
            target_id=target.target_id,
            trigger_type="MANUAL",
            actor_id=actor_id,
            idempotency_key=idempotency_key,
        )
        if existing is not None:
            artifact = await uow.runs.get_artifact_by_key(
                ops_run_id=existing.ops_run_id,
                artifact_key="workload-report:json",
            )
            if artifact is None:
                raise validation_failed("幂等报告运行缺少结果Artifact")
            return _artifact_view(
                artifact=artifact,
                report_type=report_type,
                title=str(
                    dict(artifact.provenance_json or {}).get("title")
                    or report_type
                ),
                period_start=period_start,
                period_end=period_end,
            )
        binding = await self._active_execution_binding(
            uow=uow,
            target=target,
            domain_id=domain_id,
        )
        if binding is None:
            raise validation_failed("Target缺少可用于生成报告的Active Agent绑定")
        now = datetime.now(UTC)
        run_id = uuid7()
        task_id = uuid7()
        artifact_id = uuid7()
        content_artifact_id = uuid7()
        report_id = uuid7()
        content = canonical_report_bytes(payload)
        content_hash = hashlib.sha256(content).hexdigest()
        report_slug = str(report_type).lower().replace("_", "-")
        target_slug = re.sub(
            r"[^A-Za-z0-9._-]+", "-", str(target.display_name)
        ).strip("-._") or "database-target"
        file_name = _safe_file_name(
            f"{target_slug}-{report_slug}-{period_start:%Y%m%d%H%M}-"
            f"{period_end:%Y%m%d%H%M}.json",
            suffix=".json",
        )
        title = f"{target.display_name} {_report_type_name(report_type)}"
        report_content = _report_content_payload(
            report_type=report_type,
            run_id=run_id,
            target_id=target.target_id,
            title=title,
            payload=payload,
            period_start=period_start,
            period_end=period_end,
            raw_artifact_id=artifact_id,
            raw_content_hash=content_hash,
            source_ids=source_ids,
        )
        report_content_bytes = canonical_report_bytes(report_content)
        report_content_hash = hashlib.sha256(
            report_content_bytes
        ).hexdigest()
        run = OpsRunEntity(
            ops_run_id=run_id,
            domain_id=domain_id,
            target_id=target.target_id,
            agent_id=binding.agent_id,
            agent_version_id=binding.binding_id,
            trigger_type="MANUAL",
            interaction_mode="ASYNCHRONOUS",
            workflow_kind="WORKLOAD_REPORT",
            actor_id=actor_id,
            original_request=title,
            idempotency_key=idempotency_key,
            status="COMPLETED",
            plan_snapshot_json={
                "report_type": str(report_type),
                "source_ids": [str(value) for value in source_ids],
            },
            policy_snapshot_json={"deterministic_report": True},
            final_artifact_id=content_artifact_id,
            trace_id=trace_id,
            created_at=now,
            started_at=now,
            completed_at=now,
            updated_at=now,
        )
        task = _completed_report_task(
            task_id=task_id,
            run_id=run_id,
            output_artifact_id=content_artifact_id,
            task_key="workload-report:build",
            task_type="REPORT_BUILD",
            handler_id="workload.report.build",
            input_schema_version="WORKLOAD_REPORT_REQUEST.v2",
            now=now,
        )
        artifact = OpsArtifactEntity(
            artifact_id=artifact_id,
            ops_run_id=run_id,
            ops_task_id=task_id,
            artifact_key="workload-report:json",
            artifact_type=f"{report_type}_REPORT",
            schema_version=str(payload["schema_version"]),
            payload_json=payload,
            payload_uri=None,
            content_hash=content_hash,
            byte_size=len(content),
            provenance_json={
                "target_id": str(target.target_id),
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "source_ids": [str(value) for value in source_ids],
                "report_origin": str(ReportOrigin.AIOPS_GENERATED),
                "redaction_policy_version": "database-statement-redaction.v2",
                "catalog_versions": list(
                    payload.get("catalog_versions") or ()
                ),
                "capability_probe_versions": list(
                    payload.get("capability_probe_versions") or ()
                ),
                "coverage": dict(payload.get("coverage") or {}),
                "title": title,
                "file_name": file_name,
            },
            trust_level="SOURCE_VERIFIED",
            security_level=1,
            created_at=now,
        )
        content_artifact = _report_content_artifact(
            artifact_id=content_artifact_id,
            run_id=run_id,
            task_id=task_id,
            artifact_key="report:workload:content:v2",
            payload=report_content,
            content_hash=report_content_hash,
            byte_size=len(report_content_bytes),
            source_artifact_id=artifact_id,
            producer="aiops.workload-report-builder",
            producer_version="2",
            trust_level="SOURCE_VERIFIED",
            security_level=1,
            report_origin=ReportOrigin.AIOPS_GENERATED,
            now=now,
        )
        formal_report = _formal_report(
            report_id=report_id,
            run_id=run_id,
            task_id=task_id,
            target_id=target.target_id,
            report_type=report_type,
            title=title,
            report_content=report_content,
            content_artifact_id=content_artifact_id,
            content_hash=report_content_hash,
            security_level=1,
            period_start=period_start,
            period_end=period_end,
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            after_start=after_start,
            after_end=after_end,
            result=str(payload.get("status") or "READY"),
            now=now,
        )
        await uow.runs.add_run(run)
        await uow.runs.add_task(task)
        await uow.runs.add_artifact(artifact)
        await uow.runs.add_artifact(content_artifact)
        await uow.inspections.add_report(formal_report)
        await uow.inspections.add_report_sources(
            [
                ReportSourceEntity(
                    report_id=report_id,
                    ops_run_id=run_id,
                    source_artifact_id=artifact_id,
                    source_kind="WORKLOAD",
                    content_hash=content_hash,
                    observed_at=period_end,
                    created_at=now,
                )
            ]
        )
        await uow.commit()
        return _artifact_view(
            artifact=artifact,
            report_type=report_type,
            title=title,
            period_start=period_start,
            period_end=period_end,
        )

    @staticmethod
    async def _active_execution_binding(*, uow, target, domain_id: int):
        for target_binding in await uow.targets.list_agent_bindings(
            target_id=target.target_id,
            domain_id=domain_id,
        ):
            if str(target_binding.status) != "ACTIVE":
                continue
            binding = await uow.agents.get_active(
                domain_id=domain_id,
                agent_id=target_binding.agent_id,
                target_id=target.target_id,
            )
            if binding is not None:
                return binding
        return None

    async def cleanup_expired(
        self, *, now: datetime, limit: int = 1000
    ) -> dict[str, int]:
        async with self._uow_factory() as uow:
            result = await uow.workloads.delete_expired(now=now, limit=limit)
            await uow.commit()
            return result

    @staticmethod
    async def _require_target(
        *, uow, domain_id: int, target_id: UUID, lock: bool = False
    ):
        target = await uow.targets.get_scoped(
            target_id=target_id,
            domain_id=domain_id,
            lock=lock,
        )
        if target is None:
            raise resource_not_found("Target")
        if str(target.db_type) not in {"MYSQL", "POSTGRESQL"}:
            raise validation_failed("仅MySQL或PostgreSQL Target支持工作负载采集")
        return target


def _number(value: object) -> float:
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


def _hash_dimension(value: object) -> str | None:
    if value is None or str(value) == "":
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _bounded_optional(value: object, maximum: int) -> str | None:
    text = str(value or "").strip()
    return text[:maximum] if text else None


def canonical_report_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def report_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_report_bytes(payload)).hexdigest()


def _report_type(database_type: object, kind: str) -> WorkloadReportType:
    return WorkloadReportType(f"{str(database_type).upper()}_{kind}")


def _artifact_view(
    *,
    artifact,
    report_type: WorkloadReportType,
    title: str,
    period_start: datetime,
    period_end: datetime,
    report_origin: ReportOrigin = ReportOrigin.AIOPS_GENERATED,
) -> ReportArtifactView:
    provenance = dict(artifact.provenance_json or {})
    format_name = str(provenance.get("format") or "JSON")
    content_type = str(
        provenance.get("content_type") or "application/json"
    )
    default_name = (
        "pgbadger-report.html"
        if format_name == "HTML"
        else "workload-report.json"
    )
    return ReportArtifactView(
        artifact_id=artifact.artifact_id,
        report_type=report_type,
        report_origin=report_origin,
        title=title,
        period_start=period_start,
        period_end=period_end,
        format=format_name,
        content_type=content_type,
        file_name=(
            _safe_external_file_name(
                file_name=str(provenance.get("file_name") or default_name),
                format_name=format_name,
            )
            if report_origin == ReportOrigin.EXTERNAL_IMPORTED
            else _safe_file_name(
                str(provenance.get("file_name") or default_name),
                suffix=".json",
            )
        ),
        byte_size=int(artifact.byte_size),
        content_hash=str(artifact.content_hash),
        download_url=f"/artifacts/{artifact.artifact_id}/content",
    )


def _report_template_id(report_type: WorkloadReportType) -> str:
    database_type, kind = str(report_type).split("_", 1)
    return f"system:{database_type.lower()}.{kind.lower()}"


def _report_key(report_type: WorkloadReportType) -> str:
    database_type, kind = str(report_type).split("_", 1)
    return f"{database_type.lower()}:{kind.lower()}"


def _report_type_name(report_type: WorkloadReportType) -> str:
    database_type, kind = str(report_type).split("_", 1)
    database_name = "PostgreSQL" if database_type == "POSTGRESQL" else "MySQL"
    return f"{database_name} {kind.replace('_', ' ').title()} Report"


def _completed_report_task(
    *,
    task_id: UUID,
    run_id: UUID,
    output_artifact_id: UUID,
    task_key: str,
    task_type: str,
    handler_id: str,
    input_schema_version: str,
    now: datetime,
) -> OpsTaskEntity:
    return OpsTaskEntity(
        ops_task_id=task_id,
        ops_run_id=run_id,
        task_key=task_key,
        task_type=task_type,
        handler_id=handler_id,
        handler_version="1",
        input_schema_version=input_schema_version,
        output_schema_version="REPORT_CONTENT.v1",
        depends_on_json=[],
        input_artifacts_json=[],
        output_artifact_id=output_artifact_id,
        status="COMPLETED",
        priority=50,
        available_at=now,
        attempt_count=1,
        max_attempts=1,
        timeout_seconds=120,
        started_at=now,
        completed_at=now,
        created_at=now,
        updated_at=now,
    )


def _report_content_artifact(
    *,
    artifact_id: UUID,
    run_id: UUID,
    task_id: UUID,
    artifact_key: str,
    payload: dict[str, Any],
    content_hash: str,
    byte_size: int,
    source_artifact_id: UUID,
    producer: str,
    producer_version: str,
    trust_level: str,
    security_level: int,
    report_origin: ReportOrigin,
    now: datetime,
) -> OpsArtifactEntity:
    return OpsArtifactEntity(
        artifact_id=artifact_id,
        ops_run_id=run_id,
        ops_task_id=task_id,
        artifact_key=artifact_key,
        artifact_type="REPORT_CONTENT",
        schema_version="REPORT_CONTENT.v1",
        payload_json=payload,
        payload_uri=None,
        content_hash=content_hash,
        byte_size=byte_size,
        provenance_json={
            "producer": producer,
            "producer_version": producer_version,
            "source_artifact_id": str(source_artifact_id),
            "report_origin": str(report_origin),
        },
        trust_level=trust_level,
        security_level=security_level,
        created_at=now,
    )


def _formal_report(
    *,
    report_id: UUID,
    run_id: UUID,
    task_id: UUID,
    target_id: UUID,
    report_type: WorkloadReportType,
    title: str,
    report_content: dict[str, Any],
    content_artifact_id: UUID,
    content_hash: str,
    security_level: int,
    period_start: datetime,
    period_end: datetime,
    result: str,
    now: datetime,
    baseline_start: datetime | None = None,
    baseline_end: datetime | None = None,
    after_start: datetime | None = None,
    after_end: datetime | None = None,
) -> ReportEntity:
    return ReportEntity(
        report_id=report_id,
        ops_run_id=str(run_id),
        target_id=str(target_id),
        report_key=_report_key(report_type),
        report_version=1,
        is_current=1,
        report_type=str(report_type),
        title=title,
        status=str(report_content["status"]),
        period_start=period_start,
        period_end=period_end,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        after_start=after_start,
        after_end=after_end,
        result=result,
        template_id=_report_template_id(report_type),
        template_version="1",
        generated_by_task_id=task_id,
        content_artifact_id=content_artifact_id,
        content_hash=content_hash,
        summary=str(report_content["summary"]),
        security_level=security_level,
        schema_version="REPORT_CONTENT.v1",
        created_at=now,
        updated_at=now,
    )


def _pgbadger_report_content_payload(
    *,
    run_id: UUID,
    target_id: UUID,
    title: str,
    period_start: datetime,
    period_end: datetime,
    raw_artifact_id: UUID,
    raw_content_hash: str,
    format_name: str,
    file_name: str,
) -> dict[str, Any]:
    report_type = WorkloadReportType.POSTGRESQL_PGBADGER
    template = resolve_system_template(_report_template_id(report_type))
    if template is None:
        raise RuntimeError("pgBadger报告模板未注册")
    content = ReportContent(
        report_key=_report_key(report_type),
        report_type=str(report_type),
        ops_run_id=str(run_id),
        target_id=str(target_id),
        title=title,
        status="READY",
        summary=(
            "外部pgBadger报告已通过大小、格式、解压、Hash和主动内容校验，"
            "原始结论仍保持EXTERNAL_IMPORTED信任级别。"
        ),
        period_start=period_start,
        period_end=period_end,
        scope={
            "report_origin": str(ReportOrigin.EXTERNAL_IMPORTED),
            "format": format_name,
            "file_name": file_name,
            "raw_report_artifact_id": str(raw_artifact_id),
        },
        facts=(
            {
                "kind": "external_postgresql_report",
                "summary": f"已导入静态pgBadger {format_name}报告。",
            },
        ),
        gaps=(
            {
                "code": "EXTERNAL_SOURCE_NOT_SOURCE_VERIFIED",
                "detail": (
                    "报告内容来自外部pgBadger，不自动提升为AIOps SOURCE_VERIFIED事实。"
                ),
            },
        ),
        evidence_refs=(
            {
                "artifact_id": str(raw_artifact_id),
                "content_hash": raw_content_hash,
                "schema_version": "POSTGRESQL_PGBADGER_ARTIFACT.v1",
                "source_kind": "EXTERNAL_REPORT",
            },
        ),
        provenance={
            "deterministic": True,
            "llm_used": False,
            "report_origin": str(ReportOrigin.EXTERNAL_IMPORTED),
            "template": _template_provenance(template),
        },
    )
    return content.model_dump(mode="json")


def _report_content_payload(
    *,
    report_type: WorkloadReportType,
    run_id: UUID,
    target_id: UUID,
    title: str,
    payload: Mapping[str, Any],
    period_start: datetime,
    period_end: datetime,
    raw_artifact_id: UUID,
    raw_content_hash: str,
    source_ids: tuple[UUID, ...],
) -> dict[str, Any]:
    report_kind = str(report_type).split("_", 1)[1]
    if report_kind == "ACTIVITY":
        quality = dict(payload.get("quality") or {})
        status = (
            "READY"
            if int(payload.get("sample_count") or 0) > 0
            and not any(key != "AVAILABLE" for key in quality)
            else "PARTIAL"
        )
    else:
        status = "READY" if payload.get("status") == "READY" else "PARTIAL"
    gaps: list[dict[str, Any]] = []
    for item in payload.get("discontinuities") or ():
        gaps.append({"code": "DISCONTINUITY", **dict(item)})
    for family in ("statement_gaps", "instance_metric_gaps", "metric_gaps"):
        for key, count in dict(payload.get(family) or {}).items():
            if count:
                gaps.append(
                    {"code": f"{family.upper()}_{key}", "count": int(count)}
                )
    for item in payload.get("metric_discontinuities") or ():
        gaps.append({"code": "METRIC_COUNTER_RESET", **dict(item)})
    if report_kind == "WORKLOAD_DIFF":
        for key, items in dict(payload.get("gaps") or {}).items():
            for item in items or ():
                gaps.append({"code": str(key).upper(), **dict(item)})
    facts: list[dict[str, Any]] = []
    recommendations: list[str] = []
    trends: list[str] = []
    coverage_summary = ""
    if report_kind == "WORKLOAD":
        for key, value in sorted(
            dict(payload.get("load_profile") or {}).items()
        ):
            facts.append(
                {
                    "kind": "database_load_profile",
                    "summary": f"{key}: {float(value):.6f}",
                }
            )
        for item in payload.get("top_statements") or ():
            facts.append(
                {
                    "kind": "database_top_statement",
                    "summary": (
                        f"语句 {item.get('statement_identity_value')} 总耗时 "
                        f"{item.get('total_duration_microseconds')} us，"
                        f"执行 {item.get('execution_count')} 次"
                    ),
                }
            )
        coverage_summary = "；".join(
            f"{key} {float(value.get('ratio') or 0) * 100:.1f}%"
            for key, value in sorted(
                dict(payload.get("coverage") or {}).items()
            )
        ) or "未形成可用采集覆盖率"
    elif report_kind == "WORKLOAD_DIFF":
        for key, item in sorted(
            dict(payload.get("load_profile_changes") or {}).items()
        ):
            trends.append(
                f"{key}: {item.get('baseline')} → {item.get('after')}，"
                f"状态 {item.get('change_status')}"
            )
        for item in payload.get("statement_changes") or ():
            facts.append(
                {
                    "kind": "database_statement_change",
                    "summary": (
                        f"语句 {item.get('statement_identity_value')} 耗时变化 "
                        f"{item.get('delta_total_duration_microseconds')} us，"
                        f"状态 {item.get('change_status')}"
                    ),
                }
            )
    else:
        coverage_summary = (
            f"共 {int(payload.get('sample_count') or 0)} 个样本，"
            f"样本权重 {int(payload.get('sample_weight') or 0)}"
        )
        for dimension, items in dict(
            payload.get("top_activity") or {}
        ).items():
            for item in list(items)[:5]:
                facts.append(
                    {
                        "kind": "database_activity",
                        "summary": (
                            f"{dimension}={item.get('name')}，"
                            f"活跃样本占比 {float(item.get('sample_percent') or 0):.2f}%"
                        ),
                    }
                )
        gaps.append(
            {
                "code": "SAMPLING_BOUNDARY",
                "detail": str(payload.get("disclaimer") or ""),
            }
        )
    if gaps:
        recommendations.append("先补齐报告列出的采集缺口，再提高结论确认级别。")
    template = resolve_system_template(_report_template_id(report_type))
    if template is None:
        raise RuntimeError("工作负载报告模板未注册")
    content = ReportContent(
        report_key=_report_key(report_type),
        report_type=str(report_type),
        ops_run_id=str(run_id),
        target_id=str(target_id),
        title=title,
        status=status,
        summary=(
            f"{title} 已由 {len(source_ids)} 个冻结事实确定性生成，"
            f"状态为 {status}。"
        ),
        period_start=period_start,
        period_end=period_end,
        scope={
            "report_origin": str(ReportOrigin.AIOPS_GENERATED),
            "inspection_coverage": coverage_summary,
            "trends": trends,
            "raw_report_artifact_id": str(raw_artifact_id),
        },
        facts=tuple(facts),
        gaps=tuple(gaps),
        evidence_refs=(
            {
                "artifact_id": str(raw_artifact_id),
                "content_hash": raw_content_hash,
                "schema_version": str(payload.get("schema_version") or ""),
                "source_kind": "INSPECTION",
            },
        ),
        recommendations=tuple(recommendations),
        provenance={
            "deterministic": True,
            "llm_used": False,
            "report_origin": str(ReportOrigin.AIOPS_GENERATED),
            "source_ids": [str(value) for value in source_ids],
            "template": _template_provenance(template),
        },
    )
    return content.model_dump(mode="json")


def _template_provenance(template) -> dict[str, Any]:
    return {
        "template_ref": template.template_ref,
        "version": template.version,
        "content_hash": template.content_hash,
        "definition": template.definition,
    }


def _safe_file_name(value: str, *, suffix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    if not normalized:
        normalized = f"workload-report{suffix}"
    if not normalized.lower().endswith(suffix):
        normalized = normalized.rsplit(".", 1)[0] + suffix
    return normalized[:256]


def _safe_external_file_name(*, file_name: str, format_name: str) -> str:
    suffix = ".html" if format_name == "HTML" else ".json"
    return _safe_file_name(file_name, suffix=suffix)
