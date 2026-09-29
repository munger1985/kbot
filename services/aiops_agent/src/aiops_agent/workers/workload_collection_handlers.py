"""把 DB Executor 的固定数据库 Tool 结果聚合为高频事实。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from aiops_agent.workers.handlers import TaskExecutionContext
from platform_core.contracts.aiops.workload import (
    CollectionQuality,
    StatementIdentityType,
    WorkloadMetricSnapshot,
    WorkloadSnapshotCreate,
    WorkloadStatementSnapshot,
)


class WorkloadCollectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["WORKLOAD_COLLECTION_RESULT.v2"] = (
        "WORKLOAD_COLLECTION_RESULT.v2"
    )
    target_id: UUID
    workload_snapshot_id: UUID
    status: Literal["READY", "PARTIAL", "FAILED"]
    statement_count: int
    metric_count: int


class ActivityCollectionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["ACTIVITY_COLLECTION_RESULT.v2"] = (
        "ACTIVITY_COLLECTION_RESULT.v2"
    )
    target_id: UUID
    status: Literal["READY", "DEGRADED"]
    sample_count: int
    error_code: str | None = None


class WorkloadCollectionHandler:
    def __init__(self, *, workload_service) -> None:
        self._service = workload_service

    async def execute(
        self, context: TaskExecutionContext
    ) -> WorkloadCollectionResult:
        collection = dict(context.plan_snapshot["workload_collection"])
        diagnostics = dict(context.plan_snapshot["database_diagnostics"])
        database_type = str(diagnostics["db_type"])
        observations, gaps = _observations(context.input_artifacts)
        identity = _first_row(observations.get("db.instance.identity"))
        performance = (
            _aggregate_postgresql_database_statistics(
                observations.get("db.postgresql.database.statistics")
            )
            if database_type == "POSTGRESQL"
            else _first_row(observations.get("db.instance.performance"))
        )
        captured_at = _latest_captured_at(observations) or datetime.now(UTC)
        probe = dict(
            dict(diagnostics.get("capability_snapshot") or {}).get(
                "capability_probe"
            )
            or {}
        )
        raw_capabilities = dict(
            diagnostics.get("capability_snapshot") or {}
        )
        if not probe:
            probe = dict(raw_capabilities.get("probe") or {})
        probe_details = dict(probe.get("details") or {})
        instance_identifier = str(
            performance.get("server_uuid")
            or probe.get("server_uuid")
            or "UNKNOWN"
        )
        identity_kind = "MYSQL_SERVER_UUID"
        if database_type == "POSTGRESQL":
            identity_kind = "POSTGRESQL_SYSTEM_IDENTIFIER"
        uptime = _integer(performance.get("uptime_seconds"))
        instance_started_at = (
            captured_at - timedelta(seconds=uptime)
            if uptime is not None
            else _datetime(probe.get("server_started_at")) or captured_at
        )

        coverage = _workload_coverage(
            database_type=database_type,
            observations=observations,
            gaps=gaps,
            unavailable=tuple(collection.get("unavailable_tools") or ()),
        )
        statements = _statement_rows(
            database_type=database_type,
            observation=(
                observations.get("db.mysql.statement.digest.top")
                if database_type == "MYSQL"
                else observations.get("db.postgresql.statement.top")
            ),
        )
        metrics = _metric_rows(database_type, observations)
        replication_metrics = {
            tool_id: _rows(observation)
            for tool_id, observation in observations.items()
            if tool_id
            in {
                "db.mysql.replication.channel_status",
                "db.mysql.group_replication.status",
                "db.postgresql.replication.slot_retention",
                "db.postgresql.archiver.status",
            }
        }
        fatal = not identity or instance_identifier == "UNKNOWN"
        partial = bool(gaps) or any(
            value != CollectionQuality.AVAILABLE for value in coverage.values()
        )
        status = "FAILED" if fatal else "PARTIAL" if partial else "READY"
        error_code = (
            f"{database_type}_SNAPSHOT_IDENTITY_UNAVAILABLE" if fatal else None
        )
        request = WorkloadSnapshotCreate(
            target_id=context.target_id,
            database_type=database_type,
            scheduled_for=_datetime(collection["scheduled_for"]),
            collected_at=captured_at,
            window_seconds=int(collection["window_seconds"]),
            instance_identity={
                "identity_type": identity_kind,
                "identity_value": instance_identifier,
                "database_name": (
                    probe_details.get("database_name")
                    or identity.get("instance_name")
                ),
            },
            instance_started_at=instance_started_at,
            database_version=str(
                identity.get("version")
                or diagnostics.get("configured_version")
                or "UNKNOWN"
            ),
            capability_probe_version=str(
                probe.get("version")
                or f"{database_type.lower()}-capabilities.v1"
            ),
            catalog_version=str(diagnostics["catalog_hash"]),
            continuity={
                "instance": {
                    "identity": instance_identifier,
                    "started_at": instance_started_at.isoformat(),
                }
            },
            coverage=coverage,
            instance_metrics={
                key: value
                for key, value in performance.items()
                if key != "server_uuid" and isinstance(value, (int, float))
            },
            replication_metrics=replication_metrics,
            statements=tuple(statements),
            metrics=tuple(metrics),
            status=status,
            error_code=error_code,
            retention_days=int(collection["retention_days"]),
        )
        snapshot_id = await self._service.record_snapshot(
            domain_id=int(diagnostics["domain_id"]),
            request=request,
        )
        return WorkloadCollectionResult(
            target_id=context.target_id,
            workload_snapshot_id=snapshot_id,
            status=status,
            statement_count=len(statements),
            metric_count=len(metrics),
        )


class ActivityCollectionHandler:
    def __init__(self, *, workload_service) -> None:
        self._service = workload_service

    async def execute(
        self, context: TaskExecutionContext
    ) -> ActivityCollectionResult:
        collection = dict(context.plan_snapshot["workload_collection"])
        diagnostics = dict(context.plan_snapshot["database_diagnostics"])
        database_type = str(diagnostics["db_type"])
        observations, gaps = _observations(context.input_artifacts)
        sampled_at = _latest_captured_at(observations) or datetime.now(UTC)
        activity_tool = (
            "db.mysql.activity.sample"
            if database_type == "MYSQL"
            else "db.postgresql.activity.sample"
        )
        activity = observations.get(activity_tool)
        activity_gap = gaps.get(activity_tool)
        if activity is None or activity_gap is not None:
            error_code = str(
                (activity_gap or {}).get("code")
                or f"{database_type}_ACTIVITY_SAMPLE_UNAVAILABLE"
            )
            await self._service.record_activity_failure(
                domain_id=int(diagnostics["domain_id"]),
                target_id=context.target_id,
                failed_at=sampled_at,
                error_code=error_code,
            )
            return ActivityCollectionResult(
                target_id=context.target_id,
                status="DEGRADED",
                sample_count=0,
                error_code=error_code,
            )
        policy = dict(collection.get("policy") or {})
        duration_limit = int(policy.get("query_timeout_seconds") or 2) * 1000
        if int(activity.get("duration_ms") or 0) > duration_limit:
            error_code = "ACTIVITY_SAMPLER_SELF_LOAD_LIMIT"
            await self._service.record_activity_failure(
                domain_id=int(diagnostics["domain_id"]),
                target_id=context.target_id,
                failed_at=sampled_at,
                error_code=error_code,
            )
            return ActivityCollectionResult(
                target_id=context.target_id,
                status="DEGRADED",
                sample_count=0,
                error_code=error_code,
            )
        raw_capabilities = dict(
            diagnostics.get("capability_snapshot") or {}
        )
        probe = dict(raw_capabilities.get("capability_probe") or {})
        if not probe:
            probe = dict(raw_capabilities.get("probe") or {})
        instance_identifier = str(probe.get("server_uuid") or "UNKNOWN")
        if instance_identifier == "UNKNOWN":
            error_code = f"{database_type}_ACTIVITY_IDENTITY_UNAVAILABLE"
            await self._service.record_activity_failure(
                domain_id=int(diagnostics["domain_id"]),
                target_id=context.target_id,
                failed_at=sampled_at,
                error_code=error_code,
            )
            return ActivityCollectionResult(
                target_id=context.target_id,
                status="DEGRADED",
                sample_count=0,
                error_code=error_code,
            )
        samples = _rows(activity)
        sample_ids = await self._service.record_activity_samples(
            domain_id=int(diagnostics["domain_id"]),
            target_id=context.target_id,
            sampled_at=sampled_at,
            database_type=database_type,
            instance_identity={
                "identity_type": (
                    "MYSQL_SERVER_UUID"
                    if database_type == "MYSQL"
                    else "POSTGRESQL_SYSTEM_IDENTIFIER"
                ),
                "identity_value": instance_identifier,
            },
            samples=tuple(
                _normalize_activity_sample(database_type, item)
                for item in samples
            ),
            retention_days=int(collection["retention_days"]),
        )
        return ActivityCollectionResult(
            target_id=context.target_id,
            status="READY",
            sample_count=len(sample_ids),
        )


def _observations(
    artifacts: tuple[dict[str, Any], ...]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    observations: dict[str, dict[str, Any]] = {}
    gaps: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if artifact.get("schema_version") != "DATABASE_DIAGNOSTIC_RESULT.v1":
            continue
        payload = dict(artifact.get("payload") or {})
        tool_id = str(payload.get("tool_id") or "")
        if payload.get("status") == "SUCCEEDED" and payload.get("observation"):
            observations[tool_id] = dict(payload["observation"])
        elif payload.get("gap"):
            gaps[tool_id] = dict(payload["gap"])
    return observations, gaps


def _rows(observation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not observation:
        return []
    columns = [str(item["name"]) for item in observation.get("columns", ())]
    return [
        dict(zip(columns, row, strict=True))
        for row in observation.get("rows", ())
    ]


def _first_row(observation: dict[str, Any] | None) -> dict[str, Any]:
    rows = _rows(observation)
    return rows[0] if rows else {}


def _latest_captured_at(observations: dict[str, dict[str, Any]]) -> datetime | None:
    values = [
        _datetime(item.get("captured_at")) for item in observations.values()
    ]
    materialized = [item for item in values if item is not None]
    return max(materialized) if materialized else None


def _workload_coverage(
    *, database_type: str, observations, gaps, unavailable
) -> dict[str, CollectionQuality]:
    common = {
        "db.instance.identity": "IDENTITY",
    }
    mysql = {
        "db.instance.performance": "INSTANCE",
        "db.mysql.statement.digest.top": "DIGEST",
        "db.wait.class_summary": "WAIT",
        "db.mysql.io.summary": "IO",
        "db.memory.summary": "MEMORY",
        "db.mysql.lock.current": "LOCK",
        "db.storage.capacity": "CAPACITY",
        "db.mysql.table.hotspots": "TABLE",
        "db.index.health": "INDEX",
        "db.mysql.replication.channel_status": "REPLICATION",
        "db.mysql.group_replication.status": "GROUP_REPLICATION",
    }
    postgresql = {
        "db.postgresql.database.statistics": "INSTANCE",
        "db.postgresql.statement.top": "STATEMENT",
        "db.postgresql.wait.current": "WAIT",
        "db.postgresql.io.summary": "IO",
        "db.postgresql.checkpoint.statistics": "CHECKPOINT",
        "db.postgresql.wal.statistics": "WAL",
        "db.postgresql.table.statistics": "TABLE",
        "db.postgresql.index.statistics": "INDEX",
        "db.postgresql.replication.slot_retention": "REPLICATION_SLOT",
        "db.postgresql.archiver.status": "ARCHIVE",
    }
    families = {
        **common,
        **(mysql if database_type == "MYSQL" else postgresql),
    }
    result: dict[str, CollectionQuality] = {}
    unavailable_ids = {str(item.get("tool_id")) for item in unavailable}
    for tool_id, family in families.items():
        if tool_id in observations:
            observation = observations[tool_id]
            result[family] = (
                CollectionQuality.PARTIAL
                if observation.get("truncated")
                else CollectionQuality.AVAILABLE
            )
        elif tool_id in gaps:
            code = str(gaps[tool_id].get("code") or "")
            result[family] = (
                CollectionQuality.TIMEOUT
                if "TIMEOUT" in code
                else CollectionQuality.DENIED
                if "DENIED" in code or "SECRET" in code
                else CollectionQuality.PARTIAL
            )
        elif tool_id in unavailable_ids:
            result[family] = CollectionQuality.DISABLED
        else:
            result[family] = CollectionQuality.PARTIAL
    return result


def _statement_rows(
    *, database_type: str, observation: dict[str, Any] | None
) -> list[WorkloadStatementSnapshot]:
    result: list[WorkloadStatementSnapshot] = []
    for rank, item in enumerate(_rows(observation), start=1):
        if database_type == "MYSQL":
            identity = str(item.get("digest") or "").upper()
            if len(identity) != 64:
                continue
            total_us = _picoseconds_to_microseconds(
                item.get("total_latency_picoseconds")
            )
            mean_us = _picoseconds_to_microseconds(
                item.get("average_latency_picoseconds")
            )
            max_us = _picoseconds_to_microseconds(
                item.get("maximum_latency_picoseconds")
            )
            database_metrics = {
                key: value
                for key, value in item.items()
                if key not in {
                    "schema_name", "digest", "digest_text",
                    "execution_count", "total_latency_picoseconds",
                    "average_latency_picoseconds",
                    "maximum_latency_picoseconds", "rows_sent",
                }
            }
            result.append(
                WorkloadStatementSnapshot(
                    statement_identity_type=StatementIdentityType.MYSQL_DIGEST,
                    statement_identity_value=identity,
                    database_name=str(item.get("schema_name") or "")[:128],
                    normalized_statement=(
                        str(item["digest_text"])
                        if item.get("digest_text") is not None else None
                    ),
                    execution_count=_integer(item.get("execution_count")) or 0,
                    total_duration_microseconds=total_us,
                    mean_duration_microseconds=mean_us,
                    max_duration_microseconds=max_us,
                    rows_processed=_integer(item.get("rows_sent")) or 0,
                    database_metrics=database_metrics,
                    rank_no=rank,
                )
            )
            continue
        identity = str(item.get("query_id") or "")
        if not identity:
            continue
        result.append(
            WorkloadStatementSnapshot(
                statement_identity_type=StatementIdentityType.POSTGRESQL_QUERY_ID,
                statement_identity_value=identity,
                database_name=str(item.get("database_oid") or ""),
                user_identifier=str(item.get("user_oid") or ""),
                top_level=(
                    bool(item["top_level"])
                    if item.get("top_level") is not None else None
                ),
                normalized_statement=(
                    str(item["normalized_statement"])
                    if item.get("normalized_statement") is not None else None
                ),
                execution_count=_integer(item.get("execution_count")) or 0,
                total_duration_microseconds=_milliseconds_to_microseconds(
                    item.get("total_exec_time_ms")
                ),
                mean_duration_microseconds=_milliseconds_to_microseconds(
                    item.get("mean_exec_time_ms")
                ),
                max_duration_microseconds=_milliseconds_to_microseconds(
                    item.get("max_exec_time_ms")
                ),
                rows_processed=_integer(item.get("rows_processed")) or 0,
                database_metrics={
                    key: value
                    for key, value in item.items()
                    if key not in {
                        "query_id", "database_oid", "database_name",
                        "user_oid", "username", "top_level",
                        "normalized_statement", "execution_count",
                        "total_exec_time_ms", "mean_exec_time_ms",
                        "max_exec_time_ms", "rows_processed",
                    }
                },
                rank_no=rank,
            )
        )
    return result


def _metric_rows(
    database_type: str, observations: dict[str, dict[str, Any]]
) -> list[WorkloadMetricSnapshot]:
    result: list[WorkloadMetricSnapshot] = []
    specs = {
        "db.wait.class_summary": (
            "WAIT", "wait_class", "total_wait_picoseconds", "PICOSECONDS", "COUNTER"
        ),
        "db.mysql.io.summary": (
            "IO", "event_name", "read_latency_picoseconds", "PICOSECONDS", "COUNTER"
        ),
        "db.memory.summary": (
            "MEMORY", "event_name", "current_number_of_bytes_used", "BYTES", "GAUGE"
        ),
        "db.mysql.table.hotspots": (
            "TABLE", ("object_schema", "object_name"), "read_latency_picoseconds", "PICOSECONDS", "COUNTER"
        ),
        "db.index.health": (
            "INDEX", ("object_schema", "object_name", "index_name"), "read_latency_picoseconds", "PICOSECONDS", "COUNTER"
        ),
        "db.storage.capacity": (
            "TABLE", "database_name", "allocated_mb", "MEGABYTES", "GAUGE"
        ),
        "db.postgresql.database.statistics": (
            "DATABASE", "database_name", "xact_commit", "COUNT", "COUNTER"
        ),
        "db.postgresql.checkpoint.statistics": (
            "CHECKPOINT", "stats_reset", "checkpoints_requested", "COUNT", "COUNTER"
        ),
        "db.postgresql.wal.statistics": (
            "WAL", "stats_reset", "wal_bytes", "BYTES", "COUNTER"
        ),
        "db.postgresql.table.statistics": (
            "TABLE", ("schema_name", "table_name"), "n_dead_tup", "ROWS", "GAUGE"
        ),
        "db.postgresql.index.statistics": (
            "INDEX", ("schema_name", "index_name"), "idx_scan", "COUNT", "COUNTER"
        ),
    }
    for tool_id, (family, keys, value_key, unit, value_kind) in specs.items():
        for item in _rows(observations.get(tool_id)):
            key_names = (keys,) if isinstance(keys, str) else keys
            dimension_key = ".".join(
                str(item.get(key) or "UNKNOWN") for key in key_names
            )[:256]
            value = item.get(value_key)
            if value is None:
                continue
            result.append(
                WorkloadMetricSnapshot(
                    metric_family=family,
                    metric_code=f"{database_type.lower()}.{tool_id[3:]}.{value_key}",
                    dimension_key=dimension_key,
                    dimensions={
                        key: value
                        for key, value in item.items()
                        if key != value_key
                    },
                    counter_value=(
                        float(value) if value_kind == "COUNTER" else None
                    ),
                    gauge_value=(
                        float(value) if value_kind == "GAUGE" else None
                    ),
                    unit=unit,
                    quality_status=CollectionQuality.AVAILABLE,
                )
            )
    for item in _rows(observations.get("db.mysql.lock.current")):
        result.append(
            WorkloadMetricSnapshot(
                metric_family="LOCK",
                metric_code="mysql.lock.current",
                dimension_key=(
                    f"{item.get('requesting_thread_id', 'UNKNOWN')}:"
                    f"{item.get('blocking_thread_id', 'UNKNOWN')}:"
                    f"{item.get('object_schema', '')}."
                    f"{item.get('object_name', '')}"
                )[:256],
                dimensions=item,
                gauge_value=1,
                unit="COUNT",
                quality_status=CollectionQuality.AVAILABLE,
            )
        )
    for tool_id in (
        "db.mysql.replication.channel_status",
        "db.mysql.group_replication.status",
        "db.postgresql.replication.slot_retention",
        "db.postgresql.archiver.status",
    ):
        for item in _rows(observations.get(tool_id)):
            result.append(
                WorkloadMetricSnapshot(
                    metric_family="REPLICATION",
                    metric_code=f"{database_type.lower()}.{tool_id[3:]}",
                    dimension_key=(
                        f"{tool_id}:{item.get('channel_name', '')}:"
                        f"{item.get('member_id', '')}"
                    )[:256],
                    dimensions=item,
                    gauge_value=1,
                    unit="STATE",
                    quality_status=CollectionQuality.AVAILABLE,
                )
            )
    return result[:5000]


def _normalize_activity_sample(
    database_type: str, item: dict[str, Any]
) -> dict[str, Any]:
    if database_type == "MYSQL":
        return {
            "session_identifier": item.get("mysql_connection_number"),
            "worker_identifier": item.get("mysql_thread_number"),
            "statement_identity_type": (
                "MYSQL_DIGEST" if item.get("digest") else None
            ),
            "statement_identity_value": item.get("digest"),
            "database_name": item.get("schema_name"),
            "schema_name": item.get("schema_name"),
            "username": item.get("username"),
            "client": item.get("client"),
            "command_name": item.get("command_name"),
            "session_state": item.get("session_state"),
            "stage_name": item.get("stage_name"),
            "wait_name": item.get("wait_name"),
            "transaction_active": item.get("transaction_active"),
            "lock_waiting": item.get("lock_waiting"),
            "database_sample": {},
        }
    return {
        "session_identifier": item.get("pid"),
        "worker_identifier": item.get("leader_pid"),
        "statement_identity_type": (
            "POSTGRESQL_QUERY_ID" if item.get("query_id") else None
        ),
        "statement_identity_value": item.get("query_id"),
        "database_name": item.get("database_name"),
        "username": item.get("username"),
        "client": item.get("application_name") or item.get("client_addr"),
        "command_name": item.get("backend_type"),
        "session_state": item.get("state"),
        "wait_name": item.get("wait_event"),
        "transaction_active": bool(item.get("transaction_started_at")),
        "lock_waiting": bool(item.get("wait_event_type") == "Lock"),
        "database_sample": {
            key: value
            for key, value in item.items()
            if key not in {"query_text", "username", "client_addr"}
        },
    }


def _picoseconds_to_microseconds(value: Any) -> int:
    return _duration_to_microseconds(value, divisor=Decimal("1000000"))


def _milliseconds_to_microseconds(value: Any) -> int:
    return _duration_to_microseconds(value, multiplier=Decimal("1000"))


def _duration_to_microseconds(
    value: Any,
    *,
    divisor: Decimal | None = None,
    multiplier: Decimal | None = None,
) -> int:
    try:
        amount = Decimal(str(value or 0))
        if not amount.is_finite():
            raise InvalidOperation
        if divisor is not None:
            amount /= divisor
        if multiplier is not None:
            amount *= multiplier
        result = int(amount.to_integral_value(rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("工作负载耗时不是有效数值") from exc
    if result < 0 or result > 2**63 - 1:
        raise ValueError("工作负载耗时超出bigint范围")
    return result


def _aggregate_postgresql_database_statistics(
    observation: dict[str, Any] | None,
) -> dict[str, Any]:
    rows = _rows(observation)
    if not rows:
        return {}
    result: dict[str, Any] = {}
    for key in (
        "xact_commit",
        "xact_rollback",
        "blks_read",
        "blks_hit",
        "tup_returned",
        "tup_fetched",
        "tup_inserted",
        "tup_updated",
        "tup_deleted",
        "conflicts",
        "temp_files",
        "temp_bytes",
        "deadlocks",
    ):
        values = [_integer(row.get(key)) for row in rows]
        result[key] = sum(value for value in values if value is not None)
    return result


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if value is None:
        return None
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
