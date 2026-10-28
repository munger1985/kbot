"""Worker 与来源重放共用的确定性原始观测投影，不在副本中重新推断身份。"""

from aiops_agent.application.diagnosis.provenance import database_provenance
from aiops_agent.application.diagnosis.native_measurements import MAX_NATIVE_SERIES
from aiops_agent.application.diagnosis.sql_plans import database_cursor_plan
from aiops_agent.contracts.measurements import NativeSeriesScope
from aiops_agent.contracts.turn_answer import TurnEvidenceFact


def top_sql_window_evidence_fact(*, result, artifact_id):
    """历史窗口来源按原始窗口结果重建，不能借累计或当前快照证明历史。"""
    from decimal import Decimal
    from aiops_agent.application.diagnosis.calculations import statement_window_graphs
    from aiops_agent.contracts.measurements import DerivedMeasurement

    artifact_id = str(artifact_id)

    intervals = list(result.get("covered_intervals") or ())
    requested = (
        f"请求窗口：{result.get('requested_start')} 至 "
        f"{result.get('requested_end')}"
    )
    merged = []
    for interval in intervals:
        start, end = str(interval["start"]), str(interval["end"])
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    covered = "、".join(f"{start} 至 {end}" for start, end in merged[:5]) or "无"
    if len(merged) > 5:
        covered += f"（另有 {len(merged) - 5} 段）"
    gap_counts = dict(result.get("gaps") or {})
    if not result.get("collection_enabled"):
        gap_counts["COLLECTION_DISABLED"] = 1
    gap_labels = {
        "COLLECTION_DISABLED": "周期快照未启用",
        "INSUFFICIENT_SNAPSHOTS": "有效快照不足两次",
        "WINDOW_NOT_FULLY_COVERED": "未覆盖完整窗口",
        "STATEMENT_SOURCE_UNAVAILABLE": "SQL 统计来源不可用",
        "DATABASE_OID_UNAVAILABLE": "无法核实数据库身份",
        "DATABASE_OID_CHANGED": "数据库身份发生变化",
        "DISCONTINUITY": "实例身份或启动时间不连续",
        "MISSED_SNAPSHOTS": "采样间断",
        "COUNTER_RESET": "统计计数器重置",
        "PARTIAL_SOURCE": "采集来源不完整",
        "LIMITED_CANDIDATE_SET": "累计排名截取的候选集可能遗漏窗口热点",
        "STATEMENT_APPEARED_OR_EVICTED": "语句新增或淘汰",
        "STATEMENT_UNAVAILABLE": "部分语句统计不可用",
    }
    warnings = (
        requested,
        f"实际覆盖区间：{covered}",
        *(str(item) for item in result.get("warnings") or ()),
        *(f"数据缺口：{gap_labels.get(str(code), str(code))}（{count}）"
          for code, count in gap_counts.items()),
    )
    statements = list(result.get("top_statements") or ())
    if not intervals:
        return None
    identity_type = str(statements[0].get("statement_identity_type") or "") if statements else ""
    identity_column = {
        "MYSQL_DIGEST": "digest",
        "POSTGRESQL_QUERY_ID": "query_id",
        "ORACLE_SQL_ID": "sql_id",
    }.get(identity_type, "statement_identity")
    return TurnEvidenceFact(
        evidence_ref=f"artifact:{artifact_id}#top-sql-window",
        artifact_id=artifact_id,
        source_id="workload.sql.top_window",
        step_id="top-sql-window",
        tool_id="workload.sql.top_window",
        measurement_semantics="SNAPSHOT_DELTA",
        presentation_kind="TABLE",
        captured_at=str(result["requested_end"]),
        columns=(
            {"name": identity_column, "logical_type": "STRING"},
            {"name": "normalized_statement", "logical_type": "STRING"},
            *({"name": name, "logical_type": "STRING"} for name in (("database_oid", "user_oid", "top_level") if identity_type == "POSTGRESQL_QUERY_ID" else ("schema_name",) if identity_type == "MYSQL_DIGEST" else ("plan_hash_value",) if identity_type == "ORACLE_SQL_ID" else ())),
            {"name": "execution_count", "logical_type": "INTEGER"},
            {"name": "total_duration_seconds", "logical_type": "DECIMAL"},
            {"name": "mean_duration_seconds", "logical_type": "DECIMAL"},
        ),
        rows=tuple(
            (
                item.get("statement_identity_value"),
                item.get("normalized_statement"),
                *((item.get("database_name"), item.get("user_identifier"), item.get("top_level")) if identity_type == "POSTGRESQL_QUERY_ID" else (item.get("schema_name", item.get("database_name")),) if identity_type == "MYSQL_DIGEST" else (item.get("plan_hash_value"),) if identity_type == "ORACLE_SQL_ID" else ()),
                item.get("execution_count"),
                str(Decimal(str(item.get("total_duration_microseconds", 0))) / 1_000_000),
                str(Decimal(str(item["mean_duration_microseconds"])) / 1_000_000)
                if item.get("mean_duration_microseconds") is not None else None,
            ) for item in statements
        ),
        row_count=len(statements),
        truncated=False,
        warnings=warnings,
        derivations=tuple(
            DerivedMeasurement.model_validate(value)
            for item in statements for value in item.get("derivations", ())
        ),
        calculation_graphs=statement_window_graphs(result, artifact_id),
    )


def project_evidence_rows(fact: TurnEvidenceFact, indices: tuple[int, ...], *, truncated=False):
    """筛行时同步重排原生维度；预算外行保持未知，不借用原第零行身份。"""
    if len(indices) != len(set(indices)) or any(type(index) is not int or not 0 <= index < len(fact.rows) for index in indices):
        raise ValueError("证据投影行号无效或重复")
    provenance = fact.provenance
    if provenance is not None and provenance.native_lineage is not None:
        lineage = provenance.native_lineage
        source = {scope.row_index: scope for scope in lineage.series}
        dimension_names = tuple(sorted({name for scope in lineage.series
            for name in (*dict(scope.dimensions), *scope.missing_dimensions)}))
        scopes = tuple(
            source[index].model_copy(update={"row_index": projected}) if index in source else
            NativeSeriesScope(row_index=projected, dimensions=(), missing_dimensions=dimension_names or ("SOURCE_ROW_SCOPE_UNKNOWN",))
            for projected, index in enumerate(indices[:MAX_NATIVE_SERIES])
        )
        incomplete = any(scope.missing_dimensions for scope in scopes)
        limited = lineage.series_truncated or len(indices) > MAX_NATIVE_SERIES or any(index not in source for index in indices)
        provenance = provenance.model_copy(update={
            "native_lineage": lineage.model_copy(update={"series": scopes, "series_truncated": limited}),
            "quality_flags": tuple(dict.fromkeys((*provenance.quality_flags,
                *(("NATIVE_SERIES_SCOPE_INCOMPLETE",) if incomplete else ()),
                *(("NATIVE_SERIES_TRUNCATED",) if limited else ())))),
        })
    return fact.model_copy(update={
        "rows": tuple(fact.rows[index] for index in indices), "row_count": len(indices),
        "truncated": fact.truncated or truncated, "provenance": provenance,
        "derivations": (), "calculation_graphs": (), "series_computations": (), "visualizations": (),
    })


def database_evidence_fact(*, result, outcome, artifact_id, provenance=None):
    """保留采样、未来观察与计划警告；来源复核必须使用相同的正式投影。"""
    artifact_id = str(artifact_id)
    observation = outcome.observation
    if observation is None:
        raise ValueError("原始工具结果缺少观测")
    sql_plan, plan_warnings = database_cursor_plan(observation)
    future = dict(dict(provenance or {}).get("future_observation_source") or {})
    return TurnEvidenceFact(
        evidence_ref=f"artifact:{artifact_id}#{outcome.step_id}", artifact_id=artifact_id,
        source_id=result.source_id, step_id=outcome.step_id, tool_id=outcome.tool_id,
        measurement_semantics="SNAPSHOT_DELTA" if observation.variant == "mysql_8_digest_window"
            and outcome.tool_id == "db.sql.top_current" else result.measurement_semantics,
        presentation_kind=result.presentation_kind, captured_at=observation.captured_at.isoformat(),
        columns=tuple(column.model_dump(mode="json") for column in observation.columns),
        rows=observation.rows, row_count=observation.row_count, truncated=observation.truncated,
        followup_run_id=future.get("background_run_id"),
        warnings=(*observation.warnings, *plan_warnings, *(
            (f"本轮按需采样：实际完成 {result.sampling.completed_count}/{result.sampling.requested.count} 次；间隔至少 {result.sampling.requested.interval_seconds} 秒；仅代表实际采集时点，不覆盖请求的历史窗口",)
            if result.sampling is not None else ()
        ), *(
            (f"后台未来观察：实际完成 {result.observation_series.completed_count}/{result.observation_series.requested.count} 次；仅代表原采集时点，不能当作当前值或补造过去请求窗口，未自动计算差分",)
            if result.observation_series is not None else ()
        ), *(("后台观察历史搜索达到轮次、事实、行数或字节上限，不保证覆盖完整会话",) if future.get("search_truncated") else ())),
        provenance=database_provenance(observation), sql_plan=sql_plan,
        object_access=observation.object_access, pagination=observation.pagination,
    )
