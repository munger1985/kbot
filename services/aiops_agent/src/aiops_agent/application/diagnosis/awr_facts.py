"""从已确认快照上的 AWR Fact Tool 结果投影结论，不解析 HTML。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from aiops_agent.contracts.turn_answer import TurnEvidenceFact


AWR_FACT_TOOL_IDS = frozenset(
    {
        "db.oracle.awr.load_profile",
        "db.oracle.awr.top_wait",
        "db.oracle.awr.top_sql",
    }
)
AWR_HTML_TOOL_IDS = frozenset(
    {
        "db.oracle.awr.report",
        "db.oracle.awr.diff_report",
        "db.oracle.ash.report",
    }
)
_TREND_FLAT_RATIO = 0.1


@dataclass(frozen=True)
class AwrWindowSummary:
    begin_snapshot_id: int
    end_snapshot_id: int
    interval_count: int
    evidence_refs: tuple[str, ...]
    load_profile: tuple[dict[str, Any], ...]
    top_wait: tuple[dict[str, Any], ...]
    top_sql: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "begin_snapshot_id": self.begin_snapshot_id,
            "end_snapshot_id": self.end_snapshot_id,
            "interval_count": self.interval_count,
            "evidence_refs": list(self.evidence_refs),
            "load_profile": [dict(item) for item in self.load_profile],
            "top_wait": [dict(item) for item in self.top_wait],
            "top_sql": [dict(item) for item in self.top_sql],
        }


@dataclass(frozen=True)
class AwrFactPlan:
    windows: tuple[AwrWindowSummary, ...]
    comparison: tuple[dict[str, Any], ...]
    trend: tuple[dict[str, Any], ...]
    html_report_tool_ids: tuple[str, ...]
    used_html_as_facts: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "windows": [item.as_dict() for item in self.windows],
            "comparison": [dict(item) for item in self.comparison],
            "trend": [dict(item) for item in self.trend],
            "html_report_tool_ids": list(self.html_report_tool_ids),
            "used_html_as_facts": self.used_html_as_facts,
        }


def summarize_awr_facts(
    evidence: Sequence[TurnEvidenceFact] | Iterable[TurnEvidenceFact] | None,
) -> AwrFactPlan:
    """把 Fact Tool 行投影为单窗、对比和趋势结论；HTML 只登记链接来源。"""
    facts = tuple(evidence or ())
    html_ids = tuple(
        dict.fromkeys(
            fact.tool_id for fact in facts if fact.tool_id in AWR_HTML_TOOL_IDS
        )
    )
    fact_rows = _fact_rows(facts)
    windows = _window_summaries(fact_rows)
    comparison = _compare_windows(windows) if len(windows) == 2 else ()
    trend = _trend_series(fact_rows)
    return AwrFactPlan(
        windows=windows,
        comparison=comparison,
        trend=trend,
        html_report_tool_ids=html_ids,
        used_html_as_facts=False,
    )


def _fact_rows(facts: tuple[TurnEvidenceFact, ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for fact in facts:
        if fact.tool_id in AWR_HTML_TOOL_IDS:
            continue
        if fact.tool_id not in AWR_FACT_TOOL_IDS:
            continue
        columns = [
            str(item.get("name") or "")
            if isinstance(item, Mapping)
            else str(item)
            for item in fact.columns
        ]
        for raw in fact.rows or ():
            values = {
                name.lower(): value
                for name, value in zip(columns, raw, strict=False)
                if name
            }
            snapshot_id = _as_int(values.get("snapshot_id"))
            if snapshot_id is None:
                continue
            rows.append(
                {
                    "tool_id": fact.tool_id,
                    "evidence_ref": fact.evidence_ref,
                    "snapshot_id": snapshot_id,
                    "values": values,
                }
            )
    return tuple(rows)


def _window_summaries(
    rows: tuple[dict[str, Any], ...],
) -> tuple[AwrWindowSummary, ...]:
    if not rows:
        return ()
    by_ref: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        by_ref.setdefault(str(item["evidence_ref"]), []).append(item)
    groups: dict[tuple[int, int], dict[str, Any]] = {}
    for ref, items in by_ref.items():
        snapshot_ids = sorted(
            {int(item["snapshot_id"]) for item in items}
        )
        begin_id = snapshot_ids[0]
        end_id = snapshot_ids[-1]
        key = (begin_id, end_id)
        bucket = groups.setdefault(
            key,
            {
                "begin_snapshot_id": begin_id,
                "end_snapshot_id": end_id,
                "interval_ids": set(snapshot_ids),
                "evidence_refs": [],
                "rows": [],
            },
        )
        bucket["interval_ids"].update(snapshot_ids)
        if ref not in bucket["evidence_refs"]:
            bucket["evidence_refs"].append(ref)
        bucket["rows"].extend(items)
    summaries = []
    for key in sorted(groups):
        bucket = groups[key]
        grouped_rows = tuple(bucket["rows"])
        summaries.append(
            AwrWindowSummary(
                begin_snapshot_id=bucket["begin_snapshot_id"],
                end_snapshot_id=bucket["end_snapshot_id"],
                interval_count=len(bucket["interval_ids"]),
                evidence_refs=tuple(bucket["evidence_refs"]),
                load_profile=_aggregate_load_profile(grouped_rows),
                top_wait=_aggregate_top_wait(grouped_rows),
                top_sql=_aggregate_top_sql(grouped_rows),
            )
        )
    return tuple(summaries)


def _aggregate_load_profile(
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    elapsed_by_snap: dict[int, float] = {}
    totals: dict[str, dict[str, Any]] = {}
    for item in rows:
        if item["tool_id"] != "db.oracle.awr.load_profile":
            continue
        values = item["values"]
        metric = str(values.get("metric_name") or "").strip()
        if not metric:
            continue
        elapsed = _as_number(values.get("elapsed_seconds"))
        if elapsed is not None:
            elapsed_by_snap[int(item["snapshot_id"])] = elapsed
        current = totals.setdefault(
            metric,
            {
                "metric_name": metric,
                "total_value": 0.0,
                "unit": str(values.get("unit") or ""),
            },
        )
        current["total_value"] += _as_number(values.get("total_value")) or 0.0
        if not current["unit"]:
            current["unit"] = str(values.get("unit") or "")
    window_elapsed = sum(elapsed_by_snap.values())
    result = []
    for metric, item in sorted(totals.items()):
        per_second = None
        if window_elapsed:
            per_second = round(item["total_value"] / window_elapsed, 3)
        result.append(
            {
                "metric_name": metric,
                "total_value": round(item["total_value"], 3),
                "per_second": per_second,
                "unit": item["unit"],
                "elapsed_seconds": round(window_elapsed, 3) if window_elapsed else None,
            }
        )
    return tuple(result)


def _aggregate_top_wait(
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    totals: dict[str, dict[str, Any]] = {}
    for item in rows:
        if item["tool_id"] != "db.oracle.awr.top_wait":
            continue
        values = item["values"]
        event_name = str(values.get("event_name") or "").strip()
        if not event_name:
            continue
        current = totals.setdefault(
            event_name,
            {
                "event_name": event_name,
                "wait_class": str(values.get("wait_class") or ""),
                "total_waits": 0.0,
                "time_waited_seconds": 0.0,
            },
        )
        current["total_waits"] += _as_number(values.get("total_waits")) or 0.0
        current["time_waited_seconds"] += (
            _as_number(values.get("time_waited_seconds")) or 0.0
        )
        if not current["wait_class"]:
            current["wait_class"] = str(values.get("wait_class") or "")
    ranked = sorted(
        totals.values(),
        key=lambda item: (-item["time_waited_seconds"], item["event_name"]),
    )
    result = []
    for index, item in enumerate(ranked[:15], start=1):
        waits = item["total_waits"]
        waited = item["time_waited_seconds"]
        avg_wait_ms = None
        if waits:
            avg_wait_ms = round(waited * 1000 / waits, 3)
        result.append(
            {
                "event_name": item["event_name"],
                "wait_class": item["wait_class"],
                "total_waits": round(waits, 3),
                "time_waited_seconds": round(waited, 3),
                "avg_wait_ms": avg_wait_ms,
                "wait_rank": index,
            }
        )
    return tuple(result)


def _aggregate_top_sql(
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    totals: dict[str, dict[str, Any]] = {}
    for item in rows:
        if item["tool_id"] != "db.oracle.awr.top_sql":
            continue
        values = item["values"]
        sql_id = str(values.get("sql_id") or "").strip()
        if not sql_id:
            continue
        current = totals.setdefault(
            sql_id,
            {
                "sql_id": sql_id,
                "plan_hash_value": _as_int(values.get("plan_hash_value")),
                "executions": 0.0,
                "elapsed_seconds": 0.0,
                "cpu_seconds": 0.0,
                "buffer_gets": 0.0,
                "disk_reads": 0.0,
            },
        )
        current["executions"] += _as_number(values.get("executions")) or 0.0
        current["elapsed_seconds"] += _as_number(values.get("elapsed_seconds")) or 0.0
        current["cpu_seconds"] += _as_number(values.get("cpu_seconds")) or 0.0
        current["buffer_gets"] += _as_number(values.get("buffer_gets")) or 0.0
        current["disk_reads"] += _as_number(values.get("disk_reads")) or 0.0
        if current["plan_hash_value"] is None:
            current["plan_hash_value"] = _as_int(values.get("plan_hash_value"))
    ranked = sorted(
        totals.values(),
        key=lambda item: (-item["elapsed_seconds"], item["sql_id"]),
    )
    result = []
    for index, item in enumerate(ranked[:15], start=1):
        result.append(
            {
                "sql_id": item["sql_id"],
                "plan_hash_value": item["plan_hash_value"],
                "executions": round(item["executions"], 3),
                "elapsed_seconds": round(item["elapsed_seconds"], 3),
                "cpu_seconds": round(item["cpu_seconds"], 3),
                "buffer_gets": round(item["buffer_gets"], 3),
                "disk_reads": round(item["disk_reads"], 3),
                "sql_rank": index,
            }
        )
    return tuple(result)


def _compare_windows(
    windows: tuple[AwrWindowSummary, ...],
) -> tuple[dict[str, Any], ...]:
    baseline, after = windows
    result: list[dict[str, Any]] = []
    baseline_load = {item["metric_name"]: item for item in baseline.load_profile}
    after_load = {item["metric_name"]: item for item in after.load_profile}
    for metric in sorted(set(baseline_load) | set(after_load)):
        left = baseline_load.get(metric) or {}
        right = after_load.get(metric) or {}
        result.append(
            {
                "kind": "LOAD_PROFILE",
                "name": metric,
                "baseline": left.get("per_second"),
                "after": right.get("per_second"),
                "change_pct": _change_pct(left.get("per_second"), right.get("per_second")),
                "direction": _direction(left.get("per_second"), right.get("per_second")),
            }
        )
    baseline_wait = {item["event_name"]: item for item in baseline.top_wait}
    after_wait = {item["event_name"]: item for item in after.top_wait}
    wait_names = _union_top_names(baseline_wait, after_wait, "time_waited_seconds")
    for name in wait_names:
        left = baseline_wait.get(name) or {}
        right = after_wait.get(name) or {}
        result.append(
            {
                "kind": "TOP_WAIT",
                "name": name,
                "baseline": left.get("time_waited_seconds"),
                "after": right.get("time_waited_seconds"),
                "change_pct": _change_pct(
                    left.get("time_waited_seconds"),
                    right.get("time_waited_seconds"),
                ),
                "direction": _direction(
                    left.get("time_waited_seconds"),
                    right.get("time_waited_seconds"),
                ),
            }
        )
    baseline_sql = {item["sql_id"]: item for item in baseline.top_sql}
    after_sql = {item["sql_id"]: item for item in after.top_sql}
    sql_ids = _union_top_names(baseline_sql, after_sql, "elapsed_seconds")
    for sql_id in sql_ids:
        left = baseline_sql.get(sql_id) or {}
        right = after_sql.get(sql_id) or {}
        result.append(
            {
                "kind": "TOP_SQL",
                "name": sql_id,
                "baseline": left.get("elapsed_seconds"),
                "after": right.get("elapsed_seconds"),
                "change_pct": _change_pct(
                    left.get("elapsed_seconds"),
                    right.get("elapsed_seconds"),
                ),
                "direction": _direction(
                    left.get("elapsed_seconds"),
                    right.get("elapsed_seconds"),
                ),
            }
        )
    return tuple(result)


def _trend_series(
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    series: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for item in rows:
        values = item["values"]
        snapshot_id = int(item["snapshot_id"])
        if item["tool_id"] == "db.oracle.awr.load_profile":
            name = str(values.get("metric_name") or "").strip()
            value = _as_number(values.get("per_second"))
            kind = "LOAD_PROFILE"
        elif item["tool_id"] == "db.oracle.awr.top_wait":
            name = str(values.get("event_name") or "").strip()
            value = _as_number(values.get("time_waited_seconds"))
            kind = "TOP_WAIT"
        elif item["tool_id"] == "db.oracle.awr.top_sql":
            name = str(values.get("sql_id") or "").strip()
            value = _as_number(values.get("elapsed_seconds"))
            kind = "TOP_SQL"
        else:
            continue
        if not name or value is None:
            continue
        series.setdefault((kind, name), []).append((snapshot_id, value))
    result = []
    for (kind, name), points in sorted(series.items()):
        ordered = sorted(points, key=lambda item: item[0])
        unique_snaps = [snap for snap, _value in ordered]
        if len(set(unique_snaps)) < 2:
            continue
        first = ordered[0][1]
        last = ordered[-1][1]
        result.append(
            {
                "kind": kind,
                "name": name,
                "begin_snapshot_id": ordered[0][0],
                "end_snapshot_id": ordered[-1][0],
                "interval_count": len(set(unique_snaps)),
                "first": first,
                "last": last,
                "change_pct": _change_pct(first, last),
                "direction": _direction(first, last),
            }
        )
    return tuple(result)


def _union_top_names(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    score_key: str,
) -> tuple[str, ...]:
    names = set(left) | set(right)
    ranked = sorted(
        names,
        key=lambda name: (
            -(
                max(
                    _as_number((left.get(name) or {}).get(score_key)) or 0.0,
                    _as_number((right.get(name) or {}).get(score_key)) or 0.0,
                )
            ),
            name,
        ),
    )
    return tuple(ranked[:15])


def _change_pct(before: Any, after: Any) -> float | None:
    start = _as_number(before)
    end = _as_number(after)
    if start is None or end is None:
        return None
    if start == 0:
        if end == 0:
            return 0.0
        return None
    return round((end - start) / abs(start) * 100, 3)


def _direction(before: Any, after: Any) -> str:
    start = _as_number(before)
    end = _as_number(after)
    if start is None or end is None:
        return "UNKNOWN"
    if start == 0:
        if end == 0:
            return "FLAT"
        return "UP"
    ratio = (end - start) / abs(start)
    if abs(ratio) <= _TREND_FLAT_RATIO:
        return "FLAT"
    return "UP" if ratio > 0 else "DOWN"


def _as_int(value: Any) -> int | None:
    number = _as_number(value)
    if number is None:
        return None
    return int(number)


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None
