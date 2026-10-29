"""单 SQL 诊断的确定性证据摘要，不从对象统计推断语句根因。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from aiops_agent.contracts.turn_answer import TurnEvidenceFact, TurnEvidenceGap


@dataclass(frozen=True, slots=True)
class SqlEvidenceAnswer:
    analysis: str
    solution: str
    evidence_refs: tuple[str, ...]
    has_statistics: bool
    historical_only: bool = False


_SUBJECTS = {
    "POSTGRESQL": ("POSTGRESQL_QUERY_ID", "query_id", "db.postgresql.statement.details"),
    "MYSQL": ("MYSQL_DIGEST", "digest", "db.sql.cursor_details"),
    "ORACLE": ("ORACLE_SQL_ID", "sql_id", "db.sql.cursor_details"),
}


def _number(value: Any, divisor: int = 1) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value)) / divisor
        if not number.is_finite() or number < 0:
            return None
        if 0 < number < Decimal("0.0005"):
            return f"{number:.3g}"
        return f"{number:,.3f}".rstrip("0").rstrip(".") if number % 1 else f"{number:,.0f}"
    except (InvalidOperation, ValueError, TypeError):
        return None


def _row_dicts(fact: TurnEvidenceFact) -> list[dict[str, Any]]:
    names = [str(item.get("name") or "").lower() for item in fact.columns]
    return [dict(zip(names, row, strict=True)) for row in fact.rows if len(row) == len(names)]


def _captured_at(fact: TurnEvidenceFact) -> float:
    try:
        value = datetime.fromisoformat(fact.captured_at.replace("Z", "+00:00"))
        return value.timestamp() if value.tzinfo else value.replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, OverflowError, OSError):
        return float("-inf")


def _matches(row: dict[str, Any], subject: dict[str, Any], identity: str) -> bool:
    expected = str(subject.get(identity) if subject.get(identity) is not None else "").strip()
    actual = str(row.get(identity) if row.get(identity) is not None else "").strip()
    if not expected or actual.casefold() != expected.casefold():
        return False
    for dimension in ("database_oid", "user_oid", "top_level", "schema_name", "inst_id", "child_number", "plan_hash_value", "container_id", "dbid"):
        if dimension not in subject:
            continue
        if dimension not in row:
            return False
        matches = row[dimension] == subject[dimension] if dimension == "schema_name" else str(row[dimension]).casefold() == str(subject[dimension]).casefold()
        if not matches:
            return False
    return True


def _subject(task_frame: dict, database_type: str) -> tuple[dict, tuple[str, str, str]] | None:
    if task_frame.get("diagnostic_profile") != "SINGLE_SQL_PERFORMANCE":
        return None
    if str(task_frame.get("action_intent") or "NONE") != "NONE":
        return None
    subject = task_frame.get("subject_ref") or {}
    spec = _SUBJECTS.get(database_type)
    if not isinstance(subject, dict) or not spec:
        return None
    subject_type, identity, _ = spec
    # 老的 Oracle 档案只保存 sql_id；Target 类型与唯一结构化标识必须一致。
    if subject.get("subject_type") not in (None, "", subject_type) or not subject.get(identity):
        return None
    if any(subject.get(other[1]) for other in _SUBJECTS.values() if other[1] != identity):
        return None
    return subject, spec


def sql_subject_specs(task_frame: dict, database_type: str) -> tuple:
    """比较请求共享搜索和投影预算，不为每个主体复制整轮证据。"""
    if task_frame.get("diagnostic_profile") != "SQL_COMPARISON":
        single = _subject(task_frame, database_type)
        return (single,) if single else ()
    if database_type not in _SUBJECTS or str(task_frame.get("action_intent") or "NONE") != "NONE":
        return ()
    from aiops_agent.domain.sql_subjects import normalize_sql_comparison
    subjects = normalize_sql_comparison(task_frame.get("subject_ref") or {}, database_type)
    return tuple((subject, _SUBJECTS[database_type]) for subject in subjects) if subjects else ()


def rebind_prior_sql_facts(*, payload: dict, artifact_id: str, task_frame: dict, database_type: str = "") -> tuple[TurnEvidenceFact, ...]:
    """仅把历史中同一 SQL 的来源观测作为带原采集时间的上下文。"""
    from aiops_agent.application.diagnosis.evidence_projection import project_evidence_rows
    if task_frame.get("diagnostic_profile") not in {"SINGLE_SQL_PERFORMANCE", "SQL_COMPARISON"}:
        return ()
    raw_subject = task_frame.get("subject_ref") or {}
    if not database_type and isinstance(raw_subject, dict):
        if task_frame.get("diagnostic_profile") == "SQL_COMPARISON":
            entries = raw_subject.get("subjects")
            raw_subject = entries[0] if isinstance(entries, (list, tuple)) and entries and isinstance(entries[0], dict) else {}
        database_type = next((key for key, value in _SUBJECTS.items() if value[0] == raw_subject.get("subject_type")), "")
    matches = sql_subject_specs(task_frame, database_type)
    if not matches:
        return ()
    result = []
    for index, raw in enumerate(payload.get("facts") or ()):
        try:
            fact = TurnEvidenceFact.model_validate(raw)
        except (ValidationError, TypeError, ValueError):
            continue
        if fact.trust_level != "SOURCE_VERIFIED" or fact.followup_run_id is not None:
            continue
        indices = tuple(row_index for row_index, mapped in enumerate(_row_dicts(fact))
            if any(_matches(mapped, subject, spec[1]) for subject, spec in matches)) if all(len(row) == len(fact.columns) for row in fact.rows) else ()
        if not indices:
            continue
        projected = project_evidence_rows(fact, indices)
        result.append(TurnEvidenceFact.model_validate({**projected.model_dump(mode="python"),
            "artifact_id": artifact_id, "evidence_ref": f"artifact:{artifact_id}#prior-sql-{index}",
            "source_id": "prior_turn.sql_observation", "step_id": f"prior-sql-{index}",
            "warnings": (*fact.warnings, "历史观测，保留原采集时间，不能视为当前值", *(
                ("历史证据搜索达到上限，不保证已搜索完整会话",) if dict(payload.get("search") or {}).get("truncated") else ()
            )),
        }))
    return tuple(result)


def build_sql_evidence_answer(
    *, task_frame: dict[str, Any], database_type: str, evidence: tuple[TurnEvidenceFact, ...],
    gaps: tuple[TurnEvidenceGap, ...] = (), fallback_reason: str | None = None,
) -> SqlEvidenceAnswer | None:
    """只处理结构化单 SQL 任务；不同账号、数据库及游标不合并。"""
    if task_frame.get("diagnostic_profile") == "SQL_COMPARISON":
        matches = sql_subject_specs(task_frame, database_type)
        if not matches:
            return None
        answers = [(subject, build_sql_evidence_answer(
            task_frame={**task_frame, "diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject},
            database_type=database_type, evidence=evidence, gaps=gaps, fallback_reason=fallback_reason,
        )) for subject, _ in matches]
        return SqlEvidenceAnswer(
            analysis="\n\n".join("SQL " + str(subject[next(key for key in ("query_id", "digest", "sql_id") if key in subject)]) + "：\n\n" + answer.analysis
                for subject, answer in answers),
            solution=answers[0][1].solution + "\n\n当前仅提供逐条已核验摘要，跨 SQL 的原因比较和方案尚未完成。",
            evidence_refs=tuple(dict.fromkeys(ref for _, answer in answers for ref in answer.evidence_refs)),
            has_statistics=any(answer.has_statistics for _, answer in answers),
            historical_only=all(answer.historical_only for _, answer in answers if answer.has_statistics),
        )
    match = _subject(task_frame, database_type)
    if match is None:
        return None
    subject, (_, identity, tool_id) = match
    prior_tools = {
        "POSTGRESQL": "db.postgresql.statement.top",
        "MYSQL": "db.mysql.statement.digest.top",
        "ORACLE": "db.sql.top_current",
    }
    candidates = [
        fact for fact in evidence
        if (fact.tool_id == tool_id or database_type == "POSTGRESQL" and fact.tool_id == "db.postgresql.statement.subjects"
            or fact.tool_id == "workload.sql.top_window"
            or (fact.source_id == "prior_turn.sql_observation" and fact.tool_id == prior_tools[database_type]))
        and fact.trust_level == "SOURCE_VERIFIED"
        and (str(fact.measurement_semantics) == "CUMULATIVE_SINCE_LOAD"
             or fact.tool_id == "workload.sql.top_window" and str(fact.measurement_semantics) == "SNAPSHOT_DELTA")
    ]
    metrics = {
        "POSTGRESQL": (("execution_count", "执行次数", 1), ("total_exec_time_ms", "总耗时（毫秒）", 1), ("mean_exec_time_ms", "平均耗时（毫秒）", 1), ("max_exec_time_ms", "累计最大耗时（毫秒）", 1), ("rows_processed", "处理行数", 1), ("shared_blks_hit", "共享块命中", 1), ("shared_blks_read", "共享块读取", 1), ("temp_blks_read", "临时块读取", 1), ("temp_blks_written", "临时块写入", 1), ("wal_bytes", "WAL 字节", 1), ("plan_count", "计划统计次数", 1)),
        "MYSQL": (("execution_count", "执行次数", 1), ("total_latency_picoseconds", "总耗时（毫秒）", 1_000_000_000), ("average_latency_picoseconds", "平均耗时（毫秒）", 1_000_000_000), ("maximum_latency_picoseconds", "累计最大耗时（毫秒）", 1_000_000_000), ("rows_examined", "扫描行数", 1), ("rows_sent", "返回行数", 1), ("temp_disk_tables", "磁盘临时表数", 1)),
        "ORACLE": (("executions", "执行次数", 1), ("elapsed_seconds", "总耗时（秒）", 1), ("cpu_seconds", "CPU 耗时（秒）", 1), ("buffer_gets", "逻辑读", 1), ("disk_reads", "物理读", 1)),
    }[database_type]
    window_metrics = (("execution_count", "执行次数", 1), ("total_duration_seconds", "总耗时（秒）", 1),
                      ("mean_duration_seconds", "平均耗时（秒）", 1))
    def metrics_for(fact):
        return window_metrics if str(fact.measurement_semantics) == "SNAPSHOT_DELTA" else metrics

    candidates.sort(key=lambda fact: (fact.source_id != "prior_turn.sql_observation", _captured_at(fact)), reverse=True)
    chosen = next(
        ((fact, rows) for fact in candidates
         if (rows := [row for row in _row_dicts(fact) if _matches(row, subject, identity)])
         and any(_number(row.get(key), divisor) is not None for row in rows for key, _, divisor in metrics_for(fact))),
        None,
    )
    if chosen is None:
        analysis = "本次未取得可用于指定 SQL 分析的匹配统计，诊断尚未完成，目前不能判断其性能或资源消耗。这不表示该 SQL 不存在或没有负载。"
        refs: tuple[str, ...] = ()
    else:
        fact, rows = chosen
        is_window = str(fact.measurement_semantics) == "SNAPSHOT_DELTA"
        descriptions = []
        for index, row in enumerate(rows[:10], start=1):
            values = [f"{label} {_number(row.get(key), divisor)}" for key, label, divisor in metrics_for(fact) if _number(row.get(key), divisor) is not None]
            if values:
                dimensions = [f"{key}={row[key]}" for key in ("database_oid", "user_oid", "top_level", "schema_name", "inst_id", "child_number", "plan_hash_value") if row.get(key) is not None]
                scope = f"（{'，'.join(dimensions)}）" if dimensions else ""
                descriptions.append(f"记录 {index}{scope}：" + "，".join(values) + "。")
        origin = "后台观察" if fact.followup_run_id is not None else "历史观测" if fact.source_id == "prior_turn.sql_observation" else "本轮观测"
        basis = "窗口增量统计" if is_window else "累计统计"
        analysis = f"已取得与指定 SQL 匹配的{basis}（{origin}，采集时间 {fact.captured_at}）。" + "\n\n" + "\n\n".join(descriptions)
        if fact.source_id == "prior_turn.sql_observation":
            analysis += "\n\n历史数据来自同一逻辑 Target，不能单凭此确认实际实例未切换。"
        statement = next((str(rows[0].get(key) or "").strip() for key in ("normalized_statement", "digest_text", "sql_text") if rows[0].get(key)), "")
        if statement:
            statement = " ".join(statement.split())
            statement = (statement[:160] + "…") if len(statement) > 160 else statement
            analysis += "\n\nSQL 文本：`" + statement.replace("`", "\\`") + "`。"
        if len(rows) > 1:
            analysis += "\n\n以上记录对应不同的统计维度或游标，未合并计算。"
        if fact.truncated or len(rows) > 10:
            analysis += "\n\n记录未全部展示，详情见原始证据。"
        if is_window:
            analysis += "\n\n以上仅适用于证据实际覆盖的窗口和候选集合，不能外推至未覆盖时段或全库排名。"
            if fact.warnings:
                analysis += "\n\n" + "\n".join(fact.warnings)
        else:
            analysis += "\n\n这些是累计值，不代表指定时间窗口；缺少业务基线时不能判断耗时是否异常。"
        analysis += "共享块命中或对象级扫描统计不能单独证明该 SQL 的扫描方式或根因。"
        if database_type == "POSTGRESQL" and any(row.get("statement_stats_reset") for row in rows):
            analysis += "语句统计全局重置时间见原始证据；单条语句仍可能被单独重置或淘汰，不能将该时间当作其确定起点。"
        refs = (fact.evidence_ref,)
        if fact.source_id != "prior_turn.sql_observation":
            prior = next((item for item in candidates if item.source_id == "prior_turn.sql_observation" and any(_matches(row, subject, identity) for row in _row_dicts(item))), None)
            if prior is not None:
                analysis += f"\n\n同一授权会话及逻辑 Target 存在相同标识的历史统计（采集时间 {prior.captured_at}）；口径和覆盖区间以原始证据为准，不能跨时段或跨口径直接合并，也不能单凭此确认实际实例未切换。"
                refs = (*refs, prior.evidence_ref)
    relevant_gaps = tuple(gap for gap in gaps if gap.source_id.startswith(("db.", "workload.sql.")))
    if relevant_gaps:
        analysis += "\n\n取证未完成项：" + "；".join(
            f"{gap.source_id}：{gap.detail}（{gap.code}）" for gap in relevant_gaps[:5]
        ) + "。这些是取证缺口，不能解释为 SQL 无负载或身份不匹配。"
    if fallback_reason:
        analysis += "\n\n" + fallback_reason
    solution = (
        "先核对统计身份及覆盖区间；仅在快照可比较且未发生可识别的重置或淘汰时计算执行次数、总耗时及平均耗时增量。"
        "累计最大耗时不能通过两次快照相减得到窗口最大值；需要逐次执行记录或专门采样。\n\n"
        "核对已有的执行计划是否与该 SQL、参数和时间匹配；如缺少匹配计划，补充计划证据后再判断访问路径。"
        "不要将对象级统计单独作为该 SQL 的根因。生产环境采集实际计划应先评估执行风险和权限。"
        "任何索引、参数或查询变更均需审批，并以可比窗口验证及准备回滚。"
    )
    if any(gap.source_id == tool_id for gap in relevant_gaps):
        solution = (
            f"先由平台维护者修复上述 {tool_id} 取证故障，再在同一 Target 上重跑指定 SQL 详情查询。"
            "已有历史或窗口数据可作为带时间边界的背景，不能替代本轮失败的取证。"
            "详情恢复后，继续取得匹配的执行计划和等待/资源证据，才能给出有依据的优化措施；"
            "目前不建议仅凭取证失败修改业务 SQL、索引或数据库参数。"
        )
    return SqlEvidenceAnswer(analysis=analysis, solution=solution, evidence_refs=refs, has_statistics=chosen is not None,
                             historical_only=chosen is not None and chosen[0].source_id == "prior_turn.sql_observation")
