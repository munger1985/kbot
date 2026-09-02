"""只从当前运行的可信数据库事实编译类型化 Action 参数。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class CompiledActionParameters:
    parameters: dict[str, Any]
    fact_refs: dict[str, str]
    rationale: str


def _verified_turn_rows(assessment, tool_id: str):
    for fact in assessment.evidence:
        if fact.trust_level != "SOURCE_VERIFIED" or fact.tool_id != tool_id:
            continue
        names = [str(item.get("name", "")).lower() for item in fact.columns]
        for values in fact.rows:
            yield dict(zip(names, values, strict=True)), fact.evidence_ref


def _session_from_turn(assessment, db_type: str):
    active_rows = tuple(_verified_turn_rows(assessment, "db.session.active"))
    for blocking, blocking_ref in _verified_turn_rows(
        assessment, "db.session.blocking_chain"
    ):
        session_id = blocking.get("blocking_session_id")
        if session_id is None:
            continue
        if db_type == "MYSQL":
            return CompiledActionParameters(
                parameters={"session_id": int(session_id)},
                fact_refs={"session_id": blocking_ref},
                rationale="目标连接来自本轮数据库直连的可信阻塞事实",
            )
        for active, active_ref in active_rows:
            if int(active.get("session_id", -1)) != int(session_id):
                continue
            serial_number = active.get("serial_number")
            instance_id = active.get("instance_id")
            if serial_number is None or instance_id is None:
                continue
            return CompiledActionParameters(
                parameters={
                    "session_id": int(session_id),
                    "serial_number": int(serial_number),
                    "instance_id": int(instance_id),
                },
                fact_refs={
                    "session_id": blocking_ref,
                    "serial_number": active_ref,
                    "instance_id": active_ref,
                },
                rationale="目标会话来自本轮数据库直连的可信活动会话和阻塞事实",
            )
    return None


def _oracle_cancel_sql_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.session.current_sql"
    ):
        instance_id = row.get("instance_id")
        session_id = row.get("session_id")
        serial_number = row.get("serial_number")
        sql_id = str(row.get("sql_id") or "").lower()
        status = str(row.get("status") or "").upper()
        if (
            instance_id is None
            or session_id is None
            or serial_number is None
            or len(sql_id) != 13
            or not sql_id.isalnum()
            or status != "ACTIVE"
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "session_id": int(session_id),
                "serial_number": int(serial_number),
                "instance_id": int(instance_id),
                "sql_id": sql_id,
            },
            fact_refs={
                "session_id": evidence_ref,
                "serial_number": evidence_ref,
                "instance_id": evidence_ref,
                "sql_id": evidence_ref,
            },
            rationale=(
                "会话、实例、序列号和当前 SQL_ID 均来自本轮数据库直连的可信事实"
            ),
        )
    return None


def _oracle_index_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(assessment, "db.index.health"):
        owner = row.get("owner")
        index_name = row.get("index_name")
        status = str(row.get("status") or "").upper()
        partitioned = str(row.get("partitioned") or "").upper()
        index_type = str(row.get("index_type") or "").upper()
        space_sufficient = str(row.get("space_sufficient") or "").upper()
        online = str(row.get("online_supported") or "NO").upper() == "YES"
        active_locks = int(row.get("active_table_locks") or 0)
        if (
            not owner
            or not index_name
            or status not in {"VALID", "UNUSABLE", "N/A"}
            or partitioned != "NO"
            or index_type not in {"NORMAL", "NORMAL/REV"}
            or space_sufficient != "YES"
            or (active_locks > 0 and not online)
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "index_ref": {
                    "schema": str(owner),
                    "object_type": "INDEX",
                    "object_name": str(index_name),
                },
                "online": online,
            },
            fact_refs={
                "index_ref": evidence_ref,
                "online": evidence_ref,
                "space_and_lock_check": evidence_ref,
            },
            rationale=(
                "索引身份、状态、类型、空间余量、活动锁和 ONLINE 能力均来自本轮数据库直连的可信对象事实"
            ),
        )
    return None


def _oracle_index_partition_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.index.partition.health"
    ):
        owner = row.get("owner")
        index_name = row.get("index_name")
        partition_name = row.get("partition_name")
        status = str(row.get("status") or "").upper()
        partitioned = str(row.get("partitioned") or "").upper()
        index_type = str(row.get("index_type") or "").upper()
        space_sufficient = str(row.get("space_sufficient") or "").upper()
        online = str(row.get("online_supported") or "NO").upper() == "YES"
        active_locks = int(row.get("active_table_locks") or 0)
        if (
            not owner
            or not index_name
            or not partition_name
            or status not in {"USABLE", "UNUSABLE", "N/A"}
            or partitioned != "YES"
            or index_type not in {"NORMAL", "NORMAL/REV"}
            or space_sufficient != "YES"
            or (active_locks > 0 and not online)
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "index_ref": {
                    "schema": str(owner),
                    "object_type": "INDEX",
                    "object_name": str(index_name),
                    "partition": str(partition_name),
                },
                "partition_name": str(partition_name),
                "online": online,
            },
            fact_refs={
                "index_ref": evidence_ref,
                "partition_name": evidence_ref,
                "online": evidence_ref,
                "space_and_lock_check": evidence_ref,
            },
            rationale=(
                "索引分区身份、状态、空间余量、活动锁和 ONLINE 能力均来自本轮数据库直连的可信对象事实"
            ),
        )
    return None


def _oracle_object_compile_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    allowed_types = {"PROCEDURE", "FUNCTION", "PACKAGE"}
    for row, evidence_ref in _verified_turn_rows(assessment, "db.object.status"):
        owner = row.get("owner")
        object_name = row.get("object_name")
        object_type = str(row.get("object_type") or "").upper()
        status = str(row.get("status") or "").upper()
        if (
            not owner
            or not object_name
            or object_type not in allowed_types
            or status != "INVALID"
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "object_type": object_type,
                "object_ref": {
                    "schema": str(owner),
                    "object_type": object_type,
                    "object_name": str(object_name),
                },
            },
            fact_refs={
                "object_type": evidence_ref,
                "object_ref": evidence_ref,
            },
            rationale=(
                "对象身份、类型和 INVALID 状态来自本轮数据库直连的可信对象事实"
            ),
        )
    return None


def _oracle_table_statistics_gather_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.table.statistics"
    ):
        owner = row.get("owner")
        table_name = row.get("table_name")
        temporary = str(row.get("temporary") or "").upper()
        last_analyzed = row.get("last_analyzed")
        stale_stats = str(row.get("stale_stats") or "").upper()
        stattype_locked = str(row.get("stattype_locked") or "").strip()
        if (
            not owner
            or not table_name
            or temporary != "N"
            or stattype_locked
            or (last_analyzed is not None and stale_stats != "YES")
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "table_ref": {
                    "schema": str(owner),
                    "object_type": "TABLE",
                    "object_name": str(table_name),
                }
            },
            fact_refs={"table_ref": evidence_ref},
            rationale=(
                "表身份、非临时属性、统计未锁定及统计缺失或过期状态"
                "来自本轮数据库直连的可信事实"
            ),
        )
    return None


def _oracle_scheduler_job_run_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.scheduler.job.status"
    ):
        owner = row.get("owner")
        job_name = row.get("job_name")
        enabled = str(row.get("enabled") or "").upper()
        state = str(row.get("state") or "").upper()
        try:
            run_count = int(row.get("run_count"))
            failure_count = int(row.get("failure_count"))
        except (TypeError, ValueError):
            continue
        if (
            not owner
            or not job_name
            or enabled != "TRUE"
            or state != "SCHEDULED"
            or run_count < 0
            or failure_count < 0
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "job_ref": {
                    "schema": str(owner),
                    "object_type": "SCHEDULER_JOB",
                    "object_name": str(job_name),
                },
                "previous_run_count": run_count,
                "previous_failure_count": failure_count,
            },
            fact_refs={
                "job_ref": evidence_ref,
                "previous_run_count": evidence_ref,
                "previous_failure_count": evidence_ref,
            },
            rationale=(
                "Job 身份、启用状态、SCHEDULED 状态及运行计数来自本轮"
                "数据库直连的可信事实"
            ),
        )
    return None


_TURN_COMPILERS: dict[str, Callable[[Any, str], CompiledActionParameters | None]] = {
    "session-terminate.v1": _session_from_turn,
    "oracle-session-cancel-sql.v1": _oracle_cancel_sql_from_turn,
    "oracle-index-rebuild.v1": _oracle_index_from_turn,
    "oracle-index-partition-rebuild.v1": _oracle_index_partition_from_turn,
    "oracle-object-compile.v1": _oracle_object_compile_from_turn,
    "oracle-table-statistics-gather.v1": (
        _oracle_table_statistics_gather_from_turn
    ),
    "oracle-scheduler-job-run.v1": _oracle_scheduler_job_run_from_turn,
}


class ActionCompilerRegistry:
    """按 Catalog 中的 compiler_id 选择确定性 Compiler。"""

    def compile_turn(self, *, compiler_id: str, assessment, db_type: str):
        compiler = _TURN_COMPILERS.get(compiler_id)
        if compiler is None:
            return None
        return compiler(assessment, db_type)
