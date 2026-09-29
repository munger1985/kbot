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


def _oracle_index_coalesce_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.index.coalesce_candidate"
    ):
        owner = row.get("owner")
        index_name = row.get("index_name")
        status = str(row.get("status") or "").upper()
        partitioned = str(row.get("partitioned") or "").upper()
        index_type = str(row.get("index_type") or "").upper()
        active_locks = int(row.get("active_table_locks") or 0)
        if (
            not owner
            or not index_name
            or status != "VALID"
            or partitioned != "NO"
            or index_type not in {"NORMAL", "NORMAL/REV"}
            or active_locks != 0
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "index_ref": {
                    "schema": str(owner),
                    "object_type": "INDEX",
                    "object_name": str(index_name),
                }
            },
            fact_refs={"index_ref": evidence_ref},
            rationale=(
                "索引身份、VALID 状态、非分区普通索引类型和无活动对象锁"
                "来自本轮动作专用数据库直连可信事实"
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


def _oracle_storage_from_turn(
    assessment,
    db_type: str,
    *,
    tool_id: str,
    operation: str,
):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(assessment, tool_id):
        file_name = str(row.get("file_name") or "")
        status = str(row.get("status") or "").upper()
        online_status = str(row.get("online_status") or "").upper()
        autoextensible = str(row.get("autoextensible") or "").upper()
        try:
            current_size_mb = int(row.get("current_size_mb"))
            current_next_mb = int(row.get("current_next_mb") or 0)
            current_max_size_mb = int(row.get("current_max_size_mb") or 0)
        except (TypeError, ValueError):
            continue
        if (
            not file_name
            or status != "AVAILABLE"
            or online_status != "ONLINE"
            or current_size_mb < 1
        ):
            continue
        if operation == "resize":
            try:
                requested_size_mb = int(row.get("requested_size_mb"))
            except (TypeError, ValueError):
                continue
            if not current_size_mb < requested_size_mb <= 1048576:
                continue
            return CompiledActionParameters(
                parameters={
                    "file_name": file_name,
                    "new_size_mb": requested_size_mb,
                },
                fact_refs={
                    "file_name": evidence_ref,
                    "new_size_mb": evidence_ref,
                    "current_size_mb": evidence_ref,
                },
                rationale=(
                    "文件身份、在线可用状态、当前大小和受限增长目标来自"
                    "本轮动作专用数据库直连可信事实"
                ),
            )
        try:
            requested_next_mb = int(row.get("requested_next_mb"))
            requested_max_size_mb = int(row.get("requested_max_size_mb"))
        except (TypeError, ValueError):
            continue
        if (
            not 1 <= requested_next_mb <= 1024
            or not current_size_mb < requested_max_size_mb <= 1048576
            or requested_next_mb > requested_max_size_mb - current_size_mb
            or (
                autoextensible == "YES"
                and current_next_mb == requested_next_mb
                and current_max_size_mb == requested_max_size_mb
            )
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "file_name": file_name,
                "next_mb": requested_next_mb,
                "max_size_mb": requested_max_size_mb,
            },
            fact_refs={
                "file_name": evidence_ref,
                "next_mb": evidence_ref,
                "max_size_mb": evidence_ref,
                "current_autoextend_state": evidence_ref,
            },
            rationale=(
                "文件身份、在线可用状态、当前自动扩展状态以及有限 NEXT 和"
                "MAXSIZE 目标来自本轮动作专用数据库直连可信事实"
            ),
        )
    return None


def _oracle_datafile_resize_from_turn(assessment, db_type: str):
    return _oracle_storage_from_turn(
        assessment,
        db_type,
        tool_id="db.storage.datafile.action_state",
        operation="resize",
    )


def _oracle_tempfile_resize_from_turn(assessment, db_type: str):
    return _oracle_storage_from_turn(
        assessment,
        db_type,
        tool_id="db.storage.tempfile.action_state",
        operation="resize",
    )


def _oracle_datafile_autoextend_from_turn(assessment, db_type: str):
    return _oracle_storage_from_turn(
        assessment,
        db_type,
        tool_id="db.storage.datafile.action_state",
        operation="autoextend",
    )


def _oracle_tempfile_autoextend_from_turn(assessment, db_type: str):
    return _oracle_storage_from_turn(
        assessment,
        db_type,
        tool_id="db.storage.tempfile.action_state",
        operation="autoextend",
    )


def _oracle_object_compile_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    allowed_types = {
        "PROCEDURE",
        "FUNCTION",
        "PACKAGE",
        "PACKAGE BODY",
        "TRIGGER",
        "VIEW",
        "TYPE",
        "TYPE BODY",
    }
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


def _oracle_table_statistics_state_from_turn(
    assessment,
    db_type: str,
    *,
    tool_id: str,
    require_locked: bool,
):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(assessment, tool_id):
        owner = row.get("owner")
        table_name = row.get("table_name")
        temporary = str(row.get("temporary") or "").upper()
        last_analyzed = row.get("last_analyzed")
        locked = bool(str(row.get("stattype_locked") or "").strip())
        if (
            not owner
            or not table_name
            or temporary != "N"
            or locked != require_locked
            or (not require_locked and last_analyzed is None)
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
                "表身份、非临时属性和统计锁状态来自本轮动作专用"
                "数据库直连可信事实"
            ),
        )
    return None


def _oracle_table_statistics_lock_from_turn(assessment, db_type: str):
    return _oracle_table_statistics_state_from_turn(
        assessment,
        db_type,
        tool_id="db.table.statistics.lock_candidate",
        require_locked=False,
    )


def _oracle_table_statistics_unlock_from_turn(assessment, db_type: str):
    return _oracle_table_statistics_state_from_turn(
        assessment,
        db_type,
        tool_id="db.table.statistics.unlock_candidate",
        require_locked=True,
    )


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


def _oracle_scheduler_job_state_from_turn(
    assessment,
    db_type: str,
    *,
    tool_id: str,
    expected_enabled: str,
    expected_state: str,
):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(assessment, tool_id):
        owner = row.get("owner")
        job_name = row.get("job_name")
        enabled = str(row.get("enabled") or "").upper()
        state = str(row.get("state") or "").upper()
        if (
            not owner
            or not job_name
            or enabled != expected_enabled
            or state != expected_state
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "job_ref": {
                    "schema": str(owner),
                    "object_type": "SCHEDULER_JOB",
                    "object_name": str(job_name),
                }
            },
            fact_refs={"job_ref": evidence_ref},
            rationale=(
                "Scheduler Job 身份、启用状态和运行状态来自本轮"
                "动作专用数据库直连可信事实"
            ),
        )
    return None


def _oracle_scheduler_job_enable_from_turn(assessment, db_type: str):
    return _oracle_scheduler_job_state_from_turn(
        assessment,
        db_type,
        tool_id="db.scheduler.job.enable_candidate",
        expected_enabled="FALSE",
        expected_state="DISABLED",
    )


def _oracle_scheduler_job_disable_from_turn(assessment, db_type: str):
    return _oracle_scheduler_job_state_from_turn(
        assessment,
        db_type,
        tool_id="db.scheduler.job.disable_candidate",
        expected_enabled="TRUE",
        expected_state="SCHEDULED",
    )


def _oracle_scheduler_job_stop_from_turn(assessment, db_type: str):
    return _oracle_scheduler_job_state_from_turn(
        assessment,
        db_type,
        tool_id="db.scheduler.job.stop_candidate",
        expected_enabled="TRUE",
        expected_state="RUNNING",
    )


def _oracle_user_state_from_turn(
    assessment,
    db_type: str,
    *,
    tool_id: str,
    require_locked: bool,
):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(assessment, tool_id):
        username = row.get("username")
        account_status = str(row.get("account_status") or "").upper()
        oracle_maintained = str(row.get("oracle_maintained") or "").upper()
        common = str(row.get("common") or "").upper()
        locked = "LOCKED" in account_status
        if (
            not username
            or oracle_maintained != "N"
            or common != "NO"
            or locked != require_locked
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "user_ref": {
                    "schema": str(username),
                    "object_type": "USER",
                    "object_name": str(username),
                }
            },
            fact_refs={"user_ref": evidence_ref},
            rationale=(
                "用户身份、账号锁状态、非 Oracle 维护及本地用户属性来自"
                "本轮动作专用数据库直连可信事实"
            ),
        )
    return None


def _oracle_user_lock_from_turn(assessment, db_type: str):
    return _oracle_user_state_from_turn(
        assessment,
        db_type,
        tool_id="db.user.lock_candidate",
        require_locked=False,
    )


def _oracle_user_unlock_from_turn(assessment, db_type: str):
    return _oracle_user_state_from_turn(
        assessment,
        db_type,
        tool_id="db.user.unlock_candidate",
        require_locked=True,
    )


def _oracle_user_password_expire_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.user.password_expire_candidate"
    ):
        username = row.get("username")
        account_status = str(row.get("account_status") or "").upper()
        oracle_maintained = str(row.get("oracle_maintained") or "").upper()
        common = str(row.get("common") or "").upper()
        if (
            not username
            or "EXPIRED" in account_status
            or oracle_maintained != "N"
            or common != "NO"
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "user_ref": {
                    "schema": str(username),
                    "object_type": "USER",
                    "object_name": str(username),
                }
            },
            fact_refs={"user_ref": evidence_ref},
            rationale=(
                "用户身份、密码未过期、非 Oracle 维护及本地用户属性来自"
                "本轮动作专用数据库直连可信事实"
            ),
        )
    return None


_DYNAMIC_PARAMETER_VALUES = {
    "cursor_sharing": {"EXACT", "FORCE"},
    "optimizer_mode": {"ALL_ROWS", "FIRST_ROWS"},
    "statistics_level": {"BASIC", "TYPICAL", "ALL"},
}
_SYSTEM_PRIVILEGES = {
    "CREATE SESSION",
    "CREATE TABLE",
    "CREATE VIEW",
    "CREATE PROCEDURE",
    "CREATE SEQUENCE",
    "CREATE SYNONYM",
    "CREATE TRIGGER",
    "CREATE TYPE",
}
_OBJECT_PRIVILEGE_TYPES = {
    "SELECT": {"TABLE", "VIEW", "SEQUENCE"},
    "READ": {"TABLE", "VIEW"},
    "INSERT": {"TABLE", "VIEW"},
    "UPDATE": {"TABLE", "VIEW"},
    "DELETE": {"TABLE", "VIEW"},
    "EXECUTE": {"PROCEDURE", "FUNCTION", "PACKAGE", "TYPE"},
}


def _oracle_dynamic_parameter_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.parameter.dynamic_state"
    ):
        name = str(row.get("parameter_name") or "").lower()
        current = str(row.get("current_value") or "").upper()
        requested = str(row.get("requested_value") or "").upper()
        modifiable = str(row.get("issys_modifiable") or "").upper()
        if (
            requested not in _DYNAMIC_PARAMETER_VALUES.get(name, set())
            or current == requested
            or modifiable != "IMMEDIATE"
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "parameter_name": name,
                "parameter_value": requested,
            },
            fact_refs={
                "parameter_name": evidence_ref,
                "parameter_value": evidence_ref,
                "current_value": evidence_ref,
            },
            rationale=(
                "动态参数身份、当前值、即时可修改属性和受限目标值来自"
                "本轮动作专用数据库直连可信事实"
            ),
        )
    return None


def _oracle_resource_plan_from_turn(assessment, db_type: str):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.resource_manager.plan_state"
    ):
        plan_name = str(row.get("resource_plan_name") or "")
        current_plan = str(row.get("current_plan_name") or "")
        status = str(row.get("status") or "").upper()
        if not plan_name or status != "ACTIVE" or plan_name.upper() == current_plan.upper():
            continue
        return CompiledActionParameters(
            parameters={"resource_plan_name": plan_name},
            fact_refs={
                "resource_plan_name": evidence_ref,
                "current_plan_name": evidence_ref,
            },
            rationale=(
                "Resource Manager Plan 身份、ACTIVE 状态和当前 Plan 来自"
                "本轮动作专用数据库直连可信事实"
            ),
        )
    return None


def _oracle_system_privilege_from_turn(
    assessment, db_type: str, *, require_granted: bool
):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.user.system_privilege_state"
    ):
        grantee = str(row.get("grantee_name") or "")
        privilege = str(row.get("privilege") or "").upper()
        granted = str(row.get("is_granted") or "").upper() == "YES"
        if (
            not grantee
            or privilege not in _SYSTEM_PRIVILEGES
            or granted != require_granted
            or str(row.get("oracle_maintained") or "").upper() != "N"
            or str(row.get("common") or "").upper() != "NO"
        ):
            continue
        return CompiledActionParameters(
            parameters={"grantee_name": grantee, "privilege": privilege},
            fact_refs={"grantee_name": evidence_ref, "privilege": evidence_ref},
            rationale=(
                "本地应用用户身份、精确系统权限及当前授权状态来自"
                "本轮动作专用数据库直连可信事实"
            ),
        )
    return None


def _oracle_system_privilege_grant_from_turn(assessment, db_type: str):
    return _oracle_system_privilege_from_turn(
        assessment, db_type, require_granted=False
    )


def _oracle_system_privilege_revoke_from_turn(assessment, db_type: str):
    return _oracle_system_privilege_from_turn(
        assessment, db_type, require_granted=True
    )


def _oracle_object_privilege_from_turn(
    assessment, db_type: str, *, require_granted: bool
):
    if db_type != "ORACLE":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.user.object_privilege_state"
    ):
        owner = str(row.get("owner") or "")
        object_name = str(row.get("object_name") or "")
        object_type = str(row.get("object_type") or "").upper()
        grantee = str(row.get("grantee_name") or "")
        privilege = str(row.get("privilege") or "").upper()
        granted = str(row.get("is_granted") or "").upper() == "YES"
        if (
            not owner
            or not object_name
            or not grantee
            or object_type not in _OBJECT_PRIVILEGE_TYPES.get(privilege, set())
            or granted != require_granted
            or str(row.get("oracle_maintained") or "").upper() != "N"
            or str(row.get("common") or "").upper() != "NO"
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "privilege": privilege,
                "object_ref": {
                    "schema": owner,
                    "object_type": object_type,
                    "object_name": object_name,
                },
                "grantee_name": grantee,
            },
            fact_refs={
                "privilege": evidence_ref,
                "object_ref": evidence_ref,
                "grantee_name": evidence_ref,
            },
            rationale=(
                "有效对象、本地应用用户、精确对象权限及当前授权状态来自"
                "本轮动作专用数据库直连可信事实"
            ),
        )
    return None


def _oracle_object_privilege_grant_from_turn(assessment, db_type: str):
    return _oracle_object_privilege_from_turn(
        assessment, db_type, require_granted=False
    )


def _oracle_object_privilege_revoke_from_turn(assessment, db_type: str):
    return _oracle_object_privilege_from_turn(
        assessment, db_type, require_granted=True
    )


def _mysql_query_terminate_from_turn(assessment, db_type: str):
    if db_type != "MYSQL":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.session.current_sql"
    ):
        session_id = row.get("session_id")
        digest = str(row.get("digest") or "")
        if session_id is None or not digest:
            continue
        return CompiledActionParameters(
            parameters={"session_id": int(session_id)},
            fact_refs={"session_id": evidence_ref},
            rationale=(
                "目标连接和当前Digest来自本轮MySQL直连可信事实；"
                "动作只终止当前查询，不选择断开连接"
            ),
        )
    return None


def _mysql_table_analyze_from_turn(assessment, db_type: str):
    if db_type != "MYSQL":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.mysql.statistics.health"
    ):
        schema = str(row.get("table_schema") or "")
        table = str(row.get("table_name") or "")
        missing_cardinality = int(row.get("indexes_without_cardinality") or 0)
        if not schema or not table or (
            row.get("update_time") is not None and missing_cardinality == 0
        ):
            continue
        return CompiledActionParameters(
            parameters={
                "table_ref": {
                    "schema": schema,
                    "object_type": "TABLE",
                    "object_name": table,
                }
            },
            fact_refs={"table_ref": evidence_ref},
            rationale=(
                "表身份以及统计缺失或索引基数缺失状态来自本轮MySQL直连可信事实"
            ),
        )
    return None


def _mysql_event_state_from_turn(
    assessment, db_type: str, *, desired_status: str
):
    if db_type != "MYSQL":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.scheduler.job.status"
    ):
        schema = str(row.get("event_schema") or "")
        name = str(row.get("event_name") or "")
        status = str(row.get("status") or "").upper()
        if not schema or not name or status == desired_status:
            continue
        if status not in {"ENABLED", "DISABLED", "SLAVESIDE_DISABLED"}:
            continue
        return CompiledActionParameters(
            parameters={
                "event_ref": {
                    "schema": schema,
                    "object_type": "EVENT",
                    "object_name": name,
                }
            },
            fact_refs={"event_ref": evidence_ref},
            rationale=(
                f"Event身份和当前{status}状态来自本轮MySQL直连可信事实，"
                f"批准目标为{desired_status}"
            ),
        )
    return None


def _mysql_event_enable_from_turn(assessment, db_type: str):
    return _mysql_event_state_from_turn(
        assessment, db_type, desired_status="ENABLED"
    )


def _mysql_event_disable_from_turn(assessment, db_type: str):
    return _mysql_event_state_from_turn(
        assessment, db_type, desired_status="DISABLED"
    )


def _mysql_set_persist_from_turn(assessment, db_type: str):
    if db_type != "MYSQL":
        return None
    allowed = {
        "max_connections": (10, 100_000),
        "innodb_io_capacity": (100, 200_000),
        "thread_cache_size": (0, 16_384),
    }
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.instance.parameters"
    ):
        name = str(row.get("variable_name") or "").lower()
        requested = row.get("requested_value")
        if name not in allowed or requested is None:
            continue
        try:
            value = int(requested)
        except (TypeError, ValueError):
            continue
        minimum, maximum = allowed[name]
        if not minimum <= value <= maximum:
            continue
        return CompiledActionParameters(
            parameters={
                "parameter_name": name,
                "parameter_value": value,
            },
            fact_refs={
                "parameter_name": evidence_ref,
                "parameter_value": evidence_ref,
            },
            rationale=(
                "变量身份、当前值和用户确认的目标值命中MySQL持久化参数严格白名单"
            ),
        )
    return None


def _mysql_replication_state_from_turn(
    assessment, db_type: str, *, start: bool
):
    if db_type != "MYSQL":
        return None
    for row, evidence_ref in _verified_turn_rows(
        assessment, "db.mysql.replication.channel_status"
    ):
        channel = str(row.get("channel_name") or "")
        receiver = str(row.get("receiver_state") or "").upper()
        applier = str(row.get("applier_state") or "").upper()
        receiver_error = int(row.get("receiver_error_number") or 0)
        applier_error = int(row.get("applier_error_number") or 0)
        running = receiver in {"ON", "RUNNING", "CONNECTING"} and applier in {
            "ON", "RUNNING", "APPLYING_QUEUE",
        }
        if running == start or (start and (receiver_error or applier_error)):
            continue
        return CompiledActionParameters(
            parameters={"channel_name": channel},
            fact_refs={"channel_name": evidence_ref},
            rationale=(
                "复制Channel身份、receiver/applier状态和错误码来自本轮MySQL直连可信事实"
            ),
        )
    return None


def _mysql_replication_start_from_turn(assessment, db_type: str):
    return _mysql_replication_state_from_turn(
        assessment, db_type, start=True
    )


def _mysql_replication_stop_from_turn(assessment, db_type: str):
    return _mysql_replication_state_from_turn(
        assessment, db_type, start=False
    )


_TURN_COMPILERS: dict[str, Callable[[Any, str], CompiledActionParameters | None]] = {
    "session-terminate.v1": _session_from_turn,
    "oracle-session-cancel-sql.v1": _oracle_cancel_sql_from_turn,
    "oracle-index-rebuild.v1": _oracle_index_from_turn,
    "oracle-index-coalesce.v1": _oracle_index_coalesce_from_turn,
    "oracle-index-partition-rebuild.v1": _oracle_index_partition_from_turn,
    "oracle-datafile-resize.v1": _oracle_datafile_resize_from_turn,
    "oracle-tempfile-resize.v1": _oracle_tempfile_resize_from_turn,
    "oracle-datafile-autoextend.v1": _oracle_datafile_autoextend_from_turn,
    "oracle-tempfile-autoextend.v1": _oracle_tempfile_autoextend_from_turn,
    "oracle-object-compile.v1": _oracle_object_compile_from_turn,
    "oracle-table-statistics-gather.v1": (
        _oracle_table_statistics_gather_from_turn
    ),
    "oracle-table-statistics-lock.v1": (
        _oracle_table_statistics_lock_from_turn
    ),
    "oracle-table-statistics-unlock.v1": (
        _oracle_table_statistics_unlock_from_turn
    ),
    "oracle-scheduler-job-run.v1": _oracle_scheduler_job_run_from_turn,
    "oracle-scheduler-job-enable.v1": _oracle_scheduler_job_enable_from_turn,
    "oracle-scheduler-job-disable.v1": (
        _oracle_scheduler_job_disable_from_turn
    ),
    "oracle-scheduler-job-stop.v1": _oracle_scheduler_job_stop_from_turn,
    "oracle-user-lock.v1": _oracle_user_lock_from_turn,
    "oracle-user-unlock.v1": _oracle_user_unlock_from_turn,
    "oracle-user-password-expire.v1": (
        _oracle_user_password_expire_from_turn
    ),
    "oracle-dynamic-parameter-set.v1": _oracle_dynamic_parameter_from_turn,
    "oracle-resource-manager-plan-switch.v1": _oracle_resource_plan_from_turn,
    "oracle-system-privilege-grant.v1": (
        _oracle_system_privilege_grant_from_turn
    ),
    "oracle-system-privilege-revoke.v1": (
        _oracle_system_privilege_revoke_from_turn
    ),
    "oracle-object-privilege-grant.v1": (
        _oracle_object_privilege_grant_from_turn
    ),
    "oracle-object-privilege-revoke.v1": (
        _oracle_object_privilege_revoke_from_turn
    ),
    "mysql-query-terminate.v1": _mysql_query_terminate_from_turn,
    "mysql-table-analyze.v1": _mysql_table_analyze_from_turn,
    "mysql-event-enable.v1": _mysql_event_enable_from_turn,
    "mysql-event-disable.v1": _mysql_event_disable_from_turn,
    "mysql-set-persist.v1": _mysql_set_persist_from_turn,
    "mysql-replication-start.v1": _mysql_replication_start_from_turn,
    "mysql-replication-stop.v1": _mysql_replication_stop_from_turn,
}


class ActionCompilerRegistry:
    """按 Catalog 中的 compiler_id 选择确定性 Compiler。"""

    def compile_turn(self, *, compiler_id: str, assessment, db_type: str):
        compiler = _TURN_COMPILERS.get(compiler_id)
        if compiler is None:
            return None
        return compiler(assessment, db_type)
