"""动作专用模板与渲染结果校验注册表。"""

from __future__ import annotations

import re

from .contracts import ActionTemplateDefinition


_PLACEHOLDER = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")
_COMMENT_OR_BLOCK = re.compile(r"(--|/\*|\*/)")
_SEMICOLON = re.compile(r";")
_PLSQL_VALIDATORS = frozenset(
    {
        "oracle-table-statistics-gather.v1",
        "oracle-table-statistics-lock.v1",
        "oracle-table-statistics-unlock.v1",
        "oracle-scheduler-job-run.v1",
        "oracle-scheduler-job-enable.v1",
        "oracle-scheduler-job-disable.v1",
        "oracle-scheduler-job-stop.v1",
    }
)


def _session_pattern(db_type: str, *, rendered: bool) -> str:
    if db_type == "ORACLE":
        values = (
            r"[1-9][0-9]*,[1-9][0-9]*,@[1-9][0-9]*"
            if rendered
            else r"\{\{session_id\}\},\{\{serial_number\}\},@\{\{instance_id\}\}"
        )
        return rf"ALTER SYSTEM DISCONNECT SESSION '{values}' IMMEDIATE"
    value = r"[1-9][0-9]*" if rendered else r"\{\{session_id\}\}"
    return rf"KILL CONNECTION {value}"


def _cancel_sql_pattern(*, rendered: bool) -> str:
    values = (
        r"[1-9][0-9]*,[1-9][0-9]*,@[1-9][0-9]*,[0-9a-z]{13}"
        if rendered
        else (
            r"\{\{session_id\}\},\{\{serial_number\}\},"
            r"@\{\{instance_id\}\},\{\{sql_id\}\}"
        )
    )
    return rf"ALTER SYSTEM CANCEL SQL '{values}' IMMEDIATE"


def _index_rebuild_pattern(*, rendered: bool) -> str:
    if rendered:
        identifier = r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        return rf"ALTER INDEX {identifier}\.{identifier} REBUILD(?: ONLINE)?"
    return r"ALTER INDEX \{\{index_ref\}\} REBUILD\{\{online\}\}"


def _index_partition_rebuild_pattern(*, rendered: bool) -> str:
    if rendered:
        identifier = r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        return (
            rf"ALTER INDEX {identifier}\.{identifier} REBUILD PARTITION "
            rf"{identifier}(?: ONLINE)?"
        )
    return (
        r"ALTER INDEX \{\{index_ref\}\} REBUILD PARTITION "
        r"\{\{partition_name\}\}\{\{online\}\}"
    )


def _object_compile_pattern(*, rendered: bool) -> str:
    object_type = r"(?:PROCEDURE|FUNCTION|PACKAGE|TRIGGER|VIEW|TYPE)"
    if rendered:
        identifier = r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        return (
            rf"ALTER {object_type} {identifier}\.{identifier} "
            r"COMPILE(?: BODY)?"
        )
    return r"ALTER \{\{object_type\}\} \{\{object_ref\}\} COMPILE"


def _table_statistics_gather_pattern(*, rendered: bool) -> str:
    if rendered:
        identifier = r"[A-Za-z][A-Za-z0-9_$#]{0,127}"
        table_ref = (
            rf"ownname => '{identifier}', tabname => '{identifier}'"
        )
    else:
        table_ref = r"\{\{table_ref\}\}"
    return (
        rf"BEGIN DBMS_STATS\.GATHER_TABLE_STATS\({table_ref}, "
        r"estimate_percent => DBMS_STATS\.AUTO_SAMPLE_SIZE, "
        r"method_opt => 'FOR ALL COLUMNS SIZE AUTO', cascade => TRUE, "
        r"no_invalidate => DBMS_STATS\.AUTO_INVALIDATE\); END;"
    )


def _table_statistics_state_pattern(operation: str, *, rendered: bool) -> str:
    if rendered:
        identifier = r"[A-Za-z][A-Za-z0-9_$#]{0,127}"
        table_ref = (
            rf"ownname => '{identifier}', tabname => '{identifier}'"
        )
    else:
        table_ref = r"\{\{table_ref\}\}"
    return rf"BEGIN DBMS_STATS\.{operation}\({table_ref}\); END;"


def _scheduler_job_run_pattern(*, rendered: bool) -> str:
    if rendered:
        identifier = r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        job_ref = rf"job_name => '{identifier}\.{identifier}'"
    else:
        job_ref = r"\{\{job_ref\}\}"
    return (
        rf"BEGIN DBMS_SCHEDULER\.RUN_JOB\({job_ref}, "
        r"use_current_session => FALSE\); END;"
    )


def _scheduler_job_state_pattern(operation: str, *, rendered: bool) -> str:
    if rendered:
        identifier = r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        argument = "job_name" if operation == "STOP_JOB" else "name"
        job_ref = rf"{argument} => '{identifier}\.{identifier}'"
    else:
        job_ref = r"\{\{job_ref\}\}"
    suffix = ", force => FALSE" if operation in {"DISABLE", "STOP_JOB"} else ""
    return rf"BEGIN DBMS_SCHEDULER\.{operation}\({job_ref}{suffix}\); END;"


def _user_state_pattern(operation: str, *, rendered: bool) -> str:
    user_ref = (
        r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        if rendered
        else r"\{\{user_ref\}\}"
    )
    return rf"ALTER USER {user_ref} ACCOUNT {operation}"


def _user_password_expire_pattern(*, rendered: bool) -> str:
    user_ref = (
        r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        if rendered
        else r"\{\{user_ref\}\}"
    )
    return rf"ALTER USER {user_ref} PASSWORD EXPIRE"


def _object_pattern(prefix: str, placeholder: str, *, rendered: bool) -> str:
    if rendered:
        identifier = r'"[A-Za-z][A-Za-z0-9_$#]{0,127}"'
        return rf"{prefix} {identifier}\.{identifier}"
    return rf"{prefix} \{{\{{{placeholder}\}}\}}"


def _exact(pattern: str, command: str) -> None:
    if re.fullmatch(pattern, command, re.IGNORECASE) is None:
        raise ValueError("Action 命令不在动作专用精确 Allowlist")


def validate_action_template(
    command: str, definition: ActionTemplateDefinition
) -> None:
    text = command.strip()
    if (
        not text
        or len(text) > 4000
        or "\n\n" in text
        or _COMMENT_OR_BLOCK.search(text)
        or (
            definition.validator_id not in _PLSQL_VALIDATORS
            and _SEMICOLON.search(text)
        )
    ):
        raise ValueError("Action 命令模板长度或结构无效")
    placeholders = _PLACEHOLDER.findall(text)
    expected = [item.name for item in definition.parameters]
    if definition.validator_id.startswith("oracle-scheduler-job-"):
        expected = ["job_ref"]
    if sorted(placeholders) != sorted(expected):
        raise ValueError("Action 命令占位符与参数定义不一致")
    if definition.validator_id == "session-control.v1":
        _exact(
            _session_pattern(definition.db_type, rendered=False),
            text,
        )
    elif definition.validator_id == "oracle-session-cancel-sql.v1":
        if definition.db_type != "ORACLE":
            raise ValueError("取消 SQL Validator 仅支持 Oracle")
        _exact(_cancel_sql_pattern(rendered=False), text)
    elif definition.validator_id == "oracle-index-rebuild.v1":
        if definition.db_type != "ORACLE":
            raise ValueError("索引重建 Validator 仅支持 Oracle")
        _exact(_index_rebuild_pattern(rendered=False), text)
    elif definition.validator_id == "oracle-index-partition-rebuild.v1":
        if definition.db_type != "ORACLE":
            raise ValueError("索引分区重建 Validator 仅支持 Oracle")
        _exact(_index_partition_rebuild_pattern(rendered=False), text)
    elif definition.validator_id == "oracle-object-compile.v1":
        if definition.db_type != "ORACLE":
            raise ValueError("对象编译 Validator 仅支持 Oracle")
        _exact(_object_compile_pattern(rendered=False), text)
    elif definition.validator_id == "oracle-table-statistics-gather.v1":
        if definition.db_type != "ORACLE":
            raise ValueError("统计信息收集 Validator 仅支持 Oracle")
        _exact(_table_statistics_gather_pattern(rendered=False), text)
    elif definition.validator_id == "oracle-table-statistics-lock.v1":
        _exact(
            _table_statistics_state_pattern(
                "LOCK_TABLE_STATS", rendered=False
            ),
            text,
        )
    elif definition.validator_id == "oracle-table-statistics-unlock.v1":
        _exact(
            _table_statistics_state_pattern(
                "UNLOCK_TABLE_STATS", rendered=False
            ),
            text,
        )
    elif definition.validator_id == "oracle-scheduler-job-run.v1":
        if definition.db_type != "ORACLE":
            raise ValueError("Scheduler Job Validator 仅支持 Oracle")
        _exact(_scheduler_job_run_pattern(rendered=False), text)
    elif definition.validator_id == "oracle-scheduler-job-enable.v1":
        _exact(_scheduler_job_state_pattern("ENABLE", rendered=False), text)
    elif definition.validator_id == "oracle-scheduler-job-disable.v1":
        _exact(_scheduler_job_state_pattern("DISABLE", rendered=False), text)
    elif definition.validator_id == "oracle-scheduler-job-stop.v1":
        _exact(_scheduler_job_state_pattern("STOP_JOB", rendered=False), text)
    elif definition.validator_id == "oracle-user-lock.v1":
        _exact(_user_state_pattern("LOCK", rendered=False), text)
    elif definition.validator_id == "oracle-user-unlock.v1":
        _exact(_user_state_pattern("UNLOCK", rendered=False), text)
    elif definition.validator_id == "oracle-user-password-expire.v1":
        _exact(_user_password_expire_pattern(rendered=False), text)
    elif definition.validator_id == "manual-truncate-table.v1":
        _exact(_object_pattern("TRUNCATE TABLE", "table_ref", rendered=False), text)
    elif definition.validator_id == "manual-drop-table.v1":
        _exact(_object_pattern("DROP TABLE", "table_ref", rendered=False), text)
    elif definition.validator_id == "manual-archive-cleanup.v1":
        _exact(
            r"DELETE ARCHIVELOG ALL COMPLETED BEFORE 'SYSDATE-\{\{retention_days\}\}'",
            text,
        )
    elif definition.validator_id == "manual-restore.v1":
        _exact(r"RESTORE DATABASE", text)
    elif definition.validator_id == "manual-recover.v1":
        _exact(r"RECOVER DATABASE", text)
    elif definition.validator_id == "manual-backup-delete.v1":
        _exact(r"DELETE NOPROMPT BACKUP", text)
    elif definition.validator_id == "manual-ha-failover.v1":
        _exact(r"ALTER DATABASE ACTIVATE PHYSICAL STANDBY DATABASE", text)
    elif definition.execution_mode != "UNSUPPORTED":
        raise ValueError("Action Validator 未登记")


def validate_rendered_action(
    command: str, *, definition: ActionTemplateDefinition
) -> None:
    if (
        _PLACEHOLDER.search(command)
        or _COMMENT_OR_BLOCK.search(command)
        or (
            definition.validator_id not in _PLSQL_VALIDATORS
            and _SEMICOLON.search(command)
        )
    ):
        raise ValueError("渲染后的 Action 命令结构无效")
    if definition.validator_id == "session-control.v1":
        _exact(
            _session_pattern(definition.db_type, rendered=True),
            command,
        )
    elif definition.validator_id == "oracle-session-cancel-sql.v1":
        _exact(_cancel_sql_pattern(rendered=True), command)
    elif definition.validator_id == "oracle-index-rebuild.v1":
        _exact(_index_rebuild_pattern(rendered=True), command)
    elif definition.validator_id == "oracle-index-partition-rebuild.v1":
        _exact(_index_partition_rebuild_pattern(rendered=True), command)
    elif definition.validator_id == "oracle-object-compile.v1":
        _exact(_object_compile_pattern(rendered=True), command)
    elif definition.validator_id == "oracle-table-statistics-gather.v1":
        _exact(_table_statistics_gather_pattern(rendered=True), command)
    elif definition.validator_id == "oracle-table-statistics-lock.v1":
        _exact(
            _table_statistics_state_pattern("LOCK_TABLE_STATS", rendered=True),
            command,
        )
    elif definition.validator_id == "oracle-table-statistics-unlock.v1":
        _exact(
            _table_statistics_state_pattern(
                "UNLOCK_TABLE_STATS", rendered=True
            ),
            command,
        )
    elif definition.validator_id == "oracle-scheduler-job-run.v1":
        _exact(_scheduler_job_run_pattern(rendered=True), command)
    elif definition.validator_id == "oracle-scheduler-job-enable.v1":
        _exact(_scheduler_job_state_pattern("ENABLE", rendered=True), command)
    elif definition.validator_id == "oracle-scheduler-job-disable.v1":
        _exact(_scheduler_job_state_pattern("DISABLE", rendered=True), command)
    elif definition.validator_id == "oracle-scheduler-job-stop.v1":
        _exact(_scheduler_job_state_pattern("STOP_JOB", rendered=True), command)
    elif definition.validator_id == "oracle-user-lock.v1":
        _exact(_user_state_pattern("LOCK", rendered=True), command)
    elif definition.validator_id == "oracle-user-unlock.v1":
        _exact(_user_state_pattern("UNLOCK", rendered=True), command)
    elif definition.validator_id == "oracle-user-password-expire.v1":
        _exact(_user_password_expire_pattern(rendered=True), command)
    elif definition.validator_id == "manual-truncate-table.v1":
        _exact(_object_pattern("TRUNCATE TABLE", "table_ref", rendered=True), command)
    elif definition.validator_id == "manual-drop-table.v1":
        _exact(_object_pattern("DROP TABLE", "table_ref", rendered=True), command)
    elif definition.validator_id == "manual-archive-cleanup.v1":
        _exact(
            r"DELETE ARCHIVELOG ALL COMPLETED BEFORE 'SYSDATE-[1-9][0-9]{0,3}'",
            command,
        )
    elif definition.validator_id == "manual-restore.v1":
        _exact(r"RESTORE DATABASE", command)
    elif definition.validator_id == "manual-recover.v1":
        _exact(r"RECOVER DATABASE", command)
    elif definition.validator_id == "manual-backup-delete.v1":
        _exact(r"DELETE NOPROMPT BACKUP", command)
    elif definition.validator_id == "manual-ha-failover.v1":
        _exact(
            r"ALTER DATABASE ACTIVATE PHYSICAL STANDBY DATABASE", command
        )
    else:
        raise ValueError("Action Validator 未登记")
