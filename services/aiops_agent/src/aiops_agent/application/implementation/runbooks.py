"""从已验证环境事实确定性编译数据库实施方案。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

from aiops_agent.contracts.implementation import (
    ImplementationRunbook,
    RunbookApplicability,
    RunbookCommand,
    RunbookCommandType,
    RunbookParameterStatus,
    RunbookPhase,
    RunbookRequiredInput,
    RunbookResolvedParameter,
    RunbookStatus,
    RunbookStep,
)
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import ImplementationProfile


def _command(
    command_id: str,
    command_type: RunbookCommandType,
    title: str,
    content: str,
    *notes: str,
) -> RunbookCommand:
    return RunbookCommand(
        command_id=command_id,
        command_type=command_type,
        title=title,
        content=content.strip(),
        notes=tuple(notes),
    )


def _first_row(
    evidence: Iterable[TurnEvidenceFact], tool_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    for fact in evidence:
        if fact.tool_id != tool_id or not fact.rows:
            continue
        names = [str(item.get("name") or "").lower() for item in fact.columns]
        return dict(zip(names, fact.rows[0], strict=False)), fact.evidence_ref
    return None, None


def _text(row: dict[str, Any] | None, key: str) -> str:
    if row is None:
        return ""
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _integer(row: dict[str, Any] | None, key: str) -> int:
    try:
        return int(float(_text(row, key) or "0"))
    except ValueError:
        return 0


def _setting_contains(value: str, *parts: str) -> bool:
    """忽略空白和大小写判断 Oracle 参数是否已包含目标语义。"""
    normalized = re.sub(r"\s+", "", value).upper()
    return all(
        re.sub(r"\s+", "", part).upper() in normalized
        for part in parts
        if part
    )


def _state_item(label: str, value: str, status: str) -> dict[str, Any]:
    return {"label": label, "value": value or "未取得", "status": status}


@dataclass(frozen=True)
class _ImplementationParameter:
    key: str
    label: str
    value: str
    status: RunbookParameterStatus
    source: str


_ADG_EXTERNAL_INPUTS = {
    "STANDBY_HOST": (
        "备库主机",
        "备库主机名或可解析地址，并确认主备双向网络和防火墙。",
    ),
    "ORACLE_HOME": (
        "备库 Oracle Home",
        "备库实际 Oracle Home；必须与主库数据库版本和补丁级别一致。",
    ),
    "STANDBY_STORAGE": (
        "备库存储配置",
        "确认备库 ASM/OMF 磁盘组或文件系统路径，以及数据文件、日志和 FRA 的放置规则。",
    ),
    "PRIMARY_LOG_STORAGE": (
        "主库 Standby Redo Log 路径",
        "主库未启用 OMF，需要确认新增 Standby Redo Log 的文件系统或 ASM 成员路径。",
    ),
    "PRIMARY_FRA_DEST": (
        "主库 FRA 路径",
        "当前主库未配置 FRA，需要先确定恢复区路径。",
    ),
    "PRIMARY_FRA_SIZE": (
        "主库 FRA 容量",
        "当前主库未配置 FRA，需要按归档量和保留窗口确定容量。",
    ),
}


def _explicit_parameter(context: dict[str, Any], key: str) -> str:
    values = dict(context.get("implementation_parameters") or {})
    value = values.get(key)
    return "" if value is None else str(value).strip()


def _oracle_name_with_suffix(value: str, suffix: str, limit: int) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_$#]", "_", value.strip())
    if not normalized:
        return ""
    if not normalized[0].isalpha():
        normalized = f"D{normalized}"
    return f"{normalized[: max(1, limit - len(suffix))]}{suffix}"


def _resolve_adg_parameters(
    *,
    identity: dict[str, Any] | None,
    precheck: dict[str, Any] | None,
    context: dict[str, Any],
) -> tuple[dict[str, str], tuple[RunbookResolvedParameter, ...]]:
    parameters: list[_ImplementationParameter] = []

    def add(
        key: str,
        label: str,
        value: str,
        status: RunbookParameterStatus,
        source: str,
    ) -> None:
        if value:
            parameters.append(
                _ImplementationParameter(key, label, value, status, source)
            )

    primary_db_name = _text(precheck, "database_name")
    primary_unique_name = _text(precheck, "db_unique_name")
    add(
        "PRIMARY_DB_NAME",
        "主库 DB_NAME",
        primary_db_name,
        RunbookParameterStatus.VERIFIED,
        "db.ha.adg_precheck:V$DATABASE.NAME",
    )
    add(
        "PRIMARY_DB_UNIQUE_NAME",
        "主库 DB_UNIQUE_NAME",
        primary_unique_name,
        RunbookParameterStatus.VERIFIED,
        "db.ha.adg_precheck:V$PARAMETER",
    )

    standby_unique_name = _explicit_parameter(
        context, "STANDBY_DB_UNIQUE_NAME"
    ) or _oracle_name_with_suffix(primary_unique_name, "_stby", 30)
    standby_name_source = (
        "用户确认的实施参数"
        if _explicit_parameter(context, "STANDBY_DB_UNIQUE_NAME")
        else "由主库 DB_UNIQUE_NAME 按 <主库名>_stby 规则派生"
    )
    add(
        "STANDBY_DB_UNIQUE_NAME",
        "备库 DB_UNIQUE_NAME",
        standby_unique_name,
        (
            RunbookParameterStatus.VERIFIED
            if _explicit_parameter(context, "STANDBY_DB_UNIQUE_NAME")
            else RunbookParameterStatus.DERIVED
        ),
        standby_name_source,
    )

    explicit_primary_tns_alias = _explicit_parameter(
        context, "PRIMARY_TNS_ALIAS"
    )
    explicit_standby_tns_alias = _explicit_parameter(
        context, "STANDBY_TNS_ALIAS"
    )
    primary_tns_alias = explicit_primary_tns_alias or primary_unique_name
    standby_tns_alias = explicit_standby_tns_alias or standby_unique_name
    add(
        "PRIMARY_TNS_ALIAS",
        "主库 TNS Alias",
        primary_tns_alias,
        (
            RunbookParameterStatus.VERIFIED
            if explicit_primary_tns_alias
            else RunbookParameterStatus.DERIVED
        ),
        (
            "用户确认的实施参数"
            if explicit_primary_tns_alias
            else "采用主库 DB_UNIQUE_NAME 作为稳定别名"
        ),
    )
    add(
        "STANDBY_TNS_ALIAS",
        "备库 TNS Alias",
        standby_tns_alias,
        (
            RunbookParameterStatus.VERIFIED
            if explicit_standby_tns_alias
            else RunbookParameterStatus.DERIVED
        ),
        (
            "用户确认的实施参数"
            if explicit_standby_tns_alias
            else "采用备库 DB_UNIQUE_NAME 作为稳定别名"
        ),
    )

    explicit_standby_sid = _explicit_parameter(context, "STANDBY_ORACLE_SID")
    standby_sid = explicit_standby_sid
    if not standby_sid and primary_db_name:
        standby_sid = _oracle_name_with_suffix(primary_db_name, "STBY", 12)
    add(
        "STANDBY_ORACLE_SID",
        "备库 ORACLE_SID",
        standby_sid,
        (
            RunbookParameterStatus.VERIFIED
            if explicit_standby_sid
            else RunbookParameterStatus.DERIVED
        ),
        (
            "用户确认的实施参数"
            if explicit_standby_sid
            else "由主库 DB_NAME 按 <DB_NAME>STBY 规则派生"
        ),
    )

    protection_value = _text(precheck, "protection_mode").upper()
    protection_mode = {
        "MAXIMUM PROTECTION": "PROTECTION",
        "MAXIMUM AVAILABILITY": "AVAILABILITY",
        "MAXIMUM PERFORMANCE": "PERFORMANCE",
    }.get(protection_value, "")
    transport_mode = "SYNC" if protection_mode in {
        "PROTECTION",
        "AVAILABILITY",
    } else ("ASYNC" if protection_mode == "PERFORMANCE" else "")
    transport_ack = "AFFIRM" if transport_mode == "SYNC" else (
        "NOAFFIRM" if transport_mode == "ASYNC" else ""
    )
    add(
        "PROTECTION_MODE",
        "目标保护模式",
        protection_mode,
        RunbookParameterStatus.DERIVED,
        f"沿用当前保护模式 {protection_value}",
    )
    add(
        "REDO_TRANSPORT_MODE",
        "Redo 传输模式",
        transport_mode,
        RunbookParameterStatus.DERIVED,
        "由目标保护模式确定",
    )
    add(
        "REDO_TRANSPORT_ACK",
        "Redo 确认模式",
        transport_ack,
        RunbookParameterStatus.DERIVED,
        "由 Redo 传输模式确定",
    )
    add(
        "DG_CONFIG_NAME",
        "Broker 配置名",
        _oracle_name_with_suffix(primary_unique_name, "_dg", 30),
        RunbookParameterStatus.DERIVED,
        "由主库 DB_UNIQUE_NAME 派生",
    )

    connection_profile = dict(context.get("connection_profile") or {})
    primary_host = _text(precheck, "host_name") or str(
        connection_profile.get("host") or ""
    ).strip()
    configured_primary_port = connection_profile.get("port")
    primary_port = str(configured_primary_port or "1521")
    primary_service = (
        _text(precheck, "service_names").split(",", 1)[0].strip()
        or str(connection_profile.get("service") or "").strip()
    )
    add(
        "PRIMARY_HOST",
        "主库主机",
        primary_host,
        RunbookParameterStatus.VERIFIED,
        "V$INSTANCE.HOST_NAME 或 Target 连接配置",
    )
    add(
        "PRIMARY_PORT",
        "主库监听端口",
        primary_port,
        (
            RunbookParameterStatus.VERIFIED
            if configured_primary_port
            else RunbookParameterStatus.DERIVED
        ),
        (
            "Target 连接配置"
            if configured_primary_port
            else "使用 Oracle 默认监听端口 1521"
        ),
    )
    add(
        "PRIMARY_SERVICE_NAME",
        "主库服务名",
        primary_service,
        RunbookParameterStatus.VERIFIED,
        "SERVICE_NAMES 参数或 Target 连接配置",
    )

    for key, label, column in (
        ("PRIMARY_FRA_DEST", "主库 FRA", "db_recovery_file_dest"),
        (
            "PRIMARY_FRA_SIZE",
            "主库 FRA 容量（字节）",
            "db_recovery_file_dest_size",
        ),
        ("PRIMARY_DB_CREATE_FILE_DEST", "主库 OMF 目录", "db_create_file_dest"),
    ):
        add(
            key,
            label,
            _text(precheck, column) or _explicit_parameter(context, key),
            RunbookParameterStatus.VERIFIED,
            (
                f"db.ha.adg_precheck:{column}"
                if _text(precheck, column)
                else "用户确认的实施参数"
            ),
        )

    for key, label in (
        ("STANDBY_HOST", "备库主机"),
        ("ORACLE_HOME", "备库 Oracle Home"),
        ("STANDBY_STORAGE", "备库存储配置"),
    ):
        add(
            key,
            label,
            _explicit_parameter(context, key),
            RunbookParameterStatus.VERIFIED,
            "用户确认的实施参数",
        )

    values = {item.key: item.value for item in parameters}
    rendered = tuple(
        RunbookResolvedParameter(
            key=item.key,
            label=item.label,
            value=item.value,
            status=item.status,
            source=item.source,
        )
        for item in parameters
    )
    return values, rendered


def _required_inputs(*keys: str) -> tuple[RunbookRequiredInput, ...]:
    return tuple(
        RunbookRequiredInput(
            key=key,
            label=_ADG_EXTERNAL_INPUTS[key][0],
            description=_ADG_EXTERNAL_INPUTS[key][1],
            placeholder="尚未取得",
        )
        for key in keys
    )


def _srl_commands(
    row: dict[str, Any] | None,
    *,
    command_scope: str = "primary",
    title_scope: str = "主库",
) -> tuple[RunbookCommand, ...]:
    plan = _text(row, "redo_thread_plan")
    commands: list[RunbookCommand] = []
    if plan:
        for thread_spec in plan.split(","):
            parts = thread_spec.split(":")
            if len(parts) != 5:
                continue
            thread_no, _online, _required, missing, size_mb = parts
            for ordinal in range(1, max(0, int(missing)) + 1):
                commands.append(
                    _command(
                        f"{command_scope}.srl.t{thread_no}.{ordinal}",
                        RunbookCommandType.SQLPLUS,
                        f"在{title_scope}为线程 {thread_no} 增加第 {ordinal} 个 Standby Redo Log",
                        f"""
ALTER DATABASE ADD STANDBY LOGFILE THREAD {thread_no}
  SIZE {size_mb}M;
""",
                        "OMF/ASM 环境可直接执行；文件系统环境应按存储规范显式补充日志成员路径。",
                    )
                )
    return tuple(commands)


def _compile_oracle_adg_build(
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any],
) -> ImplementationRunbook:
    identity, identity_ref = _first_row(evidence, "db.instance.identity")
    precheck, precheck_ref = _first_row(evidence, "db.ha.adg_precheck")
    evidence_refs = tuple(
        value for value in (identity_ref, precheck_ref) if value is not None
    )
    log_mode = _text(precheck, "log_mode").upper()
    force_logging = _text(precheck, "force_logging").upper()
    remote_login_passwordfile = _text(
        precheck, "remote_login_passwordfile"
    ).upper()
    standby_file_management = _text(
        precheck, "standby_file_management"
    ).upper()
    fra_dest = _text(precheck, "db_recovery_file_dest") or _explicit_parameter(
        context, "PRIMARY_FRA_DEST"
    )
    fra_size = _text(
        precheck, "db_recovery_file_dest_size"
    ) or _explicit_parameter(context, "PRIMARY_FRA_SIZE")
    srl_shortage = _integer(precheck, "standby_redo_shortage")
    parameters, resolved_parameters = _resolve_adg_parameters(
        identity=identity,
        precheck=precheck,
        context=context,
    )
    db_name = parameters.get("PRIMARY_DB_NAME", "")
    db_unique_name = parameters.get("PRIMARY_DB_UNIQUE_NAME", "")
    standby_unique_name = parameters.get("STANDBY_DB_UNIQUE_NAME", "")
    primary_tns_alias = parameters.get("PRIMARY_TNS_ALIAS", "")
    standby_tns_alias = parameters.get("STANDBY_TNS_ALIAS", "")
    standby_sid = parameters.get("STANDBY_ORACLE_SID", "")
    protection_mode = parameters.get("PROTECTION_MODE", "")
    transport_mode = parameters.get("REDO_TRANSPORT_MODE", "")
    transport_ack = parameters.get("REDO_TRANSPORT_ACK", "")
    dg_config_name = parameters.get("DG_CONFIG_NAME", "")
    primary_host = parameters.get("PRIMARY_HOST", "")
    primary_port = parameters.get("PRIMARY_PORT", "1521")
    primary_service = parameters.get("PRIMARY_SERVICE_NAME", "")
    standby_host = _explicit_parameter(context, "STANDBY_HOST")
    oracle_home = _explicit_parameter(context, "ORACLE_HOME")
    standby_storage = _explicit_parameter(context, "STANDBY_STORAGE")
    unresolved_inputs = ["STANDBY_HOST", "ORACLE_HOME", "STANDBY_STORAGE"]
    if standby_host:
        unresolved_inputs.remove("STANDBY_HOST")
    if oracle_home:
        unresolved_inputs.remove("ORACLE_HOME")
    if standby_storage:
        unresolved_inputs.remove("STANDBY_STORAGE")
    if not fra_dest:
        unresolved_inputs.extend(("PRIMARY_FRA_DEST", "PRIMARY_FRA_SIZE"))
    if (
        precheck is not None
        and srl_shortage > 0
        and not _text(precheck, "db_create_file_dest")
    ):
        unresolved_inputs.append("PRIMARY_LOG_STORAGE")
    required_inputs = _required_inputs(*unresolved_inputs)
    database_role = _text(precheck, "database_role") or _text(
        identity, "database_role"
    )
    container_name = _text(precheck, "container_name")
    container_id = _integer(precheck, "container_id")
    redo_size = _integer(precheck, "online_redo_max_size_mb")
    log_archive_dest_1 = _text(precheck, "log_archive_dest_1")
    log_archive_config = _text(precheck, "log_archive_config")
    log_archive_dest_2 = _text(precheck, "log_archive_dest_2")
    log_archive_dest_state_2 = _text(
        precheck, "log_archive_dest_state_2"
    ).upper()
    fal_server = _text(precheck, "fal_server")
    dg_broker_start = _text(precheck, "dg_broker_start").upper()
    fra_ready = bool(
        fra_dest
        and fra_size
        and _setting_contains(
            log_archive_dest_1,
            "LOCATION=USE_DB_RECOVERY_FILE_DEST",
            f"DB_UNIQUE_NAME={db_unique_name}",
        )
    )
    desired_dg_config = f"DG_CONFIG=({db_unique_name},{standby_unique_name})"
    desired_dest_2_parts = (
        f"SERVICE={standby_tns_alias}",
        transport_mode,
        transport_ack,
        "VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE)",
        f"DB_UNIQUE_NAME={standby_unique_name}",
    )
    dest_2_ready = _setting_contains(
        log_archive_dest_2, *desired_dest_2_parts
    )
    primary_parameter_commands: list[RunbookCommand] = []
    if not _setting_contains(log_archive_config, desired_dg_config):
        primary_parameter_commands.append(
            _command(
                "parameters.primary.log_archive_config",
                RunbookCommandType.SQLPLUS,
                "设置 Data Guard 成员列表",
                f"ALTER SYSTEM SET log_archive_config='{desired_dg_config}' SCOPE=BOTH SID='*';",
            )
        )
    if not dest_2_ready:
        primary_parameter_commands.append(
            _command(
                "parameters.primary.log_archive_dest_2",
                RunbookCommandType.SQLPLUS,
                "设置备库 redo 传输目标",
                (
                    "ALTER SYSTEM SET log_archive_dest_2='"
                    f"SERVICE={standby_tns_alias} {transport_mode} {transport_ack} "
                    "VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE) "
                    f"DB_UNIQUE_NAME={standby_unique_name}' SCOPE=BOTH SID='*';"
                ),
            )
        )
    if not dest_2_ready and log_archive_dest_state_2 != "DEFER":
        primary_parameter_commands.append(
            _command(
                "parameters.primary.defer_transport",
                RunbookCommandType.SQLPLUS,
                "建设期间暂缓远端传输",
                "ALTER SYSTEM SET log_archive_dest_state_2=DEFER SCOPE=BOTH SID='*';",
            )
        )
    if fal_server.upper() != standby_tns_alias.upper():
        primary_parameter_commands.append(
            _command(
                "parameters.primary.fal_server",
                RunbookCommandType.SQLPLUS,
                "设置主库角色切换后的 FAL 服务",
                f"ALTER SYSTEM SET fal_server='{standby_tns_alias}' SCOPE=BOTH SID='*';",
            )
        )
    if standby_file_management != "AUTO":
        primary_parameter_commands.append(
            _command(
                "parameters.primary.standby_file_management",
                RunbookCommandType.SQLPLUS,
                "启用自动备库文件管理",
                "ALTER SYSTEM SET standby_file_management=AUTO SCOPE=BOTH SID='*';",
            )
        )
    if dg_broker_start != "TRUE":
        primary_parameter_commands.append(
            _command(
                "parameters.primary.dg_broker_start",
                RunbookCommandType.SQLPLUS,
                "启动 Data Guard Broker",
                "ALTER SYSTEM SET dg_broker_start=TRUE SCOPE=BOTH SID='*';",
            )
        )
    status = (
        RunbookStatus.PARTIAL_EVIDENCE
        if precheck is None
        else (
            RunbookStatus.BLOCKED_BY_REQUIRED_INPUTS
            if required_inputs
            else RunbookStatus.READY
        )
    )
    baseline_backup_commands = (
        (
            _command(
                "scope.recovery_baseline.rman",
                RunbookCommandType.RMAN,
                "执行并保留变更前在线 RMAN 备份",
                """
BACKUP AS COMPRESSED BACKUPSET DATABASE PLUS ARCHIVELOG TAG 'PRE_ADG_BUILD';
BACKUP CURRENT CONTROLFILE TAG 'PRE_ADG_BUILD_CONTROLFILE';
BACKUP SPFILE TAG 'PRE_ADG_BUILD_SPFILE';
""",
                "备份必须写入已确认的恢复介质，并记录 DBID、备份片路径和恢复负责人。",
            ),
        )
        if log_mode == "ARCHIVELOG"
        else (
            _command(
                "scope.recovery_baseline.mount",
                RunbookCommandType.SQLPLUS,
                "将 NOARCHIVELOG 主库停到 MOUNT",
                """
SHUTDOWN IMMEDIATE;
STARTUP MOUNT;
""",
            ),
            _command(
                "scope.recovery_baseline.rman",
                RunbookCommandType.RMAN,
                "执行并保留变更前一致性 RMAN 备份",
                """
BACKUP AS COMPRESSED BACKUPSET DATABASE TAG 'PRE_ADG_BUILD_COLD';
BACKUP CURRENT CONTROLFILE TAG 'PRE_ADG_BUILD_CONTROLFILE';
BACKUP SPFILE TAG 'PRE_ADG_BUILD_SPFILE';
""",
                "备份必须写入已确认的恢复介质，并记录 DBID、备份片路径和恢复负责人。",
            ),
            _command(
                "scope.recovery_baseline.open",
                RunbookCommandType.SQLPLUS,
                "完成备份后恢复主库打开状态",
                "ALTER DATABASE OPEN;",
            ),
        )
    )
    current_state = (
        _state_item("DB_UNIQUE_NAME", db_unique_name, "VERIFIED" if precheck else "UNKNOWN"),
        _state_item(
            "备库 DB_UNIQUE_NAME",
            standby_unique_name,
            "DERIVED" if standby_unique_name else "UNKNOWN",
        ),
        _state_item(
            "日志传输",
            " ".join(item for item in (transport_mode, transport_ack) if item),
            "DERIVED" if transport_mode else "UNKNOWN",
        ),
        _state_item(
            "当前容器",
            f"{container_name} (CON_ID={container_id})" if container_name else "未取得",
            "SATISFIED"
            if precheck and container_name and container_id in {0, 1}
            else "REMEDIATION_REQUIRED",
        ),
        _state_item("数据库角色", database_role, "VERIFIED" if database_role else "UNKNOWN"),
        _state_item("归档模式", log_mode, "SATISFIED" if log_mode == "ARCHIVELOG" else "REMEDIATION_REQUIRED"),
        _state_item("强制日志", force_logging, "SATISFIED" if force_logging == "YES" else "REMEDIATION_REQUIRED"),
        _state_item(
            "远程密码文件",
            remote_login_passwordfile,
            "SATISFIED"
            if remote_login_passwordfile == "EXCLUSIVE"
            else "REMEDIATION_REQUIRED",
        ),
        _state_item(
            "Flashback",
            _text(precheck, "flashback_on"),
            "SATISFIED" if _text(precheck, "flashback_on").upper() == "YES" else "OPTIONAL_REMEDIATION",
        ),
        _state_item(
            "FRA 与本地归档目标",
            fra_dest,
            "SATISFIED" if fra_ready else (
                "REMEDIATION_REQUIRED" if fra_dest and fra_size else "INPUT_REQUIRED"
            ),
        ),
        _state_item(
            "Standby Redo Log",
            f"缺少 {srl_shortage} 组" if precheck else "未取得",
            "SATISFIED" if precheck and srl_shortage <= 0 else "REMEDIATION_REQUIRED",
        ),
        _state_item(
            "STANDBY_FILE_MANAGEMENT",
            standby_file_management,
            "SATISFIED" if standby_file_management == "AUTO" else "REMEDIATION_REQUIRED",
        ),
    )

    phases = (
        RunbookPhase(
            phase_id="scope",
            title="实施边界与停机窗口",
            objective="冻结拓扑、版本、存储、保护模式和回退责任人，避免带着未决设计进入变更窗口。",
            steps=(
                RunbookStep(
                    step_id="scope.confirm",
                    title="确认主备拓扑和实施输入",
                    applicability=(
                        RunbookApplicability.BLOCKED
                        if required_inputs
                        else RunbookApplicability.ALREADY_SATISFIED
                    ),
                    rationale=(
                        "数据库命名、TNS 别名、保护模式和传输方式已经由当前主库证据确定；"
                        "仅保留无法从主库查询的基础设施事实。"
                    ),
                    commands=(
                        _command(
                            "scope.checklist",
                            RunbookCommandType.MANUAL,
                            "完成实施确认单",
                            "确认页面列出的外部输入，记录主库停机窗口、备份保留点、DNS/SCAN/监听变更人和回退负责人。",
                        ),
                    ) if required_inputs else (),
                    risks=("主备版本或补丁不一致会导致 Duplicate、日志应用或切换失败。",),
                    required_inputs=tuple(item.key for item in required_inputs),
                ),
                RunbookStep(
                    step_id="scope.recovery_baseline",
                    title="建立变更前可恢复基线",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="归档模式、参数和日志组变更前必须证明能够恢复主库控制文件、SPFILE 和数据文件。",
                    commands=baseline_backup_commands,
                    verification_commands=(
                        _command(
                            "scope.recovery_baseline.verify",
                            RunbookCommandType.RMAN,
                            "核对备份可见性",
                            """
LIST BACKUP SUMMARY;
RESTORE DATABASE VALIDATE;
""",
                        ),
                    ),
                    risks=("没有经过 VALIDATE 的备份不能作为实施回退依据。",),
                    required_inputs=(
                        ("PRIMARY_FRA_DEST", "PRIMARY_FRA_SIZE")
                        if not fra_dest
                        else ()
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="primary_prerequisites",
            title="主库前置整改",
            objective="把主库改造成可持续传输并可恢复的 Data Guard 主库。",
            steps=(
                RunbookStep(
                    step_id="primary.passwordfile_mode",
                    title="设置远程密码文件模式",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if remote_login_passwordfile == "EXCLUSIVE"
                        else RunbookApplicability.REQUIRED
                    ),
                    rationale="Data Guard 远程管理和 redo 传输认证要求主备使用一致的独占密码文件。",
                    commands=() if remote_login_passwordfile == "EXCLUSIVE" else (
                        _command(
                            "primary.passwordfile_mode.configure",
                            RunbookCommandType.SQLPLUS,
                            "设置 REMOTE_LOGIN_PASSWORDFILE",
                            (
                                "ALTER SYSTEM SET remote_login_passwordfile='EXCLUSIVE' SCOPE=SPFILE SID='*';"
                                if log_mode != "ARCHIVELOG"
                                else """
ALTER SYSTEM SET remote_login_passwordfile='EXCLUSIVE' SCOPE=SPFILE SID='*';
SHUTDOWN IMMEDIATE;
STARTUP;
"""
                            ),
                            (
                                "后续启用 ARCHIVELOG 的重启会使本参数生效；重启后再复制密码文件到备库。"
                                if log_mode != "ARCHIVELOG"
                                else "重启后再复制密码文件到备库。"
                            ),
                        ),
                    ),
                    verification_commands=() if log_mode != "ARCHIVELOG" else (
                        _command(
                            "primary.passwordfile_mode.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证远程密码文件模式",
                            "SHOW PARAMETER remote_login_passwordfile;",
                        ),
                    ),
                    risks=("如本步骤发生变更，必须把主库重启纳入同一受控窗口。",),
                ),
                RunbookStep(
                    step_id="primary.archivelog",
                    title="启用 ARCHIVELOG",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if log_mode == "ARCHIVELOG"
                        else RunbookApplicability.REQUIRED
                    ),
                    rationale="物理备库依赖连续归档日志；NOARCHIVELOG 必须在建设前整改。",
                    commands=() if log_mode == "ARCHIVELOG" else (
                        _command(
                            "primary.archivelog.enable",
                            RunbookCommandType.SQLPLUS,
                            "挂载状态启用归档",
                            """
SHUTDOWN IMMEDIATE;
STARTUP MOUNT;
ALTER DATABASE ARCHIVELOG;
ALTER DATABASE OPEN;
""",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "primary.archivelog.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证归档模式",
                            (
                                "SELECT log_mode FROM v$database;"
                                if remote_login_passwordfile == "EXCLUSIVE"
                                else """
SELECT log_mode FROM v$database;
SHOW PARAMETER remote_login_passwordfile;
"""
                            ),
                        ),
                    ),
                    rollback=() if log_mode == "ARCHIVELOG" else (
                        _command(
                            "primary.archivelog.rollback",
                            RunbookCommandType.SQLPLUS,
                            "仅在正式取消 ADG 建设时回退归档模式",
                            """
SHUTDOWN IMMEDIATE;
STARTUP MOUNT;
ALTER DATABASE NOARCHIVELOG;
ALTER DATABASE OPEN;
""",
                        ),
                    ),
                    risks=("该步骤需要主库停机并重启。",),
                ),
                RunbookStep(
                    step_id="primary.force_logging",
                    title="启用 FORCE LOGGING",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if force_logging == "YES"
                        else RunbookApplicability.REQUIRED
                    ),
                    rationale="阻止 NOLOGGING 操作造成备库不可恢复的数据块缺口。",
                    commands=() if force_logging == "YES" else (
                        _command(
                            "primary.force_logging.enable",
                            RunbookCommandType.SQLPLUS,
                            "启用强制日志",
                            "ALTER DATABASE FORCE LOGGING;",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "primary.force_logging.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证强制日志",
                            "SELECT force_logging FROM v$database;",
                        ),
                    ),
                    rollback=() if force_logging == "YES" else (
                        _command(
                            "primary.force_logging.rollback",
                            RunbookCommandType.SQLPLUS,
                            "取消强制日志",
                            "ALTER DATABASE NO FORCE LOGGING;",
                        ),
                    ),
                    risks=("启用后 NOLOGGING 路径会产生额外 redo。",),
                ),
                RunbookStep(
                    step_id="primary.fra",
                    title="配置 FRA 与本地归档目标",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if fra_ready
                        else (
                            RunbookApplicability.REQUIRED
                            if fra_dest and fra_size
                            else RunbookApplicability.BLOCKED
                        )
                    ),
                    rationale="归档和恢复文件必须有明确容量、告警和清理策略。",
                    commands=() if fra_ready else (
                        _command(
                            "primary.fra.configure",
                            RunbookCommandType.SQLPLUS,
                            "配置 FRA 与本地归档目标",
                            f"""
ALTER SYSTEM SET db_recovery_file_dest_size={fra_size} SCOPE=BOTH SID='*';
ALTER SYSTEM SET db_recovery_file_dest='{fra_dest}' SCOPE=BOTH SID='*';
ALTER SYSTEM SET log_archive_dest_1='LOCATION=USE_DB_RECOVERY_FILE_DEST VALID_FOR=(ALL_LOGFILES,ALL_ROLES) DB_UNIQUE_NAME={db_unique_name}' SCOPE=BOTH SID='*';
""",
                        ),
                    ) if fra_dest and fra_size and db_unique_name else (),
                    verification_commands=(
                        _command(
                            "primary.fra.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证 FRA 和归档目标",
                            """
SELECT name, space_limit, space_used, space_reclaimable FROM v$recovery_file_dest;
SELECT dest_id, status, destination, error FROM v$archive_dest_status WHERE dest_id = 1;
""",
                        ),
                    ),
                    risks=("FRA 容量不足会阻塞归档并最终影响主库。",),
                    required_inputs=(
                        ("PRIMARY_FRA_DEST", "PRIMARY_FRA_SIZE")
                        if not (fra_dest and fra_size)
                        else ()
                    ),
                ),
                RunbookStep(
                    step_id="primary.flashback",
                    title="启用 Flashback Database",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if _text(precheck, "flashback_on").upper() == "YES"
                        else RunbookApplicability.CONDITIONAL
                    ),
                    rationale="Flashback 不是物理备库的硬前置，但可显著降低故障切换后 reinstate 原主库的成本。",
                    commands=() if _text(precheck, "flashback_on").upper() == "YES" else (
                        _command(
                            "primary.flashback.enable",
                            RunbookCommandType.SQLPLUS,
                            "在已配置 FRA 的主库启用 Flashback",
                            "ALTER DATABASE FLASHBACK ON;",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "primary.flashback.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证 Flashback",
                            "SELECT flashback_on FROM v$database;",
                        ),
                    ),
                    rollback=() if _text(precheck, "flashback_on").upper() == "YES" else (
                        _command(
                            "primary.flashback.rollback",
                            RunbookCommandType.SQLPLUS,
                            "关闭 Flashback",
                            "ALTER DATABASE FLASHBACK OFF;",
                        ),
                    ),
                    risks=("Flashback 会持续占用 FRA，必须同步调整容量和保留窗口。",),
                    required_inputs=(
                        ("PRIMARY_FRA_DEST", "PRIMARY_FRA_SIZE")
                        if not fra_dest
                        else ()
                    ),
                ),
                RunbookStep(
                    step_id="primary.srl",
                    title="补齐 Standby Redo Log",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if precheck is not None and srl_shortage <= 0
                        else (
                            RunbookApplicability.REQUIRED
                            if _text(precheck, "db_create_file_dest")
                            else RunbookApplicability.BLOCKED
                        )
                    ),
                    rationale="每个 redo thread 的 SRL 数量至少应为 online redo group 数量加一，大小不小于对应联机日志。",
                    commands=() if precheck is not None and srl_shortage <= 0 else (
                        _srl_commands(precheck)
                        if _text(precheck, "db_create_file_dest")
                        else ()
                    ),
                    verification_commands=(
                        _command(
                            "primary.srl.verify",
                            RunbookCommandType.SQLPLUS,
                            "按线程核对 SRL",
                            """
SELECT thread#, COUNT(*) group_count, MIN(bytes)/1024/1024 min_mb, MAX(bytes)/1024/1024 max_mb
FROM v$standby_log GROUP BY thread# ORDER BY thread#;
""",
                        ),
                    ),
                    rollback=(),
                    risks=("不能删除 ACTIVE 或正在归档的日志组；文件系统路径必须预先存在且空间充足。",),
                    required_inputs=(
                        ("PRIMARY_LOG_STORAGE",)
                        if srl_shortage > 0
                        and not _text(precheck, "db_create_file_dest")
                        else ()
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="dataguard_parameters",
            title="Data Guard 参数",
            objective="配置主备唯一标识、日志传输、角色切换和文件管理参数。",
            steps=(
                RunbookStep(
                    step_id="parameters.primary",
                    title="配置主库参数",
                    applicability=(
                        (
                            RunbookApplicability.REQUIRED
                            if primary_parameter_commands
                            else RunbookApplicability.ALREADY_SATISFIED
                        )
                        if all(
                            (
                                db_unique_name,
                                standby_unique_name,
                                standby_tns_alias,
                                protection_mode,
                                transport_mode,
                                transport_ack,
                            )
                        )
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="显式声明 DG_CONFIG、远端归档目标和角色相关参数。",
                    commands=tuple(primary_parameter_commands) if all(
                        (
                            db_unique_name,
                            standby_unique_name,
                            standby_tns_alias,
                            protection_mode,
                            transport_mode,
                            transport_ack,
                        )
                    ) else (),
                    verification_commands=(
                        _command(
                            "parameters.primary.verify",
                            RunbookCommandType.SQLPLUS,
                            "核对主库参数",
                            """
SELECT name, value FROM v$parameter
WHERE name IN ('db_unique_name','log_archive_config','log_archive_dest_2','log_archive_dest_state_2','fal_server','standby_file_management','dg_broker_start')
ORDER BY name;
""",
                        ),
                    ),
                    rollback=(
                        _command(
                            "parameters.primary.rollback",
                            RunbookCommandType.SQLPLUS,
                            "停用远端日志传输",
                            "ALTER SYSTEM SET log_archive_dest_state_2=DEFER SCOPE=BOTH SID='*';",
                        ),
                    ),
                    required_inputs=(),
                    risks=("错误的 SERVICE 或 DB_UNIQUE_NAME 会造成 ORA-160xx 日志传输错误。",),
                ),
                RunbookStep(
                    step_id="parameters.standby",
                    title="准备备库参数文件",
                    applicability=(
                        RunbookApplicability.REQUIRED
                        if standby_storage
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="备库必须使用独立 DB_UNIQUE_NAME，并按存储策略配置文件名转换或 OMF。",
                    commands=(
                        _command(
                            "parameters.standby.pfile",
                            RunbookCommandType.CONFIG,
                            "备库参数模板",
                            f"""
*.db_name='{db_name}'
*.db_unique_name='{standby_unique_name}'
*.log_archive_config='DG_CONFIG=({db_unique_name},{standby_unique_name})'
*.log_archive_dest_1='LOCATION=USE_DB_RECOVERY_FILE_DEST VALID_FOR=(ALL_LOGFILES,ALL_ROLES) DB_UNIQUE_NAME={standby_unique_name}'
*.log_archive_dest_2='SERVICE={primary_tns_alias} {transport_mode} {transport_ack} VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE) DB_UNIQUE_NAME={db_unique_name}'
*.fal_server='{primary_tns_alias}'
*.standby_file_management='AUTO'
{standby_storage}
""",
                        ),
                    ) if standby_storage and all(
                        (
                            db_name,
                            db_unique_name,
                            standby_unique_name,
                            primary_tns_alias,
                            transport_mode,
                            transport_ack,
                        )
                    ) else (),
                    required_inputs=(
                        ("STANDBY_STORAGE",) if not standby_storage else ()
                    ),
                    risks=("db_name 必须与主库一致，db_unique_name 必须不同。",),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="connectivity",
            title="密码文件、监听与 TNS",
            objective="建立 RMAN Duplicate、redo transport 和 Broker 所需的静态连通性。",
            steps=(
                RunbookStep(
                    step_id="connectivity.password_file",
                    title="同步密码文件",
                    applicability=(
                        RunbookApplicability.REQUIRED
                        if oracle_home and standby_sid
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="主备 SYS 密码文件必须一致，且 remote_login_passwordfile 应为 EXCLUSIVE。",
                    commands=(
                        _command(
                            "connectivity.password_file.create",
                            RunbookCommandType.SHELL,
                            "在备库准备密码文件",
                            f"""
{oracle_home}/bin/orapwd file={oracle_home}/dbs/orapw{standby_sid} format=12.2 force=y
""",
                            "优先通过受控安全通道复制主库密码文件；不要把 SYS 密码写入脚本或 Runbook。",
                        ),
                    ) if oracle_home and standby_sid else (),
                    verification_commands=(
                        _command(
                            "connectivity.password_file.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证远程密码文件模式",
                            "SHOW PARAMETER remote_login_passwordfile;",
                        ),
                    ),
                    required_inputs=(
                        ("ORACLE_HOME",) if not oracle_home else ()
                    ),
                ),
                RunbookStep(
                    step_id="connectivity.listener_tns",
                    title="配置静态监听和双向 TNS",
                    applicability=(
                        RunbookApplicability.REQUIRED
                        if all(
                            (
                                standby_host,
                                oracle_home,
                                standby_sid,
                                primary_host,
                                primary_service,
                            )
                        )
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="NOMOUNT/MOUNT 状态的辅助实例需要静态注册，主备必须双向解析。",
                    commands=(
                        _command(
                            "connectivity.listener.config",
                            RunbookCommandType.CONFIG,
                            "备库 listener.ora 静态注册片段",
                            f"""
SID_LIST_LISTENER =
  (SID_LIST =
    (SID_DESC =
      (GLOBAL_DBNAME = {standby_unique_name}_DGMGRL)
      (ORACLE_HOME = {oracle_home})
      (SID_NAME = {standby_sid})))
""",
                        ),
                        _command(
                            "connectivity.tns.config",
                            RunbookCommandType.CONFIG,
                            "tnsnames.ora 双向别名",
                            f"""
{primary_tns_alias} =
  (DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={primary_host})(PORT={primary_port}))
    (CONNECT_DATA=(SERVICE_NAME={primary_service})))

{standby_tns_alias} =
  (DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={standby_host})(PORT=1521))
    (CONNECT_DATA=(SERVICE_NAME={standby_unique_name}_DGMGRL)))
""",
                        ),
                        _command(
                            "connectivity.listener.reload",
                            RunbookCommandType.SHELL,
                            "重载监听并测试",
                            f"""
{oracle_home}/bin/lsnrctl reload
{oracle_home}/bin/tnsping {primary_tns_alias}
{oracle_home}/bin/tnsping {standby_tns_alias}
""",
                        ),
                    ) if all(
                        (
                            standby_host,
                            oracle_home,
                            standby_sid,
                            primary_host,
                            primary_service,
                        )
                    ) else (),
                    required_inputs=tuple(
                        key
                        for key, value in (
                            ("STANDBY_HOST", standby_host),
                            ("ORACLE_HOME", oracle_home),
                        )
                        if not value
                    ),
                    risks=("监听静态服务名与 RMAN AUXILIARY 连接串不一致会导致 ORA-12514/12528。",),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="standby_creation",
            title="创建物理备库",
            objective="使用 RMAN 从主库构建一致的物理备库并恢复到 MOUNT。",
            steps=(
                RunbookStep(
                    step_id="standby.nomount",
                    title="以 NOMOUNT 启动辅助实例",
                    applicability=(
                        RunbookApplicability.REQUIRED
                        if oracle_home and standby_sid and standby_storage
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="RMAN Duplicate 需要可远程连接的 NOMOUNT 辅助实例。",
                    commands=(
                        _command(
                            "standby.nomount.sql",
                            RunbookCommandType.SQLPLUS,
                            "启动备库辅助实例",
                            f"""
CREATE SPFILE FROM PFILE='{oracle_home}/dbs/init{standby_sid}.ora';
STARTUP NOMOUNT;
""",
                        ),
                    ) if oracle_home and standby_sid and standby_storage else (),
                    required_inputs=tuple(
                        key
                        for key, value in (
                            ("ORACLE_HOME", oracle_home),
                            ("STANDBY_STORAGE", standby_storage),
                        )
                        if not value
                    ),
                ),
                RunbookStep(
                    step_id="standby.duplicate",
                    title="执行 RMAN Active Duplicate",
                    applicability=(
                        RunbookApplicability.REQUIRED
                        if standby_host and standby_storage
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="从当前主库在线复制数据文件、控制文件和归档并建立物理备库。",
                    commands=(
                        _command(
                            "standby.duplicate.rman",
                            RunbookCommandType.RMAN,
                            "连接主库和辅助实例并复制",
                            f"""
CONNECT TARGET sys@{primary_tns_alias};
CONNECT AUXILIARY sys@{standby_tns_alias};
RUN {{
  ALLOCATE CHANNEL c1 DEVICE TYPE DISK;
  ALLOCATE AUXILIARY CHANNEL a1 DEVICE TYPE DISK;
  DUPLICATE TARGET DATABASE FOR STANDBY FROM ACTIVE DATABASE
    DORECOVER
    SPFILE
      SET db_unique_name='{standby_unique_name}'
      SET fal_server='{primary_tns_alias}'
      SET standby_file_management='AUTO';
}}
""",
                            "备库参数文件必须已经包含经确认的 OMF/ASM 或文件名转换配置。",
                        ),
                    ) if standby_host and standby_storage and all(
                        (
                            primary_tns_alias,
                            standby_tns_alias,
                            standby_unique_name,
                        )
                    ) else (),
                    verification_commands=(
                        _command(
                            "standby.duplicate.verify",
                            RunbookCommandType.SQLPLUS,
                            "确认备库角色与打开模式",
                            "SELECT database_role, open_mode, db_unique_name FROM v$database;",
                        ),
                    ),
                    required_inputs=tuple(
                        key
                        for key, value in (
                            ("STANDBY_HOST", standby_host),
                            ("STANDBY_STORAGE", standby_storage),
                        )
                        if not value
                    ),
                    risks=("Active Duplicate 会消耗主库网络、IO 和备库存储；大库应评估备份恢复方式。",),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="transport_apply",
            title="启动传输与实时应用",
            objective="启用主库远端归档目标，在备库启动 Managed Recovery。",
            steps=(
                RunbookStep(
                    step_id="transport.enable",
                    title="启用日志传输",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="Duplicate 完成后才正式打开主库远端传输，便于隔离建设阶段故障。",
                    commands=(
                        _command(
                            "transport.enable.sql",
                            RunbookCommandType.SQLPLUS,
                            "在主库启用并触发归档",
                            """
ALTER SYSTEM SET log_archive_dest_state_2=ENABLE SCOPE=BOTH SID='*';
ALTER SYSTEM ARCHIVE LOG CURRENT;
""",
                        ),
                    ),
                    rollback=(
                        _command(
                            "transport.enable.rollback",
                            RunbookCommandType.SQLPLUS,
                            "发生持续传输错误时暂停远端目标",
                            "ALTER SYSTEM SET log_archive_dest_state_2=DEFER SCOPE=BOTH SID='*';",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="apply.start",
                    title="启动实时日志应用",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="Managed Recovery 持续应用 standby redo，形成可切换的物理备库。",
                    commands=(
                        _command(
                            "apply.start.sql",
                            RunbookCommandType.SQLPLUS,
                            "在备库启动 MRP",
                            "ALTER DATABASE RECOVER MANAGED STANDBY DATABASE USING CURRENT LOGFILE DISCONNECT FROM SESSION;",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "apply.start.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证 RFS/MRP 进程",
                            "SELECT process, status, thread#, sequence# FROM v$managed_standby ORDER BY process;",
                        ),
                    ),
                    rollback=(
                        _command(
                            "apply.start.rollback",
                            RunbookCommandType.SQLPLUS,
                            "停止日志应用",
                            "ALTER DATABASE RECOVER MANAGED STANDBY DATABASE CANCEL;",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="apply.active_dataguard",
                    title="按许可启用 Active Data Guard 只读",
                    applicability=RunbookApplicability.CONDITIONAL,
                    rationale="只有具备 Active Data Guard 许可且业务确有只读需求时才打开。",
                    commands=(
                        _command(
                            "apply.active_dataguard.sql",
                            RunbookCommandType.SQLPLUS,
                            "只读打开后恢复实时应用",
                            """
ALTER DATABASE RECOVER MANAGED STANDBY DATABASE CANCEL;
ALTER DATABASE OPEN READ ONLY;
ALTER DATABASE RECOVER MANAGED STANDBY DATABASE USING CURRENT LOGFILE DISCONNECT FROM SESSION;
""",
                        ),
                    ),
                    risks=("未确认许可时不要启用只读实时应用。",),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="broker",
            title="配置 Data Guard Broker",
            objective="将主备纳入 Broker，统一健康检查、属性管理和后续人工切换。",
            steps=(
                RunbookStep(
                    step_id="broker.configure",
                    title="创建并启用 Broker 配置",
                    applicability=(
                        RunbookApplicability.REQUIRED
                        if standby_host and dg_config_name
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="Broker 提供一致的配置校验和切换前检查，但本 Runbook 不自动执行切换。",
                    commands=(
                        _command(
                            "broker.configure.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "建立 Broker 配置",
                            f"""
CREATE CONFIGURATION '{dg_config_name}' AS
  PRIMARY DATABASE IS '{db_unique_name}'
  CONNECT IDENTIFIER IS '{primary_tns_alias}';
ADD DATABASE '{standby_unique_name}' AS
  CONNECT IDENTIFIER IS '{standby_tns_alias}'
  MAINTAINED AS PHYSICAL;
ENABLE CONFIGURATION;
SHOW CONFIGURATION VERBOSE;
VALIDATE DATABASE VERBOSE '{db_unique_name}';
VALIDATE DATABASE VERBOSE '{standby_unique_name}';
""",
                        ),
                    ) if standby_host and all(
                        (
                            dg_config_name,
                            db_unique_name,
                            primary_tns_alias,
                            standby_unique_name,
                            standby_tns_alias,
                        )
                    ) else (),
                    rollback=(
                        _command(
                            "broker.configure.rollback",
                            RunbookCommandType.DGMGRL,
                            "禁用并移除新建 Broker 配置",
                            """
DISABLE CONFIGURATION;
REMOVE CONFIGURATION PRESERVE DESTINATIONS;
""",
                        ),
                    ),
                    required_inputs=(
                        ("STANDBY_HOST",) if not standby_host else ()
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="validation",
            title="验收与交付",
            objective="证明日志可传输、可应用、无缺口，并保留后续切换前的基线。",
            steps=(
                RunbookStep(
                    step_id="validation.runtime",
                    title="验证传输、应用和日志缺口",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="配置成功不等于数据保护有效，必须用主备实时视图和归档切换验证。",
                    commands=(
                        _command(
                            "validation.primary.sql",
                            RunbookCommandType.SQLPLUS,
                            "主库验证",
                            """
ALTER SYSTEM ARCHIVE LOG CURRENT;
SELECT dest_id, status, error, recovery_mode, archived_thread#, archived_seq#
FROM v$archive_dest_status WHERE dest_id = 2;
SELECT database_role, open_mode, protection_mode, switchover_status FROM v$database;
""",
                        ),
                        _command(
                            "validation.standby.sql",
                            RunbookCommandType.SQLPLUS,
                            "备库验证",
                            """
SELECT database_role, open_mode, protection_mode, switchover_status FROM v$database;
SELECT name, value, unit FROM v$dataguard_stats
WHERE name IN ('transport lag','apply lag','apply finish time');
SELECT thread#, low_sequence#, high_sequence# FROM v$archive_gap;
SELECT process, status, thread#, sequence# FROM v$managed_standby ORDER BY process;
""",
                        ),
                        _command(
                            "validation.broker.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "Broker 健康检查",
                            f"""
SHOW CONFIGURATION;
SHOW DATABASE VERBOSE '{standby_unique_name}';
VALIDATE NETWORK CONFIGURATION FOR ALL;
""",
                        ),
                    ),
                    risks=("存在未解决 gap、持续 apply lag 或 Broker WARNING 时不得进入切换演练。",),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="operations",
            title="日常验证与运维命令",
            objective="交付建设后的固定巡检、受控启停和切换前检查命令。",
            steps=(
                RunbookStep(
                    step_id="operations.health",
                    title="日常健康检查",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="同时核对 Broker、传输、应用延迟、日志缺口和最近 Data Guard 事件。",
                    commands=(
                        _command(
                            "operations.health.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "Broker 日常检查",
                            f"""
SHOW CONFIGURATION VERBOSE;
SHOW DATABASE VERBOSE '{db_unique_name}';
SHOW DATABASE VERBOSE '{standby_unique_name}';
VALIDATE DATABASE VERBOSE '{db_unique_name}';
VALIDATE DATABASE VERBOSE '{standby_unique_name}';
VALIDATE NETWORK CONFIGURATION FOR ALL;
""",
                        ),
                        _command(
                            "operations.health.primary",
                            RunbookCommandType.SQLPLUS,
                            "主库传输状态",
                            """
SELECT dest_id, status, target, destination, error, recovery_mode,
       archived_thread#, archived_seq#
FROM v$archive_dest_status
WHERE status <> 'INACTIVE'
ORDER BY dest_id;
SELECT timestamp, severity, message
FROM v$dataguard_status
WHERE timestamp > SYSDATE - 1
ORDER BY timestamp DESC FETCH FIRST 50 ROWS ONLY;
""",
                        ),
                        _command(
                            "operations.health.standby",
                            RunbookCommandType.SQLPLUS,
                            "备库应用状态",
                            """
SELECT name, value, unit, time_computed
FROM v$dataguard_stats
WHERE name IN ('transport lag','apply lag','apply finish time');
SELECT process, status, thread#, sequence#
FROM v$managed_standby ORDER BY process;
SELECT thread#, low_sequence#, high_sequence# FROM v$archive_gap;
SELECT timestamp, severity, message
FROM v$dataguard_status
WHERE timestamp > SYSDATE - 1
ORDER BY timestamp DESC FETCH FIRST 50 ROWS ONLY;
""",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="operations.control",
                    title="受控暂停与恢复",
                    applicability=RunbookApplicability.CONDITIONAL,
                    rationale="仅在维护窗口或故障隔离时使用，操作后必须重新执行健康检查。",
                    commands=(
                        _command(
                            "operations.control.apply",
                            RunbookCommandType.SQLPLUS,
                            "备库暂停和恢复日志应用",
                            """
ALTER DATABASE RECOVER MANAGED STANDBY DATABASE CANCEL;
ALTER DATABASE RECOVER MANAGED STANDBY DATABASE USING CURRENT LOGFILE DISCONNECT FROM SESSION;
""",
                            "两条语句分别用于暂停和恢复，不要作为一个无条件连续脚本执行。",
                        ),
                        _command(
                            "operations.control.transport",
                            RunbookCommandType.SQLPLUS,
                            "主库暂停和恢复远端传输",
                            """
ALTER SYSTEM SET log_archive_dest_state_2=DEFER SCOPE=BOTH SID='*';
ALTER SYSTEM SET log_archive_dest_state_2=ENABLE SCOPE=BOTH SID='*';
""",
                            "两条语句分别用于暂停和恢复，不要作为一个无条件连续脚本执行。",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="operations.switchover_readiness",
                    title="切换前只读检查",
                    applicability=RunbookApplicability.CONDITIONAL,
                    rationale="本步骤只验证切换条件，不执行 switchover 或 failover。",
                    commands=(
                        _command(
                            "operations.switchover_readiness.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "Broker 切换前校验",
                            f"""
SHOW CONFIGURATION VERBOSE;
VALIDATE DATABASE VERBOSE '{db_unique_name}';
VALIDATE DATABASE VERBOSE '{standby_unique_name}';
""",
                        ),
                        _command(
                            "operations.switchover_readiness.sql",
                            RunbookCommandType.SQLPLUS,
                            "数据库角色与切换状态",
                            """
SELECT db_unique_name, database_role, open_mode, protection_mode,
       switchover_status
FROM v$database;
""",
                        ),
                    ),
                    risks=("本 Runbook 不自动执行角色切换；正式切换必须另行审批并生成演练方案。",),
                ),
            ),
        ),
    )
    return ImplementationRunbook(
        profile=ImplementationProfile.ORACLE_ADG_BUILD,
        title="Oracle Active Data Guard 建设实施 Runbook",
        status=status,
        execution_policy=(
            "本产物仅生成实施步骤，不执行任何 SQL、RMAN、DGMGRL 或系统命令。"
            "页面中的可复制命令均已代入本轮解析参数；被外部输入阻断的步骤不生成伪可执行命令。"
            "用户后续明确选择某一步执行时，必须重新核验当时状态并进入受控 Action/审批。"
        ),
        current_state=current_state,
        resolved_parameters=resolved_parameters,
        required_inputs=required_inputs,
        phases=phases,
        stop_conditions=(
            "主备数据库版本、补丁级别或字符集不兼容。",
            "当前连接不是 CDB Root/NON-CDB，或目标数据库角色不是预期主库。",
            "主库无法形成可验证恢复点，或现有备份不可恢复。",
            "备库容量不足，或 ASM/文件系统路径策略尚未确认。",
            "主备监听/TNS/SYS 密码文件未通过双向连接验证。",
            "归档目标持续报错、日志缺口无法修复，或 MRP 无法稳定运行。",
            "任何命令的实际对象、路径或角色与 Runbook 假设不一致。",
        ),
        evidence_refs=evidence_refs,
    )


def compile_implementation_runbook(
    *,
    profile: ImplementationProfile,
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any] | None = None,
) -> ImplementationRunbook:
    """按结构化档案编译 Runbook，禁止模型自由拼接执行命令。"""
    if profile == ImplementationProfile.ORACLE_ADG_BUILD:
        return _compile_oracle_adg_build(evidence, dict(context or {}))
    raise ValueError(f"不支持的实施方案档案：{profile}")
