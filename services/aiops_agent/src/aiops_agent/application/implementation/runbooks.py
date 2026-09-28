"""从已验证环境事实确定性编译数据库实施方案。"""

from __future__ import annotations

from typing import Any, Iterable

from aiops_agent.contracts.implementation import (
    ImplementationRunbook,
    RunbookApplicability,
    RunbookCommand,
    RunbookCommandType,
    RunbookPhase,
    RunbookRequiredInput,
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


def _state_item(label: str, value: str, status: str) -> dict[str, Any]:
    return {"label": label, "value": value or "未取得", "status": status}


def _adg_required_inputs() -> tuple[RunbookRequiredInput, ...]:
    return (
        RunbookRequiredInput(
            key="STANDBY_HOST",
            label="备库主机",
            description="备库主机名或可解析地址，并确认主备双向网络和防火墙。",
            placeholder="${STANDBY_HOST}",
        ),
        RunbookRequiredInput(
            key="STANDBY_DB_UNIQUE_NAME",
            label="备库 DB_UNIQUE_NAME",
            description="必须与主库不同，并符合现有数据库命名规范。",
            placeholder="${STANDBY_DB_UNIQUE_NAME}",
        ),
        RunbookRequiredInput(
            key="PRIMARY_TNS_ALIAS",
            label="主库 TNS Alias",
            description="从备库连接主库的静态服务别名。",
            placeholder="${PRIMARY_TNS_ALIAS}",
        ),
        RunbookRequiredInput(
            key="STANDBY_TNS_ALIAS",
            label="备库 TNS Alias",
            description="从主库连接备库的静态服务别名。",
            placeholder="${STANDBY_TNS_ALIAS}",
        ),
        RunbookRequiredInput(
            key="ORACLE_HOME",
            label="Oracle Home",
            description="主备库实际 Oracle Home，必须使用相同数据库版本和补丁级别。",
            placeholder="${ORACLE_HOME}",
        ),
        RunbookRequiredInput(
            key="STANDBY_ORACLE_SID",
            label="备库 SID",
            description="备库实例 SID；RAC 场景应扩展为每个实例 SID。",
            placeholder="${STANDBY_ORACLE_SID}",
        ),
        RunbookRequiredInput(
            key="STORAGE_STRATEGY",
            label="存储与文件名策略",
            description="确认 ASM/OMF 或文件系统路径，以及数据文件、联机日志和 FRA 的放置规则。",
            placeholder="${STORAGE_STRATEGY}",
        ),
        RunbookRequiredInput(
            key="PROTECTION_MODE",
            label="保护模式",
            description="填写 SQL 关键字 PERFORMANCE、AVAILABILITY 或 PROTECTION，并完成业务 RPO/RTO 评审。",
            placeholder="${PROTECTION_MODE}",
        ),
        RunbookRequiredInput(
            key="REDO_TRANSPORT_MODE",
            label="Redo 传输模式",
            description="通常 MaxPerformance 使用 ASYNC，MaxAvailability/MaxProtection 使用 SYNC。",
            placeholder="${REDO_TRANSPORT_MODE}",
        ),
        RunbookRequiredInput(
            key="REDO_TRANSPORT_ACK",
            label="Redo 确认模式",
            description="通常 ASYNC 使用 NOAFFIRM，SYNC 保护模式使用 AFFIRM；须结合延迟和 RPO 评审。",
            placeholder="${REDO_TRANSPORT_ACK}",
        ),
    )


def _srl_commands(row: dict[str, Any] | None) -> tuple[RunbookCommand, ...]:
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
                        f"primary.srl.t{thread_no}.{ordinal}",
                        RunbookCommandType.SQLPLUS,
                        f"为线程 {thread_no} 增加第 {ordinal} 个 Standby Redo Log",
                        f"""
ALTER DATABASE ADD STANDBY LOGFILE THREAD {thread_no}
  SIZE {size_mb}M;
""",
                        "OMF/ASM 环境可直接执行；文件系统环境应按存储规范显式补充日志成员路径。",
                    )
                )
    if commands:
        return tuple(commands)
    return (
        _command(
            "primary.srl.template",
            RunbookCommandType.SQLPLUS,
            "按每线程联机日志组数加一补建 Standby Redo Log",
            """
ALTER DATABASE ADD STANDBY LOGFILE THREAD ${THREAD_NO}
  SIZE ${ONLINE_REDO_SIZE_MB}M;
""",
            "先按线程查询 V$LOG 和 V$STANDBY_LOG；每线程 SRL 数量至少为 online redo group 数量加一。",
        ),
    )


def _compile_oracle_adg_build(
    evidence: tuple[TurnEvidenceFact, ...],
) -> ImplementationRunbook:
    identity, identity_ref = _first_row(evidence, "db.instance.identity")
    precheck, precheck_ref = _first_row(evidence, "db.ha.adg_precheck")
    evidence_refs = tuple(
        value for value in (identity_ref, precheck_ref) if value is not None
    )
    log_mode = _text(precheck, "log_mode").upper()
    force_logging = _text(precheck, "force_logging").upper()
    standby_file_management = _text(
        precheck, "standby_file_management"
    ).upper()
    fra_dest = _text(precheck, "db_recovery_file_dest")
    srl_shortage = _integer(precheck, "standby_redo_shortage")
    db_unique_name = _text(precheck, "db_unique_name") or "${PRIMARY_DB_UNIQUE_NAME}"
    database_role = _text(precheck, "database_role") or _text(
        identity, "database_role"
    )
    container_name = _text(precheck, "container_name")
    container_id = _integer(precheck, "container_id")
    redo_size = _integer(precheck, "online_redo_max_size_mb")
    status = (
        RunbookStatus.PARTIAL_EVIDENCE
        if precheck is None
        else RunbookStatus.READY_WITH_REQUIRED_INPUTS
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
            "Flashback",
            _text(precheck, "flashback_on"),
            "SATISFIED" if _text(precheck, "flashback_on").upper() == "YES" else "OPTIONAL_REMEDIATION",
        ),
        _state_item("FRA", fra_dest, "SATISFIED" if fra_dest else "INPUT_REQUIRED"),
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
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="备库主机、命名、网络、存储和保护模式不能从数据库参数中可靠推断。",
                    commands=(
                        _command(
                            "scope.checklist",
                            RunbookCommandType.MANUAL,
                            "完成实施确认单",
                            "确认全部 required_inputs，记录主库停机窗口、备份保留点、DNS/SCAN/监听变更人和回退负责人。",
                        ),
                    ),
                    risks=("主备版本或补丁不一致会导致 Duplicate、日志应用或切换失败。",),
                    required_inputs=tuple(item.key for item in _adg_required_inputs()),
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
                    required_inputs=("STORAGE_STRATEGY",),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="primary_prerequisites",
            title="主库前置整改",
            objective="把主库改造成可持续传输并可恢复的 Data Guard 主库。",
            steps=(
                RunbookStep(
                    step_id="primary.archivelog",
                    title="启用 ARCHIVELOG",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if log_mode == "ARCHIVELOG"
                        else RunbookApplicability.REQUIRED
                    ),
                    rationale="物理备库依赖连续归档日志；NOARCHIVELOG 必须在建设前整改。",
                    commands=(
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
                            "SELECT log_mode FROM v$database;",
                        ),
                    ),
                    rollback=(
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
                    commands=(
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
                    rollback=(
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
                    step_id="primary.flashback",
                    title="启用 Flashback Database",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if _text(precheck, "flashback_on").upper() == "YES"
                        else RunbookApplicability.CONDITIONAL
                    ),
                    rationale="Flashback 不是物理备库的硬前置，但可显著降低故障切换后 reinstate 原主库的成本。",
                    commands=(
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
                    rollback=(
                        _command(
                            "primary.flashback.rollback",
                            RunbookCommandType.SQLPLUS,
                            "关闭 Flashback",
                            "ALTER DATABASE FLASHBACK OFF;",
                        ),
                    ),
                    risks=("Flashback 会持续占用 FRA，必须同步调整容量和保留窗口。",),
                    required_inputs=("STORAGE_STRATEGY",),
                ),
                RunbookStep(
                    step_id="primary.fra",
                    title="配置 FRA 与本地归档目标",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if fra_dest
                        else RunbookApplicability.REQUIRED
                    ),
                    rationale="归档和恢复文件必须有明确容量、告警和清理策略。",
                    commands=(
                        _command(
                            "primary.fra.configure",
                            RunbookCommandType.SQLPLUS,
                            "配置 FRA",
                            """
ALTER SYSTEM SET db_recovery_file_dest_size=${FRA_SIZE} SCOPE=BOTH SID='*';
ALTER SYSTEM SET db_recovery_file_dest='${PRIMARY_FRA_DEST}' SCOPE=BOTH SID='*';
ALTER SYSTEM SET log_archive_dest_1='LOCATION=USE_DB_RECOVERY_FILE_DEST VALID_FOR=(ALL_LOGFILES,ALL_ROLES) DB_UNIQUE_NAME=""" + db_unique_name + """' SCOPE=BOTH SID='*';
""",
                        ),
                    ),
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
                    required_inputs=("STORAGE_STRATEGY",),
                ),
                RunbookStep(
                    step_id="primary.srl",
                    title="补齐 Standby Redo Log",
                    applicability=(
                        RunbookApplicability.ALREADY_SATISFIED
                        if precheck is not None and srl_shortage <= 0
                        else RunbookApplicability.REQUIRED
                    ),
                    rationale="每个 redo thread 的 SRL 数量至少应为 online redo group 数量加一，大小不小于对应联机日志。",
                    commands=_srl_commands(precheck),
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
                    rollback=(
                        _command(
                            "primary.srl.rollback",
                            RunbookCommandType.SQLPLUS,
                            "删除本次新增且从未使用的 SRL",
                            "ALTER DATABASE DROP STANDBY LOGFILE GROUP ${SRL_GROUP_NO};",
                        ),
                    ),
                    risks=("不能删除 ACTIVE 或正在归档的日志组；文件系统路径必须预先存在且空间充足。",),
                    required_inputs=("STORAGE_STRATEGY",),
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
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="显式声明 DG_CONFIG、远端归档目标和角色相关参数。",
                    commands=(
                        _command(
                            "parameters.primary.sql",
                            RunbookCommandType.SQLPLUS,
                            "设置主库 Data Guard 参数",
                            f"""
ALTER SYSTEM SET log_archive_config='DG_CONFIG=({db_unique_name},${{STANDBY_DB_UNIQUE_NAME}})' SCOPE=BOTH SID='*';
ALTER SYSTEM SET log_archive_dest_2='SERVICE=${{STANDBY_TNS_ALIAS}} ${{REDO_TRANSPORT_MODE}} ${{REDO_TRANSPORT_ACK}} VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE) DB_UNIQUE_NAME=${{STANDBY_DB_UNIQUE_NAME}}' SCOPE=BOTH SID='*';
ALTER SYSTEM SET log_archive_dest_state_2=ENABLE SCOPE=BOTH SID='*';
ALTER SYSTEM SET fal_server='${{STANDBY_TNS_ALIAS}}' SCOPE=BOTH SID='*';
ALTER SYSTEM SET standby_file_management=AUTO SCOPE=BOTH SID='*';
ALTER SYSTEM SET dg_broker_start=TRUE SCOPE=BOTH SID='*';
""",
                        ),
                        _command(
                            "parameters.primary.protection_mode",
                            RunbookCommandType.SQLPLUS,
                            "在备库稳定同步后设置目标保护模式",
                            "ALTER DATABASE SET STANDBY DATABASE TO MAXIMIZE ${PROTECTION_MODE};",
                            "先用 PERFORMANCE 完成建设和追平；切换到 AVAILABILITY 或 PROTECTION 前必须确认 SYNC 目标健康。",
                        ),
                    ),
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
                    required_inputs=("STANDBY_DB_UNIQUE_NAME", "STANDBY_TNS_ALIAS", "PROTECTION_MODE", "REDO_TRANSPORT_MODE", "REDO_TRANSPORT_ACK"),
                    risks=("错误的 SERVICE 或 DB_UNIQUE_NAME 会造成 ORA-160xx 日志传输错误。",),
                ),
                RunbookStep(
                    step_id="parameters.standby",
                    title="准备备库参数文件",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="备库必须使用独立 DB_UNIQUE_NAME，并按存储策略配置文件名转换或 OMF。",
                    commands=(
                        _command(
                            "parameters.standby.pfile",
                            RunbookCommandType.CONFIG,
                            "备库参数模板",
                            f"""
*.db_name='{_text(precheck, 'database_name') or '${PRIMARY_DB_NAME}'}'
*.db_unique_name='${{STANDBY_DB_UNIQUE_NAME}}'
*.log_archive_config='DG_CONFIG=({db_unique_name},${{STANDBY_DB_UNIQUE_NAME}})'
*.log_archive_dest_1='LOCATION=USE_DB_RECOVERY_FILE_DEST VALID_FOR=(ALL_LOGFILES,ALL_ROLES) DB_UNIQUE_NAME=${{STANDBY_DB_UNIQUE_NAME}}'
*.log_archive_dest_2='SERVICE=${{PRIMARY_TNS_ALIAS}} ${{REDO_TRANSPORT_MODE}} ${{REDO_TRANSPORT_ACK}} VALID_FOR=(ONLINE_LOGFILES,PRIMARY_ROLE) DB_UNIQUE_NAME={db_unique_name}'
*.fal_server='${{PRIMARY_TNS_ALIAS}}'
*.standby_file_management='AUTO'
*.db_recovery_file_dest='${{STANDBY_FRA_DEST}}'
*.db_recovery_file_dest_size=${{FRA_SIZE}}
# 非 OMF/ASM 时再设置 db_file_name_convert 和 log_file_name_convert。
""",
                        ),
                    ),
                    required_inputs=("STANDBY_DB_UNIQUE_NAME", "PRIMARY_TNS_ALIAS", "STORAGE_STRATEGY"),
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
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="主备 SYS 密码文件必须一致，且 remote_login_passwordfile 应为 EXCLUSIVE。",
                    commands=(
                        _command(
                            "connectivity.password_file.create",
                            RunbookCommandType.SHELL,
                            "在备库准备密码文件",
                            """
${ORACLE_HOME}/bin/orapwd file=${ORACLE_HOME}/dbs/orapw${STANDBY_ORACLE_SID} format=12.2 force=y
""",
                            "优先通过受控安全通道复制主库密码文件；不要把 SYS 密码写入脚本或 Runbook。",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "connectivity.password_file.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证远程密码文件模式",
                            "SHOW PARAMETER remote_login_passwordfile;",
                        ),
                    ),
                    required_inputs=("ORACLE_HOME", "STANDBY_ORACLE_SID"),
                ),
                RunbookStep(
                    step_id="connectivity.listener_tns",
                    title="配置静态监听和双向 TNS",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="NOMOUNT/MOUNT 状态的辅助实例需要静态注册，主备必须双向解析。",
                    commands=(
                        _command(
                            "connectivity.listener.config",
                            RunbookCommandType.CONFIG,
                            "备库 listener.ora 静态注册片段",
                            """
SID_LIST_LISTENER =
  (SID_LIST =
    (SID_DESC =
      (GLOBAL_DBNAME = ${STANDBY_DB_UNIQUE_NAME}_DGMGRL)
      (ORACLE_HOME = ${ORACLE_HOME})
      (SID_NAME = ${STANDBY_ORACLE_SID})))
""",
                        ),
                        _command(
                            "connectivity.tns.config",
                            RunbookCommandType.CONFIG,
                            "tnsnames.ora 双向别名模板",
                            """
${STANDBY_TNS_ALIAS} =
  (DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=${STANDBY_HOST})(PORT=1521))
    (CONNECT_DATA=(SERVICE_NAME=${STANDBY_DB_UNIQUE_NAME}_DGMGRL)))
""",
                        ),
                        _command(
                            "connectivity.listener.reload",
                            RunbookCommandType.SHELL,
                            "重载监听并测试",
                            """
${ORACLE_HOME}/bin/lsnrctl reload
${ORACLE_HOME}/bin/tnsping ${PRIMARY_TNS_ALIAS}
${ORACLE_HOME}/bin/tnsping ${STANDBY_TNS_ALIAS}
""",
                        ),
                    ),
                    required_inputs=("STANDBY_HOST", "ORACLE_HOME", "STANDBY_ORACLE_SID", "PRIMARY_TNS_ALIAS", "STANDBY_TNS_ALIAS"),
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
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="RMAN Duplicate 需要可远程连接的 NOMOUNT 辅助实例。",
                    commands=(
                        _command(
                            "standby.nomount.sql",
                            RunbookCommandType.SQLPLUS,
                            "启动备库辅助实例",
                            """
CREATE SPFILE FROM PFILE='${STANDBY_PFILE}';
STARTUP NOMOUNT;
""",
                        ),
                    ),
                    required_inputs=("STANDBY_ORACLE_SID", "STORAGE_STRATEGY"),
                ),
                RunbookStep(
                    step_id="standby.duplicate",
                    title="执行 RMAN Active Duplicate",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="从当前主库在线复制数据文件、控制文件和归档并建立物理备库。",
                    commands=(
                        _command(
                            "standby.duplicate.rman",
                            RunbookCommandType.RMAN,
                            "连接主库和辅助实例并复制",
                            """
CONNECT TARGET sys@${PRIMARY_TNS_ALIAS};
CONNECT AUXILIARY sys@${STANDBY_TNS_ALIAS};
RUN {
  ALLOCATE CHANNEL c1 DEVICE TYPE DISK;
  ALLOCATE AUXILIARY CHANNEL a1 DEVICE TYPE DISK;
  DUPLICATE TARGET DATABASE FOR STANDBY FROM ACTIVE DATABASE
    DORECOVER
    SPFILE
      SET db_unique_name='${STANDBY_DB_UNIQUE_NAME}'
      SET fal_server='${PRIMARY_TNS_ALIAS}'
      SET standby_file_management='AUTO'
    NOFILENAMECHECK;
}
""",
                            "仅在主备使用相同 ASM/OMF 命名或已经正确设置文件名转换时使用 NOFILENAMECHECK。",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "standby.duplicate.verify",
                            RunbookCommandType.SQLPLUS,
                            "确认备库角色与打开模式",
                            "SELECT database_role, open_mode, db_unique_name FROM v$database;",
                        ),
                    ),
                    required_inputs=("PRIMARY_TNS_ALIAS", "STANDBY_TNS_ALIAS", "STANDBY_DB_UNIQUE_NAME", "STORAGE_STRATEGY"),
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
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="Broker 提供一致的配置校验和切换前检查，但本 Runbook 不自动执行切换。",
                    commands=(
                        _command(
                            "broker.configure.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "建立 Broker 配置",
                            f"""
CREATE CONFIGURATION '${{DG_CONFIG_NAME}}' AS
  PRIMARY DATABASE IS '{db_unique_name}'
  CONNECT IDENTIFIER IS '${{PRIMARY_TNS_ALIAS}}';
ADD DATABASE '${{STANDBY_DB_UNIQUE_NAME}}' AS
  CONNECT IDENTIFIER IS '${{STANDBY_TNS_ALIAS}}'
  MAINTAINED AS PHYSICAL;
ENABLE CONFIGURATION;
SHOW CONFIGURATION VERBOSE;
VALIDATE DATABASE VERBOSE '{db_unique_name}';
VALIDATE DATABASE VERBOSE '${{STANDBY_DB_UNIQUE_NAME}}';
""",
                        ),
                    ),
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
                    required_inputs=("PRIMARY_TNS_ALIAS", "STANDBY_TNS_ALIAS", "STANDBY_DB_UNIQUE_NAME"),
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
                            """
SHOW CONFIGURATION;
SHOW DATABASE VERBOSE '${STANDBY_DB_UNIQUE_NAME}';
VALIDATE NETWORK CONFIGURATION FOR ALL;
""",
                        ),
                    ),
                    risks=("存在未解决 gap、持续 apply lag 或 Broker WARNING 时不得进入切换演练。",),
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
            "用户后续明确选择某一步执行时，必须重新核验当时状态并进入受控 Action/审批。"
        ),
        current_state=current_state,
        required_inputs=_adg_required_inputs(),
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
) -> ImplementationRunbook:
    """按结构化档案编译 Runbook，禁止模型自由拼接执行命令。"""
    if profile == ImplementationProfile.ORACLE_ADG_BUILD:
        return _compile_oracle_adg_build(evidence)
    raise ValueError(f"不支持的实施方案档案：{profile}")
