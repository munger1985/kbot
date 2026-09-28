"""从已验证环境事实确定性编译数据库实施方案。"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
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


def _directory_of(path: str) -> str:
    """从主库文件事实提取可在新主机复用的目录或 ASM 磁盘组。"""
    value = path.strip()
    if not value:
        return ""
    if value.startswith("+"):
        return value.split("/", 1)[0]
    return str(PurePosixPath(value).parent)


def _oracle_release_label(version: str) -> str:
    """把数据库版本归一为安装目录和预安装包使用的发行标签。"""
    major_match = re.match(r"(\d+)", version.strip())
    major = major_match.group(1) if major_match else "26"
    return f"{major}ai" if int(major) >= 23 else f"{major}c"


def _default_oracle_home(version: str) -> str:
    """按数据库大版本生成全新备库的标准 Oracle Home。"""
    release = _oracle_release_label(version)
    return f"/u01/app/oracle/product/{release}/dbhome_1"


def _oracle_home_from_password_file(path: str) -> str:
    """优先从主库密码文件路径还原实际 Oracle Home。"""
    value = path.strip()
    if not value or value.startswith("+"):
        return ""
    password_directory = PurePosixPath(value).parent
    if password_directory.name.lower() != "dbs":
        return ""
    return str(password_directory.parent)


def _derived_standby_host(primary_host: str) -> str:
    """为全新备库生成稳定且可提前纳入 DNS 的主机名。"""
    host = primary_host.strip().split(".", 1)[0]
    domain = primary_host.strip()[len(host):]
    return f"{host}-stby{domain}" if host else "oracle-standby"


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

    source_pdb_name = _text(precheck, "container_name")
    if _integer(precheck, "container_id") > 1:
        add(
            "SOURCE_PDB_NAME",
            "源 PDB",
            source_pdb_name,
            RunbookParameterStatus.VERIFIED,
            "db.ha.adg_precheck:SYS_CONTEXT(USERENV, CON_NAME)",
        )
        add(
            "TARGET_PDB_NAME",
            "目标 PDB",
            _explicit_parameter(context, "TARGET_PDB_NAME") or source_pdb_name,
            (
                RunbookParameterStatus.VERIFIED
                if _explicit_parameter(context, "TARGET_PDB_NAME")
                else RunbookParameterStatus.DERIVED
            ),
            (
                "用户确认的实施参数"
                if _explicit_parameter(context, "TARGET_PDB_NAME")
                else "全新目标 CDB 默认沿用源 PDB 名称"
            ),
        )
        target_cdb_name = _explicit_parameter(
            context, "TARGET_CDB_DB_NAME"
        ) or _oracle_name_with_suffix(primary_db_name, "S", 8)
        target_cdb_unique_name = _explicit_parameter(
            context, "TARGET_CDB_DB_UNIQUE_NAME"
        ) or _oracle_name_with_suffix(primary_unique_name, "_dgpdb", 30)
        add(
            "TARGET_CDB_DB_NAME",
            "目标 CDB DB_NAME",
            target_cdb_name,
            (
                RunbookParameterStatus.VERIFIED
                if _explicit_parameter(context, "TARGET_CDB_DB_NAME")
                else RunbookParameterStatus.DERIVED
            ),
            "全新目标 CDB 名称由源 CDB 名称派生",
        )
        add(
            "TARGET_CDB_DB_UNIQUE_NAME",
            "目标 CDB DB_UNIQUE_NAME",
            target_cdb_unique_name,
            (
                RunbookParameterStatus.VERIFIED
                if _explicit_parameter(context, "TARGET_CDB_DB_UNIQUE_NAME")
                else RunbookParameterStatus.DERIVED
            ),
            "全新目标 CDB 唯一名由源 CDB DB_UNIQUE_NAME 派生",
        )
        add(
            "TARGET_CDB_SID",
            "目标 CDB ORACLE_SID",
            target_cdb_name,
            RunbookParameterStatus.DERIVED,
            "目标 CDB SID 与 DB_NAME 保持一致",
        )
        add(
            "TARGET_CDB_TNS_ALIAS",
            "目标 CDB TNS Alias",
            target_cdb_unique_name,
            RunbookParameterStatus.DERIVED,
            "采用目标 CDB DB_UNIQUE_NAME 作为稳定别名",
        )
        add(
            "TARGET_CDB_CONFIG_NAME",
            "目标 CDB Broker 配置名",
            _oracle_name_with_suffix(target_cdb_unique_name, "_cfg", 30),
            RunbookParameterStatus.DERIVED,
            "由目标 CDB DB_UNIQUE_NAME 派生",
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

    version = _text(identity, "version")
    explicit_standby_host = _explicit_parameter(context, "STANDBY_HOST")
    derived_standby_host = explicit_standby_host or _derived_standby_host(
        primary_host
    )
    add(
        "STANDBY_HOST",
        "备库主机",
        derived_standby_host,
        (
            RunbookParameterStatus.VERIFIED
            if explicit_standby_host
            else RunbookParameterStatus.DERIVED
        ),
        (
            "用户确认的实施参数"
            if explicit_standby_host
            else "全新备库按 <主库主机名>-stby 规则派生；需提前配置 DNS"
        ),
    )
    explicit_oracle_home = _explicit_parameter(context, "ORACLE_HOME")
    primary_password_file = _text(precheck, "password_file_path")
    observed_oracle_home = _oracle_home_from_password_file(primary_password_file)
    oracle_home = (
        explicit_oracle_home
        or observed_oracle_home
        or _default_oracle_home(version)
    )
    add(
        "ORACLE_HOME",
        "备库 Oracle Home",
        oracle_home,
        (
            RunbookParameterStatus.VERIFIED
            if explicit_oracle_home
            else RunbookParameterStatus.DERIVED
        ),
        (
            "用户确认的实施参数"
            if explicit_oracle_home
            else (
                "由主库密码文件路径还原实际 Oracle Home"
                if observed_oracle_home
                else f"按主库版本 {version or '26ai'} 采用标准安装目录"
            )
        ),
    )
    oracle_base = oracle_home.split("/product/", 1)[0]
    add(
        "ORACLE_BASE",
        "备库 Oracle Base",
        oracle_base,
        RunbookParameterStatus.DERIVED,
        "由备库 Oracle Home 派生",
    )

    for key, label, column in (
        ("PRIMARY_DATAFILE_PATH", "主库数据文件样例", "sample_datafile_path"),
        ("PRIMARY_TEMPFILE_PATH", "主库临时文件样例", "sample_tempfile_path"),
        ("PRIMARY_REDO_MEMBER_PATH", "主库 redo 成员样例", "sample_redo_member_path"),
        ("PRIMARY_AUDIT_DEST", "主库审计目录", "audit_file_dest"),
        ("PRIMARY_DIAGNOSTIC_DEST", "主库诊断目录", "diagnostic_dest"),
        ("PRIMARY_CONTROL_FILES", "主库控制文件", "control_files"),
    ):
        add(
            key,
            label,
            _text(precheck, column),
            RunbookParameterStatus.VERIFIED,
            f"db.ha.adg_precheck:{column}",
        )
    observed_password_file = _text(precheck, "password_file_path")
    derived_password_file = observed_password_file or (
        f"{oracle_home}/dbs/orapw{_text(precheck, 'instance_name')}"
    )
    add(
        "PRIMARY_PASSWORD_FILE",
        "主库密码文件",
        derived_password_file,
        (
            RunbookParameterStatus.VERIFIED
            if observed_password_file
            else RunbookParameterStatus.DERIVED
        ),
        (
            "db.ha.adg_precheck:password_file_path"
            if observed_password_file
            else "按主库 Oracle Home 和 INSTANCE_NAME 派生"
        ),
    )

    data_directory = _directory_of(_text(precheck, "sample_datafile_path"))
    redo_directory = _directory_of(_text(precheck, "sample_redo_member_path"))
    password_directory = _directory_of(_text(precheck, "password_file_path"))
    target_path_sid = (
        target_cdb_name
        if _integer(precheck, "container_id") > 1
        else standby_sid
    )
    audit_directory = _text(precheck, "audit_file_dest") or (
        f"{oracle_base}/admin/{target_path_sid}/adump"
        if target_path_sid
        else ""
    )
    add(
        "STANDBY_DATA_DEST",
        "备库数据文件目录",
        data_directory or _text(precheck, "db_create_file_dest"),
        RunbookParameterStatus.DERIVED,
        "全新备库沿用主库数据文件目录或 OMF 磁盘组",
    )
    add(
        "STANDBY_REDO_DEST",
        "备库 redo 目录",
        redo_directory or data_directory or _text(precheck, "db_create_file_dest"),
        RunbookParameterStatus.DERIVED,
        "全新备库沿用主库 redo 目录",
    )
    add(
        "STANDBY_AUDIT_DEST",
        "备库审计目录",
        audit_directory,
        RunbookParameterStatus.DERIVED,
        "全新备库沿用主库审计目录规则",
    )
    if target_path_sid:
        add(
            "STANDBY_PASSWORD_FILE",
            "备库密码文件",
            (
                f"{password_directory}/orapw{target_path_sid}"
                if password_directory and not password_directory.startswith("+")
                else f"{oracle_home}/dbs/orapw{target_path_sid}"
            ),
            RunbookParameterStatus.DERIVED,
            "沿用主库密码文件目录并替换为备库 SID 文件名",
        )

    observed_fra_dest = _text(precheck, "db_recovery_file_dest")
    explicit_fra_dest = _explicit_parameter(context, "PRIMARY_FRA_DEST")
    derived_fra_dest = (
        observed_fra_dest
        or explicit_fra_dest
        or f"{oracle_base}/fast_recovery_area"
    )
    add(
        "PRIMARY_FRA_DEST",
        "主库 FRA",
        derived_fra_dest,
        (
            RunbookParameterStatus.VERIFIED
            if observed_fra_dest or explicit_fra_dest
            else RunbookParameterStatus.DERIVED
        ),
        (
            "db.ha.adg_precheck:db_recovery_file_dest"
            if observed_fra_dest
            else (
                "用户确认的实施参数"
                if explicit_fra_dest
                else "未配置 FRA 时按 Oracle Base 派生标准恢复区"
            )
        ),
    )
    observed_fra_size = _text(precheck, "db_recovery_file_dest_size")
    explicit_fra_size = _explicit_parameter(context, "PRIMARY_FRA_SIZE")
    derived_fra_size = observed_fra_size or explicit_fra_size or str(
        max(20 * 1024**3, int(_integer(precheck, "datafile_bytes") * 0.35))
    )
    add(
        "PRIMARY_FRA_SIZE",
        "主库 FRA 容量（字节）",
        derived_fra_size,
        (
            RunbookParameterStatus.VERIFIED
            if observed_fra_size or explicit_fra_size
            else RunbookParameterStatus.DERIVED
        ),
        (
            "db.ha.adg_precheck:db_recovery_file_dest_size"
            if observed_fra_size
            else (
                "用户确认的实施参数"
                if explicit_fra_size
                else "按数据文件容量的 35% 计算且不低于 20 GiB"
            )
        ),
    )
    add(
        "PRIMARY_DB_CREATE_FILE_DEST",
        "主库 OMF 目录",
        _text(precheck, "db_create_file_dest") or data_directory,
        (
            RunbookParameterStatus.VERIFIED
            if _text(precheck, "db_create_file_dest")
            else RunbookParameterStatus.DERIVED
        ),
        (
            "db.ha.adg_precheck:db_create_file_dest"
            if _text(precheck, "db_create_file_dest")
            else "由主库数据文件样例目录派生"
        ),
    )

    storage_lines: list[str] = []
    standby_data_dest = data_directory or _text(precheck, "db_create_file_dest")
    if standby_data_dest:
        storage_lines.append(
            f"*.db_create_file_dest='{standby_data_dest}'"
        )
    if derived_fra_dest:
        storage_lines.extend(
            (
                f"*.db_recovery_file_dest='{derived_fra_dest}'",
                f"*.db_recovery_file_dest_size={derived_fra_size}",
            )
        )
    explicit_storage = _explicit_parameter(context, "STANDBY_STORAGE")
    add(
        "STANDBY_STORAGE",
        "备库存储配置",
        explicit_storage or "\n".join(storage_lines) or "SAME_PATH_AS_PRIMARY",
        (
            RunbookParameterStatus.VERIFIED
            if explicit_storage
            else RunbookParameterStatus.DERIVED
        ),
        (
            "用户确认的实施参数"
            if explicit_storage
            else "全新备库默认使用与主库相同的目录或 ASM/OMF 磁盘组"
        ),
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
    redo_destination: str = "",
) -> tuple[RunbookCommand, ...]:
    plan = _text(row, "redo_thread_plan")
    group_number = _integer(row, "max_redo_group_number")
    commands: list[RunbookCommand] = []
    if plan:
        for thread_spec in plan.split(","):
            parts = thread_spec.split(":")
            if len(parts) != 5:
                continue
            thread_no, _online, _required, missing, size_mb = parts
            for ordinal in range(1, max(0, int(missing)) + 1):
                group_number += 1
                member_clause = (
                    f" ('{redo_destination}/standby_redo_t{thread_no}_g{group_number}.log')"
                    if redo_destination and not redo_destination.startswith("+")
                    else ""
                )
                commands.append(
                    _command(
                        f"{command_scope}.srl.t{thread_no}.{ordinal}",
                        RunbookCommandType.SQLPLUS,
                        f"在{title_scope}为线程 {thread_no} 增加第 {ordinal} 个 Standby Redo Log",
                        f"""
ALTER DATABASE ADD STANDBY LOGFILE THREAD {thread_no} GROUP {group_number}{member_clause}
  SIZE {size_mb}M;
""",
                        "文件系统路径沿用主库 redo 目录；ASM/OMF 环境由数据库自动创建成员。",
                    )
                )
    return tuple(commands)


def _compile_oracle_dgpdb_build(
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any],
) -> ImplementationRunbook:
    """为 PDB Target 生成从零建设目标 CDB 的 DGPDB 操作文档。"""
    identity, identity_ref = _first_row(evidence, "db.instance.identity")
    precheck, precheck_ref = _first_row(evidence, "db.ha.adg_precheck")
    parameters, resolved_parameters = _resolve_adg_parameters(
        identity=identity,
        precheck=precheck,
        context=context,
    )
    evidence_refs = tuple(
        value for value in (identity_ref, precheck_ref) if value is not None
    )
    source_cdb = parameters.get("PRIMARY_DB_UNIQUE_NAME", "")
    source_pdb = parameters.get("SOURCE_PDB_NAME", "")
    target_pdb = parameters.get("TARGET_PDB_NAME", source_pdb)
    target_cdb_name = parameters.get("TARGET_CDB_DB_NAME", "")
    target_cdb = parameters.get("TARGET_CDB_DB_UNIQUE_NAME", "")
    target_sid = parameters.get("TARGET_CDB_SID", target_cdb_name)
    source_tns = parameters.get("PRIMARY_TNS_ALIAS", source_cdb)
    target_tns = parameters.get("TARGET_CDB_TNS_ALIAS", target_cdb)
    source_config = parameters.get("DG_CONFIG_NAME", "")
    target_config = parameters.get("TARGET_CDB_CONFIG_NAME", "")
    source_host = parameters.get("PRIMARY_HOST", "")
    target_host = parameters.get("STANDBY_HOST", "")
    port = parameters.get("PRIMARY_PORT", "1521")
    oracle_home = parameters.get("ORACLE_HOME", "")
    oracle_base = parameters.get("ORACLE_BASE", "")
    data_dest = parameters.get("STANDBY_DATA_DEST", "")
    redo_dest = parameters.get("STANDBY_REDO_DEST", data_dest)
    audit_dest = parameters.get("STANDBY_AUDIT_DEST", "")
    target_password_file = parameters.get("STANDBY_PASSWORD_FILE", "")
    source_password_file = parameters.get("PRIMARY_PASSWORD_FILE", "") or (
        f"{oracle_home}/dbs/orapw{_text(precheck, 'instance_name')}"
    )
    observed_source_password_file = _text(precheck, "password_file_path")
    fra_dest = _text(precheck, "db_recovery_file_dest") or (
        f"{oracle_base}/fast_recovery_area"
    )
    datafile_bytes = _integer(precheck, "datafile_bytes")
    fra_size = _integer(precheck, "db_recovery_file_dest_size") or max(
        20 * 1024**3,
        int(datafile_bytes * 0.35),
    )
    fra_size_mb = max(20480, fra_size // 1024 // 1024)
    character_set = _text(precheck, "character_set") or "AL32UTF8"
    national_character_set = (
        _text(precheck, "national_character_set") or "AL16UTF16"
    )
    compatible = _text(precheck, "compatible") or _text(identity, "version")
    redo_size = max(200, _integer(precheck, "online_redo_max_size_mb"))
    processes = max(500, _integer(precheck, "processes"))
    sga_target = _integer(precheck, "sga_target")
    memory_mb = max(2048, sga_target // 1024 // 1024) if sga_target else 4096
    version = _text(identity, "version") or "26ai"
    release_label = _oracle_release_label(version)
    software_archive = f"/stage/oracle/LINUX.X64_{release_label}_db_home.zip"
    filesystem_directories = tuple(
        dict.fromkeys(
            value
            for value in (
                oracle_base,
                oracle_home,
                data_dest,
                redo_dest,
                fra_dest,
                audit_dest,
                f"{oracle_base}/admin/{target_sid}/dpdump" if oracle_base else "",
                f"{oracle_base}/admin/{target_sid}/pfile" if oracle_base else "",
            )
            if value and value.startswith("/")
        )
    )
    mkdir_command = " ".join(filesystem_directories)
    storage_type = "ASM" if data_dest.startswith("+") else "FS"
    dbca_storage = (
        f"-storageType ASM -diskGroupName {data_dest.lstrip('+')}"
        if storage_type == "ASM"
        else f"-storageType FS -datafileDestination '{data_dest}'"
    )
    recovery_option = (
        f"-recoveryGroupName {fra_dest.lstrip('+')}"
        if fra_dest.startswith("+")
        else f"-recoveryAreaDestination '{fra_dest}'"
    )
    redo_member = (
        ""
        if redo_dest.startswith("+")
        else f" ('{redo_dest}/standby_redo_t1_g{{group}}.log')"
    )
    target_srl_lines = "\n".join(
        (
            f"ALTER DATABASE ADD STANDBY LOGFILE THREAD 1 GROUP {group}"
            f"{redo_member.format(group=group)} SIZE {redo_size}M;"
        )
        for group in range(11, 15)
    )
    source_archivelog_commands = (
        "SHUTDOWN IMMEDIATE;\n"
        "STARTUP MOUNT;\n"
        "ALTER DATABASE ARCHIVELOG;\n"
        "ALTER DATABASE OPEN;\n"
        if _text(precheck, "log_mode").upper() != "ARCHIVELOG"
        else ""
    )
    source_passwordfile_commands = (
        "ALTER SYSTEM SET remote_login_passwordfile='EXCLUSIVE' "
        "SCOPE=SPFILE SID='*';\n"
        if _text(precheck, "remote_login_passwordfile").upper() != "EXCLUSIVE"
        else ""
    )
    source_passwordfile_restart = (
        "SHUTDOWN IMMEDIATE;\nSTARTUP;\n"
        if source_passwordfile_commands
        else ""
    )
    source_srl_commands = "\n".join(
        command.content
        for command in _srl_commands(
            precheck,
            command_scope="source_cdb",
            title_scope="源 CDB",
            redo_destination=redo_dest,
        )
    )
    source_dgmgrl_service = f"{source_cdb}_DGMGRL"
    target_dgmgrl_service = f"{target_cdb}_DGMGRL"
    status = (
        RunbookStatus.PARTIAL_EVIDENCE
        if precheck is None
        else RunbookStatus.READY
    )
    current_state = (
        _state_item("保护范围", source_pdb, "PDB_LEVEL_DGPDB"),
        _state_item("源 CDB", source_cdb, "VERIFIED"),
        _state_item("目标 CDB", target_cdb, "DERIVED"),
        _state_item("目标主机", target_host, "DERIVED"),
        _state_item("数据库版本", version, "VERIFIED"),
        _state_item("目录策略", "沿用主库目录结构", "DERIVED"),
        _state_item(
            "归档模式",
            _text(precheck, "log_mode"),
            "SATISFIED"
            if _text(precheck, "log_mode").upper() == "ARCHIVELOG"
            else "REMEDIATION_REQUIRED",
        ),
        _state_item(
            "强制日志",
            _text(precheck, "force_logging"),
            "SATISFIED"
            if _text(precheck, "force_logging").upper() == "YES"
            else "REMEDIATION_REQUIRED",
        ),
    )
    phases = (
        RunbookPhase(
            phase_id="derived_plan",
            title="确认自动派生的目标环境",
            objective="明确系统已采用的全新目标 CDB、主机名和目录规划，不再等待用户补充参数。",
            steps=(
                RunbookStep(
                    step_id="derived_plan.defaults",
                    title="记录目标环境默认规划",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="目标环境从零建设，所有可推导名称和路径均沿用主库规则。",
                    commands=(
                        _command(
                            "derived_plan.defaults.record",
                            RunbookCommandType.MANUAL,
                            "实施前发布基础设施规划",
                            (
                                f"目标主机使用 {target_host}；目标 CDB 使用 {target_cdb_name}/"
                                f"{target_cdb}；目标 PDB 使用 {target_pdb}；Oracle Home 使用 "
                                f"{oracle_home}；数据文件、redo、FRA 和审计目录沿用主库目录结构。"
                            ),
                            "如企业实际 DNS 或存储规范不同，应在变更评审中调整派生值，但系统不再因这些事实缺失而拒绝生成文档。",
                        ),
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="source_cdb",
            title="源 CDB 前置整改",
            objective="在源 CDB Root 完成 DGPDB 所需的归档、日志和 Broker 前置条件。",
            steps=(
                RunbookStep(
                    step_id="source_cdb.prerequisites",
                    title="启用源 CDB 前置能力",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="PDB 是保护单位，但 redo、归档和 Broker 由所属 CDB 管理。",
                    commands=(
                        _command(
                            "source_cdb.prerequisites.sql",
                            RunbookCommandType.SQLPLUS,
                            "在源 CDB Root 执行",
                            f"""
ALTER SESSION SET CONTAINER=CDB$ROOT;
{source_archivelog_commands}{source_passwordfile_commands}
ALTER DATABASE FORCE LOGGING;
ALTER SYSTEM SET db_recovery_file_dest_size={fra_size} SCOPE=BOTH SID='*';
ALTER SYSTEM SET db_recovery_file_dest='{fra_dest}' SCOPE=BOTH SID='*';
ALTER SYSTEM SET dg_broker_start=TRUE SCOPE=BOTH SID='*';
{source_srl_commands}
{source_passwordfile_restart}
""",
                            "所有命令均在源 CDB Root 以 SYSDBA 执行；密码文件模式发生变化时步骤会显式重启 CDB。",
                        ),
                    ) + (
                        (
                            _command(
                                "source_cdb.prerequisites.password_file",
                                RunbookCommandType.SHELL,
                                "在源数据库主机创建独占密码文件",
                                f"""
read -rsp '输入当前 SYS 密码: ' KBOT_SYS_PASSWORD; echo
{oracle_home}/bin/orapwd file={source_password_file} password="$KBOT_SYS_PASSWORD" format=12.2 force=y
unset KBOT_SYS_PASSWORD
""",
                                "仅在 V$PASSWORDFILE_INFO 未返回现有密码文件时执行。",
                            ),
                        )
                        if not observed_source_password_file
                        else ()
                    ),
                    verification_commands=(
                        _command(
                            "source_cdb.prerequisites.verify",
                            RunbookCommandType.SQLPLUS,
                            "核对源 CDB 和源 PDB",
                            f"""
SELECT name, db_unique_name, database_role, log_mode, force_logging, open_mode
FROM v$database;
SELECT name, open_mode, restricted FROM v$pdbs WHERE name=UPPER('{source_pdb}');
SHOW PARAMETER dg_broker_start;
""",
                        ),
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="target_os",
            title="从零准备目标主机",
            objective="按主库版本和目录结构安装 Oracle 软件并准备目标 CDB 文件系统。",
            steps=(
                RunbookStep(
                    step_id="target_os.host",
                    title="创建操作系统用户和目录",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="目标主机是全新环境，目录结构直接沿用主库。",
                    commands=(
                        _command(
                            "target_os.host.shell",
                            RunbookCommandType.SHELL,
                            "在目标主机以 root 执行",
                            f"""
hostnamectl set-hostname {target_host.split('.', 1)[0]}
getent group oinstall >/dev/null || groupadd -g 54321 oinstall
getent group dba >/dev/null || groupadd -g 54322 dba
id oracle >/dev/null 2>&1 || useradd -u 54321 -g oinstall -G dba oracle
dnf install -y oracle-database-preinstall-{release_label} unzip
mkdir -p {mkdir_command}
chown -R oracle:oinstall {oracle_base} {data_dest if data_dest.startswith('/') else oracle_base} {fra_dest if fra_dest.startswith('/') else oracle_base}
chmod -R 775 {oracle_base}
""",
                        ),
                        _command(
                            "target_os.host.network",
                            RunbookCommandType.SHELL,
                            "验证主备 DNS 和端口",
                            f"""
getent hosts {source_host}
getent hosts {target_host}
nc -vz {source_host} {port}
nc -vz {target_host} {port}
""",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="target_os.software",
                    title="安装与主库一致的 Oracle 软件和 RU",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="DGPDB 两端必须使用兼容的数据库版本和补丁级别。",
                    commands=(
                        _command(
                            "target_os.software.shell",
                            RunbookCommandType.SHELL,
                            "以 oracle 用户安装软件",
                            f"""
export ORACLE_BASE={oracle_base}
export ORACLE_HOME={oracle_home}
export PATH=$ORACLE_HOME/bin:$PATH
mkdir -p "$ORACLE_HOME"
unzip -q {software_archive} -d "$ORACLE_HOME"
$ORACLE_HOME/runInstaller -silent -waitforcompletion \\
  oracle.install.option=INSTALL_DB_SWONLY \\
  UNIX_GROUP_NAME=oinstall \\
  INVENTORY_LOCATION={oracle_base}/oraInventory \\
  ORACLE_HOME="$ORACLE_HOME" \\
  ORACLE_BASE="$ORACLE_BASE" \\
  oracle.install.db.InstallEdition=EE \\
  oracle.install.db.OSDBA_GROUP=dba \\
  DECLINE_SECURITY_UPDATES=true
""",
                            "安装介质固定放置在 /stage/oracle；安装完成后由 root 执行提示的 root.sh，并应用与主库完全一致的 RU。",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "target_os.software.verify",
                            RunbookCommandType.SHELL,
                            "核对软件版本",
                            f"""
{oracle_home}/bin/sqlplus -V
{oracle_home}/OPatch/opatch lspatches
""",
                        ),
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="target_cdb",
            title="创建目标 CDB",
            objective="创建独立目标 CDB，字符集、兼容参数和目录布局与源 CDB 保持一致。",
            steps=(
                RunbookStep(
                    step_id="target_cdb.create",
                    title="使用 DBCA 创建空目标 CDB",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="DGPDB 需要一个独立目标 CDB 承载 standby PDB，而不是复制整个源 CDB。",
                    commands=(
                        _command(
                            "target_cdb.create.dbca",
                            RunbookCommandType.SHELL,
                            "在目标主机以 oracle 用户执行",
                            f"""
export ORACLE_BASE={oracle_base}
export ORACLE_HOME={oracle_home}
export ORACLE_SID={target_sid}
export PATH=$ORACLE_HOME/bin:$PATH
read -rsp '输入与源库一致的 SYS 密码: ' KBOT_SYS_PASSWORD; echo
dbca -silent -createDatabase \\
  -templateName General_Purpose.dbc \\
  -gdbname {target_cdb} \\
  -sid {target_sid} \\
  -createAsContainerDatabase true \\
  -numberOfPDBs 0 \\
  -characterSet {character_set} \\
  -nationalCharacterSet {national_character_set} \\
  -databaseType MULTIPURPOSE \\
  -memoryMgmtType auto_sga \\
  -totalMemory {memory_mb} \\
  -initParams compatible={compatible},processes={processes},db_unique_name={target_cdb},remote_login_passwordfile=EXCLUSIVE \\
  {dbca_storage} \\
  {recovery_option} \\
  -recoveryAreaSize {fra_size_mb} \\
  -sysPassword "$KBOT_SYS_PASSWORD" \\
  -systemPassword "$KBOT_SYS_PASSWORD"
unset KBOT_SYS_PASSWORD
""",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "target_cdb.create.verify",
                            RunbookCommandType.SQLPLUS,
                            "核对目标 CDB",
                            "SELECT name, db_unique_name, cdb, open_mode, log_mode FROM v$database;",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="target_cdb.configure",
                    title="配置目标 CDB 归档、Broker 和 SRL",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="目标 CDB 必须具备接收和应用源 PDB redo 的基础能力。",
                    commands=(
                        _command(
                            "target_cdb.configure.sql",
                            RunbookCommandType.SQLPLUS,
                            "在目标 CDB Root 执行",
                            f"""
SHUTDOWN IMMEDIATE;
STARTUP MOUNT;
ALTER DATABASE ARCHIVELOG;
ALTER DATABASE OPEN;
ALTER DATABASE FORCE LOGGING;
ALTER SYSTEM SET db_recovery_file_dest_size={fra_size} SCOPE=BOTH SID='*';
ALTER SYSTEM SET db_recovery_file_dest='{fra_dest}' SCOPE=BOTH SID='*';
ALTER SYSTEM SET dg_broker_start=TRUE SCOPE=BOTH SID='*';
{target_srl_lines}
""",
                        ),
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="connectivity",
            title="配置密码文件、监听和双向 TNS",
            objective="复制主库密码文件并建立 DGPDB Broker 所需的双向连接。",
            steps=(
                RunbookStep(
                    step_id="connectivity.password",
                    title="复制主库密码文件",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="目标 CDB 复用主库 SYS 密码文件内容，仅调整目标 SID 文件名。",
                    commands=(
                        _command(
                            "connectivity.password.shell",
                            RunbookCommandType.SHELL,
                            "在目标主机以 oracle 用户执行",
                            f"""
scp oracle@{source_host}:{source_password_file} {target_password_file}
chmod 600 {target_password_file}
""",
                            "密码文件必须通过受控 SSH 通道复制，不能在文档中记录 SYS 密码。",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="connectivity.listener_tns",
                    title="配置监听和双向 TNS",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="两个独立 CDB 及 DGPDB Broker 必须能够使用稳定别名双向连接。",
                    commands=(
                        _command(
                            "connectivity.listener.source_config",
                            RunbookCommandType.CONFIG,
                            "主库 listener.ora 静态注册",
                            f"""
SID_LIST_LISTENER =
  (SID_LIST =
    (SID_DESC =
      (GLOBAL_DBNAME = {source_dgmgrl_service})
      (ORACLE_HOME = {oracle_home})
      (SID_NAME = {_text(precheck, 'instance_name')})))
""",
                        ),
                        _command(
                            "connectivity.listener.target_config",
                            RunbookCommandType.CONFIG,
                            "目标主机 listener.ora 静态注册",
                            f"""
SID_LIST_LISTENER =
  (SID_LIST =
    (SID_DESC =
      (GLOBAL_DBNAME = {target_dgmgrl_service})
      (ORACLE_HOME = {oracle_home})
      (SID_NAME = {target_sid})))
""",
                        ),
                        _command(
                            "connectivity.tns.config",
                            RunbookCommandType.CONFIG,
                            "主备两端 tnsnames.ora",
                            f"""
{source_tns} =
  (DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={source_host})(PORT={port}))
    (CONNECT_DATA=(SERVICE_NAME={source_dgmgrl_service})))

{target_tns} =
  (DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST={target_host})(PORT={port}))
    (CONNECT_DATA=(SERVICE_NAME={target_dgmgrl_service})))
""",
                        ),
                        _command(
                            "connectivity.listener.reload",
                            RunbookCommandType.SHELL,
                            "分别在主库和目标主机重载监听，再测试双向别名",
                            f"""
{oracle_home}/bin/lsnrctl reload
{oracle_home}/bin/tnsping {source_tns}
{oracle_home}/bin/tnsping {target_tns}
""",
                        ),
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="dgpdb",
            title="建立 DGPDB",
            objective="把源 PDB 加入跨 CDB Broker 配置组并在目标 CDB 创建 standby PDB。",
            steps=(
                RunbookStep(
                    step_id="dgpdb.broker_configs",
                    title="创建两个 CDB 的 Broker 配置和配置组",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="DGPDB 使用配置组协调两个独立 CDB，而不是把目标 CDB 设为整库物理备库。",
                    commands=(
                        _command(
                            "dgpdb.broker_configs.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "创建并组合 Broker 配置",
                            f"""
CONNECT sys@{source_tns};
CREATE CONFIGURATION '{source_config}' AS
  PRIMARY DATABASE IS '{source_cdb}'
  CONNECT IDENTIFIER IS '{source_tns}';
ENABLE CONFIGURATION;

CONNECT sys@{target_tns};
CREATE CONFIGURATION '{target_config}' AS
  PRIMARY DATABASE IS '{target_cdb}'
  CONNECT IDENTIFIER IS '{target_tns}';
ENABLE CONFIGURATION;

CONNECT sys@{source_tns};
ADD CONFIGURATION '{target_config}' CONNECT IDENTIFIER IS '{target_tns}';
ENABLE CONFIGURATION ALL;
""",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="dgpdb.add_pdb",
                    title="创建并启用 standby PDB",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="Broker 从源 PDB 实例化目标 PDB，并持续传输和应用该 PDB 的 redo。",
                    commands=(
                        _command(
                            "dgpdb.add_pdb.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "加入 PDB 级 Data Guard",
                            f"""
CONNECT sys@{source_tns};
ADD PLUGGABLE DATABASE '{target_pdb}' AT '{target_cdb}'
  SOURCE IS '{source_pdb}' AT '{source_cdb}';
SHOW PLUGGABLE DATABASE '{target_pdb}' AT '{target_cdb}';
""",
                            "文件系统使用与源端相同的数据文件路径；目标主机必须已经创建对应目录。",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="dgpdb.active_data_guard",
                    title="按许可打开目标 PDB 只读实时应用",
                    applicability=RunbookApplicability.CONDITIONAL,
                    rationale="仅在已购买 Active Data Guard 许可且需要查询时启用。",
                    commands=(
                        _command(
                            "dgpdb.active_data_guard.sql",
                            RunbookCommandType.SQLPLUS,
                            "在目标 CDB 打开目标 PDB",
                            f"ALTER PLUGGABLE DATABASE {target_pdb} OPEN READ ONLY;",
                        ),
                    ),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="validation",
            title="验收与日常运维",
            objective="验证 PDB 级传输、应用、Broker 健康和切换准备状态。",
            steps=(
                RunbookStep(
                    step_id="validation.dgpdb",
                    title="验证 DGPDB 状态",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="必须同时验证源 PDB、目标 PDB、两个 CDB 和 Broker 配置组。",
                    commands=(
                        _command(
                            "validation.dgpdb.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "Broker 验收",
                            f"""
SHOW ALL;
SHOW CONFIGURATION VERBOSE '{source_config}';
SHOW CONFIGURATION VERBOSE '{target_config}';
SHOW PLUGGABLE DATABASE '{source_pdb}' AT '{source_cdb}';
SHOW PLUGGABLE DATABASE '{target_pdb}' AT '{target_cdb}';
VALIDATE NETWORK CONFIGURATION FOR ALL;
""",
                        ),
                        _command(
                            "validation.dgpdb.source",
                            RunbookCommandType.SQLPLUS,
                            "源端验证",
                            f"""
ALTER SESSION SET CONTAINER={source_pdb};
SELECT name, open_mode, restricted FROM v$pdbs WHERE name=UPPER('{source_pdb}');
SELECT CURRENT_TIMESTAMP FROM dual;
""",
                        ),
                        _command(
                            "validation.dgpdb.target",
                            RunbookCommandType.SQLPLUS,
                            "目标端验证",
                            f"""
ALTER SESSION SET CONTAINER=CDB$ROOT;
SELECT name, open_mode, restricted FROM v$pdbs WHERE name=UPPER('{target_pdb}');
SELECT name, value, unit FROM v$dataguard_stats
WHERE name IN ('transport lag','apply lag','apply finish time');
SELECT timestamp, severity, message FROM v$dataguard_status
WHERE timestamp > SYSDATE - 1 ORDER BY timestamp DESC FETCH FIRST 50 ROWS ONLY;
""",
                        ),
                    ),
                    risks=("Broker 存在 WARNING、目标 PDB 未追平或网络校验失败时不得进行 PDB switchover。",),
                ),
                RunbookStep(
                    step_id="validation.operations",
                    title="日常检查与切换前校验",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="交付可重复执行的 DGPDB 运维命令，但不自动切换。",
                    commands=(
                        _command(
                            "validation.operations.dgmgrl",
                            RunbookCommandType.DGMGRL,
                            "日常 Broker 检查",
                            f"""
SHOW ALL;
SHOW CONFIGURATION VERBOSE '{source_config}';
SHOW CONFIGURATION VERBOSE '{target_config}';
SHOW PLUGGABLE DATABASE '{target_pdb}' AT '{target_cdb}';
VALIDATE PLUGGABLE DATABASE '{target_pdb}' AT '{target_cdb}';
""",
                        ),
                    ),
                    risks=("正式 switchover/failover 必须单独生成演练 Runbook 并审批。",),
                ),
            ),
        ),
    )
    return ImplementationRunbook(
        profile=ImplementationProfile.ORACLE_ADG_BUILD,
        title=f"Oracle 26ai PDB 级 Data Guard（DGPDB）实施操作文档：{source_pdb}",
        status=status,
        execution_policy=(
            "本产物按全新目标环境从零建设，所有名称和路径均从主库事实自动派生；"
            "不要求用户补充备库主机、Oracle Home 或存储参数。"
            "文档仅提供操作步骤，不自动执行命令。"
        ),
        current_state=current_state,
        resolved_parameters=resolved_parameters,
        required_inputs=(),
        phases=phases,
        stop_conditions=(
            "目标主机无法解析派生主机名，或主备监听端口不通。",
            "目标 Oracle 软件版本、RU、字符集或 COMPATIBLE 与源 CDB 不兼容。",
            "目标目录或 ASM 磁盘组容量不足，无法容纳源 PDB 数据和 FRA。",
            "密码文件未安全复制，或两个 CDB 的 Broker 远程认证失败。",
            "Broker 配置组、PDB 实例化或 redo 应用出现持续错误。",
            "源 PDB 使用 TDE 但目标 CDB 尚未安全导入所需密钥。",
        ),
        evidence_refs=evidence_refs,
    )


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
    standby_host = parameters.get("STANDBY_HOST", "")
    oracle_home = parameters.get("ORACLE_HOME", "")
    oracle_base = parameters.get("ORACLE_BASE", "")
    fra_dest = parameters.get("PRIMARY_FRA_DEST", "")
    fra_size = parameters.get("PRIMARY_FRA_SIZE", "")
    standby_storage = parameters.get("STANDBY_STORAGE", "")
    standby_redo_dest = parameters.get("STANDBY_REDO_DEST", "")
    standby_data_dest = parameters.get("STANDBY_DATA_DEST", "")
    standby_audit_dest = parameters.get("STANDBY_AUDIT_DEST", "")
    source_password_file = parameters.get("PRIMARY_PASSWORD_FILE", "")
    standby_password_file = parameters.get("STANDBY_PASSWORD_FILE", "")
    observed_source_password_file = _text(precheck, "password_file_path")
    version = _text(identity, "version") or _text(precheck, "compatible") or "26ai"
    release_label = _oracle_release_label(version)
    software_archive = f"/stage/oracle/LINUX.X64_{release_label}_db_home.zip"
    target_directories = tuple(
        dict.fromkeys(
            value
            for value in (
                oracle_base,
                oracle_home,
                standby_data_dest,
                standby_redo_dest,
                fra_dest,
                standby_audit_dest,
            )
            if value and value.startswith("/")
        )
    )
    target_directory_list = " ".join(target_directories)
    primary_redo_dest = _directory_of(
        _text(precheck, "sample_redo_member_path")
    ) or _text(precheck, "db_create_file_dest")
    required_inputs: tuple[RunbookRequiredInput, ...] = ()
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
            RunbookStatus.READY
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
                        *(
                            (
                                _command(
                                    "primary.passwordfile_mode.create_file",
                                    RunbookCommandType.SHELL,
                                    "在主库创建独占密码文件",
                                    f"""
read -rsp '输入当前 SYS 密码: ' KBOT_SYS_PASSWORD; echo
{oracle_home}/bin/orapwd file={source_password_file} password="$KBOT_SYS_PASSWORD" format=12.2 force=y
unset KBOT_SYS_PASSWORD
""",
                                    "仅在 V$PASSWORDFILE_INFO 未返回现有密码文件时执行。",
                                ),
                            )
                            if not observed_source_password_file
                            else ()
                        ),
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
                        else RunbookApplicability.REQUIRED
                    ),
                    rationale="每个 redo thread 的 SRL 数量至少应为 online redo group 数量加一，大小不小于对应联机日志。",
                    commands=() if precheck is not None and srl_shortage <= 0 else (
                        _srl_commands(
                            precheck,
                            redo_destination=primary_redo_dest,
                        )
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
                    required_inputs=(),
                ),
            ),
        ),
        RunbookPhase(
            phase_id="target_environment",
            title="从零建设备库主机与 Oracle 软件",
            objective="按主库版本和文件布局准备全新备库，不等待用户补充主机、Oracle Home 或存储路径。",
            steps=(
                RunbookStep(
                    step_id="target_environment.os",
                    title="准备操作系统用户、内核前置和目录",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="备库按全新环境建设，主机名、用户组和文件系统目录由主库事实派生。",
                    commands=(
                        _command(
                            "target_environment.os.root",
                            RunbookCommandType.SHELL,
                            "在备库以 root 执行",
                            f"""
hostnamectl set-hostname {standby_host.split('.', 1)[0]}
dnf install -y oracle-database-preinstall-{release_label} unzip
getent group oinstall >/dev/null || groupadd -g 54321 oinstall
getent group dba >/dev/null || groupadd -g 54322 dba
id oracle >/dev/null 2>&1 || useradd -u 54321 -g oinstall -G dba oracle
mkdir -p {target_directory_list}
chown -R oracle:oinstall {target_directory_list}
chmod -R 775 {target_directory_list}
""",
                            "如果主库使用 ASM，基础设施团队应在此阶段以相同磁盘组名称完成 GI/ASM 建设；数据库 Runbook 继续使用已派生的磁盘组名。",
                        ),
                    ),
                ),
                RunbookStep(
                    step_id="target_environment.software",
                    title="安装与主库一致的数据库软件和 RU",
                    applicability=RunbookApplicability.REQUIRED,
                    rationale="物理备库必须与主库使用兼容的 Oracle Home 版本和补丁级别。",
                    commands=(
                        _command(
                            "target_environment.software.install",
                            RunbookCommandType.SHELL,
                            "在备库以 oracle 执行",
                            f"""
export ORACLE_BASE={oracle_base}
export ORACLE_HOME={oracle_home}
export PATH=$ORACLE_HOME/bin:$PATH
mkdir -p "$ORACLE_HOME"
unzip -q {software_archive} -d "$ORACLE_HOME"
$ORACLE_HOME/runInstaller -silent -waitforcompletion \\
  oracle.install.option=INSTALL_DB_SWONLY \\
  UNIX_GROUP_NAME=oinstall \\
  INVENTORY_LOCATION={oracle_base}/oraInventory \\
  ORACLE_HOME="$ORACLE_HOME" \\
  ORACLE_BASE="$ORACLE_BASE" \\
  oracle.install.db.InstallEdition=EE \\
  oracle.install.db.OSDBA_GROUP=dba \\
  DECLINE_SECURITY_UPDATES=true
""",
                            "安装完成后由 root 执行安装器提示的 root.sh，并应用与主库完全一致的 RU。",
                        ),
                    ),
                    verification_commands=(
                        _command(
                            "target_environment.software.verify",
                            RunbookCommandType.SHELL,
                            "核对备库软件版本和补丁",
                            f"""
{oracle_home}/bin/sqlplus -V
{oracle_home}/OPatch/opatch lspatches
""",
                        ),
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
                        if source_password_file and standby_password_file
                        else RunbookApplicability.BLOCKED
                    ),
                    rationale="主备 SYS 密码文件必须一致，且 remote_login_passwordfile 应为 EXCLUSIVE。",
                    commands=(
                        _command(
                            "connectivity.password_file.copy",
                            RunbookCommandType.SHELL,
                            "在备库复制主库密码文件并按备库 SID 命名",
                            f"""
scp oracle@{primary_host}:{source_password_file} {standby_password_file}
chmod 600 {standby_password_file}
""",
                            "通过受控 SSH 通道复制密码文件内容，不在 Runbook 中记录 SYS 密码。",
                        ),
                    ) if source_password_file and standby_password_file else (),
                    verification_commands=(
                        _command(
                            "connectivity.password_file.verify",
                            RunbookCommandType.SQLPLUS,
                            "验证远程密码文件模式",
                            "SHOW PARAMETER remote_login_passwordfile;",
                        ),
                    ),
                    required_inputs=tuple(
                        key
                        for key, value in (
                            ("PRIMARY_PASSWORD_FILE", source_password_file),
                            ("STANDBY_PASSWORD_FILE", standby_password_file),
                        )
                        if not value
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
            "本产物按全新备库环境从零建设，主机名、Oracle Home、文件路径和密码文件目标名"
            "均从主库事实自动派生；不要求用户先补充备库参数。"
            "本产物仅生成实施步骤，不执行任何 SQL、RMAN、DGMGRL 或系统命令。"
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
            "派生的 ASM/文件系统路径尚未创建，或容量不足。",
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
        precheck, _ = _first_row(evidence, "db.ha.adg_precheck")
        if _integer(precheck, "container_id") > 1:
            return _compile_oracle_dgpdb_build(evidence, dict(context or {}))
        return _compile_oracle_adg_build(evidence, dict(context or {}))
    raise ValueError(f"不支持的实施方案档案：{profile}")
