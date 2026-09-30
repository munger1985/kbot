"""数据库实施文档的可选用户参数目录与确定性校验。"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from platform_core.contracts.aiops import ImplementationProfile


def _field(
    name: str,
    label: str,
    *,
    field_type: str = "text",
    group: str = "常用参数",
    help_text: str,
    placeholder: str = "留空则自动获取或按策略派生",
    options: tuple[tuple[str, str], ...] = (),
    minimum: int | None = None,
    maximum: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "label": label,
        "type": field_type,
        "group": group,
        "required": False,
        "help": help_text,
        "placeholder": placeholder,
    }
    if options:
        payload["options"] = [
            {"value": value, "label": option_label}
            for value, option_label in options
        ]
    if minimum is not None:
        payload["min"] = minimum
    if maximum is not None:
        payload["max"] = maximum
    return payload


_INPUT_SCHEMAS: dict[ImplementationProfile, tuple[dict[str, Any], ...]] = {
    ImplementationProfile.ORACLE_ADG_BUILD: (
        _field("STANDBY_DB_UNIQUE_NAME", "备库 DB_UNIQUE_NAME", field_type="oracle_identifier", help_text="覆盖按主库唯一名自动派生的备库名称。"),
        _field("STANDBY_HOST", "备库主机名", field_type="hostname", help_text="覆盖按主库主机名自动派生的备库主机名。"),
        _field("STANDBY_ORACLE_SID", "备库 Oracle SID", field_type="oracle_identifier", help_text="覆盖按备库唯一名自动派生的实例 SID。"),
        _field("PRIMARY_TNS_ALIAS", "主库 TNS Alias", field_type="oracle_identifier", help_text="用于主备双向 Oracle Net 配置。", group="高级参数"),
        _field("STANDBY_TNS_ALIAS", "备库 TNS Alias", field_type="oracle_identifier", help_text="用于日志传输和 Broker 配置。", group="高级参数"),
        _field("ORACLE_HOME", "备库 Oracle Home", field_type="path", help_text="留空时按主库路径或数据库发行版派生。", group="高级参数"),
        _field("STANDBY_STORAGE", "备库存储位置", field_type="path", help_text="可填写 ASM 磁盘组或绝对文件系统路径。", group="高级参数"),
        _field("PRIMARY_FRA_DEST", "主库 FRA 路径", field_type="path", help_text="仅在需要覆盖当前 FRA 或补建 FRA 时使用。", group="高级参数"),
        _field("PRIMARY_FRA_SIZE", "主库 FRA 容量", help_text="例如 500G；留空时沿用现状或策略派生。", group="高级参数"),
    ),
    ImplementationProfile.ORACLE_RAC_BUILD: (
        _field("NODE1_HOST", "RAC Node 1 主机名", field_type="hostname", help_text="用于资源注册和节点范围命令。"),
        _field("NODE2_HOST", "RAC Node 2 主机名", field_type="hostname", help_text="必须与 Node 1 不同。"),
        _field("SCAN_NAME", "SCAN 名称", field_type="hostname", help_text="使用已在 DNS 或 hosts 中解析的 SCAN 名称。"),
        _field("VERSION", "Oracle 目标发行版", help_text="例如 19c 或 26ai；留空时从 Target 和数据库版本识别。"),
        _field("GRID_HOME", "Grid Home", field_type="path", help_text="留空时按数据库发行版使用标准目录。", group="高级参数"),
        _field("ORACLE_HOME", "数据库 Oracle Home", field_type="path", help_text="留空时优先采用主机和数据库事实。", group="高级参数"),
        _field("PATCH_STAGE_PATH", "安装介质目录", field_type="path", help_text="GI、数据库软件和补丁介质所在目录。", group="高级参数"),
    ),
    ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD: (
        _field("BACKUP_DEST", "备份目录", field_type="path", help_text="留空时读取 RMAN Channel、FRA 或标准备份路径。"),
        _field("PROMETHEUS_TEXTFILE_DIR", "Node Exporter Textfile 目录", field_type="path", help_text="RMAN 状态指标输出目录。", group="高级参数"),
        _field("PROMETHEUS_RULE_DIR", "Prometheus 规则目录", field_type="path", help_text="RMAN 告警规则部署目录。", group="高级参数"),
    ),
    ImplementationProfile.ORACLE_RMAN_RECOVERY: (
        _field("RECOVERY_SCENARIO", "恢复场景", field_type="select", help_text="选择本次文档的主要恢复场景。", options=(("FULL_DATABASE", "全库恢复"), ("DATABASE_PITR", "数据库时间点恢复"), ("DATAFILE", "数据文件恢复"), ("CONTROLFILE", "控制文件恢复"), ("PDB", "PDB 恢复"))),
        _field("RECOVERY_TARGET_TIME", "恢复目标时间", field_type="oracle_datetime", help_text="使用日期时间选择器填写，仅用于时间点恢复；提交后统一保存为 YYYY-MM-DD HH24:MI:SS。"),
        _field("RECOVERY_TARGET_SCN", "恢复目标 SCN", field_type="integer", help_text="与目标时间二选一。", minimum=1),
        _field("ORACLE_HOME", "恢复环境 Oracle Home", field_type="path", help_text="留空时从数据库和主机事实获取。", group="高级参数"),
    ),
    ImplementationProfile.ORACLE_RU_PATCH: (
        _field("APPROVED_RU_ID", "已批准 RU 编号", field_type="digits", help_text="留空时由暂存目录中的唯一 RU 结合 Inventory 识别。"),
        _field("PATCH_STAGE_PATH", "补丁暂存目录", field_type="path", help_text="留空时按数据库名称和发行版派生。"),
    ),
    ImplementationProfile.ORACLE_DATABASE_UPGRADE: (
        _field("TARGET_VERSION", "目标数据库版本", help_text="例如 19c、23ai 或 26ai。"),
        _field("TARGET_ORACLE_HOME", "目标 Oracle Home", field_type="path", help_text="填写已经安装并完成补丁的目标 Home。"),
        _field("ORACLE_HOME", "源 Oracle Home", field_type="path", help_text="留空时从主机和数据库事实获取。", group="高级参数"),
    ),
    ImplementationProfile.ORACLE_DATABASE_MIGRATION: (
        _field("MIGRATION_METHOD", "迁移方法", field_type="select", help_text="留空时生成 RMAN Backup Location Duplicate 方案。", options=(("RMAN_BACKUP_LOCATION_DUPLICATE", "RMAN Backup Location Duplicate"), ("RMAN_BACKUP_RESTORE", "RMAN Backup/Restore"))),
        _field("MIGRATION_STAGE_PATH", "迁移暂存目录", field_type="path", help_text="备份、摘要和迁移脚本的统一暂存位置。"),
        _field("TARGET_INSTANCE_NAME", "目标实例名", field_type="oracle_identifier", help_text="留空时沿用源数据库名称。"),
    ),
    ImplementationProfile.ORACLE_CLONE_REFRESH: (
        _field("DESTINATION_REF", "目标环境标识", help_text="用于在文档中标识目标测试或开发环境。"),
        _field("CLONE_METHOD", "克隆方式", field_type="select", help_text="选择与目标存储和版本匹配的克隆方法。", options=(("RMAN_DUPLICATE", "RMAN Active Duplicate"), ("PDB_CLONE", "PDB Clone"), ("SNAPSHOT_COPY", "Snapshot Copy"))),
        _field("TARGET_DB_NAME", "目标数据库名称", field_type="oracle_identifier", help_text="非生产目标数据库名称。"),
        _field("SOURCE_CONNECT_IDENTIFIER", "源库连接标识", help_text="只填写 TNS 服务名，不填写用户名或密码。", group="高级参数"),
        _field("AUXILIARY_CONNECT_IDENTIFIER", "辅助实例连接标识", help_text="只填写 TNS 服务名，不填写用户名或密码。", group="高级参数"),
    ),
    ImplementationProfile.ORACLE_DATAPUMP_MIGRATION: (
        _field("DATAPUMP_DIRECTORY_NAME", "Directory 对象名称", field_type="oracle_identifier", help_text="留空时使用 KBOT_DATAPUMP_DIR。"),
        _field("DATAPUMP_DIRECTORY_PATH", "Directory 操作系统路径", field_type="path", help_text="留空时读取现有目录或按数据库名称派生。"),
        _field("SOURCE_SCHEMAS", "源 Schema", field_type="identifier_list", help_text="多个 Schema 使用逗号或换行分隔；留空时自动发现业务 Schema，无法发现时生成 Full 方案。", placeholder="例如 APP_CORE, APP_REPORT"),
        _field("DATAPUMP_PARALLEL", "Data Pump 并行度", field_type="integer", help_text="同时写入 expdp 和 impdp 参数文件。", minimum=1, maximum=64),
        _field("DATAPUMP_DUMP_PREFIX", "转储文件前缀", field_type="file_prefix", help_text="留空时使用 kbot_export。", group="高级参数"),
    ),
    ImplementationProfile.ORACLE_ADG_DRILL: (
        _field("DRILL_SCENARIO", "演练场景", field_type="select", help_text="选择 Switchover、计划 Failover 或 Reinstate。", options=(("SWITCHOVER", "Switchover"), ("PLANNED_FAILOVER", "计划 Failover"), ("REINSTATE", "Reinstate"))),
        _field("DG_CONFIG_NAME", "Broker 配置名", field_type="oracle_identifier", help_text="留空时从 Broker 和数据库事实获取。"),
        _field("PRIMARY_DB_UNIQUE_NAME", "主库 DB_UNIQUE_NAME", field_type="oracle_identifier", help_text="留空时读取当前 Broker 配置。", group="高级参数"),
        _field("STANDBY_DB_UNIQUE_NAME", "备库 DB_UNIQUE_NAME", field_type="oracle_identifier", help_text="留空时读取当前 Broker 配置。", group="高级参数"),
    ),
}


def implementation_input_schema(
    profile: ImplementationProfile,
) -> tuple[dict[str, Any], ...]:
    """返回前端可展示的非秘密可选参数定义。"""
    return tuple(dict(item) for item in _INPUT_SCHEMAS.get(profile, ()))


def normalize_implementation_parameters(
    profile: ImplementationProfile,
    supplied: dict[str, Any],
) -> dict[str, Any]:
    """按档案白名单校验用户参数，禁止任意值进入命令编译器。"""
    schema = {str(item["name"]): item for item in implementation_input_schema(profile)}
    unknown = sorted(set(supplied) - set(schema))
    if unknown:
        raise ValueError("实施文档参数包含未知字段：" + ", ".join(unknown))
    normalized: dict[str, Any] = {}
    for name, raw_value in supplied.items():
        field = schema[name]
        if raw_value is None or str(raw_value).strip() == "":
            continue
        kind = str(field.get("type") or "text")
        if kind == "integer":
            try:
                value = int(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field['label']}必须是整数") from exc
            minimum = field.get("min")
            maximum = field.get("max")
            if minimum is not None and value < int(minimum):
                raise ValueError(f"{field['label']}小于允许范围")
            if maximum is not None and value > int(maximum):
                raise ValueError(f"{field['label']}超出允许范围")
            normalized[name] = value
            continue
        value = str(raw_value).strip()
        if kind == "select":
            allowed = {str(item["value"]) for item in field.get("options", ())}
            if value not in allowed:
                raise ValueError(f"{field['label']}不在允许选项中")
        elif kind == "path":
            if (
                not re.fullmatch(
                    r"(?:/[A-Za-z0-9_+.,%/@#=() -]+|\+[A-Za-z0-9_$#/-]+)",
                    value,
                )
                or ".." in value.split("/")
            ):
                raise ValueError(f"{field['label']}必须是安全的绝对路径或 ASM 路径")
        elif kind == "oracle_identifier":
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]{0,127}", value):
                raise ValueError(f"{field['label']}不是有效的 Oracle 标识")
            value = value.upper()
        elif kind == "identifier_list":
            parts = [item.strip().upper() for item in re.split(r"[,\n]+", value) if item.strip()]
            if not parts or any(
                not re.fullmatch(r"[A-Za-z][A-Za-z0-9_$#]{0,127}", item)
                for item in parts
            ):
                raise ValueError(f"{field['label']}包含无效的 Oracle 标识")
            value = ",".join(dict.fromkeys(parts))
        elif kind == "hostname":
            if len(value) > 253 or not re.fullmatch(
                r"[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?", value
            ):
                raise ValueError(f"{field['label']}不是有效主机名")
        elif kind == "digits":
            if not value.isdigit():
                raise ValueError(f"{field['label']}只能包含数字")
        elif kind == "oracle_datetime":
            candidate = value.replace("T", " ", 1)
            try:
                parsed = datetime.strptime(
                    candidate,
                    "%Y-%m-%d %H:%M:%S",
                )
            except ValueError as exc:
                raise ValueError(f"{field['label']}格式应为 YYYY-MM-DD HH24:MI:SS")
            value = parsed.strftime("%Y-%m-%d %H:%M:%S")
        elif kind == "file_prefix":
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,63}", value):
                raise ValueError(f"{field['label']}只能包含安全的文件名前缀字符")
        elif len(value) > 2048 or any(
            marker in value
            for marker in ("\n", "\r", "\x00", "`", "$", ";", "|", "&", "<", ">")
        ):
            raise ValueError(f"{field['label']}包含不允许的字符")
        normalized[name] = value
    if profile == ImplementationProfile.ORACLE_RAC_BUILD:
        if normalized.get("NODE1_HOST") == normalized.get("NODE2_HOST") and normalized.get("NODE1_HOST"):
            raise ValueError("RAC 两个节点主机名不能相同")
    if profile == ImplementationProfile.ORACLE_RMAN_RECOVERY:
        if normalized.get("RECOVERY_TARGET_TIME") and normalized.get("RECOVERY_TARGET_SCN"):
            raise ValueError("恢复目标时间和 SCN 只能填写一项")
    return normalized
