"""Oracle 整库迁移与切换实施档案。"""

from __future__ import annotations

import re
import shlex
from typing import Any

from aiops_agent.application.implementation.artifacts import GeneratedRunbookArtifact
from aiops_agent.application.implementation.profiles.common import (
    ProfileSpec,
    command,
    compile_profile,
    first_row,
    value,
)
from aiops_agent.contracts.implementation import (
    RunbookExecutor,
    RunbookRiskLevel,
)
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import ImplementationProfile


_SPEC = ProfileSpec(
    profile=ImplementationProfile.ORACLE_DATABASE_MIGRATION,
    title="Oracle 数据库目标环境迁移与切换实施操作文档",
    policy_id="oracle.migration.rman-backup-location.v1",
    fact_tool_id="db.migration.precheck",
    required_facts=(),
    phases=(
        ("assessment", "源端事实与目标差异", ("assessment",)),
        ("method_phase", "迁移方法确定与证据", ("method",)),
        ("target_phase", "目标软件、存储与网络准备", ("target",)),
        ("protection", "保护备份与回退基线", ("protection",)),
        ("sync_phase", "全量备份预演与介质准备", ("sync",)),
        ("cutover_phase", "停写、离线备份、目标复制与切换", ("cutover",)),
        ("validate", "数据、对象、服务与性能验证", ("validate",)),
        ("handover", "观察期、回退与交接", ("handover",)),
    ),
    stop_conditions=(
        "实时源库身份、角色、版本或文件布局与文档固化事实不一致。",
        "目标主机 Oracle 软件版本、补丁、字节序或字符集不兼容。",
        "目标存储未复现源库文件路径或 ASM Disk Group，且未生成批准的名称转换方案。",
        "目标主机未完成网络隔离，可能与源库同时注册相同数据库服务。",
        "切换窗口、停写责任人、业务验证责任人或回退决策人尚未获批。",
        "备份校验、传输摘要、RMAN Duplicate 或业务验证出现未解决错误。",
    ),
)


def _artifact(
    artifact_id: str,
    path: str,
    content: str,
    *,
    media_type: str = "text/plain",
    mode: str = "0640",
) -> GeneratedRunbookArtifact:
    stage = "/var/tmp/kbot-runbooks/oracle-database-migration"
    return GeneratedRunbookArtifact(
        artifact_id=artifact_id,
        relative_path=f"oracle-database-migration/{path}",
        content=content,
        media_type=media_type,
        file_mode=mode,
        run_as="oracle",
        target_path=f"{stage}/{path}",
        description=f"Oracle 数据库迁移文件：{path}",
    )


def _safe_stage_path(candidate: str) -> str:
    """只接受适合 Shell、SQL 和 RMAN 的绝对迁移暂存路径。"""
    normalized = candidate.strip().rstrip("/")
    if not normalized.startswith("/"):
        return ""
    if not re.fullmatch(r"/[A-Za-z0-9_+.,%/@#=()/-]+", normalized):
        return ""
    if ".." in normalized.split("/"):
        return ""
    return normalized


def _database_identifier(raw: str) -> str:
    """把数据库事实收敛为可放入 RMAN 和文件名的标识。"""
    candidate = re.sub(r"[^A-Za-z0-9_$#]", "_", raw.strip()).upper()
    return candidate[:30] or "ORACLE"


def compile_migration(
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any],
):
    identity, _ = first_row(evidence, "db.instance.identity")
    facts, _ = first_row(evidence, "db.migration.precheck")
    merged = {**identity, **facts}

    database_name = _database_identifier(
        value(context, merged, "DATABASE_NAME")
        or value(context, merged, "DB_UNIQUE_NAME")
        or value(context, merged, "INSTANCE_NAME")
    )
    database_key = database_name.lower()
    requested_stage = _safe_stage_path(
        value(context, merged, "MIGRATION_STAGE_PATH")
    )
    migration_stage = (
        requested_stage
        or f"/u01/app/oracle/migration/{database_key}"
    )
    quoted_stage = shlex.quote(migration_stage)
    target_sid = _database_identifier(
        value(context, merged, "TARGET_INSTANCE_NAME") or database_name
    )
    requested_method = value(context, merged, "MIGRATION_METHOD").upper()
    supported_requests = {
        "",
        "RMAN_BACKUP_RESTORE",
        "RMAN_BACKUP_LOCATION_DUPLICATE",
    }
    method_review_required = requested_method not in supported_requests
    method = "RMAN_BACKUP_LOCATION_DUPLICATE"
    source_version = str(merged.get("version") or "SOURCE_RELEASE").strip()
    source_platform = str(
        merged.get("platform_name") or "SOURCE_PLATFORM"
    ).strip()
    source_charset = str(
        merged.get("character_set") or "SOURCE_CHARACTER_SET"
    ).strip()
    source_ncharset = str(
        merged.get("national_character_set") or "SOURCE_NCHAR_CHARACTER_SET"
    ).strip()
    log_mode = str(merged.get("log_mode") or "UNKNOWN").upper()
    artifact_stage = "/var/tmp/kbot-runbooks/oracle-database-migration"

    assessment_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        "SET PAGESIZE 500 LINESIZE 240 TRIMSPOOL ON\n"
        "SELECT dbid, name, db_unique_name, database_role, open_mode, "
        "log_mode, force_logging, flashback_on, platform_name, cdb "
        "FROM v$database;\n"
        "SELECT instance_name, host_name, version, version_full, status, "
        "startup_time FROM v$instance;\n"
        "SELECT name, display_value FROM v$parameter WHERE name IN "
        "('compatible','cluster_database','control_files','db_create_file_dest',"
        "'db_recovery_file_dest','service_names','spfile') ORDER BY name;\n"
        "SELECT parameter, value FROM nls_database_parameters WHERE parameter IN "
        "('NLS_CHARACTERSET','NLS_NCHAR_CHARACTERSET') ORDER BY parameter;\n"
        "SELECT file#, name, bytes FROM v$datafile ORDER BY file#;\n"
        "SELECT file#, name, bytes FROM v$tempfile ORDER BY file#;\n"
        "SELECT group#, thread#, bytes, members, status FROM v$log ORDER BY thread#, group#;\n"
        "SELECT group#, member FROM v$logfile ORDER BY group#, member;\n"
        "SELECT comp_id, version, status FROM dba_registry ORDER BY comp_id;\n"
        "SELECT owner, object_type, COUNT(*) invalid_count FROM dba_objects "
        "WHERE status='INVALID' GROUP BY owner, object_type "
        "ORDER BY owner, object_type;\n"
        "EXIT\n"
    )
    create_pfile_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        f"CREATE PFILE='{migration_stage}/init{target_sid}.ora' FROM MEMORY;\n"
        "EXIT\n"
    )
    archive_backup = ""
    if log_mode != "NOARCHIVELOG":
        archive_backup = (
            "  BACKUP AS COMPRESSED BACKUPSET ARCHIVELOG ALL "
            f"FORMAT '{migration_stage}/arch_%U.bkp';\n"
        )
    source_backup_rman = (
        "RUN {\n"
        "  ALLOCATE CHANNEL c1 DEVICE TYPE DISK;\n"
        "  ALLOCATE CHANNEL c2 DEVICE TYPE DISK;\n"
        "  BACKUP AS COMPRESSED BACKUPSET DATABASE "
        f"FORMAT '{migration_stage}/database_%U.bkp';\n"
        + archive_backup
        + "  BACKUP CURRENT CONTROLFILE "
        f"FORMAT '{migration_stage}/controlfile.bkp';\n"
        "  RELEASE CHANNEL c1;\n"
        "  RELEASE CHANNEL c2;\n"
        "}\n"
        "LIST BACKUP SUMMARY;\n"
    )
    start_auxiliary_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        f"STARTUP FORCE NOMOUNT PFILE='{migration_stage}/init{target_sid}.ora';\n"
        "SELECT instance_name, status FROM v$instance;\n"
        "EXIT\n"
    )
    duplicate_rman = (
        f"DUPLICATE DATABASE TO {database_name}\n"
        f"  BACKUP LOCATION '{migration_stage}'\n"
        "  NOFILENAMECHECK;\n"
    )
    validation_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        "SET PAGESIZE 500 LINESIZE 240 TRIMSPOOL ON\n"
        "SELECT dbid, name, db_unique_name, database_role, open_mode, "
        "log_mode, platform_name, resetlogs_change#, resetlogs_time FROM v$database;\n"
        "SELECT instance_name, host_name, version, status FROM v$instance;\n"
        "SELECT con_id, name, open_mode, restricted FROM v$pdbs ORDER BY con_id;\n"
        "SELECT COUNT(*) datafile_count, ROUND(SUM(bytes)/POWER(1024,3),2) "
        "datafile_gb FROM v$datafile;\n"
        "SELECT comp_id, version, status FROM dba_registry ORDER BY comp_id;\n"
        "SELECT owner, object_type, COUNT(*) invalid_count FROM dba_objects "
        "WHERE status='INVALID' GROUP BY owner, object_type "
        "ORDER BY owner, object_type;\n"
        "SELECT name, network_name FROM v$services ORDER BY name;\n"
        "EXIT\n"
    )
    if log_mode == "NOARCHIVELOG":
        post_cutover_prefix = "SHUTDOWN IMMEDIATE;\nSTARTUP MOUNT;\n"
        post_cutover_database_backup = (
            "  BACKUP AS COMPRESSED BACKUPSET DATABASE "
            f"FORMAT '{migration_stage}/post_cutover_%U.bkp' "
            "TAG 'KBOT_POST_MIGRATION';\n"
        )
        post_cutover_suffix = "SQL 'ALTER DATABASE OPEN';\n"
        post_cutover_risk = RunbookRiskLevel.HIGH
    else:
        post_cutover_prefix = ""
        post_cutover_database_backup = (
            "  SQL 'ALTER SYSTEM ARCHIVE LOG CURRENT';\n"
            "  BACKUP AS COMPRESSED BACKUPSET DATABASE PLUS ARCHIVELOG "
            f"FORMAT '{migration_stage}/post_cutover_%U.bkp' "
            "TAG 'KBOT_POST_MIGRATION';\n"
        )
        post_cutover_suffix = ""
        post_cutover_risk = RunbookRiskLevel.MEDIUM
    post_cutover_backup_rman = (
        post_cutover_prefix
        + "RUN {\n"
        + post_cutover_database_backup
        + "  BACKUP CURRENT CONTROLFILE "
        f"FORMAT '{migration_stage}/post_cutover_controlfile.bkp';\n"
        "}\n"
        + post_cutover_suffix
    )

    artifacts = (
        _artifact(
            "migration.assessment",
            "sql/assess-source.sql",
            assessment_sql,
            media_type="text/x-sql",
        ),
        _artifact(
            "migration.create_pfile",
            "sql/create-source-pfile.sql",
            create_pfile_sql,
            media_type="text/x-sql",
        ),
        _artifact(
            "migration.source_backup",
            "rman/backup-source.rman",
            source_backup_rman,
            media_type="text/x-rman",
        ),
        _artifact(
            "migration.start_auxiliary",
            "sql/start-target-nomount.sql",
            start_auxiliary_sql,
            media_type="text/x-sql",
        ),
        _artifact(
            "migration.duplicate_target",
            "rman/duplicate-target.rman",
            duplicate_rman,
            media_type="text/x-rman",
        ),
        _artifact(
            "migration.validation",
            "sql/validate-database.sql",
            validation_sql,
            media_type="text/x-sql",
        ),
        _artifact(
            "migration.post_cutover_backup",
            "rman/backup-post-cutover.rman",
            post_cutover_backup_rman,
            media_type="text/x-rman",
        ),
    )

    commands: dict[str, tuple] = {
        "assessment": (
            command(
                "migration.assessment.source",
                "采集源库身份、规模、文件和组件基线",
                f"sqlplus / as sysdba @{artifact_stage}/sql/assess-source.sql",
                executor=RunbookExecutor.SQLPLUS,
                run_as="oracle",
                node_scope=("source",),
                artifact_ref="migration.assessment",
            ),
        ),
        "method": (
            command(
                "migration.method.evidence",
                "核对默认迁移方法的源端适用证据",
                "SELECT d.name, d.platform_name, d.log_mode, d.cdb, "
                "i.version, p.value compatible, "
                "ROUND(SUM(df.bytes)/POWER(1024,3),2) datafile_gb "
                "FROM v$database d CROSS JOIN v$instance i "
                "CROSS JOIN (SELECT value FROM v$parameter "
                "WHERE name='compatible') p CROSS JOIN v$datafile df "
                "GROUP BY d.name, d.platform_name, d.log_mode, d.cdb, "
                "i.version, p.value;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
                node_scope=("source",),
                notes=(
                    "本档案固定采用离线 RMAN Backup Location Duplicate；"
                    "目标默认按源库同平台、同版本、同拓扑和同文件布局建设。",
                ),
            ),
        ),
        "target": (
            command(
                "migration.source.stage",
                "在源主机创建迁移暂存目录",
                f"install -d -m 0750 -o oracle -g oinstall {quoted_stage}\n"
                f"df -P {quoted_stage}\nfindmnt -T {quoted_stage}",
                executor=RunbookExecutor.BASH,
                run_as="root",
                node_scope=("source",),
                risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "migration.target.stage",
                "在目标主机创建迁移暂存目录",
                f"install -d -m 0750 -o oracle -g oinstall {quoted_stage}\n"
                f"df -P {quoted_stage}\nfindmnt -T {quoted_stage}\n"
                f"test -z \"$(find {quoted_stage} -maxdepth 1 -type f "
                "\\( -name '*.bkp' -o -name 'init*.ora' -o "
                "-name 'migration.sha256' \\) -print -quit)\"",
                executor=RunbookExecutor.BASH,
                run_as="root",
                node_scope=("target",),
                risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "migration.target.software",
                "核验目标 Oracle 软件和监听",
                "command -v sqlplus\ncommand -v rman\ncommand -v lsnrctl\n"
                "sqlplus -V\nrman -version\nlsnrctl status",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                node_scope=("target",),
                expected=(
                    f"目标 Oracle 版本与源端 {source_version} 一致或处于批准的兼容版本。",
                ),
            ),
            command(
                "migration.target.layout",
                "复现源库文件路径、ASM Disk Group 和密码文件",
                "根据源端 assess-source.sql 输出，在目标主机创建相同文件系统路径或 ASM Disk Group；"
                "将源库密码文件通过批准的安全通道复制到目标 Oracle Home，并按目标 SID 命名。"
                "目标数据库服务在正式切换前不得对应用网络发布。",
                executor=RunbookExecutor.MANUAL,
                run_as="DBA/系统管理员",
                node_scope=("source", "target"),
                risk=RunbookRiskLevel.HIGH,
            ),
        ),
        "protection": (
            command(
                "migration.protection.validate",
                "读取并校验现有恢复链",
                "LIST BACKUP SUMMARY;\nLIST BACKUP OF DATABASE;\n"
                "RESTORE DATABASE VALIDATE CHECK LOGICAL;",
                executor=RunbookExecutor.RMAN,
                run_as="oracle",
                node_scope=("source",),
                risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "migration.protection.rollback",
                "固化回退边界",
                "目标业务验证完成前保持源库文件、配置、监听和应用连接不变。"
                "若目标复制或验收失败，在源主机执行 STARTUP，恢复原应用连接并结束切换。",
                executor=RunbookExecutor.MANUAL,
                run_as="DBA/应用负责人",
                node_scope=("source", "application"),
            ),
        ),
        "sync": (
            command(
                "migration.sync.empty_stage",
                "确认源端迁移目录不存在历史介质",
                f"test -z \"$(find {quoted_stage} -maxdepth 1 -type f "
                "\\( -name '*.bkp' -o -name 'init*.ora' -o "
                "-name 'migration.sha256' \\) -print -quit)\"",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                node_scope=("source",),
                expected=("迁移目录中不存在可能被误用的历史备份或参数文件。",),
            ),
            command(
                "migration.sync.create_pfile",
                "从源库运行内存参数生成目标启动 PFILE",
                f"sqlplus / as sysdba @{artifact_stage}/sql/create-source-pfile.sql",
                executor=RunbookExecutor.SQLPLUS,
                run_as="oracle",
                node_scope=("source",),
                artifact_ref="migration.create_pfile",
            ),
            command(
                "migration.sync.validate_source",
                "执行全库逻辑读校验和备份预演",
                "BACKUP VALIDATE CHECK LOGICAL DATABASE;\nREPORT SCHEMA;",
                executor=RunbookExecutor.RMAN,
                run_as="oracle",
                node_scope=("source",),
                risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "migration.sync.capacity",
                "核对源端与目标端迁移目录容量",
                f"df -P {quoted_stage}\ntest -w {quoted_stage}",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                node_scope=("source", "target"),
            ),
        ),
        "cutover": (
            command(
                "migration.cutover.freeze",
                "在批准窗口停止业务写入并排空连接",
                "停止应用写入、批处理和数据库作业；确认活动事务完成并记录业务停写时间。"
                "该步骤完成前不得停库或开始最终备份。",
                executor=RunbookExecutor.MANUAL,
                run_as="应用负责人/DBA",
                node_scope=("application", "source"),
                risk=RunbookRiskLevel.CRITICAL,
            ),
            command(
                "migration.cutover.mount_source",
                "关闭源库并以 MOUNT 状态启动",
                "SHUTDOWN IMMEDIATE;\nSTARTUP MOUNT;\n"
                "SELECT name, open_mode, database_role FROM v$database;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
                node_scope=("source",),
                risk=RunbookRiskLevel.CRITICAL,
            ),
            command(
                "migration.cutover.backup_source",
                "生成停写后的最终一致性 RMAN 备份",
                f"rman target / cmdfile={artifact_stage}/rman/backup-source.rman",
                executor=RunbookExecutor.RMAN,
                run_as="oracle",
                node_scope=("source",),
                artifact_ref="migration.source_backup",
                risk=RunbookRiskLevel.CRITICAL,
            ),
            command(
                "migration.cutover.checksum_source",
                "生成并验证源端迁移文件摘要",
                f"cd {quoted_stage}\n"
                "sha256sum *.bkp init*.ora > migration.sha256\n"
                "sha256sum -c migration.sha256",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                node_scope=("source",),
            ),
            command(
                "migration.cutover.transfer",
                "通过批准通道传输最终迁移文件",
                "将 migration.sha256、全部 .bkp 文件和 init*.ora 从源端迁移目录传输到"
                "目标端同一路径。传输主机和凭据使用已批准的运维通道，本静态文档不伪造。",
                executor=RunbookExecutor.MANUAL,
                run_as="DBA/系统管理员",
                node_scope=("source", "target"),
                risk=RunbookRiskLevel.CRITICAL,
            ),
            command(
                "migration.cutover.checksum_target",
                "在目标端校验迁移文件摘要",
                f"cd {quoted_stage}\nsha256sum -c migration.sha256",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                node_scope=("target",),
            ),
            command(
                "migration.cutover.start_auxiliary",
                "在目标端以复制辅助实例方式启动 NOMOUNT",
                f"export ORACLE_SID={shlex.quote(target_sid)}\n"
                f"sqlplus / as sysdba @{artifact_stage}/sql/start-target-nomount.sql",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                node_scope=("target",),
                artifact_ref="migration.start_auxiliary",
                risk=RunbookRiskLevel.HIGH,
            ),
            command(
                "migration.cutover.duplicate_target",
                "在目标端从备份目录复制数据库",
                f"export ORACLE_SID={shlex.quote(target_sid)}\n"
                f"rman auxiliary / cmdfile={artifact_stage}/rman/duplicate-target.rman",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                node_scope=("target",),
                artifact_ref="migration.duplicate_target",
                risk=RunbookRiskLevel.CRITICAL,
            ),
        ),
        "validate": (
            command(
                "migration.validate.source_state",
                "确认源库保持 MOUNT 回退状态",
                "SELECT name, open_mode, database_role FROM v$database;\n"
                "SELECT instance_name, status FROM v$instance;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
                node_scope=("source",),
            ),
            command(
                "migration.validate.target",
                "验证目标数据库、PDB、组件、对象和服务",
                f"sqlplus / as sysdba @{artifact_stage}/sql/validate-database.sql",
                executor=RunbookExecutor.SQLPLUS,
                run_as="oracle",
                node_scope=("target",),
                artifact_ref="migration.validation",
            ),
            command(
                "migration.validate.business",
                "切换应用连接并执行业务验收",
                "在目标数据库验证完成后按批准变更单更新应用连接；执行登录、查询、写入、"
                "批处理和接口验收。任一关键验收失败时恢复原连接并启动源库。",
                executor=RunbookExecutor.MANUAL,
                run_as="应用负责人/DBA",
                node_scope=("application", "target"),
                risk=RunbookRiskLevel.CRITICAL,
            ),
        ),
        "handover": (
            command(
                "migration.handover.backup",
                "在目标端建立 RESETLOGS 后的新备份基线",
                f"rman target / cmdfile={artifact_stage}/rman/backup-post-cutover.rman",
                executor=RunbookExecutor.RMAN,
                run_as="oracle",
                node_scope=("target",),
                artifact_ref="migration.post_cutover_backup",
                risk=post_cutover_risk,
            ),
            command(
                "migration.handover.observe",
                "进入观察期并保留可回退源环境",
                "在批准观察期结束前保持源库关闭但不删除文件，不复用原服务地址，不清理备份。"
                "完成监控、备份、作业、接口和性能基线交接后，再单独审批源环境下线。",
                executor=RunbookExecutor.MANUAL,
                run_as="DBA/运维负责人",
                node_scope=("source", "target"),
            ),
        ),
    }

    derived_parameters = {
        "POLICY_TEMPLATE_ID": _SPEC.policy_id,
        "MIGRATION_METHOD": method,
        "MIGRATION_METHOD_SOURCE": "STANDARD_SAME_PLATFORM_DOWNTIME_POLICY",
        "MIGRATION_STAGE_PATH": migration_stage,
        "MIGRATION_STAGE_PATH_REVIEW_REQUIRED": (
            "NO" if requested_stage else "YES"
        ),
        "SOURCE_DATABASE_NAME": database_name,
        "SOURCE_PLATFORM": source_platform,
        "SOURCE_VERSION": source_version,
        "SOURCE_CHARACTER_SET": source_charset,
        "SOURCE_NCHAR_CHARACTER_SET": source_ncharset,
        "TARGET_INSTANCE_NAME": target_sid,
        "TARGET_ENVIRONMENT": "NEW_HOST_SAME_PLATFORM_RELEASE_TOPOLOGY_AND_FILE_LAYOUT",
        "TARGET_ASSUMPTION_REVIEW_REQUIRED": "YES",
        "SOURCE_CONNECTION_MODE": "LOCAL_SYSDBA",
        "TARGET_CONNECTION_MODE": "LOCAL_SYSDBA",
        "CUTOVER_POLICY": "APPROVED_CHANGE_WINDOW_REQUIRED_AT_EXECUTION",
    }
    if requested_method:
        derived_parameters["REQUESTED_MIGRATION_METHOD"] = requested_method
    if method_review_required:
        derived_parameters["REQUESTED_METHOD_REVIEW_REQUIRED"] = "YES"

    return compile_profile(
        spec=_SPEC,
        evidence=evidence,
        context=context,
        commands_by_phase=commands,
        artifacts=artifacts,
        derived_parameters=derived_parameters,
    )
