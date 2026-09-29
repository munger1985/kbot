"""补丁、升级、迁移、克隆、Data Pump 与 ADG 演练档案公共编译器。"""

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
    RunbookFactSource,
    RunbookRiskLevel,
)
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from platform_core.contracts.aiops import ImplementationProfile


def _facts(profile: ImplementationProfile):
    common_stops = (
        "实时环境与 Runbook 固化事实不一致。",
        "保护备份、回退路径或业务验证责任人尚未确认。",
        "预检查出现未解决的 ERROR 或不可接受警告。",
    )
    specs = {
        ImplementationProfile.ORACLE_DATABASE_UPGRADE: ProfileSpec(
            profile=profile,
            title="Oracle 数据库 AutoUpgrade 跨版本升级实施操作文档",
            policy_id="oracle.upgrade.approved-target.v1",
            fact_tool_id="db.maintenance.upgrade_precheck",
            required_facts=(
                ("INSTANCE_NAME", "需要确认升级源实例 SID。", RunbookFactSource.TARGET_FACT, ("analyze", "deploy")),
                ("ORACLE_HOME", "需要确认源 Oracle Home。", RunbookFactSource.HOST_COLLECTOR, ("analyze", "deploy")),
                ("TARGET_ORACLE_HOME", "需要确认已安装的目标 Oracle Home。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("analyze", "deploy")),
                ("TARGET_VERSION", "需要从批准升级矩阵选择目标版本。", RunbookFactSource.POLICY_TEMPLATE, ("compatibility", "analyze", "deploy")),
            ),
            phases=(("scope", "升级范围与批准路径", ("scope",)), ("compatibility_phase", "组件、字符集与版本兼容性", ("compatibility",)), ("protection", "保护备份与 Guaranteed Restore Point", ("protection",)), ("analyze_phase", "AutoUpgrade Analyze 与 Fixups", ("analyze",)), ("deploy_phase", "AutoUpgrade Deploy", ("deploy",)), ("post", "Datapatch、组件与无效对象", ("post",)), ("business", "业务验收与性能基线", ("business",)), ("rollback_phase", "回退边界与交接", ("rollback",))),
            stop_conditions=common_stops + ("AutoUpgrade Analyze 或 Fixups 仍存在阻断项。",),
        ),
        ImplementationProfile.ORACLE_CLONE_REFRESH: ProfileSpec(
            profile=profile,
            title="Oracle 非生产数据库克隆与刷新实施操作文档",
            policy_id="oracle.clone.refresh.v1",
            fact_tool_id="db.clone.precheck",
            required_facts=(
                ("DESTINATION_REF", "需要引用已登记非生产目标。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("target", "clone")),
                ("CLONE_METHOD", "需要按存储和版本事实确定 RMAN/PDB/Snapshot 方法。", RunbookFactSource.POLICY_TEMPLATE, ("clone",)),
                ("SOURCE_CONNECT_IDENTIFIER", "需要确认源库安全连接标识。", RunbookFactSource.TARGET_FACT, ("clone",)),
                ("AUXILIARY_CONNECT_IDENTIFIER", "需要确认目标辅助实例连接标识。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("clone",)),
                ("TARGET_DB_NAME", "需要确认非生产目标数据库名称。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("clone", "rename")),
                ("MASKING_ARTIFACT_ID", "需要引用已批准的业务脱敏包。", RunbookFactSource.POLICY_TEMPLATE, ("masking", "release")),
            ),
            phases=(("scope", "刷新范围与数据时间点", ("scope",)), ("target_phase", "目标隔离、容量与服务准备", ("target",)), ("source", "源端保护和一致性准备", ("source",)), ("clone_phase", "克隆或恢复", ("clone",)), ("rename", "数据库重命名与服务隔离", ("rename",)), ("masking_phase", "批准脱敏包执行", ("masking",)), ("validation", "对象、数据与应用验证", ("validation",)), ("release_phase", "安全检查与交付", ("release",))),
            stop_conditions=common_stops + ("目标无法证明与生产网络、作业和外发链路隔离。",),
        ),
        ImplementationProfile.ORACLE_DATAPUMP_MIGRATION: ProfileSpec(
            profile=profile,
            title="Oracle Data Pump Schema/PDB 逻辑迁移实施操作文档",
            policy_id="oracle.datapump.standard.v1",
            fact_tool_id="db.datapump.precheck",
            required_facts=(),
            phases=(("assessment", "对象量、LOB、分区与字符集", ("assessment",)), ("mapping", "Schema、表空间与对象映射", ("mapping",)), ("directory_phase", "Directory 与目录权限", ("directory",)), ("export_phase", "一致性导出", ("export",)), ("transfer", "转储校验与传输", ("transfer",)), ("import_phase", "目标导入", ("import",)), ("validation_phase", "对象计数、无效对象和统计信息", ("validation",)), ("cutover", "业务切换与清理", ("cutover",))),
            stop_conditions=common_stops + ("源目标字符集、时区或对象兼容性未确认。",),
        ),
        ImplementationProfile.ORACLE_ADG_DRILL: ProfileSpec(
            profile=profile,
            title="Oracle Active Data Guard Switchover/Failover/Reinstate 演练操作文档",
            policy_id="oracle.adg.drill.v1",
            fact_tool_id="db.ha.adg_drill_precheck",
            required_facts=(
                ("DG_CONFIG_NAME", "需要确认已启用且健康的 Broker 配置。", RunbookFactSource.TARGET_FACT, ("precheck", "execute")),
                ("PRIMARY_DB_UNIQUE_NAME", "需要确认当前主库唯一名。", RunbookFactSource.TARGET_FACT, ("execute", "verify")),
                ("STANDBY_DB_UNIQUE_NAME", "需要确认演练备库唯一名。", RunbookFactSource.TARGET_FACT, ("execute", "verify")),
                ("DRILL_SCENARIO", "需要明确 Switchover、计划 Failover 或 Reinstate 场景。", RunbookFactSource.EXPLICIT_USER_DECISION, ("execute",)),
            ),
            phases=(("scope", "演练范围、角色与业务窗口", ("scope",)), ("precheck_phase", "Broker、日志零缺口与 Flashback 预检", ("precheck",)), ("application", "应用停写与连接排空", ("application",)), ("execute_phase", "角色切换或故障转移", ("execute",)), ("verify_phase", "数据库、服务和业务验证", ("verify",)), ("return", "回切或 Reinstate", ("return",)), ("handover", "复盘、告警与运维交接", ("handover",))),
            stop_conditions=common_stops + ("Broker 任一成员存在 ERROR、日志缺口或延迟超过批准阈值。",),
            manual_only=True,
        ),
    }
    return specs[profile]


def compile_standard(
    profile: ImplementationProfile,
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any],
):
    spec = _facts(profile)
    identity, _ = first_row(evidence, "db.instance.identity")
    facts, _ = first_row(evidence, spec.fact_tool_id)
    merged = {**identity, **facts}
    commands: dict[str, tuple] = {
        "scope": (command(
            f"{profile.value.lower()}.scope",
            "核对数据库身份、角色和版本",
            "SELECT name, db_unique_name, database_role, open_mode, log_mode, cdb FROM v$database;\nSELECT instance_name, host_name, version, status FROM v$instance;",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
        ),),
        "protection": (command(
            f"{profile.value.lower()}.protection",
            "建立并验证实施前保护备份",
            "BACKUP AS COMPRESSED BACKUPSET DATABASE PLUS ARCHIVELOG;\nBACKUP CURRENT CONTROLFILE;\nBACKUP SPFILE;\nRESTORE DATABASE VALIDATE;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
            risk=RunbookRiskLevel.MEDIUM,
        ),),
        "backup": (command(
            f"{profile.value.lower()}.backup",
            "建立并验证补丁前保护备份",
            "BACKUP AS COMPRESSED BACKUPSET DATABASE PLUS ARCHIVELOG;\nBACKUP CURRENT CONTROLFILE;\nBACKUP SPFILE;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
            risk=RunbookRiskLevel.MEDIUM,
        ),),
        "verify": (command(
            f"{profile.value.lower()}.verify",
            "验证数据库与组件状态",
            "SELECT name, open_mode, database_role FROM v$database;\nSELECT comp_id, version, status FROM dba_registry ORDER BY comp_id;\nSELECT owner, object_type, COUNT(*) FROM dba_objects WHERE status='INVALID' GROUP BY owner, object_type ORDER BY owner, object_type;",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
        ),),
        "validation": (command(
            f"{profile.value.lower()}.validation",
            "执行对象与数据库状态验证",
            "SELECT name, open_mode, database_role FROM v$database;\nSELECT owner, object_type, COUNT(*) FROM dba_objects WHERE status='INVALID' GROUP BY owner, object_type ORDER BY owner, object_type;",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
        ),),
    }
    artifacts: list[GeneratedRunbookArtifact] = []
    derived_parameters = {"POLICY_TEMPLATE_ID": spec.policy_id}
    stage_name = profile.value.lower().replace("_", "-")
    stage = f"/var/tmp/kbot-runbooks/{stage_name}"

    oracle_home = value(context, merged, "ORACLE_HOME")
    if profile == ImplementationProfile.ORACLE_DATABASE_UPGRADE:
        sid = value(context, merged, "INSTANCE_NAME") or value(context, merged, "instance_name")
        target_home = value(context, merged, "TARGET_ORACLE_HOME")
        target_version = value(context, merged, "TARGET_VERSION")
        if sid and oracle_home and target_home and target_version:
            cfg = (
                "global.autoupg_log_dir=" + stage + "/log\n"
                "upg1.sid=" + sid + "\n"
                "upg1.source_home=" + oracle_home + "\n"
                "upg1.target_home=" + target_home + "\n"
                "upg1.start_time=NOW\nupg1.run_utlrp=yes\n"
            )
            artifacts.append(GeneratedRunbookArtifact(
                artifact_id="upgrade.autoupgrade.config",
                relative_path=f"{stage_name}/autoupgrade.cfg",
                content=cfg,
                media_type="text/plain",
                file_mode="0640",
                run_as="oracle",
                target_path=f"{stage}/autoupgrade.cfg",
                description="已代入源实例与目标 Home 的 AutoUpgrade 配置。",
            ))
            jar = f"{target_home}/rdbms/admin/autoupgrade.jar"
            commands["analyze"] = (command(
                "upgrade.analyze", "执行 AutoUpgrade Analyze",
                f"{target_home}/jdk/bin/java -jar {jar} -config {stage}/autoupgrade.cfg -mode analyze",
                executor=RunbookExecutor.BASH, run_as="oracle",
                artifact_ref="upgrade.autoupgrade.config",
            ),)
            commands["deploy"] = (command(
                "upgrade.deploy", "审批后执行 AutoUpgrade Deploy",
                f"{target_home}/jdk/bin/java -jar {jar} -config {stage}/autoupgrade.cfg -mode deploy",
                executor=RunbookExecutor.BASH, run_as="oracle",
                risk=RunbookRiskLevel.CRITICAL,
                artifact_ref="upgrade.autoupgrade.config",
            ),)
    elif profile == ImplementationProfile.ORACLE_DATAPUMP_MIGRATION:
        directory_name = (
            value(context, merged, "DATAPUMP_DIRECTORY_NAME")
            or "KBOT_DATAPUMP_DIR"
        ).upper()
        dump_prefix = (
            value(context, merged, "DATAPUMP_DUMP_PREFIX")
            or "kbot_export"
        )
        parallel = value(context, merged, "DATAPUMP_PARALLEL") or "4"
        observed_directory_path = value(
            context, merged, "DATAPUMP_DIRECTORY_PATH"
        )
        supplied_directory_path = str(
            dict(context.get("implementation_parameters") or {}).get(
                "DATAPUMP_DIRECTORY_PATH"
            )
            or ""
        ).strip()
        database_key = (
            value(context, merged, "DB_UNIQUE_NAME")
            or value(context, merged, "DATABASE_NAME")
            or value(context, merged, "INSTANCE_NAME")
            or "database"
        )
        path_key = re.sub(r"[^A-Za-z0-9_-]", "_", database_key).lower()
        directory_path = observed_directory_path
        if not directory_path.startswith("/"):
            directory_path = f"/u01/app/oracle/admin/{path_key}/dpdump"

        schemas = value(context, merged, "SOURCE_SCHEMAS")
        schema_names = tuple(
            item.strip().upper() for item in schemas.split(",") if item.strip()
        )
        if schema_names:
            export_scope = "schemas=" + ",".join(schema_names)
            export_mode = "SCHEMA"
            owner_filter = "owner IN (" + ",".join(
                f"'{item}'" for item in schema_names
            ) + ")"
            user_filter = "username IN (" + ",".join(
                f"'{item}'" for item in schema_names
            ) + ")"
        else:
            export_scope = "full=yes"
            export_mode = "FULL_CURRENT_CONTAINER"
            owner_filter = (
                "owner IN (SELECT username FROM dba_users "
                "WHERE oracle_maintained='N' AND common='NO')"
            )
            user_filter = "oracle_maintained='N' AND common='NO'"

        quoted_directory_path = shlex.quote(directory_path)
        directory_sql = (
            "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
            f"CREATE OR REPLACE DIRECTORY {directory_name} AS "
            f"'{directory_path}';\n"
            "SELECT directory_name, directory_path FROM dba_directories "
            f"WHERE directory_name='{directory_name}';\n"
            "EXIT\n"
        )
        export_par = (
            f"{export_scope}\n"
            f"directory={directory_name}\n"
            f"dumpfile={dump_prefix}_%U.dmp\n"
            f"logfile={dump_prefix}.log\n"
            "flashback_time=systimestamp\n"
            f"parallel={parallel}\n"
            "compression=all\n"
            "metrics=yes\n"
            "logtime=all\n"
        )
        import_par = (
            f"{export_scope}\n"
            f"directory={directory_name}\n"
            f"dumpfile={dump_prefix}_%U.dmp\n"
            f"logfile={dump_prefix}_import.log\n"
            f"parallel={parallel}\n"
            "metrics=yes\n"
            "logtime=all\n"
        )
        preview_par = import_par.replace(
            f"logfile={dump_prefix}_import.log\n",
            f"logfile={dump_prefix}_import_preview.log\n"
            f"sqlfile={dump_prefix}_import_preview.sql\n",
        )
        assessment_sql = (
            "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
            "SET PAGESIZE 500 LINESIZE 240 TRIMSPOOL ON\n"
            "SELECT name, db_unique_name, open_mode, cdb, platform_name "
            "FROM v$database;\n"
            "SELECT parameter, value FROM nls_database_parameters "
            "WHERE parameter IN ('NLS_CHARACTERSET','NLS_NCHAR_CHARACTERSET') "
            "ORDER BY parameter;\n"
            "SELECT version FROM v$timezone_file;\n"
            "SELECT owner, object_type, COUNT(*) object_count "
            f"FROM dba_objects WHERE {owner_filter} "
            "GROUP BY owner, object_type ORDER BY owner, object_type;\n"
            "SELECT owner, segment_type, ROUND(SUM(bytes)/1024/1024,2) mb "
            f"FROM dba_segments WHERE {owner_filter} "
            "GROUP BY owner, segment_type ORDER BY owner, segment_type;\n"
            "SELECT owner, COUNT(*) lob_count FROM dba_lobs "
            f"WHERE {owner_filter} GROUP BY owner ORDER BY owner;\n"
            "SELECT table_owner owner, COUNT(*) partition_count "
            f"FROM dba_tab_partitions WHERE table_owner IN "
            "(SELECT username FROM dba_users WHERE " + user_filter + ") "
            "GROUP BY table_owner ORDER BY table_owner;\n"
            "EXIT\n"
        )
        mapping_sql = (
            "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
            "SET PAGESIZE 500 LINESIZE 240 TRIMSPOOL ON\n"
            "SELECT username, default_tablespace, temporary_tablespace, "
            "account_status FROM dba_users WHERE " + user_filter +
            " ORDER BY username;\n"
            "SELECT owner, tablespace_name, ROUND(SUM(bytes)/1024/1024,2) mb "
            f"FROM dba_segments WHERE {owner_filter} "
            "GROUP BY owner, tablespace_name ORDER BY owner, tablespace_name;\n"
            "EXIT\n"
        )
        validation_sql = (
            "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
            "SET PAGESIZE 500 LINESIZE 240 TRIMSPOOL ON\n"
            "SELECT owner, object_type, COUNT(*) object_count "
            f"FROM dba_objects WHERE {owner_filter} "
            "GROUP BY owner, object_type ORDER BY owner, object_type;\n"
            "SELECT owner, object_type, COUNT(*) invalid_count "
            f"FROM dba_objects WHERE status='INVALID' AND {owner_filter} "
            "GROUP BY owner, object_type ORDER BY owner, object_type;\n"
            "SELECT owner, COUNT(*) stale_or_missing_statistics "
            "FROM dba_tab_statistics WHERE "
            f"({owner_filter}) AND (stale_stats='YES' OR last_analyzed IS NULL) "
            "GROUP BY owner ORDER BY owner;\n"
            "EXIT\n"
        )
        artifacts.extend((
            GeneratedRunbookArtifact(
                "datapump.assessment", f"{stage_name}/sql/assess-source.sql",
                assessment_sql, "text/x-sql", "0640", "oracle",
                f"{stage}/sql/assess-source.sql", "源端对象、容量、字符集和时区评估 SQL。",
            ),
            GeneratedRunbookArtifact(
                "datapump.mapping", f"{stage_name}/sql/review-mapping.sql",
                mapping_sql, "text/x-sql", "0640", "oracle",
                f"{stage}/sql/review-mapping.sql", "源目标 Schema 与表空间映射核对 SQL。",
            ),
            GeneratedRunbookArtifact(
                "datapump.directory", f"{stage_name}/sql/create-directory.sql",
                directory_sql, "text/x-sql", "0640", "oracle",
                f"{stage}/sql/create-directory.sql", "在源端和目标端创建 Data Pump Directory。",
            ),
            GeneratedRunbookArtifact(
                "datapump.export.par", f"{stage_name}/par/expdp.par",
                export_par, "text/plain", "0640", "oracle",
                f"{stage}/par/expdp.par", "已固化导出范围与一致性时间点的 expdp 参数文件。",
            ),
            GeneratedRunbookArtifact(
                "datapump.import.preview.par", f"{stage_name}/par/impdp-preview.par",
                preview_par, "text/plain", "0640", "oracle",
                f"{stage}/par/impdp-preview.par", "正式导入前生成 SQLFILE 的 impdp 参数文件。",
            ),
            GeneratedRunbookArtifact(
                "datapump.import.par", f"{stage_name}/par/impdp.par",
                import_par, "text/plain", "0640", "oracle",
                f"{stage}/par/impdp.par", "目标端本机 SYSDBA 导入参数文件。",
            ),
            GeneratedRunbookArtifact(
                "datapump.validation", f"{stage_name}/sql/validate-objects.sql",
                validation_sql, "text/x-sql", "0640", "oracle",
                f"{stage}/sql/validate-objects.sql", "源端和目标端对象、无效对象及统计信息校验 SQL。",
            ),
        ))
        commands["assessment"] = (command(
            "datapump.assessment", "评估源端对象、容量和兼容性",
            f"sqlplus / as sysdba @{stage}/sql/assess-source.sql",
            executor=RunbookExecutor.SQLPLUS, run_as="oracle",
            artifact_ref="datapump.assessment",
        ),)
        commands["mapping"] = (
            command(
                "datapump.mapping.source", "记录源端 Schema 与表空间映射",
                f"sqlplus / as sysdba @{stage}/sql/review-mapping.sql",
                executor=RunbookExecutor.SQLPLUS, run_as="oracle",
                artifact_ref="datapump.mapping", node_scope=("source",),
            ),
            command(
                "datapump.mapping.target", "核对目标端表空间和用户冲突",
                f"sqlplus / as sysdba @{stage}/sql/review-mapping.sql",
                executor=RunbookExecutor.SQLPLUS, run_as="oracle",
                artifact_ref="datapump.mapping", node_scope=("target",),
            ),
        )
        commands["directory"] = (
            command(
                "datapump.directory.os.source", "在源端创建转储目录",
                f"install -d -m 0750 -o oracle -g oinstall {quoted_directory_path}",
                executor=RunbookExecutor.BASH, run_as="root",
                node_scope=("source",), risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "datapump.directory.db.source", "在源库创建 Directory 对象",
                f"sqlplus / as sysdba @{stage}/sql/create-directory.sql",
                executor=RunbookExecutor.SQLPLUS, run_as="oracle",
                node_scope=("source",), artifact_ref="datapump.directory",
            ),
            command(
                "datapump.directory.os.target", "在目标端创建转储目录",
                f"install -d -m 0750 -o oracle -g oinstall {quoted_directory_path}",
                executor=RunbookExecutor.BASH, run_as="root",
                node_scope=("target",), risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "datapump.directory.db.target", "在目标库创建 Directory 对象",
                f"sqlplus / as sysdba @{stage}/sql/create-directory.sql",
                executor=RunbookExecutor.SQLPLUS, run_as="oracle",
                node_scope=("target",), artifact_ref="datapump.directory",
            ),
        )
        commands["export"] = (
            command(
                "datapump.export.precheck", "确认不存在同名历史转储",
                f"test -z \"$(find {quoted_directory_path} -maxdepth 1 "
                f"-type f -name '{dump_prefix}_*.dmp' -print -quit)\"",
                executor=RunbookExecutor.BASH, run_as="oracle",
                node_scope=("source",),
            ),
            command(
                "datapump.export", "执行一致性逻辑导出",
                f"expdp '/ as sysdba' parfile={stage}/par/expdp.par",
                executor=RunbookExecutor.DATAPUMP, run_as="oracle",
                node_scope=("source",), artifact_ref="datapump.export.par",
                risk=RunbookRiskLevel.MEDIUM,
            ),
        )
        commands["transfer"] = (
            command(
                "datapump.transfer.checksum.source", "在源端生成转储摘要清单",
                f"cd {quoted_directory_path}\n"
                f"sha256sum {dump_prefix}_*.dmp {dump_prefix}.log "
                f"> {dump_prefix}.sha256\n"
                f"sha256sum -c {dump_prefix}.sha256",
                executor=RunbookExecutor.BASH, run_as="oracle",
                node_scope=("source",),
            ),
            command(
                "datapump.transfer.approved-channel", "通过批准通道传输转储文件",
                f"将 {dump_prefix}_*.dmp、{dump_prefix}.log 和 "
                f"{dump_prefix}.sha256 "
                f"从源端 {directory_path} 传输到目标端同一路径。"
                "传输工具、主机地址和网络账号以已批准的运维通道为准；"
                "本静态文档不伪造目标主机或凭据。",
                executor=RunbookExecutor.MANUAL, run_as="DBA",
                node_scope=("source", "target"), risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "datapump.transfer.checksum.target", "在目标端校验转储摘要",
                f"cd {quoted_directory_path}\n"
                f"sha256sum -c {dump_prefix}.sha256",
                executor=RunbookExecutor.BASH, run_as="oracle",
                node_scope=("target",),
            ),
        )
        commands["import"] = (
            command(
                "datapump.import.preview", "在目标端生成导入 SQL 预览",
                f"impdp '/ as sysdba' parfile={stage}/par/impdp-preview.par",
                executor=RunbookExecutor.DATAPUMP, run_as="oracle",
                node_scope=("target",),
                artifact_ref="datapump.import.preview.par",
            ),
            command(
                "datapump.import", "审批 SQL 预览后执行目标端导入",
                f"impdp '/ as sysdba' parfile={stage}/par/impdp.par",
                executor=RunbookExecutor.DATAPUMP, run_as="oracle",
                node_scope=("target",), artifact_ref="datapump.import.par",
                risk=RunbookRiskLevel.HIGH,
            ),
        )
        commands["validation"] = (
            command(
                "datapump.validation.source", "在源端输出对象基线",
                f"sqlplus / as sysdba @{stage}/sql/validate-objects.sql",
                executor=RunbookExecutor.SQLPLUS, run_as="oracle",
                node_scope=("source",), artifact_ref="datapump.validation",
            ),
            command(
                "datapump.validation.target", "在目标端核对对象、无效对象和统计信息",
                f"sqlplus / as sysdba @{stage}/sql/validate-objects.sql",
                executor=RunbookExecutor.SQLPLUS, run_as="oracle",
                node_scope=("target",), artifact_ref="datapump.validation",
            ),
        )
        commands["cutover"] = (
            command(
                "datapump.cutover.database", "执行切换前数据库最终核验",
                "SELECT name, open_mode, database_role FROM v$database;\n"
                "SELECT owner, object_type, COUNT(*) FROM dba_objects "
                f"WHERE status='INVALID' AND {owner_filter} "
                "GROUP BY owner, object_type ORDER BY owner, object_type;",
                executor=RunbookExecutor.SQLPLUS, run_as="SYSDBA",
                node_scope=("target",),
            ),
            command(
                "datapump.cutover.application", "按批准变更单切换应用连接并验证业务",
                "停止源端写入，记录最终导出日志和对象基线；按已批准变更单更新应用连接，"
                "完成登录、查询、写入和批处理验证。若任一验收失败，立即恢复原连接。",
                executor=RunbookExecutor.MANUAL, run_as="DBA/应用负责人",
                node_scope=("application",), risk=RunbookRiskLevel.CRITICAL,
            ),
        )
        derived_parameters.update({
            "DATAPUMP_DIRECTORY_NAME": directory_name,
            "DATAPUMP_DIRECTORY_PATH": directory_path,
            "DATAPUMP_PARALLEL": parallel,
            "DATAPUMP_DUMP_PREFIX": dump_prefix,
            "DATAPUMP_DIRECTORY_SOURCE": (
                "USER_SUPPLIED"
                if supplied_directory_path
                else "TARGET_OR_DATABASE_FACT"
                if observed_directory_path.startswith("/")
                else "DERIVED_STANDARD_PATH"
            ),
            "DATAPUMP_TARGET_PATH_REVIEW_REQUIRED": "YES",
            "EXPORT_MODE": export_mode,
            "SOURCE_SCHEMAS": (
                ",".join(schema_names)
                if schema_names
                else "ALL_BUSINESS_SCHEMAS_IN_CURRENT_CONTAINER"
            ),
            "TARGET_EXECUTION_MODE": "LOCAL_SYSDBA",
        })
    elif profile == ImplementationProfile.ORACLE_CLONE_REFRESH:
        method = value(context, merged, "CLONE_METHOD").upper()
        source_tns = value(context, merged, "SOURCE_CONNECT_IDENTIFIER")
        auxiliary_tns = value(context, merged, "AUXILIARY_CONNECT_IDENTIFIER")
        target_name = value(context, merged, "TARGET_DB_NAME")
        if method == "RMAN_DUPLICATE" and source_tns and auxiliary_tns and target_name:
            script = (
                f"CONNECT TARGET /@{source_tns}\n"
                f"CONNECT AUXILIARY /@{auxiliary_tns}\n"
                f"DUPLICATE TARGET DATABASE TO {target_name} FROM ACTIVE DATABASE NOFILENAMECHECK;\n"
            )
            artifacts.append(GeneratedRunbookArtifact(
                "clone.rman.duplicate", f"{stage_name}/rman/duplicate-refresh.rman",
                script, "text/x-rman", "0640", "oracle",
                f"{stage}/rman/duplicate-refresh.rman", "刷新已登记非生产目标的 RMAN Duplicate 脚本。",
            ))
            commands["clone"] = (command(
                "clone.execute.rman", "执行非生产目标克隆",
                f"rman cmdfile={stage}/rman/duplicate-refresh.rman",
                executor=RunbookExecutor.RMAN, run_as="oracle",
                artifact_ref="clone.rman.duplicate", risk=RunbookRiskLevel.CRITICAL,
            ),)
    elif profile == ImplementationProfile.ORACLE_ADG_DRILL:
        primary = value(context, merged, "PRIMARY_DB_UNIQUE_NAME")
        standby = value(context, merged, "STANDBY_DB_UNIQUE_NAME")
        scenario = value(context, merged, "DRILL_SCENARIO").upper()
        if primary and standby:
            commands["precheck"] = (command(
                "adg.drill.precheck", "执行 Broker 演练预检查",
                f"SHOW CONFIGURATION VERBOSE;\nSHOW DATABASE VERBOSE '{primary}';\nSHOW DATABASE VERBOSE '{standby}';\nVALIDATE DATABASE VERBOSE '{primary}';\nVALIDATE DATABASE VERBOSE '{standby}';",
                executor=RunbookExecutor.DGMGRL, run_as="oracle",
            ),)
        if primary and standby and scenario:
            if scenario == "SWITCHOVER":
                body = f"SWITCHOVER TO '{standby}';\nSHOW CONFIGURATION;"
            elif scenario == "FAILOVER":
                body = f"FAILOVER TO '{standby}';\nSHOW CONFIGURATION;"
            elif scenario == "REINSTATE":
                body = f"REINSTATE DATABASE '{primary}';\nSHOW CONFIGURATION;"
            else:
                body = ""
            if body:
                artifacts.append(GeneratedRunbookArtifact(
                    "adg.drill.command", f"{stage_name}/dgmgrl/execute-drill.dgmgrl",
                    body, "text/plain", "0640", "oracle",
                    f"{stage}/dgmgrl/execute-drill.dgmgrl", "已冻结角色和场景的 Broker 演练命令。",
                ))
                commands["execute"] = (command(
                    "adg.drill.execute", f"人工执行 {scenario}",
                    f"dgmgrl / @{stage}/dgmgrl/execute-drill.dgmgrl",
                    executor=RunbookExecutor.DGMGRL, run_as="oracle",
                    risk=RunbookRiskLevel.CRITICAL,
                    artifact_ref="adg.drill.command",
                ),)
    return compile_profile(
        spec=spec,
        evidence=evidence,
        context=context,
        commands_by_phase=commands,
        artifacts=tuple(artifacts),
        derived_parameters=derived_parameters,
    )
