"""Oracle RAC 两节点建设实施档案。"""

from __future__ import annotations

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


_SPEC = ProfileSpec(
    profile=ImplementationProfile.ORACLE_RAC_BUILD,
    title="Oracle 两节点 RAC/ASM 从零建设与现有数据库迁移实施操作文档",
    policy_id="oracle.rac.standard-2node.v1",
    fact_tool_id="db.ha.rac_precheck",
    required_facts=(
        ("DATABASE_NAME", "需要确认源数据库名称。", RunbookFactSource.TARGET_FACT, ("migration", "rac_config")),
        ("ORACLE_HOME", "需要由主机采集确认数据库软件目录。", RunbookFactSource.HOST_COLLECTOR, ("db_home", "migration")),
        ("GRID_HOME", "需要确认目标 Grid Infrastructure 目录。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("grid_install",)),
        ("NODE1_HOST", "需要登记第一个 RAC 节点主机名。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("os_prepare", "network", "grid_install", "register")),
        ("NODE2_HOST", "需要登记第二个 RAC 节点主机名。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("os_prepare", "network", "grid_install", "register")),
        ("NODE1_VIP", "需要登记第一个节点 VIP 名称或地址。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("network", "grid_install")),
        ("NODE2_VIP", "需要登记第二个节点 VIP 名称或地址。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("network", "grid_install")),
        ("SCAN_NAME", "需要登记已完成 DNS 解析的 SCAN 名称。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("network", "grid_install", "services")),
        ("ASM_DISK_WWIDS", "需要由主机采集确认两节点一致的共享盘 WWID。", RunbookFactSource.HOST_COLLECTOR, ("shared_storage", "asm")),
        ("PATCH_STAGE_PATH", "需要登记平台批准的 GI/DB Home 介质目录。", RunbookFactSource.POLICY_TEMPLATE, ("grid_install", "db_home")),
    ),
    phases=(
        ("scope", "范围与目标拓扑", ("scope",)),
        ("source_protection", "源库保护与回退边界", ("source_protection",)),
        ("os", "操作系统准备", ("os_prepare",)),
        ("network_phase", "网络、VIP 与 SCAN", ("network",)),
        ("storage", "共享存储与设备一致性", ("shared_storage",)),
        ("grid", "Grid Infrastructure 安装", ("grid_install",)),
        ("asm_phase", "ASM 磁盘组建设", ("asm",)),
        ("database_home", "数据库 Home 安装与补丁", ("db_home",)),
        ("migration_phase", "数据库迁移到 ASM", ("migration",)),
        ("rac", "数据库 RAC 化", ("rac_config",)),
        ("cluster", "集群资源注册", ("register",)),
        ("service_phase", "CDB/PDB 与业务服务", ("services",)),
        ("acceptance", "集群验收与故障漂移", ("validate",)),
        ("operations", "运维接管与回退", ("handover",)),
    ),
    stop_conditions=(
        "SCAN、VIP、私网互通、MTU 或时间同步核验失败。",
        "两节点看到的共享磁盘 WWID、容量或多路径映射不一致。",
        "GI、数据库 Home、RU 或 OPatch 版本不一致。",
        "源库没有可验证备份，或目标容量不足以容纳数据库与恢复区。",
        "数据库组件、字符集或 COMPATIBLE 与目标软件不兼容。",
        "实际主机、设备或路径与本 Runbook 固化事实不一致。",
    ),
)


def compile_rac(
    evidence: tuple[TurnEvidenceFact, ...], context: dict[str, Any]
):
    identity, _ = first_row(evidence, "db.instance.identity")
    facts, _ = first_row(evidence, "db.ha.rac_precheck")
    merged = {**identity, **facts}
    db_name = value(context, merged, "DATABASE_NAME") or value(context, merged, "database_name")
    db_unique = value(context, merged, "DB_UNIQUE_NAME") or db_name
    node1 = value(context, merged, "NODE1_HOST")
    node2 = value(context, merged, "NODE2_HOST")
    scan = value(context, merged, "SCAN_NAME")
    grid_home = value(context, merged, "GRID_HOME")
    oracle_home = value(context, merged, "ORACLE_HOME")
    patch_stage = value(context, merged, "PATCH_STAGE_PATH")
    stage = "/var/tmp/kbot-runbooks/oracle-rac-build"
    commands: dict[str, tuple] = {
        "scope": (command(
            "rac.scope.verify",
            "核对源库是否仍为单实例",
            "SELECT name, db_unique_name, database_role, open_mode, cdb FROM v$database;\nSELECT instance_name, host_name, version FROM v$instance;\nSHOW PARAMETER cluster_database;",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
            expected=("cluster_database 在迁移前为 FALSE",),
        ),),
        "source_protection": (command(
            "rac.source.backup",
            "执行源库全库保护备份并校验",
            "BACKUP AS COMPRESSED BACKUPSET DATABASE PLUS ARCHIVELOG;\nBACKUP CURRENT CONTROLFILE;\nBACKUP SPFILE;\nRESTORE DATABASE VALIDATE;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
            risk=RunbookRiskLevel.MEDIUM,
            expected=("RMAN 任务以零错误结束", "控制文件和 SPFILE 备份可列出"),
        ),),
        "validate": (command(
            "rac.validate.database",
            "验证 RAC 实例、线程、Undo 和服务",
            "SELECT inst_id, instance_name, host_name, status FROM gv$instance ORDER BY inst_id;\nSELECT thread#, enabled, status FROM v$thread ORDER BY thread#;\nSELECT inst_id, name, value FROM gv$parameter WHERE name IN ('cluster_database','instance_number','thread','undo_tablespace') ORDER BY name, inst_id;",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
            node_scope=("node1",),
        ),),
    }
    if db_unique:
        commands["handover"] = (command(
            "rac.operations.status",
            "交接日常集群状态检查",
            f"crsctl stat res -t\nsrvctl status database -db {db_unique}",
            executor=RunbookExecutor.CRSCTL,
            run_as="grid",
            node_scope=("node1",),
        ),)
    artifacts: list[GeneratedRunbookArtifact] = [GeneratedRunbookArtifact(
        artifact_id="rac.topology.summary",
        relative_path="oracle-rac-build/01-topology-summary.md",
        content=(
            "# RAC 固化拓扑摘要\n\n"
            f"- 源数据库：{db_unique or '尚未取得'}\n"
            f"- 节点一：{node1 or '尚未登记'}\n"
            f"- 节点二：{node2 or '尚未登记'}\n"
            f"- SCAN：{scan or '尚未登记'}\n"
            "- 数据磁盘组：+DATA\n- 恢复磁盘组：+RECO\n"
        ),
        media_type="text/markdown",
        file_mode="0644",
        run_as="DBA",
        target_path=f"{stage}/01-topology-summary.md",
        description="本次文档固化的源库与目标 RAC 拓扑。",
    )]
    if db_name:
        source_sql = (
            "ALTER SYSTEM SET remote_login_passwordfile='EXCLUSIVE' SCOPE=SPFILE;\n"
            "ALTER DATABASE FORCE LOGGING;\n"
            "ALTER SYSTEM ARCHIVE LOG CURRENT;\n"
            "SELECT force_logging, log_mode FROM v$database;\n"
        )
        artifacts.append(GeneratedRunbookArtifact(
            artifact_id="rac.prepare.source",
            relative_path="oracle-rac-build/database/prepare-source.sql",
            content=source_sql,
            media_type="text/x-sql",
            file_mode="0640",
            run_as="oracle",
            target_path=f"{stage}/database/prepare-source.sql",
            description="迁移前源库保护和日志模式核验脚本。",
        ))
        commands["source_protection"] += (command(
            "rac.source.prepare",
            "执行源库准备脚本",
            f"sqlplus / as sysdba @{stage}/database/prepare-source.sql",
            executor=RunbookExecutor.SQLPLUS,
            run_as="oracle",
            risk=RunbookRiskLevel.HIGH,
            artifact_ref="rac.prepare.source",
            target_path=f"{stage}/database/prepare-source.sql",
        ),)
    if all((node1, node2, scan, grid_home, patch_stage)):
        verify = (
            f"{grid_home}/bin/olsnodes -n -s -t\n"
            f"{grid_home}/bin/crsctl check cluster -all\n"
            f"{grid_home}/bin/srvctl config scan\n"
            f"getent hosts {scan}\n"
        )
        artifacts.append(GeneratedRunbookArtifact(
            artifact_id="rac.verify.cluster",
            relative_path="oracle-rac-build/grid/verify-cluster.sh",
            content="#!/usr/bin/env bash\nset -euo pipefail\n" + verify,
            media_type="text/x-shellscript",
            file_mode="0750",
            run_as="grid",
            target_path=f"{stage}/grid/verify-cluster.sh",
            description="验证两节点 CRS、SCAN 和节点清单。",
        ))
        commands["grid_install"] = (command(
            "rac.grid.install",
            "按响应文件安装并验证 Grid Infrastructure",
            f"cd {patch_stage}\n{grid_home}/gridSetup.sh -silent -responseFile {stage}/grid/gridsetup.rsp\n{stage}/grid/verify-cluster.sh",
            executor=RunbookExecutor.BASH,
            run_as="grid",
            node_scope=("node1",),
            risk=RunbookRiskLevel.CRITICAL,
            artifact_ref="rac.verify.cluster",
            notes=("root.sh 只能在安装程序明确提示后由 root 按节点顺序人工执行。",),
        ),)
    if db_unique and oracle_home and node1 and node2:
        register = (
            f"{oracle_home}/bin/srvctl add database -db {db_unique} -oraclehome {oracle_home} -spfile +DATA/{db_unique}/PARAMETERFILE/spfile.ora -role PRIMARY -startoption OPEN -stopoption IMMEDIATE -policy AUTOMATIC\n"
            f"{oracle_home}/bin/srvctl add instance -db {db_unique} -instance {db_name}1 -node {node1}\n"
            f"{oracle_home}/bin/srvctl add instance -db {db_unique} -instance {db_name}2 -node {node2}\n"
            f"{oracle_home}/bin/srvctl enable database -db {db_unique}\n"
            f"{oracle_home}/bin/srvctl config database -db {db_unique}\n"
        )
        artifacts.append(GeneratedRunbookArtifact(
            artifact_id="rac.register.resources",
            relative_path="oracle-rac-build/database/register-resources.sh",
            content="#!/usr/bin/env bash\nset -euo pipefail\n" + register,
            media_type="text/x-shellscript",
            file_mode="0750",
            run_as="oracle",
            target_path=f"{stage}/database/register-resources.sh",
            description="把已恢复数据库注册为两实例 RAC 资源。",
        ))
        commands["register"] = (command(
            "rac.cluster.register",
            "注册数据库和两个 RAC 实例",
            f"{stage}/database/register-resources.sh",
            executor=RunbookExecutor.SRVCTL,
            run_as="oracle",
            node_scope=("node1",),
            risk=RunbookRiskLevel.HIGH,
            artifact_ref="rac.register.resources",
        ),)
    return compile_profile(
        spec=_SPEC,
        evidence=evidence,
        context=context,
        commands_by_phase=commands,
        artifacts=tuple(artifacts),
        derived_parameters={
            "POLICY_TEMPLATE_ID": _SPEC.policy_id,
            "DATA_DISKGROUP": "+DATA",
            "RECO_DISKGROUP": "+RECO",
            "TARGET_INSTANCE_1": f"{db_name}1" if db_name else "待数据库名称确认后派生",
            "TARGET_INSTANCE_2": f"{db_name}2" if db_name else "待数据库名称确认后派生",
        },
    )
