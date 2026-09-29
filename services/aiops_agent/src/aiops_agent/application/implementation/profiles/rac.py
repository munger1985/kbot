"""Oracle RAC 两节点建设实施档案。"""

from __future__ import annotations

import re
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
    RunbookCommandType,
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
        ("VERSION", "需要从 Target 配置或数据库实例取得真实 Oracle 版本。", RunbookFactSource.TARGET_FACT, ("os_prepare", "grid_install", "asm")),
        ("ORACLE_HOME", "需要由主机采集确认数据库软件目录。", RunbookFactSource.HOST_COLLECTOR, ("db_home", "migration")),
        ("NODE1_HOST", "资源注册前需要确认第一个 RAC 节点主机名。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("register",)),
        ("NODE2_HOST", "资源注册前需要确认第二个 RAC 节点主机名。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("register",)),
        ("SCAN_NAME", "业务服务发布前需要确认 GI 已登记的 SCAN。", RunbookFactSource.DEPLOYMENT_TOPOLOGY, ("services",)),
        ("PATCH_STAGE_PATH", "数据库 Home 安装前需要确认数据库软件介质目录。", RunbookFactSource.POLICY_TEMPLATE, ("db_home",)),
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


def _release_candidate(raw_value: object) -> tuple[str, int]:
    """把 Target/数据库版本转换为 Oracle 安装介质发行标签。"""
    value_text = str(raw_value or "").strip()
    if not value_text:
        return "", 0
    named = re.search(r"(?i)(\d+)\s*(ai|c)\b", value_text)
    if named:
        return f"{int(named.group(1))}{named.group(2).lower()}", 100
    numbers = [int(item) for item in re.findall(r"\d+", value_text)]
    if not numbers:
        return "", 0
    major = numbers[0]
    minor = numbers[1] if len(numbers) > 1 else None
    if major == 23 and minor == 26:
        return "26ai", 95
    if major >= 23:
        return f"{major}ai", 80 if minor is not None else 70
    return f"{major}c", 80 if minor is not None else 70


def _resolve_release(
    context: dict[str, Any],
    identity: dict[str, Any],
    facts: dict[str, Any],
) -> tuple[str, str, str]:
    """优先使用 Target 版本，并用 V$INSTANCE 完整版本纠正 26ai 映射。"""
    configured = (
        value(context, {}, "VERSION")
        or str(context.get("configured_version") or "").strip()
    )
    candidates = (
        ("TARGET_CONFIGURATION", configured),
        ("V$INSTANCE.VERSION_FULL", identity.get("version")),
        ("V$INSTANCE.VERSION", facts.get("version")),
        ("V$PARAMETER.COMPATIBLE", facts.get("compatible")),
    )
    resolved: list[tuple[int, int, str, str, str]] = []
    for order, (source, raw_value) in enumerate(candidates):
        label, confidence = _release_candidate(raw_value)
        if label:
            resolved.append((confidence, -order, label, source, str(raw_value)))
    if not resolved:
        return "", "UNRESOLVED", ""
    _, _, label, source, raw_value = max(resolved)
    return label, source, raw_value


def compile_rac(
    evidence: tuple[TurnEvidenceFact, ...], context: dict[str, Any]
):
    identity, _ = first_row(evidence, "db.instance.identity")
    facts, _ = first_row(evidence, "db.ha.rac_precheck")
    merged = {**identity, **facts}
    db_name = value(context, merged, "DATABASE_NAME") or value(context, merged, "database_name")
    db_unique = value(context, merged, "DB_UNIQUE_NAME") or db_name
    release_label, release_source, observed_version = _resolve_release(
        context,
        identity,
        facts,
    )
    node1 = value(context, merged, "NODE1_HOST")
    node2 = value(context, merged, "NODE2_HOST")
    scan = value(context, merged, "SCAN_NAME")
    grid_home = (
        value(context, merged, "GRID_HOME")
        or (f"/u01/app/{release_label}/grid" if release_label else "")
    )
    oracle_home = value(context, merged, "ORACLE_HOME")
    patch_stage = value(context, merged, "PATCH_STAGE_PATH") or "/stage/oracle"
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
        "network": (
            command(
                "rac.network.collect",
                "在两个节点核对主机名、网卡、路由、名称解析和时间同步",
                "hostnamectl\nhostname -f\nip -br link\nip -br address\nip route\ngetent hosts \"$(hostname -s)\"\ntimedatectl status\nchronyc tracking",
                executor=RunbookExecutor.BASH,
                run_as="root",
                node_scope=("node1", "node2"),
            ),
            command(
                "rac.network.gridsetup",
                "在 gridSetup.sh 中登记公共网、私网、VIP 和 SCAN",
                (
                    "在 Grid 安装器的 Cluster Node Information、Network Interface Usage 和 "
                    "Grid Naming Service 页面中登记两个节点；使用安装器自带的 SSH Connectivity "
                    "完成互信，选择 Public/ASM & Private 接口，并录入已在 DNS 或 hosts 中解析的 "
                    "VIP 与 SCAN。安装器预检查未通过时不得继续。"
                ),
                executor=RunbookExecutor.MANUAL,
                command_type=RunbookCommandType.MANUAL,
                run_as="grid",
                node_scope=("node1",),
            ),
        ),
        "shared_storage": (
            command(
                "rac.storage.discover",
                "在两个节点核对 GI 安装器可见的共享磁盘",
                "lsblk -e7 -o NAME,KNAME,TYPE,SIZE,FSTYPE,MOUNTPOINTS,WWN,SERIAL\nmultipath -ll || true\nfind /dev/mapper -maxdepth 1 -type l -printf '%f -> %l\\n' | sort",
                executor=RunbookExecutor.BASH,
                run_as="root",
                node_scope=("node1", "node2"),
                expected=("两个节点看到相同的共享设备、WWN、容量和多路径映射。",),
            ),
            command(
                "rac.storage.gridsetup",
                "由 GI 安装器配置 ASM Filter Driver 和 OCR 磁盘组",
                (
                    "在 gridSetup.sh 的 Create ASM Disk Group 页面选择两个节点均可见且未挂载的"
                    "共享设备，启用安装器支持的 ASM Filter Driver，并创建用于 OCR/Voting 的"
                    "+DATA 磁盘组；不要选择系统盘、本地盘或已有文件系统的设备。"
                ),
                executor=RunbookExecutor.MANUAL,
                command_type=RunbookCommandType.MANUAL,
                run_as="grid",
                node_scope=("node1",),
                risk=RunbookRiskLevel.CRITICAL,
            ),
        ),
    }
    if release_label and grid_home:
        commands["os_prepare"] = (
            command(
                "rac.os.prepare",
                "在两个目标节点安装依赖并创建 Grid 用户和目录",
                "\n".join((
                    f"dnf install -y oracle-database-preinstall-{release_label} unzip tar ksh nfs-utils device-mapper-multipath",
                    "getent group oinstall >/dev/null || groupadd oinstall",
                    "getent group dba >/dev/null || groupadd dba",
                    "getent group asmdba >/dev/null || groupadd asmdba",
                    "getent group asmoper >/dev/null || groupadd asmoper",
                    "getent group asmadmin >/dev/null || groupadd asmadmin",
                    "getent group racdba >/dev/null || groupadd racdba",
                    "id oracle >/dev/null 2>&1 || useradd -g oinstall -G dba,asmdba,racdba oracle",
                    "id grid >/dev/null 2>&1 || useradd -g oinstall -G asmadmin,asmdba,asmoper,racdba grid",
                    "usermod -a -G asmdba,racdba oracle",
                    "install -d -o grid -g oinstall -m 0775 /u01/app/grid",
                    f"install -d -o grid -g oinstall -m 0775 {grid_home}",
                    f"install -d -o root -g oinstall -m 0775 {patch_stage}",
                    "systemctl enable --now chronyd",
                    "timedatectl status",
                    "chronyc tracking",
                )),
                executor=RunbookExecutor.BASH,
                run_as="root",
                node_scope=("node1", "node2"),
                risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "rac.os.extract.grid",
                "在两个目标节点解压已经下载的 GI 安装包",
                "\n".join((
                    f"gi_archive_count=$(find {patch_stage} -maxdepth 1 -type f -iname '*grid*home*.zip' | wc -l)",
                    "test \"$gi_archive_count\" -eq 1",
                    f"gi_archive=$(find {patch_stage} -maxdepth 1 -type f -iname '*grid*home*.zip' -print)",
                    f"unzip -q \"$gi_archive\" -d {grid_home}",
                    f"chown -R grid:oinstall {grid_home}",
                    f"rpm -Uvh {grid_home}/cv/rpm/cvuqdisk-*.rpm",
                    f"test -x {grid_home}/gridSetup.sh",
                    f"test -x {grid_home}/runcluvfy.sh",
                )),
                executor=RunbookExecutor.BASH,
                run_as="root",
                node_scope=("node1", "node2"),
                risk=RunbookRiskLevel.MEDIUM,
                notes=(f"每个节点的 {patch_stage} 中必须只有一份 GI grid home ZIP。",),
            ),
        )
        commands["asm"] = (
            command(
                "rac.asm.configure",
                "使用 GI 自带 ASMCA 创建或核对 DATA 与 RECO 磁盘组",
                f"{grid_home}/bin/asmca",
                executor=RunbookExecutor.BASH,
                run_as="grid",
                node_scope=("node1",),
                risk=RunbookRiskLevel.CRITICAL,
                notes=("在 ASMCA 中使用剩余共享磁盘创建 +RECO；+DATA 已存在时只做核验。",),
            ),
            command(
                "rac.asm.verify",
                "验证 ASM 实例和磁盘组",
                f"{grid_home}/bin/srvctl status asm -detail\n{grid_home}/bin/asmcmd lsdg",
                executor=RunbookExecutor.ASMCMD,
                run_as="grid",
                node_scope=("node1",),
            ),
        )
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
            f"- 节点一：{node1 or '由 gridSetup.sh 交互登记'}\n"
            f"- 节点二：{node2 or '由 gridSetup.sh 交互登记'}\n"
            f"- SCAN：{scan or '由 gridSetup.sh 交互登记'}\n"
            f"- Oracle 发行版：{release_label or '尚未从 Target 或数据库证据解析'}\n"
            f"- 版本证据：{observed_version or '尚未取得'}（{release_source}）\n"
            f"- Grid Home：{grid_home or '待版本解析后派生'}\n"
            f"- GI 介质目录：{patch_stage}\n"
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
    if release_label and grid_home:
        verify = (
            f"{grid_home}/bin/crsctl check cluster -all\n"
            f"{grid_home}/bin/olsnodes -n -s -t\n"
            f"{grid_home}/bin/srvctl config scan\n"
            f"{grid_home}/bin/srvctl status scan\n"
            f"{grid_home}/bin/cluvfy comp healthcheck -collect cluster -bestpractice -deviations -verbose\n"
        )
        artifacts.append(GeneratedRunbookArtifact(
            artifact_id="rac.verify.cluster",
            relative_path="oracle-rac-build/grid/verify-cluster.sh",
            content="#!/usr/bin/env bash\nset -euo pipefail\n" + verify,
            media_type="text/x-shellscript",
            file_mode="0750",
            run_as="grid",
            target_path=f"{stage}/grid/verify-cluster.sh",
            description="验证两节点 CRS、SCAN、节点清单和集群健康状态。",
        ))
        commands["grid_install"] = (
            command(
                "rac.grid.configure",
                "使用 GI 安装包自带 gridSetup.sh 配置两节点集群",
                (
                    "通过支持 X11 转发的 grid 会话进入第一个节点；在安装器中选择 Configure "
                    "Oracle Grid Infrastructure for a New Cluster 和 Configure an Oracle "
                    "Standalone Cluster，完成节点、SSH、网络、SCAN、VIP、ASM 与安装前置检查。"
                ),
                executor=RunbookExecutor.MANUAL,
                command_type=RunbookCommandType.MANUAL,
                run_as="grid",
                node_scope=("node1",),
                risk=RunbookRiskLevel.CRITICAL,
            ),
            command(
                "rac.grid.install",
                "启动 Oracle Grid Infrastructure 图形安装器",
                f"test -n \"$DISPLAY\"\n{grid_home}/gridSetup.sh",
                executor=RunbookExecutor.BASH,
                run_as="grid",
                node_scope=("node1",),
                risk=RunbookRiskLevel.CRITICAL,
                notes=("在安装器显示 Execute Configuration Scripts 页面后暂停，按下一条命令执行 root 脚本。",),
            ),
            command(
                "rac.grid.root.scripts",
                "按安装器提示在两个节点依次执行 root 脚本",
                f"test -x /u01/app/oraInventory/orainstRoot.sh && /u01/app/oraInventory/orainstRoot.sh\n{grid_home}/root.sh",
                executor=RunbookExecutor.BASH,
                run_as="root",
                node_scope=("node1", "node2"),
                risk=RunbookRiskLevel.CRITICAL,
                notes=("严格按照安装器显示的节点顺序逐台执行，前一节点成功后再执行下一节点。",),
            ),
            command(
                "rac.grid.verify",
                "完成安装器后验证 Clusterware、SCAN 和集群健康",
                f"{stage}/grid/verify-cluster.sh",
                executor=RunbookExecutor.BASH,
                run_as="grid",
                node_scope=("node1",),
                artifact_ref="rac.verify.cluster",
            ),
        )
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
    compile_context = dict(context)
    implementation_parameters = dict(
        compile_context.get("implementation_parameters") or {}
    )
    if observed_version and not value(context, {}, "VERSION"):
        implementation_parameters["VERSION"] = observed_version
    compile_context["implementation_parameters"] = implementation_parameters
    return compile_profile(
        spec=_SPEC,
        evidence=evidence,
        context=compile_context,
        commands_by_phase=commands,
        artifacts=tuple(artifacts),
        derived_parameters={key: item for key, item in {
            "POLICY_TEMPLATE_ID": _SPEC.policy_id,
            "ORACLE_RELEASE": release_label,
            "ORACLE_RELEASE_SOURCE": release_source if release_label else "",
            "ORACLE_OBSERVED_VERSION": observed_version,
            "GRID_HOME": grid_home,
            "GI_MEDIA_STAGE": patch_stage,
            "DATA_DISKGROUP": "+DATA",
            "RECO_DISKGROUP": "+RECO",
            "TARGET_INSTANCE_1": f"{db_name}1" if db_name else "待数据库名称确认后派生",
            "TARGET_INSTANCE_2": f"{db_name}2" if db_name else "待数据库名称确认后派生",
        }.items() if item},
    )
