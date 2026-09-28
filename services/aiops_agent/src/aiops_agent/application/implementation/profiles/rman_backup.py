"""Oracle RMAN 标准备份体系建设实施档案。"""

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
    profile=ImplementationProfile.ORACLE_RMAN_BACKUP_BUILD,
    title="Oracle RMAN 标准备份体系建设与调度实施操作文档",
    policy_id="oracle.rman.standard-disk.v1",
    fact_tool_id="db.backup.rman_configuration",
    required_facts=(
        ("INSTANCE_NAME", "需要确认本机 OS 认证使用的 Oracle SID。", RunbookFactSource.TARGET_FACT, ("scripts", "schedule")),
        ("ORACLE_HOME", "需要由主机采集确认 Oracle Home。", RunbookFactSource.HOST_COLLECTOR, ("scripts", "schedule")),
        ("BACKUP_DEST", "需要确认备份目录及其容量、挂载和权限。", RunbookFactSource.HOST_COLLECTOR, ("capacity", "scripts", "schedule")),
    ),
    phases=(
        ("assessment", "当前备份状态与恢复风险", ("assessment",)),
        ("policy", "RPO/RTO 与备份策略", ("policy",)),
        ("prerequisites", "ARCHIVELOG、BCT、FRA 与控制文件自动备份", ("prerequisites",)),
        ("configuration", "RMAN CONFIGURE 基线", ("configuration",)),
        ("backup_jobs", "Level 0、Level 1 与归档备份", ("scripts",)),
        ("maintenance", "Crosscheck、Catalog 与清理", ("maintenance",)),
        ("validation", "备份和恢复可用性校验", ("validation",)),
        ("scheduler", "Systemd 调度部署", ("schedule",)),
        ("monitoring", "日志、退出码与监控接入", ("monitoring",)),
        ("drill", "隔离恢复演练计划", ("drill",)),
        ("capacity_phase", "容量、失败处理与日常运维", ("capacity",)),
    ),
    stop_conditions=(
        "数据库不是 ARCHIVELOG 模式，且尚未批准切换维护窗口。",
        "备份目录不是独立持久挂载、权限不正确或可用容量低于保守需求。",
        "当前数据库角色与登记的 Data Guard 备份职责不一致。",
        "RMAN、ORA 错误未被脚本转化为非零退出码。",
        "首次 Level 0 和 RESTORE VALIDATE 未成功完成。",
    ),
)


def _artifact(
    artifact_id: str,
    path: str,
    content: str,
    *,
    mode: str = "0640",
    media_type: str = "text/plain",
    run_as: str = "oracle",
) -> GeneratedRunbookArtifact:
    stage = "/var/tmp/kbot-runbooks/oracle-rman-backup"
    return GeneratedRunbookArtifact(
        artifact_id=artifact_id,
        relative_path=f"oracle-rman-backup/{path}",
        content=content,
        media_type=media_type,
        file_mode=mode,
        run_as=run_as,
        target_path=f"{stage}/{path}",
        description=f"RMAN 备份体系文件：{path}",
    )


def compile_rman_backup(
    evidence: tuple[TurnEvidenceFact, ...], context: dict[str, Any]
):
    identity, _ = first_row(evidence, "db.instance.identity")
    facts, _ = first_row(evidence, "db.backup.rman_configuration")
    merged = {**identity, **facts}
    sid = value(context, merged, "INSTANCE_NAME") or value(context, merged, "instance_name")
    oracle_home = value(context, merged, "ORACLE_HOME")
    backup_dest = value(context, merged, "BACKUP_DEST") or value(context, merged, "db_recovery_file_dest")
    stage = "/var/tmp/kbot-runbooks/oracle-rman-backup"
    commands: dict[str, tuple] = {
        "assessment": (command(
            "rman.assessment.list",
            "核对现有配置和最近备份",
            "SHOW ALL;\nLIST BACKUP SUMMARY;\nLIST FAILURE;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
        ),),
        "prerequisites": (command(
            "rman.prerequisites.database",
            "核验归档、BCT、FRA 和数据库角色",
            "SELECT name, database_role, log_mode, open_mode, flashback_on FROM v$database;\nSELECT status, filename FROM v$block_change_tracking;\nSELECT name, value FROM v$parameter WHERE name IN ('db_recovery_file_dest','db_recovery_file_dest_size');",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
        ),),
        "validation": (command(
            "rman.validation.restore",
            "验证数据库备份可恢复性",
            "RESTORE DATABASE VALIDATE;\nRESTORE ARCHIVELOG ALL VALIDATE;\nVALIDATE DATABASE CHECK LOGICAL;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
            risk=RunbookRiskLevel.MEDIUM,
        ),),
    }
    artifacts: list[GeneratedRunbookArtifact] = []
    if sid and oracle_home and backup_dest:
        env = (
            f"ORACLE_SID={sid}\nORACLE_HOME={oracle_home}\n"
            f"BACKUP_DEST={backup_dest}\nLOG_DEST={backup_dest}/log\n"
        )
        configure = (
            "CONFIGURE RETENTION POLICY TO RECOVERY WINDOW OF 7 DAYS;\n"
            "CONFIGURE CONTROLFILE AUTOBACKUP ON;\n"
            f"CONFIGURE CONTROLFILE AUTOBACKUP FORMAT FOR DEVICE TYPE DISK TO '{backup_dest}/autobackup/%F';\n"
            "CONFIGURE DEVICE TYPE DISK PARALLELISM 2 BACKUP TYPE TO COMPRESSED BACKUPSET;\n"
            "CONFIGURE BACKUP OPTIMIZATION ON;\n"
            "CONFIGURE ARCHIVELOG DELETION POLICY TO BACKED UP 1 TIMES TO DISK;\n"
        )
        jobs = {
            "backup_level0.rman": (
                "RUN {\n  SQL 'ALTER SYSTEM ARCHIVE LOG CURRENT';\n"
                f"  BACKUP AS COMPRESSED BACKUPSET INCREMENTAL LEVEL 0 DATABASE FORMAT '{backup_dest}/level0/%d_%T_%U.bkp' TAG 'KBOT_WEEKLY_L0';\n"
                f"  BACKUP ARCHIVELOG ALL NOT BACKED UP 1 TIMES FORMAT '{backup_dest}/arch/%d_%T_%U.arc' DELETE INPUT;\n"
                "  BACKUP CURRENT CONTROLFILE;\n  BACKUP SPFILE;\n}\n"
            ),
            "backup_level1.rman": (
                "RUN {\n  SQL 'ALTER SYSTEM ARCHIVE LOG CURRENT';\n"
                f"  BACKUP AS COMPRESSED BACKUPSET INCREMENTAL LEVEL 1 CUMULATIVE DATABASE FORMAT '{backup_dest}/level1/%d_%T_%U.bkp' TAG 'KBOT_DAILY_L1';\n"
                f"  BACKUP ARCHIVELOG ALL NOT BACKED UP 1 TIMES FORMAT '{backup_dest}/arch/%d_%T_%U.arc' DELETE INPUT;\n"
                "}\n"
            ),
            "backup_archivelog.rman": (
                "RUN {\n  SQL 'ALTER SYSTEM ARCHIVE LOG CURRENT';\n"
                f"  BACKUP ARCHIVELOG ALL NOT BACKED UP 1 TIMES FORMAT '{backup_dest}/arch/%d_%T_%U.arc' DELETE INPUT;\n"
                "}\n"
            ),
            "crosscheck_cleanup.rman": (
                "CROSSCHECK BACKUP;\nCROSSCHECK ARCHIVELOG ALL;\n"
                "DELETE NOPROMPT EXPIRED BACKUP;\nDELETE NOPROMPT EXPIRED ARCHIVELOG ALL;\n"
                "DELETE NOPROMPT OBSOLETE;\n"
            ),
            "restore_validate.rman": "RESTORE DATABASE VALIDATE;\nRESTORE ARCHIVELOG ALL VALIDATE;\n",
        }
        artifacts.extend((
            _artifact("rman.env", "env.conf", env),
            _artifact("rman.configure", "rman/configure.rman", configure, media_type="text/x-rman"),
        ))
        for file_name, content in jobs.items():
            artifact_id = "rman." + file_name.removesuffix(".rman").replace("_", ".")
            artifacts.append(_artifact(
                artifact_id, f"rman/{file_name}", content, media_type="text/x-rman"
            ))
        runner = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"source {stage}/env.conf\n"
            "job_name=$1\n"
            "case $job_name in\n"
            "  level0) script=backup_level0.rman ;;\n"
            "  level1) script=backup_level1.rman ;;\n"
            "  archivelog) script=backup_archivelog.rman ;;\n"
            "  cleanup) script=crosscheck_cleanup.rman ;;\n"
            "  validate) script=restore_validate.rman ;;\n"
            "  *) echo '不支持的 RMAN 任务' >&2; exit 64 ;;\n"
            "esac\n"
            "mkdir -p \"$LOG_DEST\"\n"
            "exec 9>\"$LOG_DEST/$job_name.lock\"\n"
            "flock -n 9 || { echo '同类任务正在运行' >&2; exit 75; }\n"
            "log_file=\"$LOG_DEST/$job_name-$(date +%Y%m%d-%H%M%S).log\"\n"
            f"{oracle_home}/bin/rman target / cmdfile={stage}/rman/\"$script\" log=\"$log_file\"\n"
            "grep -Eq 'RMAN-[0-9]{5}|ORA-[0-9]{5}' \"$log_file\" && exit 1\n"
        )
        artifacts.append(_artifact(
            "rman.runner", "bin/run-rman-job.sh", runner,
            mode="0750", media_type="text/x-shellscript"
        ))
        service = (
            "[Unit]\nDescription=Oracle RMAN job %i\nAfter=network.target\n\n"
            "[Service]\nType=oneshot\nUser=oracle\nGroup=oinstall\n"
            f"ExecStart={stage}/bin/run-rman-job.sh %i\n"
        )
        artifacts.append(_artifact(
            "rman.systemd.service", "systemd/oracle-rman@.service", service,
            run_as="root"
        ))
        timers = {
            "oracle-rman-level0.timer": "Sun *-*-* 02:00:00",
            "oracle-rman-level1.timer": "Mon..Sat *-*-* 02:00:00",
            "oracle-rman-archivelog.timer": "*-*-* *:05:00",
        }
        for file_name, schedule in timers.items():
            job = file_name.removeprefix("oracle-rman-").removesuffix(".timer")
            artifacts.append(_artifact(
                "rman.timer." + job,
                f"systemd/{file_name}",
                "[Unit]\nDescription=Oracle RMAN schedule\n\n[Timer]\n"
                f"OnCalendar={schedule}\nPersistent=true\nUnit=oracle-rman@{job}.service\n\n[Install]\nWantedBy=timers.target\n",
                run_as="root",
            ))
        commands.update({
            "configuration": (command(
                "rman.configure.apply", "应用 RMAN 配置基线",
                f"{oracle_home}/bin/rman target / cmdfile={stage}/rman/configure.rman",
                executor=RunbookExecutor.RMAN, run_as="oracle",
                artifact_ref="rman.configure", risk=RunbookRiskLevel.MEDIUM,
            ),),
            "scripts": (command(
                "rman.backup.level0", "首次执行 Level 0 并核验日志",
                f"{stage}/bin/run-rman-job.sh level0",
                executor=RunbookExecutor.BASH, run_as="oracle",
                artifact_ref="rman.runner", risk=RunbookRiskLevel.MEDIUM,
            ),),
            "maintenance": (command(
                "rman.maintenance.cleanup", "执行交叉核对和策略清理",
                f"{stage}/bin/run-rman-job.sh cleanup",
                executor=RunbookExecutor.BASH, run_as="oracle",
                artifact_ref="rman.crosscheck.cleanup", risk=RunbookRiskLevel.HIGH,
            ),),
            "schedule": (command(
                "rman.schedule.install", "安装并启用 Systemd Timer",
                f"install -o root -g root -m 0644 {stage}/systemd/oracle-rman@.service /etc/systemd/system/oracle-rman@.service\n"
                f"install -o root -g root -m 0644 {stage}/systemd/oracle-rman-level0.timer /etc/systemd/system/oracle-rman-level0.timer\n"
                f"install -o root -g root -m 0644 {stage}/systemd/oracle-rman-level1.timer /etc/systemd/system/oracle-rman-level1.timer\n"
                f"install -o root -g root -m 0644 {stage}/systemd/oracle-rman-archivelog.timer /etc/systemd/system/oracle-rman-archivelog.timer\n"
                "systemctl daemon-reload\nsystemctl enable --now oracle-rman-level0.timer oracle-rman-level1.timer oracle-rman-archivelog.timer\nsystemctl list-timers 'oracle-rman-*'",
                executor=RunbookExecutor.BASH, run_as="root",
                risk=RunbookRiskLevel.MEDIUM,
            ),),
        })
    return compile_profile(
        spec=_SPEC,
        evidence=evidence,
        context=context,
        commands_by_phase=commands,
        artifacts=tuple(artifacts),
        derived_parameters={
            "POLICY_TEMPLATE_ID": _SPEC.policy_id,
            "RECOVERY_WINDOW_DAYS": "7",
            "WEEKLY_LEVEL0": "Sunday 02:00",
            "DAILY_LEVEL1_CUMULATIVE": "Monday-Saturday 02:00",
            "ARCHIVELOG_INTERVAL": "hourly",
        },
    )
