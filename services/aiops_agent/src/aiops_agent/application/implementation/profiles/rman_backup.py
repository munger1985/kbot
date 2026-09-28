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
        (
            "INSTANCE_NAME",
            "需要确认本机 OS 认证使用的 Oracle SID。",
            RunbookFactSource.TARGET_FACT,
            (
                "configuration", "scripts", "maintenance",
                "schedule", "monitoring", "capacity",
            ),
        ),
        (
            "ORACLE_HOME",
            "需要由主机采集确认 Oracle Home。",
            RunbookFactSource.HOST_COLLECTOR,
            (
                "configuration", "scripts", "maintenance",
                "schedule", "monitoring", "capacity",
            ),
        ),
        (
            "BACKUP_DEST",
            "需要确认备份目录及其容量、挂载和权限。",
            RunbookFactSource.HOST_COLLECTOR,
            (
                "configuration", "capacity", "scripts",
                "maintenance", "schedule", "monitoring",
            ),
        ),
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
    backup_dest = value(context, merged, "BACKUP_DEST")
    textfile_dir = (
        value(context, merged, "PROMETHEUS_TEXTFILE_DIR")
        or "/var/lib/node_exporter/textfile_collector"
    )
    prometheus_rule_dir = (
        value(context, merged, "PROMETHEUS_RULE_DIR")
        or "/etc/prometheus/rules"
    )
    log_mode = str(merged.get("log_mode") or "").upper()
    database_role = str(merged.get("database_role") or "ANY").upper()
    if database_role not in {
        "PRIMARY",
        "PHYSICAL STANDBY",
        "LOGICAL STANDBY",
        "SNAPSHOT STANDBY",
        "FAR SYNC",
    }:
        database_role = "ANY"
    stage = "/var/tmp/kbot-runbooks/oracle-rman-backup"
    commands: dict[str, tuple] = {
        "assessment": (
            command(
                "rman.assessment.database",
                "核对数据库身份、归档模式和最近 RMAN 任务",
                "SELECT name, db_unique_name, dbid, database_role, open_mode, log_mode, force_logging, flashback_on, cdb FROM v$database;\n"
                "SELECT instance_name, host_name, version, status, startup_time FROM v$instance;\n"
                "SELECT session_key, input_type, status, start_time, end_time, elapsed_seconds, input_bytes_display, output_bytes_display FROM v$rman_backup_job_details WHERE start_time >= SYSDATE - 30 ORDER BY start_time DESC FETCH FIRST 30 ROWS ONLY;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
            ),
            command(
                "rman.assessment.list",
                "核对现有配置、备份目录和失败记录",
                "SHOW ALL;\nREPORT SCHEMA;\nLIST BACKUP SUMMARY;\nLIST FAILURE;",
                executor=RunbookExecutor.RMAN,
                run_as="oracle",
            ),
        ),
        "policy": (
            command(
                "rman.policy.database",
                "取得策略计算所需的数据库与归档事实",
                "SELECT ROUND(SUM(bytes) / POWER(1024, 3), 2) AS datafile_allocated_gb FROM dba_data_files;\n"
                "SELECT TRUNC(first_time) AS archive_day, ROUND(SUM(blocks * block_size) / POWER(1024, 3), 2) AS archive_gb FROM v$archived_log WHERE first_time >= SYSDATE - 30 AND archived = 'YES' GROUP BY TRUNC(first_time) ORDER BY archive_day;\n"
                "SELECT name, space_limit, space_used, space_reclaimable, number_of_files FROM v$recovery_file_dest;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
            ),
            command(
                "rman.policy.coverage",
                "核对七天恢复窗口的备份覆盖",
                "REPORT NEED BACKUP RECOVERY WINDOW OF 7 DAYS;\nLIST BACKUP OF DATABASE;\nLIST BACKUP OF ARCHIVELOG FROM TIME 'SYSDATE-7';",
                executor=RunbookExecutor.RMAN,
                run_as="oracle",
            ),
        ),
        "prerequisites": (
            command(
                "rman.prerequisites.database",
                "核验归档、BCT、FRA 和数据库角色",
                "SELECT name, database_role, log_mode, open_mode, force_logging FROM v$database;\n"
                "SELECT status, filename, bytes FROM v$block_change_tracking;\n"
                "SELECT name, display_value FROM v$parameter WHERE name IN ('db_create_file_dest','db_recovery_file_dest','db_recovery_file_dest_size','control_file_record_keep_time') ORDER BY name;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
            ),
            command(
                "rman.prerequisites.bct",
                "在 BCT 未启用时开启块变化跟踪",
                "DECLARE\n"
                "  current_status VARCHAR2(16);\n"
                "BEGIN\n"
                "  SELECT status INTO current_status FROM v$block_change_tracking;\n"
                "  IF current_status = 'DISABLED' THEN\n"
                "    EXECUTE IMMEDIATE 'ALTER DATABASE ENABLE BLOCK CHANGE TRACKING';\n"
                "  END IF;\n"
                "END;\n/\n"
                "SELECT status, filename, bytes FROM v$block_change_tracking;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
                risk=RunbookRiskLevel.MEDIUM,
                notes=("数据库必须已配置可用的 OMF 目标；否则本命令会停止并保留原状态。",),
            ),
        ),
        "validation": (command(
            "rman.validation.restore",
            "验证数据库、控制文件、SPFILE 和归档备份可恢复性",
            "RESTORE DATABASE VALIDATE CHECK LOGICAL;\nRESTORE CONTROLFILE VALIDATE;\nRESTORE SPFILE VALIDATE;\nRESTORE ARCHIVELOG FROM TIME 'SYSDATE-7' VALIDATE;\nVALIDATE DATABASE CHECK LOGICAL;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
            risk=RunbookRiskLevel.MEDIUM,
        ),),
        "drill": (command(
            "rman.drill.restore.validate",
            "执行月度恢复介质演练",
            "RUN {\n  RESTORE DATABASE VALIDATE CHECK LOGICAL;\n  RESTORE CONTROLFILE VALIDATE;\n  RESTORE SPFILE VALIDATE;\n  RESTORE ARCHIVELOG FROM TIME 'SYSDATE-7' VALIDATE;\n}\nLIST FAILURE;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
            risk=RunbookRiskLevel.MEDIUM,
            expected=("数据库、控制文件、SPFILE 和最近七天归档均可从备份读取。",),
            notes=("真正的异机 Restore/Recover 必须使用独立的 RMAN 恢复实施档案，并冻结恢复目标与隔离环境。",),
        ),),
    }
    if log_mode == "NOARCHIVELOG":
        commands["prerequisites"] += (command(
            "rman.prerequisites.archivelog",
            "在批准维护窗口切换为 ARCHIVELOG",
            "SHUTDOWN IMMEDIATE;\nSTARTUP MOUNT;\nALTER DATABASE ARCHIVELOG;\nALTER DATABASE OPEN;\nALTER SYSTEM ARCHIVE LOG CURRENT;\nSELECT log_mode, open_mode FROM v$database;",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
            risk=RunbookRiskLevel.HIGH,
            expected=("LOG_MODE 返回 ARCHIVELOG。",),
        ),)
    artifacts: list[GeneratedRunbookArtifact] = []
    if sid and oracle_home and backup_dest:
        env = (
            f"ORACLE_SID={sid}\nORACLE_HOME={oracle_home}\n"
            f"BACKUP_DEST={backup_dest}\nLOG_DEST={backup_dest}/log\n"
            f"STATUS_DEST={backup_dest}/status\n"
            f"EXPECTED_DATABASE_ROLE='{database_role}'\n"
            + (f"PROMETHEUS_TEXTFILE_DIR={textfile_dir}\n" if textfile_dir else "")
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
            "backup_controlfile_spfile.rman": (
                "RUN {\n  BACKUP CURRENT CONTROLFILE;\n  BACKUP SPFILE;\n}\n"
            ),
            "crosscheck_cleanup.rman": (
                "CROSSCHECK BACKUP;\nCROSSCHECK ARCHIVELOG ALL;\n"
                "DELETE NOPROMPT EXPIRED BACKUP;\nDELETE NOPROMPT EXPIRED ARCHIVELOG ALL;\n"
                "DELETE NOPROMPT OBSOLETE;\n"
            ),
            "validate_database.rman": "VALIDATE DATABASE CHECK LOGICAL;\nLIST FAILURE;\n",
            "restore_validate.rman": (
                "RESTORE DATABASE VALIDATE CHECK LOGICAL;\n"
                "RESTORE CONTROLFILE VALIDATE;\nRESTORE SPFILE VALIDATE;\n"
                "RESTORE ARCHIVELOG FROM TIME 'SYSDATE-7' VALIDATE;\n"
            ),
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
            "  controlfile) script=backup_controlfile_spfile.rman ;;\n"
            "  cleanup) script=crosscheck_cleanup.rman ;;\n"
            "  validate) script=validate_database.rman ;;\n"
            "  restore-validate) script=restore_validate.rman ;;\n"
            "  *) echo '不支持的 RMAN 任务' >&2; exit 64 ;;\n"
            "esac\n"
            "current_role=$(\"$ORACLE_HOME/bin/sqlplus\" -s / as sysdba <<'SQL'\n"
            "WHENEVER SQLERROR EXIT SQL.SQLCODE\nSET HEADING OFF FEEDBACK OFF PAGES 0 VERIFY OFF ECHO OFF\n"
            "SELECT database_role FROM v$database;\nEXIT\nSQL\n)\n"
            "current_role=$(printf '%s' \"$current_role\" | xargs)\n"
            "printf '数据库角色：%s\\n' \"$current_role\"\n"
            "if [ \"$EXPECTED_DATABASE_ROLE\" != 'ANY' ] && [ \"$current_role\" != \"$EXPECTED_DATABASE_ROLE\" ]; then echo '数据库角色与文档固化事实不一致' >&2; exit 78; fi\n"
            "mkdir -p \"$BACKUP_DEST/level0\" \"$BACKUP_DEST/level1\" \"$BACKUP_DEST/arch\" \"$BACKUP_DEST/autobackup\" \"$LOG_DEST\" \"$STATUS_DEST\"\n"
            "exec 9>\"$LOG_DEST/$job_name.lock\"\n"
            "flock -n 9 || { echo '同类任务正在运行' >&2; exit 75; }\n"
            "started_epoch=$(date +%s)\n"
            "log_file=\"$LOG_DEST/$job_name-$(date +%Y%m%d-%H%M%S).log\"\n"
            "set +e\n"
            f"{oracle_home}/bin/rman target / cmdfile={stage}/rman/\"$script\" log=\"$log_file\"\n"
            "job_status=$?\nset -e\n"
            "if grep -Eq 'RMAN-[0-9]{5}|ORA-[0-9]{5}' \"$log_file\"; then job_status=1; fi\n"
            "ended_epoch=$(date +%s)\n"
            "status_tmp=\"$STATUS_DEST/$job_name.status.tmp\"\n"
            "printf 'job=%s\\nstatus=%s\\nstarted_epoch=%s\\nended_epoch=%s\\nduration_seconds=%s\\nlog_file=%s\\n' \"$job_name\" \"$job_status\" \"$started_epoch\" \"$ended_epoch\" \"$((ended_epoch-started_epoch))\" \"$log_file\" > \"$status_tmp\"\n"
            "mv \"$status_tmp\" \"$STATUS_DEST/$job_name.status\"\n"
            "exit \"$job_status\"\n"
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
            "oracle-rman-cleanup.timer": "*-*-* 04:00:00",
            "oracle-rman-restore-validate.timer": "Sun *-*-* 05:00:00",
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
        check_last_backup = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"source {stage}/env.conf\n"
            "\"$ORACLE_HOME/bin/sqlplus\" -s / as sysdba <<'SQL'\n"
            "WHENEVER SQLERROR EXIT SQL.SQLCODE\nSET PAGES 100 LINES 240 FEEDBACK ON VERIFY OFF\n"
            "SELECT session_key, input_type, status, start_time, end_time, elapsed_seconds, output_bytes_display FROM v$rman_backup_job_details WHERE start_time >= SYSDATE - 7 ORDER BY start_time DESC FETCH FIRST 20 ROWS ONLY;\n"
            "DECLARE\n  successful_jobs PLS_INTEGER;\nBEGIN\n  SELECT COUNT(*) INTO successful_jobs FROM v$rman_backup_job_details WHERE status LIKE 'COMPLETED%' AND end_time >= SYSDATE - 26/24;\n  IF successful_jobs = 0 THEN\n    RAISE_APPLICATION_ERROR(-20001, '最近 26 小时没有成功的 RMAN 备份');\n  END IF;\nEND;\n/\nEXIT\nSQL\n"
        )
        artifacts.append(_artifact(
            "rman.check.last.backup", "bin/check-last-backup.sh",
            check_last_backup, mode="0750", media_type="text/x-shellscript"
        ))
        install_schedule = (
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"install -o root -g root -m 0644 {stage}/systemd/oracle-rman@.service /etc/systemd/system/oracle-rman@.service\n"
            + "\n".join(
                f"install -o root -g root -m 0644 {stage}/systemd/{file_name} /etc/systemd/system/{file_name}"
                for file_name in timers
            )
            + "\nsystemctl daemon-reload\n"
            "systemctl enable --now oracle-rman-level0.timer oracle-rman-level1.timer oracle-rman-archivelog.timer oracle-rman-cleanup.timer oracle-rman-restore-validate.timer\n"
            "systemctl list-timers 'oracle-rman-*'\n"
        )
        artifacts.append(_artifact(
            "rman.install.schedule", "bin/install-schedule.sh",
            install_schedule, mode="0750", media_type="text/x-shellscript",
            run_as="root",
        ))
        commands.update({
            "configuration": (command(
                "rman.configure.apply", "应用 RMAN 配置基线",
                configure,
                executor=RunbookExecutor.RMAN, run_as="oracle",
                artifact_ref="rman.configure", risk=RunbookRiskLevel.MEDIUM,
            ),),
            "scripts": (
                command(
                    "rman.backup.level0", "首次执行 Level 0 全库备份",
                    f"{stage}/bin/run-rman-job.sh level0",
                    executor=RunbookExecutor.BASH, run_as="oracle",
                    artifact_ref="rman.runner", risk=RunbookRiskLevel.MEDIUM,
                ),
                command(
                    "rman.backup.level1", "执行 Level 1 累积增量备份",
                    f"{stage}/bin/run-rman-job.sh level1",
                    executor=RunbookExecutor.BASH, run_as="oracle",
                    artifact_ref="rman.runner", risk=RunbookRiskLevel.MEDIUM,
                ),
                command(
                    "rman.backup.archivelog", "执行归档日志备份",
                    f"{stage}/bin/run-rman-job.sh archivelog",
                    executor=RunbookExecutor.BASH, run_as="oracle",
                    artifact_ref="rman.runner", risk=RunbookRiskLevel.MEDIUM,
                ),
            ),
            "maintenance": (command(
                "rman.maintenance.cleanup", "执行交叉核对和策略清理",
                f"{stage}/bin/run-rman-job.sh cleanup",
                executor=RunbookExecutor.BASH, run_as="oracle",
                artifact_ref="rman.crosscheck.cleanup", risk=RunbookRiskLevel.HIGH,
            ),),
            "schedule": (command(
                "rman.schedule.install", "安装并启用 Systemd Timer",
                f"{stage}/bin/install-schedule.sh",
                executor=RunbookExecutor.BASH, run_as="root",
                artifact_ref="rman.install.schedule",
                risk=RunbookRiskLevel.MEDIUM,
            ),),
            "capacity": (
                command(
                    "rman.capacity.database", "核对数据库、FRA 和近三十天归档量",
                    "SELECT ROUND(SUM(bytes) / POWER(1024, 3), 2) AS datafile_allocated_gb FROM dba_data_files;\n"
                    "SELECT ROUND(SUM(bytes) / POWER(1024, 3), 2) AS segment_used_gb FROM dba_segments;\n"
                    "SELECT name, ROUND(space_limit / POWER(1024, 3), 2) AS limit_gb, ROUND(space_used / POWER(1024, 3), 2) AS used_gb, ROUND(space_reclaimable / POWER(1024, 3), 2) AS reclaimable_gb FROM v$recovery_file_dest;\n"
                    "SELECT TRUNC(first_time) AS archive_day, ROUND(SUM(blocks * block_size) / POWER(1024, 3), 2) AS archive_gb FROM v$archived_log WHERE first_time >= SYSDATE - 30 AND archived = 'YES' GROUP BY TRUNC(first_time) ORDER BY archive_day;",
                    executor=RunbookExecutor.SQLPLUS, run_as="SYSDBA",
                ),
                command(
                    "rman.capacity.filesystem", "核对备份挂载容量和现有占用",
                    f"df -P {backup_dest}\ndu -sh {backup_dest}\nfindmnt -T {backup_dest}",
                    executor=RunbookExecutor.BASH, run_as="oracle",
                ),
                command(
                    "rman.operations.status", "检查调度、最近结果和失败日志",
                    "systemctl list-timers 'oracle-rman-*'\n"
                    f"{stage}/bin/check-last-backup.sh\n"
                    f"find {backup_dest}/log -type f -name '*.log' -mtime -7 -print\n"
                    f"grep -ER 'RMAN-[0-9]{{5}}|ORA-[0-9]{{5}}' {backup_dest}/log || true",
                    executor=RunbookExecutor.BASH, run_as="oracle",
                    artifact_ref="rman.check.last.backup",
                ),
            ),
        })
        if textfile_dir and prometheus_rule_dir:
            prometheus_exporter = (
                "#!/usr/bin/env bash\nset -euo pipefail\n"
                f"source {stage}/env.conf\n"
                "mkdir -p \"$PROMETHEUS_TEXTFILE_DIR\"\n"
                "metric_file=\"$PROMETHEUS_TEXTFILE_DIR/oracle_rman_$ORACLE_SID.prom\"\n"
                "metric_tmp=\"$metric_file.tmp\"\n"
                "\"$ORACLE_HOME/bin/sqlplus\" -s / as sysdba > \"$metric_tmp\" <<SQL\n"
                "WHENEVER SQLERROR EXIT SQL.SQLCODE\nSET HEADING OFF FEEDBACK OFF PAGES 0 VERIFY OFF ECHO OFF TRIMSPOOL ON\n"
                f"SELECT 'kbot_oracle_rman_last_success_timestamp_seconds{{database=\"{sid}\"}} ' || NVL(TO_CHAR(MAX(CASE WHEN status LIKE 'COMPLETED%' THEN (end_time - DATE '1970-01-01') * 86400 END), 'FM9999999999999990'), '0') FROM v\\$rman_backup_job_details;\n"
                f"SELECT 'kbot_oracle_rman_last_failure_timestamp_seconds{{database=\"{sid}\"}} ' || NVL(TO_CHAR(MAX(CASE WHEN status NOT LIKE 'COMPLETED%' THEN (end_time - DATE '1970-01-01') * 86400 END), 'FM9999999999999990'), '0') FROM v\\$rman_backup_job_details;\n"
                f"SELECT 'kbot_oracle_rman_running_jobs{{database=\"{sid}\"}} ' || COUNT(*) FROM v\\$rman_status WHERE status = 'RUNNING' AND operation = 'BACKUP';\n"
                "EXIT\nSQL\n"
                "mv \"$metric_tmp\" \"$metric_file\"\n"
            )
            alert_rules = (
                "groups:\n"
                "  - name: kbot-oracle-rman\n"
                "    rules:\n"
                "      - alert: OracleRmanBackupMissing\n"
                f"        expr: absent(kbot_oracle_rman_last_success_timestamp_seconds{{database=\"{sid}\"}})\n"
                "        for: 15m\n"
                "        labels:\n          severity: critical\n"
                "        annotations:\n          summary: Oracle RMAN backup metric is missing\n"
                "      - alert: OracleRmanBackupStale\n"
                "        expr: time() - kbot_oracle_rman_last_success_timestamp_seconds > 93600\n"
                "        for: 15m\n"
                "        labels:\n          severity: critical\n"
                "        annotations:\n          summary: Oracle RMAN backup has no success in 26 hours\n"
                "      - alert: OracleRmanBackupStillRunning\n"
                "        expr: kbot_oracle_rman_running_jobs > 0\n"
                "        for: 6h\n"
                "        labels:\n          severity: warning\n"
                "        annotations:\n          summary: Oracle RMAN backup has been running for over 6 hours\n"
            )
            metric_service = (
                "[Unit]\nDescription=Export Oracle RMAN metrics for Node Exporter\nAfter=network.target\n\n"
                "[Service]\nType=oneshot\nUser=oracle\nGroup=oinstall\n"
                f"ExecStart={stage}/monitoring/prometheus-textfile.sh\n"
            )
            metric_timer = (
                "[Unit]\nDescription=Refresh Oracle RMAN metrics\n\n"
                "[Timer]\nOnBootSec=2min\nOnUnitActiveSec=5min\nPersistent=true\n"
                "Unit=oracle-rman-metrics.service\n\n[Install]\nWantedBy=timers.target\n"
            )
            artifacts.extend((
                _artifact("rman.prometheus.exporter", "monitoring/prometheus-textfile.sh", prometheus_exporter, mode="0750", media_type="text/x-shellscript"),
                _artifact("rman.prometheus.rules", "monitoring/alert-rules.yml", alert_rules, media_type="application/yaml", run_as="root"),
                _artifact("rman.prometheus.service", "systemd/oracle-rman-metrics.service", metric_service, run_as="root"),
                _artifact("rman.prometheus.timer", "systemd/oracle-rman-metrics.timer", metric_timer, run_as="root"),
            ))
            commands["monitoring"] = (
                command(
                    "rman.monitoring.exporter", "在数据库主机安装 RMAN 状态采集",
                    f"install -d -o oracle -g oinstall -m 0750 {textfile_dir}\n"
                    f"install -o root -g root -m 0644 {stage}/systemd/oracle-rman-metrics.service /etc/systemd/system/oracle-rman-metrics.service\n"
                    f"install -o root -g root -m 0644 {stage}/systemd/oracle-rman-metrics.timer /etc/systemd/system/oracle-rman-metrics.timer\n"
                    "systemctl daemon-reload\nsystemctl enable --now oracle-rman-metrics.timer\n"
                    "systemctl start oracle-rman-metrics.service\nsystemctl status oracle-rman-metrics.service --no-pager\n"
                    f"test -s {textfile_dir}/oracle_rman_{sid}.prom",
                    executor=RunbookExecutor.BASH, run_as="root",
                    node_scope=("source",),
                    artifact_ref="rman.prometheus.exporter",
                    risk=RunbookRiskLevel.MEDIUM,
                ),
                command(
                    "rman.monitoring.rules", "在 Prometheus 主机安装并加载告警规则",
                    f"install -o root -g root -m 0644 {stage}/monitoring/alert-rules.yml {prometheus_rule_dir}/oracle-rman-backup.yml\n"
                    f"promtool check rules {prometheus_rule_dir}/oracle-rman-backup.yml\n"
                    "curl -fsS -X POST http://127.0.0.1:9090/-/reload",
                    executor=RunbookExecutor.BASH, run_as="root",
                    node_scope=("monitoring",),
                    artifact_ref="rman.prometheus.rules",
                    risk=RunbookRiskLevel.MEDIUM,
                ),
                command(
                    "rman.monitoring.verify", "检查最近备份并验证指标文件",
                    f"{stage}/bin/check-last-backup.sh\ncat {textfile_dir}/oracle_rman_{sid}.prom",
                    executor=RunbookExecutor.BASH, run_as="oracle",
                    node_scope=("source",),
                    artifact_ref="rman.check.last.backup",
                ),
            )
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
            "PROMETHEUS_TEXTFILE_DIR": textfile_dir,
            "PROMETHEUS_RULE_DIR": prometheus_rule_dir,
            "EXPECTED_DATABASE_ROLE": database_role,
        },
    )
