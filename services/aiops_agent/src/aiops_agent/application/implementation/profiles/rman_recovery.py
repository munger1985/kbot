"""Oracle RMAN 恢复与灾难恢复实施档案。"""

from __future__ import annotations

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


_SPEC = ProfileSpec(
    profile=ImplementationProfile.ORACLE_RMAN_RECOVERY,
    title="Oracle RMAN 恢复、时间点恢复与灾难恢复实施操作文档",
    policy_id="oracle.rman.recovery-manual.v1",
    fact_tool_id="db.recovery.capabilities",
    required_facts=(
        ("RECOVERY_SCENARIO", "需要明确恢复场景和业务影响范围。", RunbookFactSource.EXPLICIT_USER_DECISION, ("scenario",)),
        ("BACKUP_CHAIN_STATUS", "需要冻结可用备份链和归档覆盖。", RunbookFactSource.TARGET_FACT, ("backup_chain", "restore")),
        ("INSTANCE_NAME", "需要确认恢复实例 SID。", RunbookFactSource.TARGET_FACT, ("environment", "restore")),
    ),
    phases=(
        ("scope", "故障范围与恢复目标冻结", ("scenario",)),
        ("chain", "备份链、归档覆盖与校验", ("backup_chain",)),
        ("environment_phase", "恢复环境、目录和认证准备", ("environment",)),
        ("control", "SPFILE 与控制文件恢复", ("controlfile",)),
        ("restore_phase", "数据文件 Restore 与 Recover", ("restore",)),
        ("open_phase", "数据库打开与 RESETLOGS 决策", ("open",)),
        ("validation_phase", "组件、数据、业务和备份验证", ("validation",)),
        ("handover_phase", "恢复后保护备份与交接", ("handover",)),
    ),
    stop_conditions=(
        "恢复目标时间或 SCN 未由业务和 DBA 双方冻结。",
        "备份链、控制文件/SPFILE 或所需归档日志存在缺口。",
        "恢复主机版本、补丁、字符集、字节序或存储路径不兼容。",
        "恢复过程中出现新的介质错误、块损坏或归档缺口。",
        "OPEN RESETLOGS 前未完成业务确认和不可逆边界复核。",
    ),
    manual_only=True,
)


def compile_rman_recovery(
    evidence: tuple[TurnEvidenceFact, ...], context: dict[str, Any]
):
    identity, _ = first_row(evidence, "db.instance.identity")
    facts, _ = first_row(evidence, "db.recovery.capabilities")
    merged = {**identity, **facts}
    scenario = value(context, merged, "RECOVERY_SCENARIO")
    target_time = value(context, merged, "RECOVERY_TARGET_TIME")
    target_scn = value(context, merged, "RECOVERY_TARGET_SCN")
    oracle_home = value(context, merged, "ORACLE_HOME")
    sid = value(context, merged, "INSTANCE_NAME") or value(context, merged, "instance_name")
    stage = "/var/tmp/kbot-runbooks/oracle-rman-recovery"
    commands: dict[str, tuple] = {
        "backup_chain": (command(
            "recovery.chain.verify",
            "列出备份链和归档覆盖",
            "LIST BACKUP SUMMARY;\nLIST BACKUP OF CONTROLFILE;\nLIST BACKUP OF SPFILE;\nLIST ARCHIVELOG ALL;\nRESTORE DATABASE PREVIEW SUMMARY;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
        ),),
        "validation": (command(
            "recovery.validate.database",
            "恢复后执行数据库一致性检查",
            "SELECT name, open_mode, database_role, resetlogs_change#, resetlogs_time FROM v$database;\nSELECT status, COUNT(*) FROM v$datafile GROUP BY status;\nSELECT owner, object_type, COUNT(*) FROM dba_objects WHERE status='INVALID' GROUP BY owner, object_type ORDER BY owner, object_type;",
            executor=RunbookExecutor.SQLPLUS,
            run_as="SYSDBA",
            risk=RunbookRiskLevel.MEDIUM,
        ),),
        "handover": (command(
            "recovery.protection.backup",
            "恢复完成后立即建立新的保护基线",
            "BACKUP AS COMPRESSED BACKUPSET DATABASE PLUS ARCHIVELOG;\nBACKUP CURRENT CONTROLFILE;\nBACKUP SPFILE;",
            executor=RunbookExecutor.RMAN,
            run_as="oracle",
            risk=RunbookRiskLevel.MEDIUM,
        ),),
    }
    artifacts: list[GeneratedRunbookArtifact] = []
    if sid:
        env_content = (
            f"ORACLE_SID={shlex.quote(sid)}\n"
            f"ORACLE_HOME_HINT={shlex.quote(oracle_home)}\n"
            "export ORACLE_SID ORACLE_HOME_HINT\n"
        )
        oracle_env_loader = (
            "#!/usr/bin/env bash\n"
            "oracle_env_loaded=0\n"
            "if [ -n \"$ORACLE_HOME_HINT\" ] "
            "&& [ -x \"$ORACLE_HOME_HINT/bin/rman\" ] "
            "&& [ -x \"$ORACLE_HOME_HINT/bin/sqlplus\" ]; then\n"
            "  ORACLE_HOME=$ORACLE_HOME_HINT\n"
            "  export ORACLE_HOME PATH=\"$ORACLE_HOME/bin:$PATH\"\n"
            "  oracle_env_loaded=1\n"
            "fi\n"
            "if [ \"$oracle_env_loaded\" -eq 0 ]; then\n"
            "  export ORAENV_ASK=NO\n"
            "  oraenv_path=$(command -v oraenv 2>/dev/null || true)\n"
            "  if [ -z \"$oraenv_path\" ] && [ -x /usr/local/bin/oraenv ]; then oraenv_path=/usr/local/bin/oraenv; fi\n"
            "  if [ -z \"$oraenv_path\" ] && [ -x /usr/bin/oraenv ]; then oraenv_path=/usr/bin/oraenv; fi\n"
            "  if [ -n \"$oraenv_path\" ]; then\n"
            "    set +e\n  set +u\n  . \"$oraenv_path\" >/dev/null\n  oraenv_status=$?\n  set -u\n  set -e\n"
            "    if [ \"$oraenv_status\" -eq 0 ] && command -v sqlplus >/dev/null && command -v rman >/dev/null; then oracle_env_loaded=1; fi\n"
            "  fi\n"
            "fi\n"
            "if [ \"$oracle_env_loaded\" -eq 0 ]; then\n"
            "  pmon_pid=$(pgrep -f \"ora_pmon_$ORACLE_SID\" | head -n 1 || true)\n"
            "  if [ -n \"$pmon_pid\" ]; then\n"
            "    oracle_binary=$(readlink -f \"/proc/$pmon_pid/exe\")\n"
            "    ORACLE_HOME=$(dirname \"$(dirname \"$oracle_binary\")\")\n"
            "    export ORACLE_HOME PATH=\"$ORACLE_HOME/bin:$PATH\"\n"
            "    oracle_env_loaded=1\n"
            "  fi\n"
            "fi\n"
            "if [ \"$oracle_env_loaded\" -eq 0 ]; then\n"
            "  echo '无法通过显式提示、oraenv 或 PMON 解析 Oracle 环境' >&2\n"
            "  return 1\n"
            "fi\n"
            "command -v sqlplus >/dev/null\n"
            "command -v rman >/dev/null\n"
            "printf 'ORACLE_SID=%s\\nORACLE_HOME=%s\\n' \"$ORACLE_SID\" \"$ORACLE_HOME\"\n"
        )
        artifacts.extend((
            GeneratedRunbookArtifact(
                artifact_id="recovery.environment",
                relative_path="oracle-rman-recovery/env.conf",
                content=env_content,
                media_type="text/plain",
                file_mode="0640",
                run_as="oracle",
                target_path=f"{stage}/env.conf",
                description="恢复实例 SID 与可选 Oracle Home 提示。",
            ),
            GeneratedRunbookArtifact(
                artifact_id="recovery.oracle.env",
                relative_path=(
                    "oracle-rman-recovery/bin/load-oracle-env.sh"
                ),
                content=oracle_env_loader,
                media_type="text/x-shellscript",
                file_mode="0750",
                run_as="oracle",
                target_path=f"{stage}/bin/load-oracle-env.sh",
                description=(
                    "通过显式提示、oraenv 或 PMON 自动解析 Oracle 环境。"
                ),
            ),
        ))
        commands["environment"] = (command(
            "recovery.environment.verify",
            "解析 Oracle 环境并验证本机 OS 认证",
            (
                f"source {stage}/env.conf\n"
                f"source {stage}/bin/load-oracle-env.sh\n"
                "id\n"
                "sqlplus -L / as sysdba <<'SQL'\n"
                "EXIT\n"
                "SQL\n"
                "rman target / <<'RMAN'\n"
                "EXIT\n"
                "RMAN"
            ),
            executor=RunbookExecutor.BASH,
            run_as="oracle",
            artifact_ref="recovery.oracle.env",
            notes=(
                "无法解析 Oracle Home 或本机 OS 认证失败时立即停止，不执行恢复。",
            ),
        ),)
    until_clause = ""
    if target_scn:
        until_clause = f"SET UNTIL SCN {target_scn};\n"
    elif target_time:
        until_clause = f"SET UNTIL TIME \"TO_DATE('{target_time}','YYYY-MM-DD HH24:MI:SS')\";\n"
    if scenario and sid:
        if scenario.upper() in {"PITR", "DATABASE_PITR", "FULL_DATABASE_PITR"} and not until_clause:
            # 不输出可执行 SET UNTIL；缺失事实由附加上下文转为阻断。
            context = {**context, "implementation_parameters": {
                **dict(context.get("implementation_parameters") or {}),
                "RECOVERY_SCENARIO": scenario,
            }}
        else:
            restore_script = (
                "RUN {\n"
                + ("  " + until_clause.replace("\n", "\n  ") if until_clause else "")
                + "  RESTORE DATABASE;\n  RECOVER DATABASE;\n}\n"
            )
            artifacts.append(GeneratedRunbookArtifact(
                artifact_id="recovery.restore.database",
                relative_path="oracle-rman-recovery/rman/restore-database.rman",
                content=restore_script,
                media_type="text/x-rman",
                file_mode="0640",
                run_as="oracle",
                target_path=f"{stage}/rman/restore-database.rman",
                description="按已冻结恢复目标执行全库 Restore/Recover。",
            ))
            commands["restore"] = (command(
                "recovery.restore.execute",
                "人工复核后执行 Restore 与 Recover",
                (
                    f"source {stage}/env.conf\n"
                    f"source {stage}/bin/load-oracle-env.sh\n"
                    f"rman target / cmdfile={stage}/rman/restore-database.rman"
                ),
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                risk=RunbookRiskLevel.CRITICAL,
                artifact_ref="recovery.restore.database",
                notes=("执行前必须确认当前实例是隔离恢复环境或已批准的故障库。",),
            ),)
            open_sql = (
                "ALTER DATABASE OPEN RESETLOGS;\n"
                if until_clause else "ALTER DATABASE OPEN;\n"
            )
            commands["open"] = (command(
                "recovery.database.open",
                "在业务确认后打开数据库",
                open_sql,
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
                risk=RunbookRiskLevel.CRITICAL,
                notes=("RESETLOGS 是不可逆边界；打开后必须立即执行新的全库备份。",),
            ),)
    extra_required = list(_SPEC.required_facts)
    if scenario.upper() in {"PITR", "DATABASE_PITR", "FULL_DATABASE_PITR"} and not (target_time or target_scn):
        extra_required.append((
            "RECOVERY_TARGET",
            "时间点恢复必须由业务确认唯一目标时间或 SCN。",
            RunbookFactSource.EXPLICIT_USER_DECISION,
            ("restore", "open"),
        ))
    spec = ProfileSpec(**{**_SPEC.__dict__, "required_facts": tuple(extra_required)})
    return compile_profile(
        spec=spec,
        evidence=evidence,
        context=context,
        commands_by_phase=commands,
        artifacts=tuple(artifacts),
        derived_parameters={
            "POLICY_TEMPLATE_ID": _SPEC.policy_id,
            "EXECUTION_MODE": "MANUAL_ONLY",
            "ORACLE_HOME_RESOLUTION": "EXPLICIT_THEN_ORAENV_THEN_PMON",
            **({"ORACLE_HOME_HINT": oracle_home} if oracle_home else {}),
            **({"RECOVERY_TARGET_TIME": target_time} if target_time else {}),
            **({"RECOVERY_TARGET_SCN": target_scn} if target_scn else {}),
        },
    )
