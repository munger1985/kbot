"""Oracle RU 与 OJVM 补丁实施档案。"""

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
    profile=ImplementationProfile.ORACLE_RU_PATCH,
    title="Oracle GI/数据库 Home RU 与 OJVM 补丁实施操作文档",
    policy_id="oracle.patch.runtime-discovery.v1",
    fact_tool_id="db.maintenance.patch_precheck",
    required_facts=(),
    phases=(
        ("scope", "范围、拓扑与滚动能力", ("scope",)),
        ("backup", "保护备份与回退准备", ("backup",)),
        ("inventory_phase", "Inventory、OPatch 与空间核验", ("inventory",)),
        ("conflict_phase", "补丁冲突分析", ("conflict",)),
        ("apply_phase", "GI/DB Home 补丁实施", ("apply",)),
        ("sql_phase", "Datapatch 与组件更新", ("datapatch",)),
        ("verify", "组件、服务与告警验证", ("verify",)),
        ("rollback_phase", "回退与交接", ("rollback",)),
    ),
    stop_conditions=(
        "实时数据库、实例、Home 或集群拓扑与文档固化事实不一致。",
        "补丁暂存目录没有且仅有一套已审批 RU，或出现多套候选介质。",
        "SHA-256 摘要、OPatch Inventory、空间或补丁冲突分析未通过。",
        "数据库保护备份、恢复校验或 Home/Inventory 回退材料不完整。",
        "RAC/GI 环境未通过 OPatchAuto Analyze，或任一节点状态异常。",
        "维护窗口、业务停写、回退决策人或变更审批尚未确认。",
    ),
)


def _artifact(
    artifact_id: str,
    path: str,
    content: str,
    *,
    media_type: str = "text/plain",
    mode: str = "0640",
    run_as: str = "oracle",
) -> GeneratedRunbookArtifact:
    stage = "/var/tmp/kbot-runbooks/oracle-ru-patch"
    return GeneratedRunbookArtifact(
        artifact_id=artifact_id,
        relative_path=f"oracle-ru-patch/{path}",
        content=content,
        media_type=media_type,
        file_mode=mode,
        run_as=run_as,
        target_path=f"{stage}/{path}",
        description=f"Oracle RU 补丁文件：{path}",
    )


def _safe_path(candidate: str) -> str:
    """只接受可安全写入 Shell 脚本的绝对补丁暂存路径。"""
    normalized = candidate.strip().rstrip("/")
    if not normalized.startswith("/"):
        return ""
    if not re.fullmatch(r"/[A-Za-z0-9_+.,%/@#=()/-]+", normalized):
        return ""
    if ".." in normalized.split("/"):
        return ""
    return normalized


def _release_candidate(raw_value: object) -> tuple[str, int]:
    """把数据库版本转换为补丁暂存目录使用的发行标签。"""
    text = str(raw_value or "").strip()
    if not text:
        return "", 0
    named = re.search(r"(?i)(\d+)\s*(ai|c)\b", text)
    if named:
        return f"{int(named.group(1))}{named.group(2).lower()}", 100
    numbers = [int(item) for item in re.findall(r"\d+", text)]
    if not numbers:
        return "", 0
    major = numbers[0]
    minor = numbers[1] if len(numbers) > 1 else None
    if major == 23 and minor == 26:
        return "26ai", 95
    if major >= 23:
        return f"{major}ai", 80
    return f"{major}c", 80


def _resolve_release(
    context: dict[str, Any],
    identity: dict[str, Any],
    facts: dict[str, Any],
) -> tuple[str, str, str]:
    """按 Target 配置、实例版本和 compatible 顺序解析 Oracle 发行标签。"""
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
            resolved.append(
                (confidence, -order, label, source, str(raw_value))
            )
    if not resolved:
        return "oracle", "UNVERIFIED_SOURCE_VERSION", ""
    _, _, label, source, raw_value = max(resolved)
    return label, source, raw_value


def compile_patch(
    evidence: tuple[TurnEvidenceFact, ...],
    context: dict[str, Any],
):
    identity, _ = first_row(evidence, "db.instance.identity")
    facts, _ = first_row(evidence, "db.maintenance.patch_precheck")
    merged = {**identity, **facts}
    release, release_source, observed_version = _resolve_release(
        context, identity, facts
    )
    database_name = (
        value(context, merged, "DB_UNIQUE_NAME")
        or value(context, merged, "DATABASE_NAME")
        or "oracle"
    )
    database_key = re.sub(
        r"[^A-Za-z0-9_.-]", "_", database_name
    ).lower()
    oracle_sid = (
        value(context, merged, "INSTANCE_NAME")
        or re.sub(r"[^A-Za-z0-9_$#]", "_", database_name).upper()
    )
    requested_stage = _safe_path(value(context, merged, "PATCH_STAGE_PATH"))
    patch_stage = (
        requested_stage
        or f"/u01/stage/oracle/patches/{release}/{database_key}"
    )
    expected_ru_id = value(context, merged, "APPROVED_RU_ID")
    cluster_database = str(
        merged.get("cluster_database") or "UNKNOWN"
    ).upper()
    stage = "/var/tmp/kbot-runbooks/oracle-ru-patch"
    quoted_patch_stage = shlex.quote(patch_stage)

    environment_script = f'''#!/usr/bin/env bash
set -euo pipefail

ORACLE_SID={shlex.quote(oracle_sid)}
DB_UNIQUE_NAME={shlex.quote(database_name)}
PATCH_STAGE={shlex.quote(patch_stage)}
EXPECTED_RU_ID={shlex.quote(expected_ru_id)}
export ORACLE_SID DB_UNIQUE_NAME PATCH_STAGE EXPECTED_RU_ID

resolve_db_home() {{
  local home_value=""
  if [ -r /etc/oratab ]; then
    home_value=$(awk -F: -v sid="$ORACLE_SID" '$1 == sid {{print $2; exit}}' /etc/oratab)
  fi
  if [ -z "$home_value" ]; then
    local pmon_pid
    pmon_pid=$(pgrep -o -f "ora_pmon_$ORACLE_SID" || true)
    if [ -n "$pmon_pid" ] && [ -e "/proc/$pmon_pid/exe" ]; then
      home_value=$(dirname "$(dirname "$(readlink -f "/proc/$pmon_pid/exe")")")
    fi
  fi
  if [ -z "$home_value" ]; then
    local oraenv_path
    oraenv_path=$(command -v oraenv 2>/dev/null || true)
    if [ -n "$oraenv_path" ]; then
      export ORAENV_ASK=NO
      set +u
      . "$oraenv_path" >/dev/null
      set -u
      home_value=$(printenv ORACLE_HOME 2>/dev/null || true)
    fi
  fi
  test -n "$home_value"
  test -x "$home_value/bin/sqlplus"
  test -x "$home_value/OPatch/opatch"
  DB_HOME="$home_value"
  export DB_HOME ORACLE_HOME="$DB_HOME"
}}

resolve_gi_home() {{
  GI_HOME=""
  if [ -r /etc/oracle/olr.loc ]; then
    GI_HOME=$(awk -F= '$1 == "crs_home" {{print $2; exit}}' /etc/oracle/olr.loc)
    GI_HOME=$(printf '%s' "$GI_HOME" | tr -d '[:space:]')
  fi
  export GI_HOME
}}

find_patch_root() {{
  local kind="$1"
  local expected=""
  if [ "$#" -ge 2 ]; then expected="$2"; fi
  local base="$PATCH_STAGE/$kind"
  test -d "$base"
  local candidate_count candidate
  candidate_count=$(find "$base" -mindepth 1 -maxdepth 1 -type d -print | wc -l)
  test "$candidate_count" -eq 1
  candidate=$(find "$base" -mindepth 1 -maxdepth 1 -type d -print | sort | head -n 1)
  find "$candidate" -type f -path '*/etc/config/inventory.xml' -print -quit \
    | grep -q .
  if [ -n "$expected" ]; then
    test "$(basename "$candidate")" = "$expected"
  fi
  printf '%s\n' "$candidate"
}}

find_optional_patch_root() {{
  local kind="$1"
  local base="$PATCH_STAGE/$kind"
  if [ ! -d "$base" ]; then return 0; fi
  local candidate_count candidate
  candidate_count=$(find "$base" -mindepth 1 -maxdepth 1 -type d -print | wc -l)
  test "$candidate_count" -le 1
  if [ "$candidate_count" -eq 1 ]; then
    candidate=$(find "$base" -mindepth 1 -maxdepth 1 -type d -print | sort | head -n 1)
    find "$candidate" -type f -path '*/etc/config/inventory.xml' \
      -print -quit | grep -q .
    printf '%s\n' "$candidate"
  fi
}}

validate_media() {{
  test -f "$PATCH_STAGE/SHA256SUMS"
  (cd "$PATCH_STAGE" && sha256sum -c SHA256SUMS)
  RU_DIR=$(find_patch_root ru "$EXPECTED_RU_ID")
  OJVM_DIR=$(find_optional_patch_root ojvm)
  export RU_DIR OJVM_DIR
}}
'''

    prepare_media_script = f'''#!/usr/bin/env bash
set -euo pipefail
. {stage}/lib/patch-env.sh
test "$(id -u)" -eq 0
command -v unzip >/dev/null
oracle_group=$(id -gn oracle)
install -d -m 0750 -o oracle -g "$oracle_group" \
  "$PATCH_STAGE/media/ru" "$PATCH_STAGE/media/ojvm" \
  "$PATCH_STAGE/ru" "$PATCH_STAGE/ojvm" "$PATCH_STAGE/logs"
test -f "$PATCH_STAGE/SHA256SUMS"
(cd "$PATCH_STAGE" && sha256sum -c SHA256SUMS)
ru_archive_count=$(find "$PATCH_STAGE/media/ru" -maxdepth 1 -type f -iname '*.zip' -print | wc -l)
ojvm_archive_count=$(find "$PATCH_STAGE/media/ojvm" -maxdepth 1 -type f -iname '*.zip' -print | wc -l)
test "$ru_archive_count" -eq 1
test "$ojvm_archive_count" -le 1
ru_archive=$(find "$PATCH_STAGE/media/ru" -maxdepth 1 -type f -iname '*.zip' -print | sort | head -n 1)
unzip -oq "$ru_archive" -d "$PATCH_STAGE/ru"
if [ "$ojvm_archive_count" -eq 1 ]; then
  ojvm_archive=$(find "$PATCH_STAGE/media/ojvm" -maxdepth 1 -type f -iname '*.zip' -print | sort | head -n 1)
  unzip -oq "$ojvm_archive" -d "$PATCH_STAGE/ojvm"
fi
chown -R oracle:"$oracle_group" \
  "$PATCH_STAGE/media" "$PATCH_STAGE/ru" "$PATCH_STAGE/ojvm" "$PATCH_STAGE/logs"
validate_media
printf 'RU_DIR=%s\nOJVM_DIR=%s\n' "$RU_DIR" "$OJVM_DIR"
'''

    inventory_script = f'''#!/usr/bin/env bash
set -euo pipefail
. {stage}/lib/patch-env.sh
test "$(id -u)" -eq 0
resolve_db_home
resolve_gi_home
validate_media
install -d -m 0750 "$PATCH_STAGE/logs"
media_kb=$(du -sk "$PATCH_STAGE/ru" "$PATCH_STAGE/ojvm" | awk '{{total += $1}} END {{print total + 0}}')
printf 'EXTRACTED_PATCH_KB=%s\n' "$media_kb"
df -Pk "$PATCH_STAGE" "$DB_HOME" /tmp
"$DB_HOME/OPatch/opatch" version
"$DB_HOME/OPatch/opatch" lsinventory -detail \
  | tee "$PATCH_STAGE/logs/db-home-inventory-before.txt"
if [ -n "$GI_HOME" ] && [ -x "$GI_HOME/OPatch/opatch" ]; then
  "$GI_HOME/OPatch/opatch" version
  "$GI_HOME/OPatch/opatch" lsinventory -detail \
    | tee "$PATCH_STAGE/logs/gi-home-inventory-before.txt"
fi
printf 'RU_DIR=%s\nOJVM_DIR=%s\nDB_HOME=%s\nGI_HOME=%s\n' \
  "$RU_DIR" "$OJVM_DIR" "$DB_HOME" "$GI_HOME"
'''

    analyze_script = f'''#!/usr/bin/env bash
set -euo pipefail
. {stage}/lib/patch-env.sh
test "$(id -u)" -eq 0
resolve_db_home
resolve_gi_home
validate_media
if [ -n "$GI_HOME" ] && [ -x "$GI_HOME/OPatch/opatchauto" ]; then
  "$GI_HOME/OPatch/opatchauto" apply "$RU_DIR" -analyze
  if [ -n "$OJVM_DIR" ]; then
    "$GI_HOME/OPatch/opatchauto" apply "$OJVM_DIR" -analyze
  fi
else
  runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/OPatch:$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/OPatch/opatch" prereq CheckConflictAgainstOHWithDetail -phBaseDir "$RU_DIR"
  if [ -n "$OJVM_DIR" ]; then
    runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/OPatch:$DB_HOME/bin:/usr/bin:/bin" \
      "$DB_HOME/OPatch/opatch" prereq CheckConflictAgainstOHWithDetail -phBaseDir "$OJVM_DIR"
  fi
fi
'''

    shutdown_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        "SHUTDOWN IMMEDIATE;\n"
        "EXIT\n"
    )
    startup_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        "STARTUP;\n"
        "SELECT name, open_mode, database_role FROM v$database;\n"
        "EXIT\n"
    )
    prepare_datapatch_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        "DECLARE\n"
        "  is_cdb VARCHAR2(3);\n"
        "BEGIN\n"
        "  SELECT cdb INTO is_cdb FROM v$database;\n"
        "  IF is_cdb = 'YES' THEN\n"
        "    EXECUTE IMMEDIATE 'ALTER PLUGGABLE DATABASE ALL OPEN';\n"
        "  END IF;\n"
        "END;\n"
        "/\n"
        "SELECT con_id, name, open_mode FROM v$pdbs ORDER BY con_id;\n"
        "EXIT\n"
    )

    apply_script = f'''#!/usr/bin/env bash
set -euo pipefail
. {stage}/lib/patch-env.sh
test "$(id -u)" -eq 0
resolve_db_home
resolve_gi_home
validate_media
if [ -n "$GI_HOME" ] && [ -x "$GI_HOME/OPatch/opatchauto" ]; then
  "$GI_HOME/OPatch/opatchauto" apply "$RU_DIR"
  if [ -n "$OJVM_DIR" ]; then
    "$GI_HOME/OPatch/opatchauto" apply "$OJVM_DIR"
  fi
else
  runuser -u oracle -- env ORACLE_SID="$ORACLE_SID" ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/sqlplus" -s / as sysdba @{stage}/sql/shutdown-database.sql
  runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/lsnrctl" stop || true
  runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/OPatch:$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/OPatch/opatch" apply -silent "$RU_DIR"
  if [ -n "$OJVM_DIR" ]; then
    runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/OPatch:$DB_HOME/bin:/usr/bin:/bin" \
      "$DB_HOME/OPatch/opatch" apply -silent "$OJVM_DIR"
  fi
  runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/lsnrctl" start
  runuser -u oracle -- env ORACLE_SID="$ORACLE_SID" ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/sqlplus" -s / as sysdba @{stage}/sql/startup-database.sql
fi
'''

    datapatch_script = f'''#!/usr/bin/env bash
set -euo pipefail
. {stage}/lib/patch-env.sh
resolve_db_home
"$DB_HOME/bin/sqlplus" -s / as sysdba @{stage}/sql/prepare-datapatch.sql
"$DB_HOME/OPatch/datapatch" -verbose
"$DB_HOME/bin/sqlplus" -s / as sysdba @"$DB_HOME/rdbms/admin/utlrp.sql"
'''

    verify_script = f'''#!/usr/bin/env bash
set -euo pipefail
. {stage}/lib/patch-env.sh
resolve_db_home
resolve_gi_home
"$DB_HOME/OPatch/opatch" lspatches
"$DB_HOME/OPatch/opatch" lsinventory -detail
if [ -n "$GI_HOME" ] && [ -x "$GI_HOME/bin/crsctl" ]; then
  "$GI_HOME/bin/crsctl" check cluster -all || "$GI_HOME/bin/crsctl" check has
  "$GI_HOME/bin/crsctl" stat res -t
else
  "$DB_HOME/bin/lsnrctl" status
fi
'''

    rollback_script = f'''#!/usr/bin/env bash
set -euo pipefail
. {stage}/lib/patch-env.sh
test "$(id -u)" -eq 0
resolve_db_home
resolve_gi_home
validate_media
if [ -n "$GI_HOME" ] && [ -x "$GI_HOME/OPatch/opatchauto" ]; then
  if [ -n "$OJVM_DIR" ]; then "$GI_HOME/OPatch/opatchauto" rollback "$OJVM_DIR"; fi
  "$GI_HOME/OPatch/opatchauto" rollback "$RU_DIR"
else
  runuser -u oracle -- env ORACLE_SID="$ORACLE_SID" ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/sqlplus" -s / as sysdba @{stage}/sql/shutdown-database.sql
  runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/lsnrctl" stop || true
  if [ -n "$OJVM_DIR" ]; then
    runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/OPatch:$DB_HOME/bin:/usr/bin:/bin" \
      "$DB_HOME/OPatch/opatch" rollback -silent -id "$(basename "$OJVM_DIR")"
  fi
  runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/OPatch:$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/OPatch/opatch" rollback -silent -id "$(basename "$RU_DIR")"
  runuser -u oracle -- env ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/lsnrctl" start
  runuser -u oracle -- env ORACLE_SID="$ORACLE_SID" ORACLE_HOME="$DB_HOME" PATH="$DB_HOME/bin:/usr/bin:/bin" \
    "$DB_HOME/bin/sqlplus" -s / as sysdba @{stage}/sql/startup-database.sql
fi
'''

    verify_sql = (
        "WHENEVER SQLERROR EXIT SQL.SQLCODE\n"
        "SET PAGESIZE 500 LINESIZE 240 TRIMSPOOL ON\n"
        "SELECT dbid, name, db_unique_name, open_mode, database_role FROM v$database;\n"
        "SELECT instance_name, host_name, version, status FROM v$instance;\n"
        "SELECT patch_id, patch_uid, patch_type, action, status, action_time, "
        "description FROM dba_registry_sqlpatch ORDER BY action_time DESC;\n"
        "SELECT comp_id, version, status FROM dba_registry ORDER BY comp_id;\n"
        "SELECT owner, object_type, COUNT(*) invalid_count FROM dba_objects "
        "WHERE status='INVALID' GROUP BY owner, object_type "
        "ORDER BY owner, object_type;\n"
        "SELECT con_id, name, open_mode, restricted FROM v$pdbs ORDER BY con_id;\n"
        "EXIT\n"
    )

    artifacts = (
        _artifact(
            "patch.environment", "lib/patch-env.sh", environment_script,
            media_type="text/x-shellscript", mode="0750", run_as="root",
        ),
        _artifact(
            "patch.prepare_media", "bin/prepare-media.sh", prepare_media_script,
            media_type="text/x-shellscript", mode="0750", run_as="root",
        ),
        _artifact(
            "patch.inventory", "bin/inventory-and-media-check.sh", inventory_script,
            media_type="text/x-shellscript", mode="0750", run_as="root",
        ),
        _artifact(
            "patch.analyze", "bin/analyze-patch.sh", analyze_script,
            media_type="text/x-shellscript", mode="0750", run_as="root",
        ),
        _artifact(
            "patch.apply", "bin/apply-patch.sh", apply_script,
            media_type="text/x-shellscript", mode="0750", run_as="root",
        ),
        _artifact(
            "patch.datapatch", "bin/run-datapatch.sh", datapatch_script,
            media_type="text/x-shellscript", mode="0750", run_as="oracle",
        ),
        _artifact(
            "patch.verify", "bin/verify-patch.sh", verify_script,
            media_type="text/x-shellscript", mode="0750", run_as="root",
        ),
        _artifact(
            "patch.rollback", "bin/rollback-patch.sh", rollback_script,
            media_type="text/x-shellscript", mode="0750", run_as="root",
        ),
        _artifact(
            "patch.shutdown", "sql/shutdown-database.sql", shutdown_sql,
            media_type="text/x-sql",
        ),
        _artifact(
            "patch.startup", "sql/startup-database.sql", startup_sql,
            media_type="text/x-sql",
        ),
        _artifact(
            "patch.prepare_datapatch", "sql/prepare-datapatch.sql",
            prepare_datapatch_sql, media_type="text/x-sql",
        ),
        _artifact(
            "patch.verify_sql", "sql/verify-patch.sql", verify_sql,
            media_type="text/x-sql",
        ),
    )

    commands: dict[str, tuple] = {
        "scope": (
            command(
                "patch.scope.database",
                "核对数据库、实例、拓扑和当前 SQL Patch",
                "SELECT name, db_unique_name, database_role, open_mode, log_mode, cdb "
                "FROM v$database;\n"
                "SELECT instance_name, host_name, version, status FROM v$instance;\n"
                "SELECT name, display_value FROM v$parameter WHERE name IN "
                "('cluster_database','compatible','spfile') ORDER BY name;\n"
                "SELECT patch_id, action, status, action_time, description "
                "FROM dba_registry_sqlpatch ORDER BY action_time DESC;",
                executor=RunbookExecutor.SQLPLUS,
                run_as="SYSDBA",
            ),
        ),
        "backup": (
            command(
                "patch.backup.media_layout",
                "创建补丁介质、解压和日志目录",
                "oracle_group=$(id -gn oracle)\n"
                f"install -d -m 0750 -o oracle -g \"$oracle_group\" {quoted_patch_stage}/media/ru\n"
                f"install -d -m 0750 -o oracle -g \"$oracle_group\" {quoted_patch_stage}/media/ojvm\n"
                f"install -d -m 0750 -o oracle -g \"$oracle_group\" {quoted_patch_stage}/ru\n"
                f"install -d -m 0750 -o oracle -g \"$oracle_group\" {quoted_patch_stage}/ojvm\n"
                f"install -d -m 0750 -o oracle -g \"$oracle_group\" {quoted_patch_stage}/logs",
                executor=RunbookExecutor.BASH,
                run_as="root",
                risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "patch.backup.stage_media",
                "放置已审批补丁压缩包和官方 SHA-256 清单",
                f"将唯一已审批 RU ZIP 保存到 {patch_stage}/media/ru/，可选的唯一 OJVM ZIP "
                f"保存到 {patch_stage}/media/ojvm/；把批准来源提供的摘要写入 "
                f"{patch_stage}/SHA256SUMS。清单路径必须相对 {patch_stage}，例如 "
                "media/ru/p12345678.zip。不得用下载后自行计算的摘要替代批准来源摘要。",
                executor=RunbookExecutor.MANUAL,
                run_as="DBA",
                risk=RunbookRiskLevel.HIGH,
            ),
            command(
                "patch.backup.prepare_media",
                "校验摘要并解压唯一 RU/OJVM 介质",
                f"{stage}/bin/prepare-media.sh",
                executor=RunbookExecutor.BASH,
                run_as="root",
                artifact_ref="patch.prepare_media",
                risk=RunbookRiskLevel.MEDIUM,
            ),
            command(
                "patch.backup.database",
                "执行补丁前保护备份和恢复校验",
                "BACKUP AS COMPRESSED BACKUPSET DATABASE PLUS ARCHIVELOG;\n"
                "BACKUP CURRENT CONTROLFILE;\nBACKUP SPFILE;\n"
                "RESTORE DATABASE VALIDATE CHECK LOGICAL;",
                executor=RunbookExecutor.RMAN,
                run_as="oracle",
                risk=RunbookRiskLevel.MEDIUM,
            ),
        ),
        "inventory": (
            command(
                "patch.inventory.execute",
                "自动定位 Home 并核验 Inventory、OPatch、介质和空间",
                f"{stage}/bin/inventory-and-media-check.sh",
                executor=RunbookExecutor.BASH,
                run_as="root",
                artifact_ref="patch.inventory",
            ),
        ),
        "conflict": (
            command(
                "patch.conflict.execute",
                "执行 OPatch 或 OPatchAuto 冲突分析",
                f"{stage}/bin/analyze-patch.sh",
                executor=RunbookExecutor.BASH,
                run_as="root",
                artifact_ref="patch.analyze",
                risk=RunbookRiskLevel.MEDIUM,
            ),
        ),
        "apply": (
            command(
                "patch.apply.execute",
                "在批准维护窗口应用唯一已审批补丁集",
                f"{stage}/bin/apply-patch.sh",
                executor=RunbookExecutor.BASH,
                run_as="root",
                artifact_ref="patch.apply",
                risk=RunbookRiskLevel.CRITICAL,
            ),
        ),
        "datapatch": (
            command(
                "patch.datapatch.execute",
                "打开 PDB、执行 Datapatch 并重编译无效对象",
                f"{stage}/bin/run-datapatch.sh",
                executor=RunbookExecutor.BASH,
                run_as="oracle",
                artifact_ref="patch.datapatch",
                risk=RunbookRiskLevel.HIGH,
            ),
        ),
        "verify": (
            command(
                "patch.verify.binary",
                "验证二进制 Inventory 和集群或监听状态",
                f"{stage}/bin/verify-patch.sh",
                executor=RunbookExecutor.BASH,
                run_as="root",
                artifact_ref="patch.verify",
            ),
            command(
                "patch.verify.database",
                "验证 SQL Patch、组件、PDB 和无效对象",
                f"sqlplus / as sysdba @{stage}/sql/verify-patch.sql",
                executor=RunbookExecutor.SQLPLUS,
                run_as="oracle",
                artifact_ref="patch.verify_sql",
            ),
        ),
        "rollback": (
            command(
                "patch.rollback.retain",
                "保留补丁证据和观察期回退材料",
                f"tar -C {quoted_patch_stage} -czf "
                f"{quoted_patch_stage}/logs/patch-evidence.tgz logs SHA256SUMS",
                executor=RunbookExecutor.BASH,
                run_as="root",
            ),
            command(
                "patch.rollback.execute",
                "仅在回退获批后执行二进制补丁回退",
                f"{stage}/bin/rollback-patch.sh\n"
                f"{stage}/bin/run-datapatch.sh\n"
                f"{stage}/bin/verify-patch.sh",
                executor=RunbookExecutor.BASH,
                run_as="root",
                artifact_ref="patch.rollback",
                risk=RunbookRiskLevel.CRITICAL,
                notes=("正常成功流程不得执行本命令；仅在已批准回退决定后使用。",),
            ),
        ),
    }

    derived_parameters = {
        "POLICY_TEMPLATE_ID": _SPEC.policy_id,
        "ORACLE_RELEASE": release,
        "ORACLE_RELEASE_SOURCE": release_source,
        "ORACLE_SID": oracle_sid,
        "DB_UNIQUE_NAME": database_name,
        "ORACLE_HOME_RESOLUTION": "ORATAB_THEN_PMON_THEN_ORAENV",
        "PATCH_STAGE_PATH": patch_stage,
        "PATCH_STAGE_PATH_REVIEW_REQUIRED": (
            "NO" if requested_stage else "YES"
        ),
        "PATCH_MEDIA_SELECTION": "EXACTLY_ONE_TOP_LEVEL_RU_VALIDATED_BY_INVENTORY",
        "PATCH_EXECUTION_MODE": "RUNTIME_GI_OPATCHAUTO_OR_DB_HOME_OPATCH",
        "CLUSTER_DATABASE_OBSERVED": cluster_database,
    }
    if observed_version:
        derived_parameters["ORACLE_OBSERVED_VERSION"] = observed_version
    if expected_ru_id:
        derived_parameters["APPROVED_RU_ID"] = expected_ru_id
        derived_parameters["RU_ID_SOURCE"] = "EXPLICIT_APPROVAL"
    else:
        derived_parameters["RU_ID_SOURCE"] = "UNIQUE_STAGE_DIRECTORY_AT_EXECUTION"

    return compile_profile(
        spec=_SPEC,
        evidence=evidence,
        context=context,
        commands_by_phase=commands,
        artifacts=artifacts,
        derived_parameters=derived_parameters,
    )
