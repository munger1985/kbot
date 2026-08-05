"""验收 KBot 4.0 按服务拆分的 Oracle 全量建库脚本。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_ROOT = ROOT / "database" / "oracle"
SERVICE_TABLES = {
    "platform_core": {
        "KBOT_PLATFORM_DOMAIN",
        "KBOT_PLATFORM_PROMPT",
        "KBOT_PLATFORM_PROMPT_VERSION",
        "KBOT_NOTIFICATION_OUTBOX",
        "KBOT_NOTIFICATION_INBOX",
        "KBOT_NOTIFICATION_PREF",
        "KBOT_WORK_ITEM",
        "KBOT_BACKGROUND_OPERATION",
        "KBOT_OPERATION_WATCH",
        "KBOT_COMPOSITION_RECEIPT",
    },
    "main_api": {
        "KBOT_MAIN_SLACK_INBOX",
        "KBOT_MAIN_SLACK_THREAD",
        "KBOT_MAIN_SLACK_DELIVERY",
    },
    "model_serving": {"KBOT_AI_MODEL"},
    "knowledge_core": {
        "KBOT_KC_COLLECTION",
        "KBOT_KC_COLLECTION_BINDING",
        "KBOT_KC_INGESTION_RECEIPT",
        "KBOT_KC_BUNDLE",
        "KBOT_KC_BUNDLE_REVISION",
        "KBOT_KC_DOCUMENT",
        "KBOT_KC_DOCUMENT_VERSION",
        "KBOT_KC_BUNDLE_REVISION_DOCUMENT",
        "KBOT_KC_PARSE_VIEW",
        "KBOT_KC_INGESTION_JOB",
        "KBOT_KC_EVIDENCE",
        "KBOT_KC_DISCOVERY_OBJECT",
        "KBOT_KC_RELATION",
        "KBOT_KC_VISUAL_ASSET",
    },
    "agent_runtime": {
        "KBOT_AGENT_DEFINITION",
        "KBOT_AGENT_RUN",
        "KBOT_AGENT_TASK",
        "KBOT_AGENT_ARTIFACT",
        "KBOT_AGENT_RUN_EVENT",
        "KBOT_AGENT_DELEGATION",
        "KBOT_AGENT_CONVERSATION",
        "KBOT_AGENT_CONVERSATION_TURN",
        "KBOT_AGENT_CONVERSATION_ITEM",
        "KBOT_AGENT_MEMORY_SNAPSHOT",
        "KBOT_AGENT_MEMORY_INDEX_PROFILE",
        "KBOT_AGENT_MEMORY_ITEM",
        "KBOT_AGENT_MEMORY_SOURCE",
        "KBOT_AGENT_MEMORY_JOB",
    },
    "data_query": {
        "KBOT_DQ_CREDENTIAL",
        "KBOT_DQ_DATA_SOURCE",
        "KBOT_DQ_SCHEMA_SNAPSHOT",
        "KBOT_DQ_SNAPSHOT_OBJECT",
        "KBOT_DQ_SEMANTIC_MODEL",
        "KBOT_DQ_MODEL_VERSION",
        "KBOT_DQ_MODEL_GEN_JOB",
        "KBOT_DQ_POLICY",
        "KBOT_DQ_AGENT_BINDING",
        "KBOT_DQ_VERIFIED_QUERY",
        "KBOT_DQ_RUN",
        "KBOT_DQ_EXECUTION",
        "KBOT_DQ_RESULT",
        "KBOT_DQ_EVENT",
        "KBOT_DQ_AUDIT",
    },
    "aiops_agent": {
        "KBOT_OPS_CREDENTIAL",
        "KBOT_OPS_TARGET",
        "KBOT_OPS_POLICY",
        "KBOT_OPS_TARGET_BINDING",
        "KBOT_OPS_MONITOR_SOURCE",
        "KBOT_OPS_TARGET_MONITOR",
        "KBOT_OPS_EVENT",
        "KBOT_OPS_ALERT",
        "KBOT_OPS_RUN",
        "KBOT_OPS_TASK",
        "KBOT_OPS_ARTIFACT",
        "KBOT_OPS_RUN_EVENT",
        "KBOT_OPS_CHANGE_PROPOSAL",
        "KBOT_OPS_HITL",
        "KBOT_OPS_APPROVAL_TOKEN",
        "KBOT_OPS_EXECUTION",
        "KBOT_OPS_INSPECTION_PLAN",
        "KBOT_OPS_INSPECTION_TARGET",
        "KBOT_OPS_INSPECTION_FIRE",
        "KBOT_OPS_REPORT",
        "KBOT_OPS_INBOX",
        "KBOT_OPS_OUTBOX",
    },
}
SERVICE_VIEWS = {
    "platform_core": {"KBOT_V_PLATFORM_DOMAIN"},
    "main_api": set(),
    "model_serving": set(),
    "knowledge_core": set(),
    "agent_runtime": set(),
    "data_query": {
        "KBOT_V_DQ_SOURCE",
        "KBOT_V_DQ_RUN",
        "KBOT_V_DQ_SCHEMA_VERSION",
    },
    "aiops_agent": {
        "KBOT_V_OPS_TARGET",
        "KBOT_V_OPS_MONITOR_SOURCE",
        "KBOT_V_OPS_POLICY",
        "KBOT_V_OPS_INSPECTION_PLAN",
        "KBOT_V_OPS_INSPECTION_FIRE",
        "KBOT_V_OPS_RUN",
        "KBOT_V_OPS_PENDING_APPROVAL",
        "KBOT_V_OPS_REPORT",
        "KBOT_V_OPS_CHAT_PENDING",
        "KBOT_V_OPS_SCHEMA_VERSION",
    },
}
FORBIDDEN_TOKENS = (
    "KBOT_MD_",
    "KBOT_BIZ_",
    "TXTCHUNK",
    "INSERT INTO",
    "MERGE INTO",
    "CREATE TABLE AS SELECT",
    "DATABASE LINK",
)
FORBIDDEN_RESOURCE_COLUMNS = {
    "AGENT_KEY",
    "COLLECTION_KEY",
    "TARGET_KEY",
    "SOURCE_KEY",
    "PLAN_KEY",
    "MODEL_KEY",
}
KC_UUID_COLUMNS = (
    "COLLECTION_ID",
    "BINDING_ID",
    "INGESTION_RECEIPT_ID",
    "BUNDLE_ID",
    "CURRENT_REVISION_ID",
    "BUNDLE_REVISION_ID",
    "DOCUMENT_ID",
    "DOCUMENT_VERSION_ID",
    "BUNDLE_REVISION_DOCUMENT_ID",
    "PARSE_VIEW_ID",
    "INGESTION_JOB_ID",
    "EVIDENCE_ID",
    "DISCOVERY_OBJECT_ID",
    "RELATION_ID",
    "SUBJECT_ID",
    "OBJECT_ID",
)
AGENT_UUID_COLUMNS = (
    "AGENT_ID",
    "RUN_ID",
    "PARENT_RUN_ID",
    "TASK_ID",
    "PARENT_TASK_ID",
    "OUTPUT_ARTIFACT_ID",
    "ARTIFACT_ID",
    "DELEGATION_ID",
    "CHILD_RUN_ID",
    "RESULT_ARTIFACT_ID",
    "CONVERSATION_ID",
    "TURN_ID",
    "USER_ITEM_ID",
    "ASSISTANT_ITEM_ID",
    "ITEM_ID",
    "SNAPSHOT_ID",
    "MEMORY_ID",
    "MEMORY_SOURCE_ID",
    "MEMORY_JOB_ID",
)
AIOPS_UUID_COLUMNS = (
    "TARGET_ID",
    "POLICY_ID",
    "BINDING_ID",
    "AGENT_ID",
    "MONITOR_SOURCE_ID",
    "TARGET_MONITOR_ID",
    "EVENT_ID",
    "ALERT_ID",
    "OPS_RUN_ID",
    "OPS_TASK_ID",
    "ARTIFACT_ID",
    "PROPOSAL_ID",
    "HITL_ID",
    "APPROVAL_TOKEN_ID",
    "EXECUTION_ID",
    "INSPECTION_PLAN_ID",
    "INSPECTION_TARGET_ID",
    "INSPECTION_FIRE_ID",
    "REPORT_ID",
    "INBOX_ID",
    "OUTBOX_ID",
)
DATA_QUERY_UUID_COLUMNS = (
    "CREDENTIAL_ID",
    "DATA_SOURCE_ID",
    "SCHEMA_SNAPSHOT_ID",
    "SCHEMA_SNAPSHOT_OBJECT_ID",
    "SEMANTIC_MODEL_ID",
    "SEMANTIC_MODEL_VERSION_ID",
    "GENERATION_JOB_ID",
    "POLICY_BINDING_ID",
    "AGENT_BINDING_ID",
    "VERIFIED_QUERY_ID",
    "DATA_QUERY_RUN_ID",
    "DATA_QUERY_EXECUTION_ID",
    "DATA_QUERY_RESULT_ID",
)
NOTIFICATION_UUID_COLUMNS = (
    "OUTBOX_ID",
    "INBOX_ID",
    "OPERATION_ID",
    "WORK_ITEM_ID",
    "OPENED_OUTBOX_ID",
    "RESOLVED_OUTBOX_ID",
    "LAST_OUTBOX_ID",
)
COMPOSITION_UUID_COLUMNS = ("RECEIPT_ID",)


def _ordered_scripts(service_dir: Path, errors: list[str]) -> list[Path]:
    scripts = sorted(service_dir.glob("[0-9][0-9][0-9]_*.sql"))
    numbers = [
        int(re.match(r"(\d+)_", script.name).group(1))
        for script in scripts
    ]
    expected = list(range(1, len(scripts) + 1))
    if not scripts:
        errors.append(f"{service_dir.name} 缺少建库脚本")
    elif numbers != expected:
        errors.append(
            f"{service_dir.name} 建库顺序为 {numbers}，预期为 {expected}"
        )
    return scripts


def main() -> int:
    errors: list[str] = []
    if (ROOT / "migrations").exists():
        errors.append("4.0 禁止保留 migrations/；请维护规范全量建库脚本")

    all_tables: dict[str, str] = {}
    all_views: dict[str, str] = {}
    service_sql: dict[str, str] = {}
    for service, expected_tables in SERVICE_TABLES.items():
        service_dir = SCHEMA_ROOT / service
        scripts = _ordered_scripts(service_dir, errors)
        combined = "\n".join(
            script.read_text(encoding="utf-8").upper()
            for script in scripts
        )
        service_sql[service] = combined
        for token in FORBIDDEN_TOKENS:
            if token in combined:
                errors.append(f"{service} 禁止出现 3.x 或数据导入语句：{token}")
        for column in sorted(FORBIDDEN_RESOURCE_COLUMNS):
            if re.search(rf"\b{column}\b", combined):
                errors.append(f"{service} 禁止出现冗余资源标识列：{column}")
        if re.search(r"\bDROP\s+(TABLE|VIEW|INDEX)\b", combined):
            errors.append(f"{service} 全量建库脚本禁止包含 DROP")

        created_tables = set(
            re.findall(r"\bCREATE\s+TABLE\s+([A-Z][A-Z0-9_]*)", combined)
        )
        missing = expected_tables - created_tables
        unexpected = created_tables - expected_tables
        if missing:
            errors.append(f"{service} 缺少表：{sorted(missing)}")
        if unexpected:
            errors.append(f"{service} 出现未登记表：{sorted(unexpected)}")
        for table in created_tables:
            owner = all_tables.setdefault(table, service)
            if owner != service:
                errors.append(f"{table} 被 {owner} 和 {service} 重复拥有")

        created_views = set(
            re.findall(
                r"\bCREATE\s+OR\s+REPLACE\s+VIEW\s+([A-Z][A-Z0-9_]*)",
                combined,
            )
        )
        expected_views = SERVICE_VIEWS[service]
        missing_views = expected_views - created_views
        unexpected_views = created_views - expected_views
        if missing_views:
            errors.append(f"{service} 缺少视图：{sorted(missing_views)}")
        if unexpected_views:
            errors.append(f"{service} 出现未登记视图：{sorted(unexpected_views)}")
        for view in created_views:
            owner = all_views.setdefault(view, service)
            if owner != service:
                errors.append(f"{view} 被 {owner} 和 {service} 重复拥有")

        constraints = re.findall(
            r"\bCONSTRAINT\s+([A-Z][A-Z0-9_]*)",
            combined,
        )
        for constraint in constraints:
            if len(constraint) > 30:
                errors.append(
                    f"{service} 的 Oracle Constraint 超过 30 字符：{constraint}"
                )
        for length in re.findall(r"\bVARCHAR2\s*\(\s*(\d+)\s+CHAR\s*\)", combined):
            if int(length) > 4000:
                errors.append(
                    f"{service} 的 VARCHAR2({length} CHAR) 超过 Oracle 上限"
                )

    kc_sql = service_sql.get("knowledge_core", "")
    if "GENERATED BY DEFAULT ON NULL AS IDENTITY" in kc_sql:
        errors.append("Knowledge Core UUID 主键禁止使用数据库自增 Identity")
    for column in KC_UUID_COLUMNS:
        if not re.search(rf"\b{column}\s+RAW\s*\(\s*16\s*\)", kc_sql):
            errors.append(f"{column} 必须至少声明一次为 UUIDv7 RAW(16)")
        if re.search(rf"\b{column}\s+NUMBER\s*\(", kc_sql):
            errors.append(f"{column} 禁止声明为 NUMBER")

    model_sql = service_sql.get("model_serving", "")
    if not re.search(r"\bMODEL_ID\s+RAW\s*\(\s*16\s*\)", model_sql):
        errors.append("Model Serving 的 MODEL_ID 必须为 UUIDv7 RAW(16)")

    platform_sql = service_sql.get("platform_core", "")
    for column in NOTIFICATION_UUID_COLUMNS:
        if not re.search(rf"\b{column}\s+RAW\s*\(\s*16\s*\)", platform_sql):
            errors.append(f"Notification 的 {column} 必须至少声明一次为 UUIDv7 RAW(16)")
    for column in COMPOSITION_UUID_COLUMNS:
        if not re.search(rf"\b{column}\s+RAW\s*\(\s*16\s*\)", platform_sql):
            errors.append(f"Composition 的 {column} 必须为 UUIDv7 RAW(16)")
    for forbidden in ("TENANT_ID", "ROLE_ID", "PERMISSION_ID", "USER_ID"):
        if re.search(rf"\b{forbidden}\b", platform_sql):
            errors.append(f"Platform Core 通知禁止依赖 {forbidden}")

    agent_sql = service_sql.get("agent_runtime", "")
    for column in AGENT_UUID_COLUMNS:
        if not re.search(rf"\b{column}\s+RAW\s*\(\s*16\s*\)", agent_sql):
            errors.append(f"Agent Runtime 的 {column} 必须为 UUIDv7 RAW(16)")
        if re.search(rf"\b{column}\s+NUMBER\s*\(", agent_sql):
            errors.append(f"Agent Runtime 的 {column} 禁止声明为 NUMBER")
    if not re.search(r"\bLEASE_TOKEN\s+RAW\s*\(\s*16\s*\)", agent_sql):
        errors.append("Agent Runtime 的 LEASE_TOKEN 必须为 128-bit RAW(16)")
    if "CONSTRAINT UK_AGENT_DELEGATION_CHILD UNIQUE" in agent_sql:
        errors.append("Delegation 子运行唯一性不得阻止 CHILD_RUN_ID 为 NULL 的记录")
    if not re.search(
        r"\bCREATE\s+UNIQUE\s+INDEX\s+UX_AGENT_DELEGATION_CHILD\b",
        agent_sql,
    ):
        errors.append("Agent Runtime 缺少 Delegation 子运行条件唯一索引")

    data_query_sql = service_sql.get("data_query", "")
    for column in DATA_QUERY_UUID_COLUMNS:
        if not re.search(rf"\b{column}\s+RAW\s*\(\s*16\s*\)", data_query_sql):
            errors.append(f"Data Query 的 {column} 必须至少声明一次为 UUIDv7 RAW(16)")
        if re.search(rf"\b{column}\s+NUMBER\s*\(", data_query_sql):
            errors.append(f"Data Query 的 {column} 禁止声明为 NUMBER")
    for index_name in ("UX_DQ_MODEL_ACTIVE", "UX_DQ_BINDING_ACTIVE", "UX_DQ_EVENT_KEY"):
        if not re.search(
            rf"\bCREATE\s+UNIQUE\s+INDEX\s+{index_name}\b",
            data_query_sql,
        ):
            errors.append(f"Data Query 缺少函数唯一索引：{index_name}")
    if "SUBJECT_SELECTOR" in data_query_sql or "ROLE_ID" in data_query_sql:
        errors.append("Data Query 禁止引入 User/Role 策略选择器")
    if "SECRET_REF" in data_query_sql:
        errors.append("Data Query 禁止保存外部 SecretRef")

    aiops_sql = service_sql.get("aiops_agent", "")
    for column in AIOPS_UUID_COLUMNS:
        if not re.search(rf"\b{column}\s+RAW\s*\(\s*16\s*\)", aiops_sql):
            errors.append(f"AIOps 的 {column} 必须至少声明一次为 UUIDv7 RAW(16)")
        if re.search(rf"\b{column}\s+NUMBER\s*\(", aiops_sql):
            errors.append(f"AIOps 的 {column} 禁止声明为 NUMBER")
    for index_name in (
        "UX_OPS_POLICY_ACTIVE",
        "UX_OPS_ALERT_ACTIVE",
        "UX_OPS_HITL_PENDING",
        "UX_OPS_REPORT_CURRENT",
        "UX_OPS_RUN_EVENT_KEY",
    ):
        if not re.search(
            rf"\bCREATE\s+UNIQUE\s+INDEX\s+{index_name}\b",
            aiops_sql,
        ):
            errors.append(f"AIOps 缺少函数唯一索引：{index_name}")
    if aiops_sql.count("DEFERRABLE INITIALLY DEFERRED") != 5:
        errors.append("AIOps 必须包含 5 个延后 Artifact 当前指针外键")
    if re.search(r"\bMODE\s+VARCHAR2\b", aiops_sql):
        errors.append("AIOps 禁止使用 Oracle 26ai 保留字 MODE 作为列名")
    if not re.search(
        r"\bSCHEDULED_FOR_UTC\s+TIMESTAMP\s*\(\s*6\s*\)"
        r"\s+GENERATED\s+ALWAYS\s+AS\s*"
        r"\(\s*SYS_EXTRACT_UTC\s*\(\s*SCHEDULED_FOR\s*\)\s*\)\s+VIRTUAL",
        aiops_sql,
    ):
        errors.append("AIOps 缺少巡检时点 UTC 唯一键虚拟列")
    for forbidden_projection in (
        "KBOT_V_OPS_TARGET AS\nSELECT\n    T.ENDPOINT_JSON",
        "KBOT_V_OPS_MONITOR_SOURCE AS\nSELECT\n    M.ENDPOINT",
        "KBOT_V_OPS_PENDING_APPROVAL AS\nSELECT\n    P.PARAMETERS_JSON",
    ):
        if forbidden_projection in aiops_sql:
            errors.append("AIOps APEX 视图暴露了受保护字段")

    for manifest_service, display_name in (
        ("platform_core", "Platform Core"),
        ("data_query", "Data Query"),
        ("aiops_agent", "AIOps"),
    ):
        manifest_path = SCHEMA_ROOT / manifest_service / "schema_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{display_name} Schema Manifest 无法读取：{exc}")
            continue
        if set(manifest.get("tables", ())) != SERVICE_TABLES[manifest_service]:
            errors.append(f"{display_name} Schema Manifest 表清单与 DDL 登记不一致")
        if set(manifest.get("views", ())) != SERVICE_VIEWS[manifest_service]:
            errors.append(f"{display_name} Schema Manifest 视图清单与 DDL 登记不一致")
        manifest_scripts = manifest.get("scripts", ())
        actual_scripts = sorted(
            (SCHEMA_ROOT / manifest_service).glob("[0-9][0-9][0-9]_*.sql")
        )
        if [item.get("name") for item in manifest_scripts] != [
            path.name for path in actual_scripts
        ]:
            errors.append(f"{display_name} Schema Manifest 脚本顺序不一致")
        else:
            for item, path in zip(manifest_scripts, actual_scripts):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if item.get("sha256") != digest:
                    errors.append(f"{display_name} Schema Manifest Hash 失配：{path.name}")

    if errors:
        print("Oracle 全量建库脚本检查失败：")
        print("\n".join(errors))
        return 1
    script_count = sum(
        len(list((SCHEMA_ROOT / service).glob("*.sql")))
        for service in SERVICE_TABLES
    )
    print(
        f"Oracle 全量建库脚本检查通过："
        f"{len(SERVICE_TABLES)} 个服务，{script_count} 个脚本"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
