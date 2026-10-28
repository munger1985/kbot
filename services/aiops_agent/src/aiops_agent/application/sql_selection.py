"""SQL 结果行选择：引用不授予权限，身份只能由服务端不可变证据提取。"""

from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import UUID

from aiops_agent.application.diagnosis.sql_diagnosis import exact_wire
from aiops_agent.contracts.turn_answer import DbaSufficiencyAssessment, TurnEvidenceFact


def _row_hash(fact: TurnEvidenceFact, row_index: int) -> str:
    value = {"columns": [column["name"] for column in fact.columns], "row": exact_wire(fact.rows[row_index])}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def selection_ref(*, artifact_id: str, fact: TurnEvidenceFact, row_index: int) -> str:
    payload = {"v": 1, "artifact_id": artifact_id, "evidence_ref": fact.evidence_ref, "row_index": row_index, "row_sha256": _row_hash(fact, row_index)}
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")


def decode_selection(value: str) -> dict:
    try:
        data = json.loads(base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True))
        if not isinstance(data, dict) or set(data) != {"v", "artifact_id", "evidence_ref", "row_index", "row_sha256"}:
            raise ValueError
        if data["v"] != 1 or type(data["row_index"]) is not int or not 0 <= data["row_index"] < 2000:
            raise ValueError
        UUID(data["artifact_id"])
        if not isinstance(data["row_sha256"], str) or len(data["row_sha256"]) != 64:
            raise ValueError
        return data
    except (ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise ValueError("SQL 选择引用无效，请从结果列表重新选择") from exc


def subject_from_row(fact: TurnEvidenceFact, row_index: int, database_type: str) -> dict:
    row = dict(zip((str(column["name"]).lower() for column in fact.columns), fact.rows[row_index], strict=True))
    identity = {"POSTGRESQL": ("POSTGRESQL_QUERY_ID", "query_id"), "MYSQL": ("MYSQL_DIGEST", "digest"), "ORACLE": ("ORACLE_SQL_ID", "sql_id")}.get(database_type)
    if not identity or row.get(identity[1]) in (None, ""):
        raise ValueError("该记录没有可核验的 SQL 标识，请重新获取列表")
    subject = {"subject_type": identity[0], identity[1]: str(row[identity[1]])}
    dimensions = {
        "POSTGRESQL": ("database_oid", "user_oid", "top_level"),
        "MYSQL": ("schema_name",),
        "ORACLE": ("dbid", "inst_id", "container_id", "child_number", "plan_hash_value"),
    }[database_type]
    for name in dimensions:
        if name in row:
            if row[name] is None and name != "schema_name":
                continue
            subject[name] = bool(row[name]) if name == "top_level" and row[name] in (0, 1) else exact_wire(row[name])
    return subject


async def resolve_sql_selection(*, uow, reference: str, conversation, actor_id, ttl_seconds: int = 86400, now: datetime | None = None) -> dict:
    """重新校验会话、主体、Agent 版本、Target 版本、来源及行内容，不执行客户端 SQL。"""
    data = decode_selection(reference)
    artifact = await uow.runs.get_artifact(artifact_id=UUID(data["artifact_id"]))
    if artifact is None or artifact.schema_version != "DBA_SUFFICIENCY.v1":
        raise ValueError("SQL 来源证据已失效，请重新查询")
    source_run = await uow.runs.get_run_scoped(ops_run_id=artifact.ops_run_id, domain_id=conversation.domain_id)
    if source_run is None or any(str(getattr(source_run, key)) != str(getattr(conversation, key)) for key in ("agent_id", "agent_version_id", "domain_id")) or str(source_run.actor_id) != str(actor_id):
        raise PermissionError("SQL 来源不属于当前用户及 Agent")
    link = await uow.turns.get_run_link_by_ops_run_id(ops_run_id=source_run.ops_run_id)
    source_turn = await uow.turns.get_turn(domain_id=conversation.domain_id, turn_id=link.turn_id) if link else None
    if source_turn is None or str(source_turn.conversation_id) != str(conversation.conversation_id) or str(source_turn.created_by) != str(actor_id) or str(source_turn.sufficiency_artifact_id) != str(artifact.artifact_id):
        raise PermissionError("SQL 来源不属于当前会话")
    if now is None:
        now = datetime.now(UTC)
    created_at = artifact.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if not created_at <= now <= created_at + timedelta(seconds=ttl_seconds):
        raise ValueError("SQL 选择引用已过期，请重新查询")
    target = await uow.targets.get_scoped(target_id=source_run.target_id, domain_id=conversation.domain_id)
    frozen = dict(dict(source_run.plan_snapshot_json or {}).get("target") or {})
    if target is None or str(target.status) != "ENABLED" or str(frozen.get("row_version")) != str(target.row_version) or str(frozen.get("db_type")).upper() != str(target.db_type).upper():
        raise ValueError("Target 已变更或下线，请重新查询并选择")
    if not await uow.agents.version_has_target(agent_version_id=conversation.agent_version_id, target_id=target.target_id):
        raise PermissionError("Agent 已无权访问此 SQL 的 Target")
    assessment = DbaSufficiencyAssessment.model_validate(artifact.payload_json)
    fact = next((item for item in assessment.evidence if item.evidence_ref == data["evidence_ref"]), None)
    index = data["row_index"]
    if fact is None or fact.trust_level != "SOURCE_VERIFIED" or index >= len(fact.rows) or _row_hash(fact, index) != data["row_sha256"]:
        raise ValueError("SQL 结果行校验失败，请重新查询")
    subject = subject_from_row(fact, index, str(target.db_type).upper())
    selected_fact = fact.model_copy(update={
        "rows": (fact.rows[index],), "row_count": 1,
        "derivations": (), "calculation_graphs": (), "series_computations": (),
    })
    return {
        "subject_ref": subject, "target_id": str(target.target_id), "target_row_version": str(target.row_version),
        "source_turn_id": str(source_turn.turn_id), "source_run_id": str(source_run.ops_run_id),
        "source_artifact_id": str(artifact.artifact_id), "captured_at": fact.captured_at,
        "measurement_semantics": str(fact.measurement_semantics),
        "facts": [selected_fact.model_dump(mode="json")],
    }
