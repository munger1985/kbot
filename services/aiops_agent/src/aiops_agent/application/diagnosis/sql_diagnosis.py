"""将多源观察按 SQL 身份投影，并校验、呈现可追溯诊断。"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime
from typing import Any

from aiops_agent.application.diagnosis.sql_evidence_answer import _matches, _subject
from aiops_agent.contracts.sql_diagnosis import SqlDiagnosisDraft
from aiops_agent.contracts.turn_answer import TurnEvidenceFact


def exact_wire(value: Any) -> Any:
    """避免大整数和 Decimal 在浏览器或模型输入中丢失精度。"""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {str(key): exact_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_wire(item) for item in value]
    return value


def build_sql_evidence_bundle(*, task_frame: dict, database_type: str, evidence: tuple[TurnEvidenceFact, ...]) -> dict | None:
    """保留原字段和行号；没有计划、SLO 或某类指标不阻止其他原因的调查。"""
    subject_spec = _subject(task_frame, database_type)
    if subject_spec is None:
        return None
    subject, (_, identity, _) = subject_spec
    observations = []
    for fact in evidence:
        names = [str(column.get("name") or "").lower() for column in fact.columns]
        records = []
        for index, row in enumerate(fact.rows[:40]):
            if len(row) != len(names):
                continue
            mapped = dict(zip(names, row, strict=True))
            if identity in mapped:
                matches = _matches(mapped, subject, identity)
                required_dimensions = [key for key in ("database_oid", "user_oid", "top_level", "schema_name", "inst_id", "child_number", "plan_hash_value") if key in subject]
                complete = all(key in mapped for key in required_dimensions)
                role = "SUBJECT" if matches and complete else "SUBJECT_CANDIDATE" if matches else "CONTEXT"
            else:
                role = "CONTEXT"
            records.append({"row_index": index, "role": role, "fields": exact_wire(mapped)})
        observations.append({
            "evidence_ref": fact.evidence_ref,
            "source_id": fact.source_id,
            "tool_id": fact.tool_id,
            "trust_level": fact.trust_level,
            "measurement_semantics": str(fact.measurement_semantics),
            "captured_at": fact.captured_at,
            "historical": fact.source_id == "prior_turn.sql_observation",
            "columns": list(fact.columns),
            "records": records,
            "row_count": fact.row_count,
            "truncated": fact.truncated or len(fact.rows) > 40,
            "warnings": list(fact.warnings),
            "lineage": {"artifact_id": str(fact.artifact_id), "step_id": fact.step_id},
        })
    relationships = []
    # 仅对相近时点的已核验活动会话建立 SQL→会话→锁/事务关系，不能跨历史累计视图拼接。
    sessions = []
    for observation in observations:
        if observation["trust_level"] != "SOURCE_VERIFIED" or observation["measurement_semantics"] != "CURRENT_ACTIVITY":
            continue
        for record in observation["records"]:
            if record["role"] == "SUBJECT":
                for key in ("pid", "session_id", "sid"):
                    value = record["fields"].get(key)
                    if value is not None:
                        sessions.append((observation, record, str(value)))
    for observation in observations:
        if observation["trust_level"] != "SOURCE_VERIFIED" or observation["measurement_semantics"] != "CURRENT_ACTIVITY":
            continue
        for record in observation["records"]:
            if record["role"] != "CONTEXT":
                continue
            for anchor, source, session_id in sessions:
                try:
                    age = abs((datetime.fromisoformat(observation["captured_at"].replace("Z", "+00:00")) - datetime.fromisoformat(anchor["captured_at"].replace("Z", "+00:00"))).total_seconds())
                except (ValueError, TypeError):
                    continue
                dimensions = ("inst_id", "instance_id", "container_id", "database_oid")
                if age > 5 or any(key in source["fields"] and key in record["fields"] and source["fields"][key] != record["fields"][key] for key in dimensions):
                    continue
                for field in ("pid", "session_id", "sid", "blocked_pid", "waiting_pid", "blocked_session_id"):
                    if record["fields"].get(field) is not None and str(record["fields"][field]) == session_id:
                        record["role"] = "RELATED_SESSION"
                        relationships.append({
                            "kind": "SAME_OBSERVED_SESSION",
                            "anchor": {"evidence_ref": anchor["evidence_ref"], "row_index": source["row_index"], "field": identity},
                            "related": {"evidence_ref": observation["evidence_ref"], "row_index": record["row_index"], "field": field},
                            "time_skew_seconds": age,
                            "limitation": "相近时点的会话关联，不证明其他时段或对象访问路径",
                        })
    return {
        "schema_version": "SQL_EVIDENCE_BUNDLE.v1",
        "subject": exact_wire(subject),
        "database_type": database_type,
        "task": task_frame,
        "observations": observations,
        "relationships": relationships,
        "interpretation_policy": {
            "plan_is_not_universal_prerequisite": True,
            "context_is_not_sql_attribution": True,
            "same_counter_is_not_independent_confirmation": True,
            "missing_is_not_negative_evidence": True,
            "cumulative_is_not_window": True,
            "draft_is_not_execution_authority": True,
        },
    }


def validate_sql_diagnosis(draft: SqlDiagnosisDraft, bundle: dict) -> None:
    """校验实际提供给模型的行/字段，不把引用存在误称为已证明因果。"""
    records = {
        (item["evidence_ref"], row["row_index"]): (item, row)
        for item in bundle["observations"]
        for row in item["records"]
    }
    for conclusion in draft.conclusions:
        has_subject = False
        for ref in conclusion.evidence_fields:
            entry = records.get((ref.evidence_ref, ref.row_index))
            if entry is None or ref.field.lower() not in entry[1]["fields"]:
                raise ValueError("SQL 诊断引用了不存在或未投影的证据字段")
            observation, record = entry
            if conclusion.level in {"OBSERVED", "SUPPORTED"} and observation["trust_level"] != "SOURCE_VERIFIED":
                raise ValueError("未经来源核验的材料不能作为已确认事实或原因的依据")
            if record["role"] == "SUBJECT" and observation["trust_level"] == "SOURCE_VERIFIED":
                has_subject = True
        # 关联会话的原因解释还必须引用 SQL 身份锚点，不允许孤立的全库锁行自证。
        if conclusion.scope == "SUBJECT" and conclusion.level in {"OBSERVED", "SUPPORTED"} and not has_subject:
            raise ValueError("对象或数据库背景证据不能单独升级为该 SQL 已确认的事实或原因")
    if draft.goal_satisfied and not any(item.level in {"OBSERVED", "SUPPORTED"} for item in draft.conclusions):
        raise ValueError("只有假设时不能声明诊断目标已经完成")


def render_sql_diagnosis(draft: SqlDiagnosisDraft) -> tuple[str, str, tuple[str, ...]]:
    """正文不介绍 Finding 状态，建议草案不连接执行通道。"""
    levels = {"OBSERVED": "已确认", "SUPPORTED": "有证据支持", "HYPOTHESIS": "待验证", "UNKNOWN": "无法判断"}
    analysis = [draft.summary]
    refs = []
    for item in draft.conclusions:
        scope = "背景·" if item.scope == "CONTEXT" else ""
        analysis.append(f"{scope}{levels[item.level]}：{item.statement}\n\n{item.reasoning}\n\n时间口径：{item.applicable_time}")
        if item.alternatives:
            analysis.append("其他解释：" + "；".join(item.alternatives))
        if item.limitations:
            analysis.append("适用边界：" + "；".join(item.limitations))
        for field in item.evidence_fields:
            refs.append(field.evidence_ref)
    solutions = []
    for item in draft.recommendations:
        parts = [item.action, "适用条件：" + "；".join(item.preconditions),
                 "风险：" + "；".join(item.risks), "验证：" + "；".join(item.validation),
                 "回滚或停止：" + item.rollback_or_stop]
        if item.requires_approval:
            parts.append("此项为待确认/审批建议，本轮未执行。")
        if item.draft_sql:
            parts.append("待验证、待审批草案（不可直接视为已批准执行）：\n\n```sql\n" + item.draft_sql + "\n```")
        solutions.append("\n\n".join(parts))
    if draft.unresolved_questions:
        solutions.append("仍影响判断的问题：" + "；".join(draft.unresolved_questions))
    return "\n\n".join(analysis), "\n\n".join(solutions), tuple(dict.fromkeys(refs))
