"""将多源观察按 SQL 身份投影，并校验、呈现可追溯诊断。"""

from __future__ import annotations

from decimal import Decimal
from datetime import datetime
from typing import Any

from aiops_agent.application.diagnosis.sql_evidence_answer import _matches, _subject
from aiops_agent.contracts.sql_diagnosis import SqlDiagnosisDraft
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.playbooks import canonical_hash


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



_SQL_SCOPE_DIMENSIONS = {
    "POSTGRESQL": ("database_oid", "user_oid", "top_level"),
    "MYSQL": ("schema_name",),
    "ORACLE": ("dbid", "inst_id", "container_id", "child_number", "plan_hash_value"),
}
_SQL_STATISTICS_TOOLS = {
    "POSTGRESQL": ("db.postgresql.statement.details", "db.postgresql.statement.subjects", "db.postgresql.statement.top", "workload.sql.top_window"),
    "MYSQL": ("db.sql.cursor_details", "db.mysql.statement.digest.top", "workload.sql.top_window"),
    "ORACLE": ("db.sql.cursor_details", "db.sql.top_current", "workload.sql.top_window"),
}


def sql_scope_resolution(subject, identity, database_type, evidence):
    """当前完整主体可分别诊断与比较；不合并统计，不让历史或活动行扩大主体集合。"""
    tools = _SQL_STATISTICS_TOOLS[database_type]
    sources = [fact for fact in evidence if (
        fact.tool_id in tools and fact.trust_level == "SOURCE_VERIFIED"
        and fact.source_id != "prior_turn.sql_observation"
    )]
    sources.sort(key=lambda fact: (tools.index(fact.tool_id), -_capture_order(fact.captured_at)))
    candidates, seen, conflicts, truncated = [], {}, False, False
    for fact in sources:
        names = [str(column.get("name") or "").lower() for column in fact.columns]
        if len(names) != len(set(names)):
            continue
        matched_source = False
        for index, row in enumerate(fact.rows[:256]):
            if len(row) != len(names):
                continue
            mapped = dict(zip(names, row, strict=True))
            if not _matches(mapped, subject, identity):
                continue
            matched_source = True
            dimensions = {key: exact_wire(mapped[key]) for key in _SQL_SCOPE_DIMENSIONS[database_type] if key in mapped}
            signature = canonical_hash(dimensions)
            statement = next((mapped[key] for key in ("normalized_statement", "sql_text", "digest_text") if mapped.get(key)), None)
            if signature in seen:
                if statement is not None and seen[signature] is not None and statement != seen[signature]:
                    conflicts = True
                continue
            seen[signature] = statement
            required = {"POSTGRESQL": ("database_oid", "user_oid"), "MYSQL": ("schema_name",),
                        "ORACLE": ("dbid", "inst_id", "container_id", "child_number", "plan_hash_value")}[database_type]
            if database_type == "POSTGRESQL" and "top_level" in names:
                required = (*required, "top_level")
            missing = [key for key in required if key not in dimensions or dimensions[key] is None and key != "schema_name"]
            resolved = {**subject, **dimensions}
            candidates.append({"evidence_ref": fact.evidence_ref, "row_index": index, "dimensions": dimensions,
                               "subject_id": "sql-" + canonical_hash(resolved)[:24], "subject_ref": resolved,
                               "missing_dimensions": missing})
        if matched_source:
            truncated = truncated or fact.truncated or len(fact.rows) > 256
    if candidates:
        truncated = truncated or len(candidates) > 20
        comparable = len(candidates) > 1 and not truncated and not conflicts and all(not item["missing_dimensions"] for item in candidates)
        return {
            "status": "COMPARISON" if comparable else "AMBIGUOUS" if len(candidates) > 1 or conflicts else "BOUNDED" if truncated else "RESOLVED",
            "candidates": candidates[:20], "observed_candidate_count": len(candidates),
            "truncated": truncated, "identity_conflict": conflicts,
            "limitation": "仅对本轮凭据可见主体分别诊断；不同库、用户、Schema、顶层或游标/计划不合并，不把累计值比较当作窗口或效果归因。",
        }
    return {"status": "UNOBSERVED", "candidates": [], "observed_candidate_count": 0, "truncated": False}


def _capture_order(value):
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.timestamp() if parsed.tzinfo is not None else float("-inf")
    except (ValueError, OverflowError, OSError):
        return float("-inf")

def build_sql_evidence_bundle(*, task_frame: dict, database_type: str, evidence: tuple[TurnEvidenceFact, ...]) -> dict | None:
    """保留原字段和行号；没有计划、SLO 或某类指标不阻止其他原因的调查。"""
    if task_frame.get("diagnostic_profile") == "SQL_COMPARISON":
        return _comparison_evidence_bundle(task_frame=task_frame, database_type=database_type, evidence=evidence)
    subject_spec = _subject(task_frame, database_type)
    if subject_spec is None:
        return None
    subject, (_, identity, _) = subject_spec
    resolution = sql_scope_resolution(subject, identity, database_type, evidence)
    observations = []
    for fact in evidence:
        names = [str(column.get("name") or "").lower() for column in fact.columns]
        records = []
        # 将命中主体放入有界投影，避免目标位于第 41 行后时只向模型提供其他 SQL。
        indexed = list(enumerate(fact.rows[:256]))
        def matching(entry):
            return len(entry[1]) == len(names) and _matches(dict(zip(names, entry[1], strict=True)), subject, identity)
        indexed.sort(key=lambda entry: not matching(entry))
        for index, row in sorted(indexed[:40]):
            if len(row) != len(names):
                continue
            mapped = dict(zip(names, row, strict=True))
            if identity in mapped:
                matches = _matches(mapped, subject, identity)
                required_dimensions = [key for key in ("database_oid", "user_oid", "top_level", "schema_name", "inst_id", "child_number", "plan_hash_value") if key in subject]
                complete = all(key in mapped for key in required_dimensions)
                role = "SUBJECT" if matches and complete else "SUBJECT_CANDIDATE" if matches else "CONTEXT"
                if role == "SUBJECT" and resolution["status"] == "AMBIGUOUS":
                    role = "SUBJECT_CANDIDATE"
            else:
                role = "CONTEXT"
            subject_ids = [item["subject_id"] for item in resolution["candidates"]
                           if _matches(mapped, item["subject_ref"], identity)] if role == "SUBJECT" else []
            if resolution["status"] == "COMPARISON" and len(subject_ids) != 1:
                role = "SUBJECT_CANDIDATE" if _matches(mapped, subject, identity) else "CONTEXT"
            records.append({"row_index": index, "role": role, "subject_ids": subject_ids, "fields": exact_wire(mapped)})
        subject_paths = {
            f"artifact:{fact.artifact_id}#/result/top_statements/{record['row_index']}/"
            for record in records if record["role"] == "SUBJECT"
        }
        calculation_graphs = [graph for graph in fact.calculation_graphs if any(
            root.input_ref.startswith(path) for root in graph.graph.roots.values() for path in subject_paths
        )]
        observations.append({
            "evidence_ref": fact.evidence_ref,
            "source_id": fact.source_id,
            "tool_id": fact.tool_id,
            "trust_level": fact.trust_level,
            "measurement_semantics": str(fact.measurement_semantics),
            "captured_at": fact.captured_at,
            "historical": fact.source_id == "prior_turn.sql_observation" or fact.followup_run_id is not None,
            "followup_run_id": str(fact.followup_run_id) if fact.followup_run_id else None,
            "columns": list(fact.columns),
            "records": records,
            "row_count": fact.row_count,
            "truncated": fact.truncated or len(fact.rows) > 40,
            "warnings": list(fact.warnings),
            "object_access": [item.model_dump(mode="json") for item in fact.object_access],
            "pagination": fact.pagination.model_dump(mode="json") if fact.pagination is not None else None,
            "lineage": {"artifact_id": str(fact.artifact_id), "step_id": fact.step_id,
                "provenance": fact.provenance.model_dump(mode="json") if fact.provenance else None},
            "derivations": [item.model_dump(mode="json") for item in fact.derivations],
            "calculation_graphs": [item.model_dump(mode="json") for item in calculation_graphs[:4]],
            "calculation_graphs_truncated": len(calculation_graphs) > 4,
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
        "subject_resolution": resolution,
        "database_type": database_type,
        "task": task_frame,
        "observations": observations,
        "relationships": relationships,
        "object_graph": sql_object_graph_for_resolution(
            database_type=database_type, subject=subject, resolution=resolution, evidence=evidence,
        ),
        "subjects": [{**item, "object_graph": sql_object_graph_for_resolution(
            database_type=database_type, subject=item["subject_ref"],
            resolution={"status": "RESOLVED", "candidates": [item]}, evidence=evidence,
        ) if resolution["status"] in {"RESOLVED", "BOUNDED", "COMPARISON"} else None} for item in resolution["candidates"]],
        "comparison_policy": {"separate_subjects": True, "implicit_aggregation": False,
                              "cumulative_difference_is_window": False, "comparison_is_causal_effect": False},
        "interpretation_policy": {
            "plan_is_not_universal_prerequisite": True,
            "context_is_not_sql_attribution": True,
            "same_counter_is_not_independent_confirmation": True,
            "missing_is_not_negative_evidence": True,
            "cumulative_is_not_window": True,
            "draft_is_not_execution_authority": True,
            "syntactic_reference_is_not_runtime_access": True,
            "catalog_index_is_not_plan_usage": True,
        },
    }


def _comparison_evidence_bundle(*, task_frame, database_type, evidence):
    """复用单主体取证规则，共享投影预算；一项缺证不抹去其他项的可交付结果。"""
    from aiops_agent.application.diagnosis.sql_evidence_answer import sql_subject_specs
    specs = sql_subject_specs(task_frame, database_type)
    if not specs:
        return None
    parts = [build_sql_evidence_bundle(
        task_frame={**task_frame, "diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": subject},
        database_type=database_type, evidence=evidence,
    ) for subject, _ in specs]
    requests, subjects = [], []
    for (subject, _), part in zip(specs, parts, strict=True):
        resolution = part["subject_resolution"]
        status = resolution["status"]
        if status in {"RESOLVED", "COMPARISON"} and any(item["missing_dimensions"] for item in part["subjects"]):
            status = "INCOMPLETE"
        requests.append({
            "request_id": "request-" + canonical_hash(subject)[:24], "subject_ref": exact_wire(subject),
            "status": status, "truncated": resolution["truncated"],
            "identity_conflict": resolution.get("identity_conflict", False),
            "subject_ids": [item["subject_id"] for item in part["subjects"]],
        })
        subjects.extend(part["subjects"])
    overflow = len(subjects) > 20
    subjects = subjects[:20]
    allowed = {item["subject_id"] for item in subjects}
    observations = {}
    for part in parts:
        for item in part["observations"]:
            target = observations.setdefault(item["evidence_ref"], {**item, "records": [], "calculation_graphs": []})
            by_index = {row["row_index"]: row for row in target["records"]}
            for row in item["records"]:
                previous = by_index.get(row["row_index"])
                if previous is None or row["role"] in {"SUBJECT", "SUBJECT_CANDIDATE"} and previous["role"] not in {"SUBJECT", "SUBJECT_CANDIDATE"}:
                    by_index[row["row_index"]] = {**row, "subject_ids": [key for key in row["subject_ids"] if key in allowed]}
            for row in by_index.values():
                if row["role"] == "SUBJECT" and not row["subject_ids"]:
                    row["role"] = "SUBJECT_CANDIDATE"
            ranked = sorted(by_index.values(), key=lambda row: (row["role"] not in {"SUBJECT", "SUBJECT_CANDIDATE"}, row["row_index"]))
            target["truncated"] = target["truncated"] or len(ranked) > 40
            target["records"] = sorted(ranked[:40], key=lambda row: row["row_index"])
            graphs = {canonical_hash(graph): graph for graph in (*target["calculation_graphs"], *item["calculation_graphs"])}
            target["calculation_graphs_truncated"] = target["calculation_graphs_truncated"] or len(graphs) > 4
            target["calculation_graphs"] = list(graphs.values())[:4]
    for request in requests:
        if not set(request["subject_ids"]) <= allowed:
            request.update(status="BOUNDED", truncated=True)
    complete = not overflow and all(
        request["status"] in {"RESOLVED", "COMPARISON"} and not request["truncated"] and request["subject_ids"]
        and all(not item["missing_dimensions"] for item in subjects if item["subject_id"] in request["subject_ids"])
        for request in requests
    )
    return {
        **parts[0], "task": task_frame, "subject": exact_wire(task_frame["subject_ref"]),
        "requests": requests, "subjects": subjects, "object_graph": None,
        "subject_resolution": {
            "status": "COMPARISON", "candidates": subjects, "observed_candidate_count": sum(part["subject_resolution"]["observed_candidate_count"] for part in parts),
            "truncated": overflow or any(request["truncated"] for request in requests),
            "requests_complete": complete,
            "limitation": "各 SQL 按独立完整主体取证；逐行保留来源、时间与口径。缺失不能补零，累计值不能替代窗口或效果归因。",
        },
        "observations": list(observations.values()),
        "relationships": list({canonical_hash(item): item for part in parts for item in part["relationships"]}.values()),
    }


def sql_object_graph_for_resolution(*, database_type, subject, resolution, evidence):
    """先使用已消歧的完整统计主体，再建立对象关系；不隐选多个用户或游标。"""
    if resolution["status"] not in {"RESOLVED", "BOUNDED"} or len(resolution["candidates"]) != 1:
        return None
    from aiops_agent.domain.evidence.sql_object_graph import build_sql_object_graph
    keys = {"subject_type", "query_id", "sql_id", "digest", "database_oid", "user_oid", "top_level",
            "schema_name", "inst_id", "child_number", "plan_hash_value", "container_id", "dbid"}
    graph = build_sql_object_graph(database_type=database_type,
        subject={key: value for key, value in {**subject, **resolution["candidates"][0]["dimensions"]}.items() if key in keys}, evidence=evidence)
    return graph.model_dump(mode="json")


def validate_sql_diagnosis(draft: SqlDiagnosisDraft, bundle: dict) -> None:
    """校验实际提供给模型的行/字段，不把引用存在误称为已证明因果。"""
    records = {
        (item["evidence_ref"], row["row_index"]): (item, row)
        for item in bundle["observations"]
        for row in item["records"]
    }
    for conclusion in draft.conclusions:
        has_subject = False
        cited_subjects = set()
        for ref in conclusion.evidence_fields:
            entry = records.get((ref.evidence_ref, ref.row_index))
            if entry is None or ref.field.lower() not in entry[1]["fields"]:
                raise ValueError("SQL 诊断引用了不存在或未投影的证据字段")
            observation, record = entry
            if conclusion.level in {"OBSERVED", "SUPPORTED"} and observation["trust_level"] != "SOURCE_VERIFIED":
                raise ValueError("未经来源核验的材料不能作为已确认事实或原因的依据")
            if record["role"] == "SUBJECT" and observation["trust_level"] == "SOURCE_VERIFIED":
                has_subject = True
                cited_subjects.update(record.get("subject_ids", ()))
        # 关联会话的原因解释还必须引用 SQL 身份锚点，不允许孤立的全库锁行自证。
        if conclusion.scope == "SUBJECT" and conclusion.level in {"OBSERVED", "SUPPORTED"} and not has_subject:
            raise ValueError("对象或数据库背景证据不能单独升级为该 SQL 已确认的事实或原因")
        known_subjects = {item["subject_id"] for item in bundle.get("subjects", ())}
        if not set(conclusion.subject_ids) <= known_subjects:
            raise ValueError("SQL 诊断引用了未知统计主体")
        if bundle["subject_resolution"]["status"] == "COMPARISON" and conclusion.scope == "SUBJECT":
            if len(conclusion.subject_ids) != 1 or not cited_subjects <= set(conclusion.subject_ids):
                raise ValueError("多主体诊断的每条 SQL 结论必须定位单一主体；跨主体比较使用 COMPARISON 范围")
        if conclusion.scope == "COMPARISON":
            if (len(conclusion.subject_ids) < 2 or not set(conclusion.subject_ids) <= cited_subjects
                or not cited_subjects <= set(conclusion.subject_ids)):
                raise ValueError("比较判断必须引用所有声明主体各自的直接证据，不能用背景或其他主体代替")
    if draft.goal_satisfied and bundle.get("requests"):
        if not bundle["subject_resolution"]["requests_complete"]:
            raise ValueError("比较请求尚有主体未观测、歧义或截断，不能声明全部完成")
        diagnosed = {key for item in draft.conclusions if item.scope == "SUBJECT"
                     and item.level in {"OBSERVED", "SUPPORTED"} for key in item.subject_ids}
        if not {item["subject_id"] for item in bundle["subjects"]} <= diagnosed:
            raise ValueError("比较完成必须保留每个独立主体的已核验判断")
        owners = {key: index for index, request in enumerate(bundle["requests"]) for key in request["subject_ids"]}
        neighbors = {index: set() for index in range(len(bundle["requests"]))}
        for item in draft.conclusions:
            if item.scope != "COMPARISON" or item.level not in {"OBSERVED", "SUPPORTED"}:
                continue
            participating = {owners[key] for key in item.subject_ids}
            for index in participating:
                neighbors[index].update(participating - {index})
        connected, pending = set(), [0]
        while pending:
            index = pending.pop()
            if index not in connected:
                connected.add(index)
                pending.extend(neighbors[index] - connected)
        if len(connected) != len(bundle["requests"]):
            raise ValueError("比较完成必须有连接全部请求的跨 SQL 比较判断，不能只在各自标识内部比较")
    if draft.goal_satisfied and not any(item.level in {"OBSERVED", "SUPPORTED"} for item in draft.conclusions):
        raise ValueError("只有假设时不能声明诊断目标已经完成")
