"""逐主体核验已有计划的声明，不以人工材料自证生产执行。"""

from datetime import datetime

from aiops_agent.contracts.sql_plan import SqlPlanAssessment
from aiops_agent.domain.evidence.sql_plan import _PHYSICAL_KEYS, _SUBJECT_KEYS
from aiops_agent.domain.sql_subjects import SQL_STATISTICS_TOOLS as _SQL_STATISTICS_TOOLS


def _same(left, right, key=""):
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if key in {"digest", "sql_id", "server_uuid"}:
        return str(left).casefold() == str(right).casefold()
    if key == "instance_started_at":
        try:
            return datetime.fromisoformat(str(left).replace("Z", "+00:00")) == datetime.fromisoformat(str(right).replace("Z", "+00:00"))
        except ValueError:
            return False
    return str(left) == str(right)


def _match(declared, expected, required):
    if any(key in declared and key in expected and not _same(declared[key], expected[key], key) for key in required):
        return "MISMATCH"
    return "DECLARED_MATCH" if all(key in declared and key in expected for key in required) else "INCOMPLETE"


def assess_sql_plans(*, database_type, subject, evidence):
    anchors = []
    required = _SUBJECT_KEYS[database_type]
    for fact in evidence:
        if fact.trust_level != "SOURCE_VERIFIED" or fact.tool_id not in _SQL_STATISTICS_TOOLS[database_type] or fact.source_id == "prior_turn.sql_observation":
            continue
        names = [str(column.get("name") or "").lower() for column in fact.columns]
        if len(names) != len(set(names)):
            continue
        for row in fact.rows[:256]:
            if len(row) == len(names):
                mapped = dict(zip(names, row, strict=True))
                if all(key in mapped and _same(mapped[key], subject[key], key) for key in required if key in subject):
                    anchors.append(fact)
                    break
    def captured_order(fact):
        try:
            stamp = datetime.fromisoformat(fact.captured_at.replace("Z", "+00:00"))
            return stamp.timestamp() if stamp.tzinfo is not None else float("-inf")
        except (ValueError, OverflowError, OSError):
            return float("-inf")
    anchors.sort(key=captured_order, reverse=True)
    anchor = anchors[0] if anchors else None
    expected_physical = dict(anchor.provenance.physical_identity) if anchor and anchor.provenance else {}
    results = []
    # 每主体最多四份计划；仍保留原始附件清单，不声称检索了所有计划。
    for fact in [item for item in evidence if item.sql_plan is not None][:4]:
        document = fact.sql_plan
        identity_match = _match(document.subject, subject, required) if document.database_type == database_type else "MISMATCH"
        physical_match = _match(document.physical_identity, expected_physical, _PHYSICAL_KEYS[database_type]) if document.database_type == database_type else "MISMATCH"
        time_relation = "UNKNOWN"
        if anchor and document.declared_captured_at:
            try:
                age = abs((datetime.fromisoformat(anchor.captured_at.replace("Z", "+00:00")) - datetime.fromisoformat(document.declared_captured_at)).total_seconds())
                time_relation = "DECLARED_NEAR_OBSERVATION" if age <= 300 else "HISTORICAL_OR_DIFFERENT_TIME"
            except (ValueError, TypeError):
                pass
        results.append(SqlPlanAssessment(evidence_ref=fact.evidence_ref, anchor_ref=anchor.evidence_ref if anchor else None, database_type=database_type,
            subject=subject, identity_match=identity_match, physical_match=physical_match, time_relation=time_relation,
            runtime_statistics=document.runtime_statistics, node_count=len(document.nodes), limitations=(
                "来源为用户材料；身份或时间声明一致不等于来源已核验，不可据此确认生产访问路径或原因。",
                "参数仅保留类型；代表性、环境以及与某次执行的绑定关系均未核验。",
                "估算行数偏差按引擎的每循环/每启动语义计算；零循环或零估算不计算比率。",
                "每主体最多投影四份已有计划；计划缺失或不匹配不阻断其他诊断。",
            )))
    return tuple(results)


def replay_sql_plan_assessments(assessments, graphs, evidence):
    subjects = {(graph.database_type, tuple(sorted(graph.subject.items()))): graph.subject for graph in graphs}
    expected = []
    for (engine, _), subject in subjects.items():
        expected.extend(assess_sql_plans(database_type=engine, subject=subject, evidence=evidence))
    if tuple(assessments) != tuple(expected):
        raise ValueError("计划匹配判断与原计划、SQL 主体及来源观察不一致")
