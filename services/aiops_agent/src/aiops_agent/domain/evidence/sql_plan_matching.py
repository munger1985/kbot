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


def cursor_physical_match(document_physical, expected_physical, document_subject, subject):
    """Root 可读取其他容器；不能把本地连接启动代次借给 RAC 的另一个实例。"""
    if not document_physical or not expected_physical:
        return "INCOMPLETE"
    for physical, identity in ((document_physical, document_subject), (expected_physical, subject)):
        if not all(key in identity for key in ("dbid", "container_id", "inst_id")):
            return "INCOMPLETE"
        if "dbid" not in physical:
            return "INCOMPLETE"
        if not _same(physical["dbid"], identity["dbid"]):
            return "MISMATCH"
        if "container_id" not in physical:
            return "INCOMPLETE"
        if str(physical["container_id"]) != "1" and not _same(physical["container_id"], identity["container_id"]):
            return "MISMATCH"
        if "instance_number" not in physical or not _same(physical["instance_number"], identity["inst_id"]):
            return "INCOMPLETE"
    keys = ("dbid", "instance_number", "instance_started_at")
    match = _match(document_physical, expected_physical, keys)
    return "SOURCE_MATCH" if match == "DECLARED_MATCH" else match


def _near(left, right):
    try:
        a = datetime.fromisoformat(left.replace("Z", "+00:00"))
        b = datetime.fromisoformat(right.replace("Z", "+00:00"))
        return a.tzinfo is not None and b.tzinfo is not None and abs((a - b).total_seconds()) <= 300
    except (ValueError, TypeError, AttributeError):
        return False


def _bind_refs(document, subject, evidence, anchor):
    if anchor is None:
        return ()
    results = []
    for fact in evidence:
        if (fact.tool_id != "db.oracle.statement.bind_metadata" or fact.trust_level != "SOURCE_VERIFIED"
            or fact.measurement_semantics != "CURRENT_ACTIVITY" or fact.source_id == "prior_turn.sql_observation"
            or fact.truncated or fact.row_count == 0 or fact.provenance is None
            or not _near(fact.captured_at, document.observed_at)):
            continue
        names = [str(column.get("name") or "").lower() for column in fact.columns]
        if len(names) != len(set(names)):
            continue
        if not all(len(row) == len(names) and _match(dict(zip(names, row, strict=True)), subject, _SUBJECT_KEYS["ORACLE"]) == "DECLARED_MATCH" for row in fact.rows):
            continue
        physical = dict(fact.provenance.physical_identity)
        if cursor_physical_match(document.physical_identity, physical, document.subject, subject) != "SOURCE_MATCH":
            continue
        results.append(fact.evidence_ref)
        if len(results) == 4:
            break
    return tuple(results)


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
    # 每主体优先保留匹配材料；其他主体的前四份不能挤掉当前主体的有效计划。
    plans = [item for item in evidence if item.sql_plan is not None]
    plans.sort(key=lambda item: (
        item.sql_plan.database_type != database_type,
        _match(item.sql_plan.subject, subject, required) != "DECLARED_MATCH",
    ))
    for fact in plans[:4]:
        document = fact.sql_plan
        native = document.origin == "DATABASE_CURSOR"
        estimate = document.origin == "CONTROLLED_ESTIMATE"
        identity_match = _match(document.subject, subject, required) if document.database_type == database_type else "MISMATCH"
        physical_match = _match(document.physical_identity, expected_physical, _PHYSICAL_KEYS[database_type]) if document.database_type == database_type else "MISMATCH"
        if native and document.database_type == database_type:
            identity_match = "SOURCE_MATCH" if identity_match == "DECLARED_MATCH" else identity_match
            physical_match = cursor_physical_match(document.physical_identity, expected_physical, document.subject, subject)
        if estimate and document.database_type == database_type:
            from aiops_agent.domain.evidence.sql_plan_capture import source_sql_text
            from aiops_agent.playbooks import canonical_hash
            # 完整统计主体相同仍不够；Query ID/Digest碰撞、截断或文本变化不得建立计划匹配。
            source_texts = []
            if anchor is not None:
                names = [str(column.get("name") or "").lower() for column in anchor.columns]
                for row in anchor.rows[:256]:
                    if len(row) == len(names):
                        fields = dict(zip(names, row, strict=True))
                        if _match(fields, subject, required) == "DECLARED_MATCH":
                            text = source_sql_text(fields, database_type)
                            if text is not None:
                                source_texts.append(canonical_hash(text))
            if identity_match == "DECLARED_MATCH":
                identity_match = ("SOURCE_MATCH" if source_texts and all(value == document.source_sql_sha256 for value in source_texts)
                    else "MISMATCH" if source_texts else "INCOMPLETE")
            physical_match = "SOURCE_MATCH" if physical_match == "DECLARED_MATCH" else physical_match
        time_relation = "UNKNOWN"
        plan_time = document.observed_at if native or estimate else document.declared_captured_at
        if anchor and plan_time:
            try:
                age = abs((datetime.fromisoformat(anchor.captured_at.replace("Z", "+00:00")) - datetime.fromisoformat(plan_time)).total_seconds())
                time_relation = ("SOURCE_NEAR_OBSERVATION" if native or estimate else "DECLARED_NEAR_OBSERVATION") if age <= 300 else "HISTORICAL_OR_DIFFERENT_TIME"
            except (ValueError, TypeError):
                pass
        results.append(SqlPlanAssessment(evidence_ref=fact.evidence_ref, anchor_ref=anchor.evidence_ref if anchor else None, database_type=database_type,
            origin=document.origin,
            subject=subject, identity_match=identity_match, physical_match=physical_match, time_relation=time_relation,
            bind_metadata_refs=_bind_refs(document, subject, evidence, anchor) if native and identity_match == physical_match == "SOURCE_MATCH" and time_relation == "SOURCE_NEAR_OBSERVATION" else (),
            native_verification=document.native_verification if estimate else None,
            runtime_statistics=document.runtime_statistics, node_count=len(document.nodes), limitations=((
                "已核验的数据库来源仅证明缓存计划或 LAST 行源；采集时点不等于执行时点，不自证过去指定执行或参数。",
                "绑定元数据仅为位置、类型、采集标记及原生时间；不是绑定值或代表性证明。",
                "Root 的行容器需显式核验；RAC 远端实例不可借用本地连接启动代次。",
                "缺失或零启动行源不能当作实际执行；LAST 各节点也不保证来自一个可定位的完整执行。",
            ) if native else (
                "来源为独立审批的受控估算；节点仅证明采集环境和所提交参数下的优化器估算，不证明生产实际路径。",
                "原解析环境按逐项原生核验解释，未知项和总体参数代表性不补全；同主体、文本、物理身份或相近时点不能代替运行时证明。",
                "不能从估算计划生成实际行数偏差、实际耗时或根因确认；成本是优化器单位，不是毫秒。",
                "Oracle来源游标标识不是本次估算计划哈希，估算不证明参数窥探；MySQL对象名可能是别名。",
            ) if estimate else (
                "来源为用户材料；身份或时间声明一致不等于来源已核验，不可据此确认生产访问路径或原因。",
                "参数仅保留类型；代表性、环境以及与某次执行的绑定关系均未核验。",
                "估算行数偏差按引擎的每循环/每启动语义计算；零循环或零估算不计算比率。",
                "每主体最多投影四份已有计划；计划缺失或不匹配不阻断其他诊断。",
            ))))
    return tuple(results)


def replay_sql_plan_assessments(assessments, graphs, evidence):
    subjects = {(graph.database_type, tuple(sorted(graph.subject.items()))): graph.subject for graph in graphs}
    expected = []
    for (engine, _), subject in subjects.items():
        expected.extend(assess_sql_plans(database_type=engine, subject=subject, evidence=evidence))
    if tuple(assessments) != tuple(expected):
        raise ValueError("计划匹配判断与原计划、SQL 主体及来源观察不一致")
