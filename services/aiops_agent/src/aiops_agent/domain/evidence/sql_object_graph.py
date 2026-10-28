"""有界生成并重放 SQL 与原生目录对象的关系；不猜 search_path 或运行时扫描方式。"""

from datetime import datetime
from collections import deque

from sqlglot import exp, parse

from aiops_agent.contracts.sql_object_graph import SqlObjectGraph, SqlObjectRelationship
from aiops_agent.contracts.sql_diagnosis import EvidenceFieldReference


_TEXT_TOOLS = {
    "POSTGRESQL": {"db.postgresql.statement.details", "db.postgresql.statement.subjects", "db.postgresql.statement.top"},
    "MYSQL": {"db.sql.cursor_details", "db.mysql.statement.digest.top"},
    "ORACLE": {"db.sql.cursor_details", "db.sql.top_current"},
}
_IDENTITIES = {"POSTGRESQL": "query_id", "MYSQL": "digest", "ORACLE": "sql_id"}
_SCOPES = {"POSTGRESQL": ("system_identifier", "database_oid"),
           "MYSQL": ("server_uuid",), "ORACLE": ("dbid", "container_id")}
_DEPENDENCY_TOOL = "db.sql.object_dependencies"
_PLAN_TOOLS = {"db.sql.execution_plan", "db.oracle.statement.plan"}
_CATALOG_TOOLS = {"db.sql.object_indexes", "db.sql.object_statistics", _DEPENDENCY_TOOL}
_DIMENSIONS = ("database_oid", "user_oid", "top_level", "schema_name", "inst_id", "child_number", "plan_hash_value", "container_id", "dbid")


def explicit_relation_names(sql, database_type):
    """只接收完整单语句中显式二段名称；引号保留大小写，未知默认 Schema 不猜。"""
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 12000 or sql.rstrip().endswith(("...", "…")):
        return ()
    try:
        statements = parse(sql, read={"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[database_type])
        if len(statements) != 1 or statements[0] is None:
            return ()
        names = set()
        for table in statements[0].find_all(exp.Table):
            if table.catalog or not table.db or not all(isinstance(table.args.get(key), exp.Identifier) for key in ("this", "db")):
                continue
            values = []
            for key in ("db", "this"):
                identifier = table.args[key]
                name = identifier.name
                if not identifier.args.get("quoted"):
                    name = name.lower() if database_type == "POSTGRESQL" else name.upper() if database_type == "ORACLE" else name
                values.append(name)
            if all(0 < len(value) <= 256 for value in values):
                names.add(tuple(values))
            if len(names) > 32:
                return ()
        return tuple(sorted(names))
    except Exception:
        return ()


def _rows(fact):
    names = [str(column.get("name") or "").lower() for column in fact.columns]
    if len(names) != len(set(names)):
        return ()
    return tuple((index, dict(zip(names, row, strict=True))) for index, row in enumerate(fact.rows[:40]) if len(row) == len(names))


def _subject_row(row, subject, engine):
    identity = _IDENTITIES[engine]
    if not subject.get(identity) or str(row.get(identity) or "").casefold() != str(subject[identity]).casefold():
        return False
    return all(key not in subject or key in row and (
        row[key] == subject[key] if key == "schema_name" else str(row[key]).casefold() == str(subject[key]).casefold()
    ) for key in _DIMENSIONS)


def _physical_scope(fact, row, engine):
    if fact.provenance is None or "PHYSICAL_IDENTITY_UNKNOWN" in fact.provenance.quality_flags:
        return None
    physical = dict(fact.provenance.physical_identity)
    keys = _SCOPES[engine]
    if any(not physical.get(key) for key in keys):
        return None
    if engine == "ORACLE" and physical["container_id"] in {"0", "1"} and "container_id" not in row:
        # Root 的 GV$SQL 可混有其他容器；缺 CON_ID 时不能拿当前连接容器替代游标容器。
        return None
    if engine == "ORACLE" and physical["container_id"] == "1" and "container_id" in row:
        container = str(row["container_id"])
        if not container.isdigit():
            return None
        physical["container_id"] = container
    # PG 语句统计可读取其他数据库，连接数据库身份不能替代该行的数据库身份。
    if any(key in row and str(row[key]) != str(physical[key]) for key in keys):
        return None
    return tuple((key, physical[key]) for key in keys)


def _comparable(left, left_row, right, right_row, engine):
    scope = _physical_scope(left, left_row, engine)
    if not scope or scope != _physical_scope(right, right_row, engine):
        return False
    boots = [dict(item.provenance.physical_identity).get("instance_started_at") for item in (left, right)]
    if all(boots) and boots[0] != boots[1]:
        return False
    try:
        times = [datetime.fromisoformat(fact.captured_at.replace("Z", "+00:00")) for fact in (left, right)]
        return all(item.tzinfo is not None for item in times) and abs((times[1] - times[0]).total_seconds()) <= 300
    except (ValueError, TypeError, OverflowError):
        return False


def build_sql_object_graph(*, database_type, subject, evidence):
    """关系只在可信当前观测、核验物理数据库范围及五分钟内建立。"""
    facts = tuple(fact for fact in evidence[:64] if fact.trust_level == "SOURCE_VERIFIED"
                  and fact.source_id != "prior_turn.sql_observation")
    relationships = []
    catalog_relationships = []
    dependency_rows = []
    remote_nodes = set()
    remote_references = False
    truncated = len(evidence) > 64 or any(fact.truncated or len(fact.rows) > 40 for fact in facts)

    def append(kind, left, left_index, left_field, right, right_index, right_field, schema, relation, limitation):
        nonlocal truncated
        destination = catalog_relationships if kind.startswith("CATALOG_") else relationships
        if len(destination) >= (256 if kind.startswith("CATALOG_") else 128):
            truncated = True
            return
        edge = SqlObjectRelationship(kind=kind,
            anchor=EvidenceFieldReference(evidence_ref=left.evidence_ref, row_index=left_index, field=left_field),
            related=EvidenceFieldReference(evidence_ref=right.evidence_ref, row_index=right_index, field=right_field),
            schema_name=schema, relation_name=relation, limitation=limitation)
        if edge not in destination:
            destination.append(edge)
        return edge

    catalog = []
    for fact in facts:
        if fact.tool_id not in _CATALOG_TOOLS:
            continue
        rows = _rows(fact)
        for index, row in rows:
            field = "relation_name" if row.get("relation_name") else "table_name"
            schema, relation = row.get("schema_name"), row.get(field)
            if not isinstance(schema, str) or not isinstance(relation, str) or not schema or not relation or max(len(schema), len(relation)) > 256:
                continue
            if len(catalog) < 256:
                catalog.append((fact, index, row, field, schema, relation))
            else:
                truncated = True
                continue
            if fact.tool_id == _DEPENDENCY_TOOL:
                if row.get("reference_scope") == "REMOTE":
                    remote_nodes.add((fact.evidence_ref, index, field))
                elif row.get("reference_scope") == "LOCAL":
                    target_schema, target_name = row.get("dependency_schema_name"), row.get("dependency_name")
                    kind = row.get("relation_kind")
                    if (kind in {"VIEW", "MATERIALIZED VIEW", "SYNONYM"}
                            and (kind != "SYNONYM" or database_type == "ORACLE")
                            and all(isinstance(value, str) and 0 < len(value) <= 256 for value in (target_schema, target_name))
                            and _comparable(fact, row, fact, row, database_type)):
                        dependency_rows.append((fact, index, row, schema, relation, target_schema, target_name, kind))
                continue
            if fact.tool_id != "db.sql.object_indexes" or not _comparable(fact, row, fact, row, database_type):
                continue
            if row.get("index_name"):
                append("CATALOG_INDEX_ON", fact, index, "index_name", fact, index, field, schema, relation,
                       "目录证明索引归属，不证明该 SQL 使用、适用性或索引有效。")
            if database_type == "POSTGRESQL" and row.get("table_name") != relation:
                roots = [(root_index, root) for root_index, root in rows if root.get("relation_name") == row.get("table_name") and root.get("table_name") == row.get("table_name")]
                # 递归目录保留的是根对象，不把孙分区误写为直接子分区；跨 Schema 不靠同名猜根。
                identities = {(root.get("schema_name"), root.get("relation_name")) for _, root in roots}
                if len(identities) == 1:
                    root_index, root = roots[0]
                    if isinstance(root.get("schema_name"), str):
                        append("CATALOG_DESCENDANT_OF", fact, index, field, fact, root_index, "relation_name", schema, relation,
                               "原生递归目录证明属于根对象的后代，不保证直接父子关系，也不证明运行时分区裁剪。")

    parsed_texts = {}
    anchors = 0
    for fact in facts:
        if fact.tool_id not in _TEXT_TOOLS[database_type] and not (database_type == "ORACLE" and fact.tool_id in _PLAN_TOOLS):
            continue
        for index, row in _rows(fact):
            if not _subject_row(row, subject, database_type):
                continue
            anchors += 1
            if anchors > 64:
                truncated = True
                break
            candidates = []
            if fact.tool_id in _TEXT_TOOLS[database_type]:
                for field in ("normalized_statement", "digest_text", "sql_text"):
                    sql = row.get(field)
                    if not isinstance(sql, str) or len(sql) > 12000:
                        continue
                    if sql not in parsed_texts:
                        if len(parsed_texts) >= 16:
                            truncated = True
                            continue
                        parsed_texts[sql] = explicit_relation_names(sql, database_type)
                    candidates.extend(("SQL_TEXT_REFERENCES_OBJECT", field, schema, relation) for schema, relation in parsed_texts[sql])
            elif row.get("object_owner") and row.get("object_name"):
                candidates.append(("PLAN_ROW_REFERENCES_OBJECT", "object_name", row["object_owner"], row["object_name"]))
            for kind, field, schema, relation in candidates:
                for other, other_index, mapped, other_field, other_schema, other_relation in catalog:
                    if (schema, relation) != (other_schema, other_relation) or not _comparable(fact, row, other, mapped, database_type):
                        continue
                    append(kind, fact, index, field, other, other_index, other_field, schema, relation,
                           "语法明确引用该对象；不证明执行时扫描、索引使用或对象累计活动由该 SQL 造成。" if kind == "SQL_TEXT_REFERENCES_OBJECT" else
                           "匹配 SQL/游标/计划的目录计划行；不是指定绑定参数的实际执行计划或实际行数。")
    # 逐字段追踪对象端点：同一依赖行的源、目标不能因共用行号而被误认成同一个对象。
    nodes = {(item[0].evidence_ref, item[1], item[3]): item for item in catalog}
    source_rows = {(fact.evidence_ref, index): (fact, row) for fact in facts for index, row in _rows(fact)}
    queue = deque()
    for edge in relationships:
        node = nodes.get((edge.related.evidence_ref, edge.related.row_index, edge.related.field))
        source = source_rows.get((edge.anchor.evidence_ref, edge.anchor.row_index))
        if node is not None and source is not None:
            queue.append((node, source[0], source[1], 0))
    expanded = set()

    def retain(edge):
        nonlocal truncated
        if edge in relationships:
            return True
        if len(relationships) >= 128:
            truncated = True
            return False
        relationships.append(edge)
        return True

    def link(kind, left, right, limitation):
        return retain(SqlObjectRelationship(kind=kind,
            anchor=EvidenceFieldReference(evidence_ref=left[0].evidence_ref, row_index=left[1], field=left[3]),
            related=EvidenceFieldReference(evidence_ref=right[0].evidence_ref, row_index=right[1], field=right[3]),
            schema_name=right[4], relation_name=right[5], limitation=limitation))

    while queue:
        node, origin, origin_row, depth = queue.popleft()
        fact, index, row, field, schema, relation = node
        key = (origin.evidence_ref, schema, relation)
        if key in expanded:
            continue
        expanded.add(key)
        # 每一跳仍与最初的观测比较，不能靠连续五分钟的中转无限延伸时间范围。
        matches = [item for item in catalog if item[4:] == (schema, relation)
                   and _comparable(fact, row, item[0], item[2], database_type)
                   and _comparable(origin, origin_row, item[0], item[2], database_type)]
        for matched in matches:
            if matched[:2] != node[:2] or matched[3] != field:
                if not link("CATALOG_NAME_MATCH", node, matched,
                            "同一核验物理范围与近时点的目录名称匹配；不保证 DDL 期间对象生命周期不变，不证明运行时访问。"):
                    break
            reference = (matched[0].evidence_ref, matched[1], matched[3])
            if reference in remote_nodes:
                remote_references = True
            for edge in catalog_relationships:
                if (edge.related.evidence_ref, edge.related.row_index, edge.related.field) != reference:
                    continue
                if not retain(edge):
                    break
                if edge.kind == "CATALOG_DESCENDANT_OF":
                    child = nodes.get((edge.anchor.evidence_ref, edge.anchor.row_index, edge.anchor.field))
                    if child is not None:
                        queue.append((child, origin, origin_row, depth))
            for other, other_index, mapped, source_schema, source_name, target_schema, target_name, kind in dependency_rows:
                if (other.evidence_ref, other_index, "table_name") != reference:
                    continue
                if depth >= 4:
                    truncated = True
                    continue
                target = (other, other_index, mapped, "dependency_name", target_schema, target_name)
                if link("CATALOG_SYNONYM_REFERENCES" if kind == "SYNONYM" else "CATALOG_VIEW_DEPENDS_ON", matched, target,
                        "原生目录记录的本地依赖；不证明实际扫描、参数适配、视图已内联或物化结果已刷新，不能还原动态 SQL。"):
                    queue.append((target, origin, origin_row, depth + 1))
        if len(relationships) >= 128 and queue:
            truncated = True
            break
    return SqlObjectGraph(database_type=database_type, subject=subject, relationships=tuple(relationships),
        truncated=truncated, limitations=("仅覆盖可核验物理数据库范围内的显式名称、当前目录及最多四层已采集视图/同义词依赖；未解析默认 Schema、运行时动态 SQL和代表参数。",
            "未建立关系不代表 SQL 未访问对象；缺失、未解析或范围未知不能作为否定证据。",
            *(("可达目录含远端引用；只保留原始记录，不跨 DB Link、外部服务器或未知物理范围展开。",) if remote_references else ())))


def replay_sql_object_graph(graph, evidence):
    expected = build_sql_object_graph(database_type=graph.database_type, subject=graph.subject, evidence=evidence)
    if expected != graph:
        raise ValueError("SQL 对象关系与原始观测、物理身份或时间范围不一致")
