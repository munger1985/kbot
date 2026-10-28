"""有界生成并重放 SQL 与原生目录对象的关系；不猜 search_path 或运行时扫描方式。"""

from datetime import datetime

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
_CATALOG_TOOLS = {"db.sql.object_indexes", "db.sql.object_statistics"}
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
        if fact.tool_id not in _TEXT_TOOLS[database_type] and not (database_type == "ORACLE" and fact.tool_id == "db.sql.execution_plan"):
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
    # 只冻结由指定 SQL 的已核验名称/计划连接到的对象分量，不带入无关表的索引关系。
    reachable = {(edge.related.evidence_ref, edge.related.row_index) for edge in relationships}
    for edge in catalog_relationships:
        if edge.kind == "CATALOG_DESCENDANT_OF" and (edge.related.evidence_ref, edge.related.row_index) in reachable:
            if len(relationships) < 128:
                relationships.append(edge)
                reachable.add((edge.anchor.evidence_ref, edge.anchor.row_index))
            else:
                truncated = True
    for edge in catalog_relationships:
        if edge.kind == "CATALOG_INDEX_ON" and (edge.related.evidence_ref, edge.related.row_index) in reachable:
            if len(relationships) < 128:
                relationships.append(edge)
            else:
                truncated = True
    return SqlObjectGraph(database_type=database_type, subject=subject, relationships=tuple(relationships),
        truncated=truncated, limitations=("仅覆盖可核验物理数据库范围内的显式名称及当前目录；未解析默认 Schema、动态 SQL、视图展开和代表参数。",
            "未建立关系不代表 SQL 未访问对象；缺失、未解析或范围未知不能作为否定证据。"))


def replay_sql_object_graph(graph, evidence):
    expected = build_sql_object_graph(database_type=graph.database_type, subject=graph.subject, evidence=evidence)
    if expected != graph:
        raise ValueError("SQL 对象关系与原始观测、物理身份或时间范围不一致")
