"""有界生成并重放 SQL 与原生目录对象的关系；不猜 search_path 或运行时扫描方式。"""

from datetime import datetime
from collections import deque

from sqlglot import exp, parse
from sqlglot.optimizer.scope import traverse_scope

from aiops_agent.contracts.sql_object_graph import (
    SqlObjectGraph, SqlObjectRelationship, SqlObjectReference, SqlObjectCatalogCandidate, SqlObjectCatalogLookup,
)
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
_CANDIDATES_TOOL = "db.sql.object_candidates"
_PLAN_TOOLS = {"db.sql.execution_plan", "db.oracle.statement.plan"}
_CATALOG_TOOLS = {"db.sql.object_indexes", "db.sql.object_statistics", _DEPENDENCY_TOOL, _CANDIDATES_TOOL}
_DIMENSIONS = ("database_oid", "user_oid", "top_level", "schema_name", "inst_id", "child_number", "plan_hash_value", "container_id", "dbid")


def _inspect_relation_names(sql, database_type):
    """保留未限定名称，按词法作用域排除 CTE；不解析 SQL 字符串、函数或远端名称。"""
    if not isinstance(sql, str) or not sql.strip():
        return (), False
    if len(sql) > 12000 or sql.rstrip().endswith(("...", "…")):
        return (), True
    try:
        statements = parse(sql, read={"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[database_type])
        if len(statements) != 1 or statements[0] is None:
            return (), False
        statement = statements[0]
        if not isinstance(statement, (exp.Query, exp.Insert, exp.Update, exp.Delete, exp.Merge)):
            return (), False
        scoped = {}
        for scope in traverse_scope(statement):
            for table in scope.tables:
                scoped[id(table)] = isinstance(scope.sources.get(table.alias_or_name), exp.Table)
        # DML 的目标不在 Query Scope 中；遇到 CTE 同名目标保守保留未知，不推断可更新性。
        cte_names = {cte.alias_or_name.casefold() for cte in statement.find_all(exp.CTE)}

        def identifier_name(identifier):
            name = identifier.name
            return name if identifier.args.get("quoted") else (
                name.lower() if database_type == "POSTGRESQL" else name.upper() if database_type == "ORACLE" else name)

        def recursive_cte_reference(table):
            owner = table.parent
            while owner is not None:
                block = owner.args.get("with_")
                if block is not None and block.args.get("recursive") and any(
                    isinstance(cte.args.get("alias"), exp.TableAlias)
                    and isinstance(cte.args["alias"].this, exp.Identifier)
                    and identifier_name(cte.args["alias"].this) == identifier_name(table.this)
                    for cte in block.expressions
                ):
                    return True
                owner = owner.parent
            return False

        names = set()
        for table in statement.find_all(exp.Table):
            if (table.catalog or scoped.get(id(table)) is False
                    or not isinstance(table.this, exp.Identifier)
                    or table.db and not isinstance(table.args.get("db"), exp.Identifier)):
                continue
            if (id(table) not in scoped and not table.db and table.name.casefold() in cte_names
                    or table.parent is statement and table.arg_key == "tables"):
                continue
            # sqlglot 的顺序 Scope 构造不能识别所有递归 WITH 前向引用；按实际祖先词法块补齐。
            if not table.db and recursive_cte_reference(table):
                continue
            values = []
            for key in ("db", "this"):
                identifier = table.args.get(key)
                if identifier is None:
                    values.append(None)
                    continue
                name = identifier.name
                if database_type == "ORACLE" and not identifier.args.get("quoted") and "@" in name:
                    values = []
                    break
                values.append(identifier_name(identifier))
            if len(values) == 2 and values[1] and all(value is None or 0 < len(value) <= 256 for value in values):
                names.add(tuple(values))
            if len(names) > 32:
                return (), True
        return tuple(sorted(names, key=lambda item: (item[0] or "", item[1]))), False
    except Exception:
        return (), False


def relation_names(sql, database_type):
    """返回有界名称集合；图生产者另保留预算/截断状态。"""
    return _inspect_relation_names(sql, database_type)[0]


def explicit_relation_names(sql, database_type):
    """显式二段名称仍可由原调用方单独读取；未限定名称不猜默认 Schema。"""
    return tuple(item for item in relation_names(sql, database_type) if item[0] is not None)


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


def matched_plan_objects(*, database_type, subject, evidence):
    """从完整匹配计划读取底层对象；不执行动态SQL，不借连接Schema消歧。"""
    from aiops_agent.domain.evidence.sql_plan_matching import assess_sql_plans
    from aiops_agent.contracts.sql_object_graph import SqlPlanObjectReference

    facts = {fact.evidence_ref: fact for fact in evidence}
    result = []
    for assessment in assess_sql_plans(database_type=database_type, subject=subject, evidence=evidence):
        if (assessment.identity_match != "SOURCE_MATCH" or assessment.physical_match != "SOURCE_MATCH"
            or assessment.time_relation != "SOURCE_NEAR_OBSERVATION" or assessment.anchor_ref is None):
            continue
        fact = facts[assessment.evidence_ref]
        if fact.trust_level != "SOURCE_VERIFIED":
            continue
        document = fact.sql_plan
        if document.origin == "CONTROLLED_ESTIMATE" and document.native_verification is not None:
            row_indices = {node.node_id: index for index, node in enumerate(document.nodes)}
            for item in document.native_verification.resolved_objects:
                result.append(SqlPlanObjectReference(node_id=item.node_id, source="CONTROLLED_ESTIMATE_OBJECT",
                    object_kind=item.object_kind, schema_name=item.schema_name, object_name=item.object_name,
                    native_object_id=item.native_object_id, subject_anchor_ref=assessment.anchor_ref,
                    field_ref={"evidence_ref":fact.evidence_ref, "row_index":row_indices[item.node_id], "field":"resolved_object_name"}))
        elif document.origin == "DATABASE_CURSOR":
            names = [str(column.get("name") or "").lower() for column in fact.columns]
            for node in document.nodes:
                name = node.index_name or node.relation_name
                if not name or not node.schema_name:
                    continue
                rows = [index for index, row in enumerate(fact.rows) if len(row) == len(names)
                    and dict(zip(names,row,strict=True)).get("object_owner") == node.schema_name
                    and dict(zip(names,row,strict=True)).get("object_name") == name]
                if not rows:
                    continue
                result.append(SqlPlanObjectReference(node_id=node.node_id, source="DATABASE_CURSOR_OBJECT",
                    object_kind="INDEX" if node.index_name else "TABLE", schema_name=node.schema_name, object_name=name,
                    subject_anchor_ref=assessment.anchor_ref,
                    field_ref={"evidence_ref":fact.evidence_ref, "row_index":rows[0], "field":"object_name"}))
    return tuple(result)


def build_sql_object_graph(*, database_type, subject, evidence):
    """关系只在可信当前观测、核验物理数据库范围及五分钟内建立。"""
    facts = tuple(fact for fact in evidence[:64] if fact.trust_level == "SOURCE_VERIFIED"
                  and fact.source_id != "prior_turn.sql_observation")
    relationships = []
    references = []
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
                    if not isinstance(sql, str):
                        continue
                    if sql not in parsed_texts:
                        if len(parsed_texts) >= 16:
                            truncated = True
                            continue
                        parsed_texts[sql], text_bounded = _inspect_relation_names(sql, database_type)
                        truncated = truncated or text_bounded
                    for schema, relation in parsed_texts[sql]:
                        resolution, context_ref = "EXPLICIT_SCHEMA", None
                        if schema is None:
                            resolution = "CONTEXT_REQUIRED"
                            default = row.get("schema_name")
                            # MySQL Digest 原生保留默认数据库；其他引擎的连接环境不能替代原解析环境。
                            if database_type == "MYSQL" and isinstance(default, str) and 0 < len(default) <= 256:
                                schema, resolution = default, "NATIVE_DEFAULT_SCHEMA"
                                context_ref = EvidenceFieldReference(evidence_ref=fact.evidence_ref, row_index=index, field="schema_name")
                        matches = [item for item in catalog if item[5] == relation
                                   and (schema is None or item[4] == schema)
                                   and _comparable(fact, row, item[0], item[2], database_type)]
                        discovered = tuple(SqlObjectCatalogCandidate(schema_name=item[4], relation_name=item[5],
                            catalog_ref=EvidenceFieldReference(evidence_ref=item[0].evidence_ref, row_index=item[1], field=item[3]))
                            for item in matches if item[0].tool_id == _CANDIDATES_TOOL)
                        if len(discovered) > 16:
                            truncated = True
                        reference = SqlObjectReference(
                            anchor=EvidenceFieldReference(evidence_ref=fact.evidence_ref, row_index=index, field=field),
                            relation_name=relation, schema_name=schema, resolution=resolution, context_ref=context_ref,
                            candidates=discovered[:16],
                            lookup=None if discovered or max(len(relation), len(schema or "")) > {
                                "POSTGRESQL": 63, "MYSQL": 64, "ORACLE": 128}[database_type] else SqlObjectCatalogLookup(input={
                                "table_name": relation, "schema_name": schema or "", "limit": 40}),
                            limitation=("未限定名称需要原语句解析环境或匹配计划；目录唯一候选也不能证明历史 search_path、同义词优先级或实际访问。"
                                if schema is None else "名称与原生默认数据库只界定语法候选；临时表遮蔽、DDL 及历史解析仍需核验，不证明实际访问。"
                                if resolution == "NATIVE_DEFAULT_SCHEMA" else "显式名称及当前目录只界定语法对象，不证明运行时访问或历史对象生命周期。"))
                        if reference not in references:
                            if len(references) < 64:
                                references.append(reference)
                            else:
                                truncated = True
                        if schema is not None:
                            candidates.append(("SQL_TEXT_DEFAULT_SCHEMA_CANDIDATE" if resolution == "NATIVE_DEFAULT_SCHEMA"
                                else "SQL_TEXT_REFERENCES_OBJECT", field, schema, relation))
            elif row.get("object_owner") and row.get("object_name"):
                candidates.append(("PLAN_ROW_REFERENCES_OBJECT", "object_name", row["object_owner"], row["object_name"]))
            for kind, field, schema, relation in candidates:
                for other, other_index, mapped, other_field, other_schema, other_relation in catalog:
                    if (schema, relation) != (other_schema, other_relation) or not _comparable(fact, row, other, mapped, database_type):
                        continue
                    append(kind, fact, index, field, other, other_index, other_field, schema, relation,
                           "语法明确引用该对象；不证明执行时扫描、索引使用或对象累计活动由该 SQL 造成。" if kind == "SQL_TEXT_REFERENCES_OBJECT" else
                           "Digest 原生默认数据库下的语法候选；不排除临时表遮蔽或 DDL，不证明实际访问或扫描。" if kind == "SQL_TEXT_DEFAULT_SCHEMA_CANDIDATE" else
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
    plan_objects = matched_plan_objects(database_type=database_type, subject=subject, evidence=facts)
    truncated = truncated or len(plan_objects) > 128
    return SqlObjectGraph(database_type=database_type, subject=subject, references=tuple(references), relationships=tuple(relationships),
        plan_objects=plan_objects[:128],
        truncated=truncated, limitations=("仅覆盖可核验物理数据库范围内的语法名称、MySQL 原生默认数据库、当前目录及最多四层已采集依赖；PG 历史 search_path、Oracle 同义词优先级、运行时动态 SQL和代表参数不猜测。",
            "未建立关系不代表 SQL 未访问对象；缺失、未解析或范围未知不能作为否定证据。",
            *(("可达目录含远端引用；只保留原始记录，不跨 DB Link、外部服务器或未知物理范围展开。",) if remote_references else ())))


def replay_sql_object_graph(graph, evidence):
    expected = build_sql_object_graph(database_type=graph.database_type, subject=graph.subject, evidence=evidence)
    if expected != graph:
        raise ValueError("SQL 对象关系与原始观测、物理身份或时间范围不一致")
