"""K2 固定表清单、实体、持久化边界与规范建库工件的离线检查。"""

import ast
import importlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(p) for p in ROOT.glob("packages/*/src")] + [str(p) for p in ROOT.glob("services/*/src")]


def entity_tables():
    result = {}
    for app in ("knowledge_retrieval_app","aiops_agent"):
        module = importlib.import_module(app + ".entities.knowledge_ingestion")
        result[app] = [getattr(module,name).__table__ for name in module.__all__]
    return result


def render_oracle():
    from sqlalchemy.dialects.oracle import dialect
    from sqlalchemy.schema import CreateTable,CreateIndex,AddConstraint
    from sqlalchemy.sql.ddl import sort_tables_and_constraints
    result = {}
    for app,tables in entity_tables().items():
        statements = []
        for table,constraints in sort_tables_and_constraints(tables):
            if table is not None:
                statements.append(str(CreateTable(table,include_foreign_key_constraints=constraints).compile(dialect=dialect())).strip())
            else:
                for constraint in sorted(constraints,key=lambda c:c.name):
                    statements.append(str(AddConstraint(constraint).compile(dialect=dialect())).strip())
        for table in tables:
            for index in sorted(table.indexes,key=lambda i:i.name):
                statements.append(str(CreateIndex(index).compile(dialect=dialect())).strip())
        ddl = "-- App 独占知识数据；只定义主键、外键、非空及非唯一访问索引。\n\n"
        ddl += ";\n\n".join(s.replace('"','').upper() for s in statements) + ";\n"
        ddl = "\n".join(line.rstrip() for line in ddl.splitlines()) + "\n"
        result[app] = {"ddl":ddl,"tables":sorted(t.name for t in tables),"count":len(statements)}
    return result


def check():
    from sqlalchemy import CheckConstraint,UniqueConstraint
    inventory = json.loads((ROOT / "tests/contract/knowledge_algorithms/inventory-v1.json").read_text())
    oracle = (ROOT / "database/oracle").is_dir()
    all_names, errors = set(), []
    for app,tables in entity_tables().items():
        owner = "aiops" if app == "aiops_agent" else "knowledge_retrieval"
        registered = inventory["apps"][owner]["tables"]
        expected = {value["oracle" if oracle else "postgresql"]:value for value in registered.values()}
        if len(tables) != 18 or {t.name for t in tables} != set(expected):
            errors.append(app + " 固定表集合不匹配")
        for table in tables:
            if table.name in all_names:
                errors.append(table.name + " 跨 App 共用物理表")
            all_names.add(table.name)
            if not set(expected[table.name]["columns"]) <= set(table.columns.keys()):
                errors.append(table.name + " 缺少冻结列")
            if any(isinstance(c,(CheckConstraint,UniqueConstraint)) for c in table.constraints):
                errors.append(table.name + " 出现禁止的业务 CHECK/UNIQUE")
            if any(index.unique for index in table.indexes):
                errors.append(table.name + " 出现禁止的唯一索引")
            for fk in table.foreign_key_constraints:
                if fk.referred_table.name not in expected:
                    errors.append(table.name + " 外键越出当前 App")
                if not fk.name or len(fk.name) > 30:
                    errors.append(table.name + " 外键名缺失或超长")
        base = ROOT / "services" / app / "src" / app
        for folder in ("application","api"):
            for path in base.joinpath(folder).glob("knowledge_*.py"):
                if path.stem not in {"knowledge_ingestion", "knowledge_projection", "knowledge_source",
                    "knowledge_source_auth", "knowledge_http", "knowledge_retrieval", "knowledge_retrieval_auth",
                    "knowledge_hydration", "knowledge_pipeline"}:
                    continue
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node,ast.ImportFrom) and node.module and (
                            node.module.startswith("sqlalchemy") or ".entities" in node.module):
                        errors.append(str(path.relative_to(ROOT)) + " 越层访问持久化")
        for logical in registered:
            if not (base / "repositories" / (logical + ".py")).is_file():
                errors.append(app + " 缺少逐表 Repository: " + logical)
    if oracle:
        for app,value in render_oracle().items():
            name = "014_ops_knowledge_ingestion.sql" if app == "aiops_agent" else "003_knowledge_ingestion.sql"
            if (ROOT / "database/oracle" / app / name).read_text().rstrip() + "\n" != value["ddl"]:
                errors.append(app + " 规范 Oracle DDL 与实体不一致")
    if errors:
        raise SystemExit("\n".join(errors))
    print("K2 固定 36 表、逐表仓储、App 隔离与规范 Schema 离线检查通过")


if __name__ == "__main__":
    if "--render" in sys.argv:
        print(json.dumps(render_oracle(),ensure_ascii=False))
    else:
        check()
