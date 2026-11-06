"""真实开发库的 K2 知识检索专用验收；不建删表，不操作 AIOps 数据。"""

import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from dataclasses import replace
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "scripts/deployment"), str(ROOT / "tests/integration")]
from knowledge_retrieval_k2 import configured_engine, inspect_tables
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.sql.ddl import sort_tables_and_constraints

ORACLE = (ROOT / "database/oracle").is_dir()
base = importlib.import_module("test_app_owned_knowledge_ingestion_oracle" if ORACLE
                              else "test_app_owned_knowledge_ingestion_postgresql")
BaseTests = base.AppIngestionOracleTests if ORACLE else base.AppIngestionPostgresTests
ENABLED = os.environ.get("KNOWLEDGE_K2_DEVELOPMENT_ACCEPTANCE") == "knowledge_retrieval_only"


class KnowledgeRetrievalDevelopmentTests(BaseTests):
    __unittest_skip__ = not ENABLED
    __unittest_skip_why__ = "需要 KNOWLEDGE_K2_DEVELOPMENT_ACCEPTANCE=knowledge_retrieval_only 显式授权"

    async def asyncSetUp(self):
        self.engine, self.tables, _ = configured_engine()
        self.scope = None
        self.tmp = tempfile.TemporaryDirectory(prefix="knowledge-k2-kr-only-")
        self.addAsyncCleanup(self.cleanup_synthetic_scope)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        async with self.engine.connect() as connection:
            missing = await connection.run_sync(lambda sync: inspect_tables(sync, self.tables))
            if missing:
                raise RuntimeError("请先显式执行知识检索定向建表；验收不自动建表")
        fixture = base.Fixture("knowledge_retrieval_app", self.sessions, self.tmp.name)
        # 不登记平台 Domain/Tenant，不碰平台表；18 张新表没有平台外键。
        domain_id = 10**30 + uuid4().int % 10**30 if ORACLE else uuid4()
        scope = replace(fixture.scope, domain_id=domain_id,
                        subject_ref="synthetic-k2-" + uuid4().hex)
        async with self.engine.connect() as connection:
            for table in self.tables:
                count = await connection.scalar(select(func.count()).select_from(table).where(
                    table.c.domain_id == scope.domain_id))
                if count:
                    raise RuntimeError("合成范围已存在，拒绝验收和清理")
        fixture.scope = self.scope = scope
        self.fixtures = [fixture]

    async def asyncTearDown(self):
        # 不调用父类 drop_all/DROP SCHEMA；清理由已登记的精确合成范围处理。
        pass

    async def cleanup_synthetic_scope(self):
        try:
            if self.scope is not None:
                async with self.engine.begin() as connection:
                    def predicate(table):
                        conditions = [table.c.domain_id == self.scope.domain_id]
                        if not ORACLE:
                            conditions.append(table.c.tenant_id == self.scope.tenant_id)
                        return conditions
                    version = next(table for table in self.tables if "published_parse_run_id" in table.c)
                    await connection.execute(update(version).where(*predicate(version)).values(
                        published_parse_run_id=None, active_generation_id=None))
                    # use_alter 发布指针已解开，只逆序删除本例合成范围的行。
                    ordered = [table for table, _ in sort_tables_and_constraints(self.tables)
                               if table is not None]
                    for table in reversed(ordered):
                        await connection.execute(delete(table).where(*predicate(table)))
                    for table in self.tables:
                        if await connection.scalar(select(func.count()).select_from(table).where(*predicate(table))):
                            raise RuntimeError("合成验收数据清理未闭合")
        finally:
            await self.engine.dispose()
            self.tmp.cleanup()

    async def test_physical_app_domain_tenant_subject_isolation(self):
        fixture = self.fixtures[0]
        await fixture.register()
        await fixture.parse()
        wrong_domain = self.scope.domain_id + 1 if ORACLE else uuid4()
        for scope in (replace(self.scope, domain_id=wrong_domain),
                      replace(self.scope, tenant_id=uuid4()),
                      replace(self.scope, subject_ref="other")):
            with self.assertRaises(base.AlgorithmError):
                await fixture.service.get_job(scope, fixture.job)
        # 另一 App 只核对离线映射和对象目录，绝不连接其表或执行 DDL/DML。
        other = base.Fixture("aiops_agent", self.sessions, self.tmp.name)
        names = {getattr(fixture.entities, name).__tablename__ for name in fixture.entities.__all__}
        other_names = {getattr(other.entities, name).__tablename__ for name in other.entities.__all__}
        self.assertFalse(names & other_names)
        self.assertNotEqual(fixture.store.root, other.store.root)


del BaseTests


if __name__ == "__main__":
    print("验收范围：仅知识检索 App；AIOps 只做离线映射比较，不计其真库验收", flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(KnowledgeRetrievalDevelopmentTests)
    result = unittest.TextTestRunner(verbosity=2, failfast=True).run(suite)
    raise SystemExit(0 if result.wasSuccessful() and not result.skipped else 1)
