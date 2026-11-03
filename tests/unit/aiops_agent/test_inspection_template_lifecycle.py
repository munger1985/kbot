"""巡检模板正式草稿与版本隔离的两端共享回归。"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from aiops_agent.application.inspection_template_lifecycle import InspectionTemplateService, metadata
from aiops_agent.application.inspections.check_catalog import default_selected_check_ids
from aiops_agent.application.inspections.daily_sections import normalize_daily_layout
from platform_core.identity import uuid7

IS_KBOT = Path(__file__).resolve().parents[3].name == "kbot4"


class Store:
    """内存事务替身只测试业务规则；不声称验证真实数据库锁。"""

    def __init__(self):
        self.domain = 7 if IS_KBOT else uuid7()
        self.actor = "tester" if IS_KBOT else uuid7()
        self.identity = {"domain_id": self.domain, "actor_id": self.actor, **({} if IS_KBOT else {"tenant_id": uuid7()})}
        self.rows, self.versions, self.plans, self.fires, self.snapshots = {}, {}, [], [], {}
        self.repo = SimpleNamespace(
            add_inspection_template=AsyncMock(side_effect=self.add),
            add_inspection_template_version=AsyncMock(side_effect=self.add_version),
            get_inspection_template=AsyncMock(side_effect=self.get),
            get_inspection_template_version=AsyncMock(side_effect=lambda **kw:self.versions.get(kw["inspection_template_version_id"])),
            list_inspection_templates=AsyncMock(side_effect=lambda **kw:[row for row in self.rows.values() if row.domain_id == kw["domain_id"]]),
            list_inspection_template_versions=AsyncMock(side_effect=lambda **kw:sorted([v for v in self.versions.values() if v.inspection_template_id == kw["inspection_template_id"]], key=lambda v:v.version_no, reverse=True)),
            count_plans_by_inspection_templates=AsyncMock(side_effect=lambda **kw:{key:sum(p.inspection_template_id == key and p.domain_id == kw["domain_id"] for p in self.plans) for key in kw["template_ids"]}),
            count_fires_by_inspection_templates=AsyncMock(side_effect=lambda **kw:{key:sum(f.inspection_template_id == key for f in self.fires) for key in kw["template_ids"]}),
            delete_inspection_template_with_versions=AsyncMock(side_effect=self.delete_with_versions),
            list_plans_by_inspection_template=AsyncMock(side_effect=lambda **kw:[p for p in self.plans if p.domain_id == kw["domain_id"] and p.inspection_template_id == kw["template_id"]]),
            lock_inspection_template_defaults=AsyncMock(),
            delete_inspection_template=AsyncMock(side_effect=lambda row:self.rows.pop(row.inspection_template_id)),
        )
        self.service = InspectionTemplateService(uow_factory=self.context)

    def delete_with_versions(self, row):
        self.versions = {key:version for key,version in self.versions.items() if version.inspection_template_id != row.inspection_template_id}
        self.rows.pop(row.inspection_template_id)

    def add(self, row):
        row.row_version = 1
        row.updated_at = datetime.now(UTC)
        self.rows[row.inspection_template_id] = row
        self.snapshots[row.inspection_template_id] = deepcopy((row.management_json, row.current_version_id, row.status, row.display_name))

    def add_version(self, version):
        version.created_at = datetime.now(UTC)
        self.versions[version.inspection_template_version_id] = version

    def get(self, **kw):
        row = self.rows.get(kw["inspection_template_id"])
        return row if row and row.domain_id == kw["domain_id"] else None

    async def commit(self):
        for key, row in self.rows.items():
            state = (row.management_json, row.current_version_id, row.status, row.display_name)
            if state != self.snapshots[key]:
                row.row_version += 1
                self.snapshots[key] = deepcopy(state)

    @asynccontextmanager
    async def context(self):
        yield SimpleNamespace(inspections=self.repo, commit=self.commit)

    async def create(self, name="业务日报"):
        data = await self.service.create(**self.identity, display_name=name, selected_check_ids=default_selected_check_ids("DAILY"))
        return next(row for row in self.rows.values() if str(row.inspection_template_id) == data["inspection_template_id"])

    def command(self, row):
        return {"domain_id":self.domain, "actor_id":self.actor, "template_id":row.inspection_template_id, "expected_row_version":row.row_version}

    async def publish(self, row):
        return await self.service.publish(**self.command(row), **({} if IS_KBOT else {"tenant_id":self.identity["tenant_id"]}))

    async def save(self, row, title="业务日报第二版"):
        layout = normalize_daily_layout()
        layout["title"] = title
        return await self.service.save_draft(**self.command(row), display_name=row.display_name, selected_check_ids=default_selected_check_ids("DAILY"), daily_report_layout=layout)


def run(case):
    asyncio.run(case())


def test_draft_reopen_publish_and_old_plan_isolation():
    async def case():
        store = Store()
        row = await store.create()
        detail = await store.service.get(domain_id=store.domain, template_id=row.inspection_template_id)
        assert detail["has_draft"] and detail["version_no"] is None and not store.versions
        await store.save(row)
        assert not store.versions
        first = await store.publish(row)
        frozen = row.current_version_id
        plan = SimpleNamespace(inspection_plan_id=uuid7(), domain_id=store.domain, inspection_template_id=row.inspection_template_id,
                               inspection_template_version_id=frozen, display_name="生产库巡检", status="ACTIVE")
        store.plans.append(plan)
        original = deepcopy(store.versions[frozen].definition_json)
        saved = await store.save(row, "第三版展示")
        assert saved["version_no"] == 1 and saved["has_unpublished_changes"]
        reopened = await store.service.get(domain_id=store.domain, template_id=row.inspection_template_id)
        assert reopened["draft"]["daily_report_layout"]["title"] == "第三版展示"
        assert reopened["definition"] == original
        second = await store.publish(row)
        assert first["version_no"] == 1 and second["version_no"] == 2
        assert plan.inspection_template_version_id == frozen and store.versions[frozen].definition_json == original
        refs = await store.service.list_plans(domain_id=store.domain, template_id=row.inspection_template_id)
        assert refs[0]["template_version_no"] == 1 and not refs[0]["is_latest"]
        history = await store.service.list_versions(domain_id=store.domain, template_id=row.inspection_template_id)
        assert [v["version_no"] for v in history] == [2,1] and history[0]["created_by"] == str(store.actor)
    run(case)


def test_identical_publication_and_historical_duplicate_are_rejected():
    async def case():
        store = Store()
        row = await store.create()
        await store.publish(row)
        await store.save(row, normalize_daily_layout()["title"])
        with pytest.raises(Exception, match="相同"):
            await store.publish(row)
        assert len(store.versions) == 1
        await store.save(row)
        await store.publish(row)
        await store.save(row, normalize_daily_layout()["title"])
        with pytest.raises(Exception, match="相同"):
            await store.publish(row)
        assert len(store.versions) == 2
    run(case)


@pytest.mark.parametrize("operation", ["save", "publish", "discard", "status", "delete"])
def test_builtin_is_readonly(operation):
    async def case():
        store = Store()
        await store.service.list(**store.identity)
        row = next(row for row in store.rows.values() if metadata(row)["template_kind"] == "DAILY")
        with pytest.raises(Exception, match="只读"):
            if operation == "save":
                await store.save(row)
            elif operation == "publish":
                await store.publish(row)
            elif operation == "discard":
                await store.service.discard_draft(**store.command(row))
            elif operation == "delete":
                await store.service.delete(**store.command(row))
            else:
                await store.service.update(**store.command(row), status="INACTIVE")
    run(case)


def test_defaults_are_unique_scoped_and_do_not_create_versions():
    async def case():
        store = Store()
        await store.service.list(**store.identity)
        builtin = next(row for row in store.rows.values() if metadata(row)["template_kind"] == "DAILY")
        row = await store.create()
        with pytest.raises(Exception, match="已发布"):
            await store.service.set_default(**store.command(row))
        await store.publish(row)
        count = len(store.versions)
        await store.service.set_default(**store.command(row))
        assert metadata(row)["is_default"] and not metadata(builtin)["is_default"]
        assert sum(metadata(item)["is_default"] for item in store.rows.values()) == 2
        assert len(store.versions) == count
        with pytest.raises(Exception, match="替代"):
            await store.service.update(**store.command(row), status="INACTIVE")
        await store.service.set_default(**store.command(builtin))
        await store.service.update(**store.command(row), status="INACTIVE")
        assert row.status == "INACTIVE" and len(store.versions) == count
        with pytest.raises(Exception, match="已发布"):
            await store.service.set_default(**store.command(row))
        assert store.repo.lock_inspection_template_defaults.await_count >= 5
    run(case)


def test_discard_preserves_history_and_only_deletes_unpublished_root():
    async def case():
        store = Store()
        row = await store.create()
        await store.service.discard_draft(**store.command(row))
        assert row.inspection_template_id not in store.rows
        row = await store.create()
        await store.publish(row)
        frozen = row.current_version_id
        await store.save(row)
        await store.service.discard_draft(**store.command(row))
        assert row.current_version_id == frozen and not metadata(row)["draft"] and len(store.versions) == 1
    run(case)


@pytest.mark.parametrize("operation", ["get","versions","plans","save","publish","discard","default","status","delete"])
def test_cross_domain_access_is_rejected(operation):
    async def case():
        store = Store()
        row = await store.create()
        args = store.command(row)
        args["domain_id"] = 8 if IS_KBOT else uuid7()
        read = {"domain_id":args["domain_id"],"template_id":row.inspection_template_id}
        with pytest.raises(Exception):
            if operation == "get": await store.service.get(**read)
            elif operation == "versions": await store.service.list_versions(**read)
            elif operation == "plans": await store.service.list_plans(**read)
            elif operation == "save": await store.service.save_draft(**args,display_name="其他",selected_check_ids=default_selected_check_ids("DAILY"))
            elif operation == "publish": await store.service.publish(**args)
            elif operation == "discard": await store.service.discard_draft(**args)
            elif operation == "default": await store.service.set_default(**args)
            elif operation == "delete": await store.service.delete(**args)
            else: await store.service.update(**args,status="INACTIVE")
        assert not store.versions
    run(case)


def test_stale_draft_cannot_overwrite_and_named_custom_is_not_builtin():
    async def case():
        store = Store()
        row = await store.create("日常巡检")
        old = store.command(row)
        await store.save(row)
        with pytest.raises(Exception,match="刷新"):
            await store.service.save_draft(**old,display_name="冲突",selected_check_ids=default_selected_check_ids("DAILY"))
        result = await store.service.list(**store.identity)
        assert len(result) == 3
        assert sum(item["source_kind"] == "BUILTIN" for item in result) == 2
        assert metadata(row)["source_kind"] == "CUSTOM"
        await store.service.list(**store.identity)
        assert len(store.rows) == 3
    run(case)


def test_internal_lifecycle_routes_remove_direct_version_write_and_validate_payload():
    from aiops_agent.api import report_templates as api
    from unittest.mock import patch
    async def case():
        app = FastAPI()
        app.include_router(api.router)
        context = SimpleNamespace(domain_id=7 if IS_KBOT else uuid7(),tenant_id=uuid7(),principal_id=uuid7(),asserted_user_id="tester",client_id="portal")
        app.dependency_overrides[api.get_aiops_auth_context] = lambda:context
        service = SimpleNamespace(**{name:AsyncMock(return_value={}) for name in ("save_draft","publish","set_default","discard_draft","delete","list_plans")})
        app.state.inspection_template_service = service
        async with AsyncClient(transport=ASGITransport(app=app),base_url="http://local") as client:
            with patch.object(api,"require_service_scope"):
                template_id=uuid7()
                old=await client.post(f"/internal/v1/aiops/inspection-templates/{template_id}/versions",json={})
                assert old.status_code == 405
                for method,suffix in (("post",":publish"),("post",":default"),("delete","/draft"),("delete","")):
                    response=await client.request(method.upper(),f"/internal/v1/aiops/inspection-templates/{template_id}{suffix}",json={"expected_row_version":1})
                    assert response.status_code in (200,201)
                invalid=await client.post(f"/internal/v1/aiops/inspection-templates/{template_id}:publish",json={"expected_row_version":1,"source_kind":"BUILTIN"})
                assert invalid.status_code == 422
                assert service.publish.await_count == 1
                assert service.delete.await_count == 1
                service.delete.assert_awaited_once_with(domain_id=context.domain_id, actor_id=context.asserted_user_id if IS_KBOT else context.principal_id, template_id=template_id, expected_row_version=1)
                for payload in ({}, {"expected_row_version":0}, {"expected_row_version":1,"force":True}):
                    invalid=await client.request("DELETE",f"/internal/v1/aiops/inspection-templates/{template_id}",json=payload)
                    assert invalid.status_code == 422
                assert service.delete.await_count == 1
    run(case)

@pytest.mark.parametrize("reason", ["active", "default", "plan", "inactive_plan", "fire", "stale"])
def test_delete_revalidates_state_and_all_references(reason):
    async def case():
        store = Store()
        row = await store.create()
        await store.publish(row)
        await store.service.update(**store.command(row), status="INACTIVE")
        args = store.command(row)
        if reason == "active":
            await store.service.update(**store.command(row), status="ACTIVE")
            args = store.command(row)
        elif reason == "default":
            row.management_json = {**metadata(row), "is_default":True}
        elif reason in {"plan", "inactive_plan"}:
            store.plans.append(SimpleNamespace(domain_id=store.domain, inspection_template_id=row.inspection_template_id,
                                               status="ACTIVE" if reason == "plan" else "DISABLED"))
        elif reason == "fire":
            store.fires.append(SimpleNamespace(inspection_template_id=row.inspection_template_id))
        else:
            args["expected_row_version"] -= 1
        before = deepcopy((row.management_json, row.current_version_id, row.status))
        with pytest.raises(Exception):
            await store.service.delete(**args)
        assert row.inspection_template_id in store.rows and len(store.versions) == 1
        assert (row.management_json, row.current_version_id, row.status) == before
        store.repo.delete_inspection_template_with_versions.assert_not_awaited()
        item = next(item for item in await store.service.list(**store.identity) if item["inspection_template_id"] == str(row.inspection_template_id))
        assert item["can_delete"] is (reason == "stale")
    run(case)


def test_delete_removes_only_unused_inactive_custom_root_all_versions_and_draft():
    async def case():
        store = Store()
        row, other = await store.create(), await store.create("保留模板")
        await store.publish(row)
        await store.save(row)
        await store.publish(row)
        await store.save(row, "尚未发布修改")
        await store.publish(other)
        preserved = deepcopy(store.versions[other.current_version_id].definition_json)
        await store.service.update(**store.command(row), status="INACTIVE")
        detail = await store.service.get(domain_id=store.domain, template_id=row.inspection_template_id)
        assert detail["can_delete"] and detail["has_draft"] and detail["usage_count"] == detail["history_count"] == 0
        result = await store.service.delete(**store.command(row))
        assert result == {"deleted":True}
        assert list(store.rows) == [other.inspection_template_id]
        assert list(store.versions) == [other.current_version_id]
        assert store.versions[other.current_version_id].definition_json == preserved
        store.repo.delete_inspection_template_with_versions.assert_awaited_once_with(row)
        with pytest.raises(Exception):
            await store.service.delete(**store.command(row))
    run(case)


def test_repository_delete_order_scope_and_reference_queries():
    from unittest.mock import Mock
    from sqlalchemy.dialects import oracle, postgresql
    from aiops_agent.repositories.inspection import InspectionRepository
    from aiops_agent.entities import InspectionTemplateEntity, InspectionTemplateVersionEntity
    async def case():
        store = Store()
        row = await store.create()
        await store.publish(row)
        statements, events = [], []
        async def execute(statement):
            statements.append(statement)
            events.append("sql")
            return Mock(all=lambda:[])
        session = SimpleNamespace(
            flush=AsyncMock(side_effect=lambda:events.append(("flush", row.current_version_id))),
            execute=AsyncMock(side_effect=execute),
            delete=AsyncMock(side_effect=lambda entity:events.append(("delete", entity))),
        )
        repo = InspectionRepository(session)
        await repo.delete_inspection_template_with_versions(row)
        assert events[0] == ("flush", None) and events[-1] == ("delete", row)
        deletion = statements[0]
        sql = str(deletion.compile(dialect=oracle.dialect() if IS_KBOT else postgresql.dialect())).lower().replace('"', '')
        assert sql.startswith("delete from " + InspectionTemplateVersionEntity.__tablename__.lower())
        assert "domain_id" in sql and "inspection_template_id" in sql
        assert set(deletion.compile().params.values()) == {row.domain_id, row.inspection_template_id}
        assert len(statements) == 1
        for counter, table in ((repo.count_plans_by_inspection_templates, "inspection_plan"),
                               (repo.count_fires_by_inspection_templates, "inspection_fire")):
            assert await counter(domain_id=store.domain, template_ids=[]) == {}
            await counter(domain_id=store.domain, template_ids=[row.inspection_template_id])
            for dialect in (oracle.dialect(), postgresql.dialect()):
                compiled = statements[-1].compile(dialect=dialect)
                query = str(compiled).lower().replace('"', '')
                assert table in query and "inspection_template_version_id in (select" in query
                assert " or " in query and "group by" in query
                assert InspectionTemplateEntity.__tablename__.lower() + ".domain_id" in query
            assert len(statements) >= 2
    run(case)

@pytest.mark.parametrize("kind", ["plan", "fire"])
def test_reference_sql_counts_root_and_old_version_without_double_counting(kind):
    """隔离 SQLite 执行正式引用查询，不连接应用数据库。"""
    from sqlalchemy import Column, MetaData, Table, create_engine
    from aiops_agent.entities import (
        InspectionTemplateEntity as Root, InspectionTemplateVersionEntity as Version,
        InspectionPlanEntity, InspectionFireEntity,
    )
    from aiops_agent.repositories.inspection import InspectionRepository
    async def case():
        store = Store()
        engine = create_engine("sqlite://")
        schema = MetaData()
        def minimal(entity, names):
            return Table(entity.__tablename__, schema, *(
                Column(name, entity.__table__.c[name].type) for name in names
            ))
        roots = minimal(Root, ["inspection_template_id", "domain_id"])
        versions = minimal(Version, ["inspection_template_version_id", "inspection_template_id"])
        entity = InspectionPlanEntity if kind == "plan" else InspectionFireEntity
        refs = minimal(entity, ["inspection_template_id", "inspection_template_version_id"])
        schema.create_all(engine)
        a, b, foreign, old, current, bv, fv = [uuid7() for _ in range(7)]
        other_domain = 8 if IS_KBOT else uuid7()
        try:
            with engine.begin() as connection:
                connection.execute(roots.insert(), [
                    {"inspection_template_id":key, "domain_id":domain}
                    for key,domain in ((a,store.domain),(b,store.domain),(foreign,other_domain))
                ])
                connection.execute(versions.insert(), [
                    {"inspection_template_version_id":key, "inspection_template_id":root}
                    for key,root in ((old,a),(current,a),(bv,b),(fv,foreign))
                ])
                connection.execute(refs.insert(), [
                    {"inspection_template_id":root, "inspection_template_version_id":version}
                    for root,version in ((a,old),(b,current),(foreign,fv))
                ])
                repo = InspectionRepository(SimpleNamespace(execute=AsyncMock(side_effect=connection.execute)))
                counter = repo.count_plans_by_inspection_templates if kind == "plan" else repo.count_fires_by_inspection_templates
                assert await counter(domain_id=store.domain, template_ids=[a,b,foreign]) == {a:2,b:1}
                assert await counter(domain_id=other_domain, template_ids=[a,b,foreign]) == {foreign:1}
        finally:
            engine.dispose()
    run(case)

def test_plan_binding_locks_root_and_rejects_disabled_or_deleted_template():
    from aiops_agent.application.configuration.inspection_service import InspectionConfigurationMixin
    async def case():
        store = Store()
        row = await store.create()
        await store.publish(row)
        uow = SimpleNamespace(inspections=store.repo)
        service = InspectionConfigurationMixin()
        template, version = await service._inspection_template_version(
            uow=uow, domain_id=store.domain, template_id=row.inspection_template_id,
        )
        assert template is row and version.inspection_template_version_id == row.current_version_id
        store.repo.get_inspection_template.assert_awaited_with(
            domain_id=store.domain, inspection_template_id=row.inspection_template_id, lock=True,
        )
        await store.service.update(**store.command(row), status="INACTIVE")
        with pytest.raises(Exception, match="已启用"):
            await service._inspection_template_version(uow=uow, domain_id=store.domain, template_id=row.inspection_template_id)
        await store.service.delete(**store.command(row))
        with pytest.raises(Exception, match="已启用"):
            await service._inspection_template_version(uow=uow, domain_id=store.domain, template_id=row.inspection_template_id)
    run(case)
