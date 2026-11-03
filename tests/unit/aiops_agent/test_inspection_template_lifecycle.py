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
        self.rows, self.versions, self.plans, self.snapshots = {}, {}, [], {}
        self.repo = SimpleNamespace(
            add_inspection_template=AsyncMock(side_effect=self.add),
            add_inspection_template_version=AsyncMock(side_effect=self.add_version),
            get_inspection_template=AsyncMock(side_effect=self.get),
            get_inspection_template_version=AsyncMock(side_effect=lambda **kw:self.versions.get(kw["inspection_template_version_id"])),
            list_inspection_templates=AsyncMock(side_effect=lambda **kw:[row for row in self.rows.values() if row.domain_id == kw["domain_id"]]),
            list_inspection_template_versions=AsyncMock(side_effect=lambda **kw:sorted([v for v in self.versions.values() if v.inspection_template_id == kw["inspection_template_id"]], key=lambda v:v.version_no, reverse=True)),
            count_plans_by_inspection_templates=AsyncMock(side_effect=lambda **kw:{key:sum(p.inspection_template_id == key and p.domain_id == kw["domain_id"] for p in self.plans) for key in kw["template_ids"]}),
            list_plans_by_inspection_template=AsyncMock(side_effect=lambda **kw:[p for p in self.plans if p.domain_id == kw["domain_id"] and p.inspection_template_id == kw["template_id"]]),
            lock_inspection_template_defaults=AsyncMock(),
            delete_inspection_template=AsyncMock(side_effect=lambda row:self.rows.pop(row.inspection_template_id)),
        )
        self.service = InspectionTemplateService(uow_factory=self.context)

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


@pytest.mark.parametrize("operation", ["save", "publish", "discard", "status"])
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


@pytest.mark.parametrize("operation", ["get","versions","plans","save","publish","discard","default","status"])
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
        service = SimpleNamespace(**{name:AsyncMock(return_value={}) for name in ("save_draft","publish","set_default","discard_draft","list_plans")})
        app.state.inspection_template_service = service
        async with AsyncClient(transport=ASGITransport(app=app),base_url="http://local") as client:
            with patch.object(api,"require_service_scope"):
                template_id=uuid7()
                old=await client.post(f"/internal/v1/aiops/inspection-templates/{template_id}/versions",json={})
                assert old.status_code == 405
                for method,suffix in (("post",":publish"),("post",":default"),("delete","/draft")):
                    response=await client.request(method.upper(),f"/internal/v1/aiops/inspection-templates/{template_id}{suffix}",json={"expected_row_version":1})
                    assert response.status_code in (200,201)
                invalid=await client.post(f"/internal/v1/aiops/inspection-templates/{template_id}:publish",json={"expected_row_version":1,"source_kind":"BUILTIN"})
                assert invalid.status_code == 422
                assert service.publish.await_count == 1
    run(case)
