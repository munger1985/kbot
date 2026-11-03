"""报告显式分类、两层查询合同及筛选先于分页的离线回归。"""

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Column, MetaData, Table, create_engine

from aiops_agent.api.runtime import routes
from aiops_agent.application.configuration.common import SignedCursorCodec
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.entities import ReportEntity, TargetEntity
from aiops_agent.repositories.inspection import InspectionRepository
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts.aiops.report_filters import (
    REPORT_CATEGORY_TYPES, ReportCategory, category_report_types,
)
from platform_core.contracts.aiops.types import ReportType
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_output_job_api import api
from tests.unit.aiops_agent.test_aiops_output_jobs import Store
from tests.unit.aiops_agent.test_aiops_report_queries import _context, _scope


def service():
    repository = SimpleNamespace(page_current_reports=AsyncMock(return_value=[]))
    runtime = AIOpsRuntimeService(
        uow_factory=lambda: _context(SimpleNamespace(inspections=repository)),
        blueprint_registry=Mock(), handler_registry=Mock(),
        cursor_codec=SignedCursorCodec(secret="report-filter-test-key-" * 2, ttl_seconds=60),
    )
    return runtime, repository


def test_categories_are_exhaustive_and_disjoint():
    classified = [item for types in REPORT_CATEGORY_TYPES.values() for item in types]
    assert len(classified) == len(set(classified))
    assert set(classified) == set(ReportType)
    assert set(REPORT_CATEGORY_TYPES) == set(ReportCategory)
    assert category_report_types(None) is None
    assert category_report_types("DIAGNOSIS") == ("INCIDENT", "PERFORMANCE")


@pytest.mark.parametrize("report_type", list(ReportType))
def test_all_canonical_report_types_can_be_filtered(report_type):
    async def run():
        runtime, repository = service()
        await runtime.list_reports(scope=_scope(), target_id=None,
            report_type=report_type.value, cursor=None, limit=50)
        assert repository.page_current_reports.await_args.kwargs["report_type"] == report_type.value
    asyncio.run(run())


@pytest.mark.parametrize("category", list(ReportCategory))
def test_category_and_template_are_passed_before_repository_pagination(category):
    async def run():
        runtime, repository = service()
        await runtime.list_reports(scope=_scope(), target_id=None,
            report_type=None, category=category, template_ref="domain:exact-template",
            cursor=None, limit=3)
        assert repository.page_current_reports.await_args.kwargs == {
            "domain_id": _scope().domain_id, "target_id": None, "report_type": None,
            "report_types": category_report_types(category), "template_ref": "domain:exact-template",
            "before_created_at": None, "before_id": None, "limit": 4,
        }
    asyncio.run(run())


@pytest.mark.parametrize("filters", [
    {"category": "OTHER"}, {"category": ""}, {"report_type": "UNKNOWN"},
    {"template_ref": ""}, {"template_ref": " "}, {"template_ref": "x" * 129},
    {"template_ref": " system:diagnosis.standard"},
])
def test_invalid_filters_do_not_query_repository(filters):
    async def run():
        runtime, repository = service()
        with pytest.raises(AIOpsApplicationError):
            await runtime.list_reports(**{
                "scope": _scope(), "target_id": None, "report_type": None,
                "cursor": None, "limit": 50, **filters,
            })
        repository.page_current_reports.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("change", ["category", "template_ref", "target_id", "report_type", "domain", "principal"])
def test_signed_cursor_cannot_move_between_filters_or_authorities(change):
    async def run():
        runtime, repository = service()
        scope = _scope()
        filters = {"target_id": None, "report_type": None,
            "category": "DIAGNOSIS", "template_ref": "system:diagnosis.standard"}
        token = runtime._cursor_codec.encode(scope=scope, filters=filters,
            sort_at=datetime.now(UTC), resource_id=uuid7())
        args = {"scope": scope, "target_id": None, "report_type": None,
            "category": "DIAGNOSIS", "template_ref": "system:diagnosis.standard",
            "cursor": token, "limit": 50}
        if change == "domain":
            domain = uuid7() if isinstance(scope.domain_id, type(uuid7())) else scope.domain_id + 1
            args["scope"] = replace(scope, domain_id=domain)
        elif change == "principal":
            args["scope"] = replace(scope, principal_id=str(uuid7()))
        else:
            args[change] = {"category": "INSPECTION", "template_ref": "domain:other",
                "target_id": uuid7(), "report_type": "INCIDENT"}[change]
        with pytest.raises(AIOpsApplicationError):
            await runtime.list_reports(**args)
        repository.page_current_reports.assert_not_awaited()
        args.update(scope=scope, **{key: filters[key] for key in ("target_id", "report_type", "category", "template_ref")})
        await runtime.list_reports(**args)
        assert repository.page_current_reports.await_args.kwargs["before_id"] is not None
    asyncio.run(run())


def test_actual_repository_sql_filters_before_keyset_and_limit():
    """SQLite 只执行真实仓储查询，不代表 PostgreSQL／Oracle 部署和并发验收。"""
    async def run():
        domain = _scope().domain_id
        other_domain = uuid7() if isinstance(domain, type(uuid7())) else domain + 1
        metadata = MetaData()
        original = ReportEntity.__table__
        reports = Table(original.name, metadata, *(
            Column(column.name, column.type, primary_key=column.primary_key, nullable=not column.primary_key)
            for column in original.columns
        ))
        target_table = TargetEntity.__table__
        targets = Table(target_table.name, metadata,
            Column("target_id", target_table.c.target_id.type, primary_key=True),
            Column("domain_id", target_table.c.domain_id.type))
        engine = create_engine("sqlite:///:memory:")
        now, target, other_target = datetime.now(UTC), uuid7(), uuid7()
        wanted = [uuid7() for _ in range(7)]
        try:
            with patch.object(engine.dialect.type_compiler_instance, "visit_JSONB", return_value="JSON", create=True):
                metadata.create_all(engine)
            with engine.begin() as connection:
                connection.execute(targets.insert(), [
                    {"target_id": target, "domain_id": domain},
                    {"target_id": other_target, "domain_id": other_domain},
                ])
                def row(identity, **changes):
                    return {"report_id": identity, "target_id": target, "is_current": True,
                        "report_type": "INCIDENT", "template_id": "domain:exact",
                        "title": "巡检文字不决定分类", "created_at": now, **changes}
                # 超过旧前端 100 条上限的干扰报告更晚，不能挤掉正确筛选的报告。
                distractors = [row(uuid7(), report_type="INSPECTION_DAILY",
                    created_at=now + timedelta(seconds=i + 1)) for i in range(130)]
                connection.execute(reports.insert(), distractors + [row(item) for item in wanted] + [
                    row(uuid7(), target_id=other_target), row(uuid7(), template_id="domain:wrong"),
                    row(uuid7(), is_current=False),
                ])
                repository = InspectionRepository(SimpleNamespace(
                    execute=AsyncMock(side_effect=lambda statement: connection.execute(statement))))
                kwargs = dict(domain_id=domain, target_id=None, report_type=None,
                    report_types=category_report_types("DIAGNOSIS"), template_ref="domain:exact",
                    before_created_at=None, before_id=None, limit=3)
                actual = []
                while True:
                    page = await repository.page_current_reports(**kwargs)
                    actual.extend(page)
                    if not page:
                        break
                    kwargs.update(before_created_at=now, before_id=page[-1])
                assert actual == sorted(wanted, reverse=True)
                # 分类、精确类型和模板是交集，不得因类型冲突扩大到其它分类。
                kwargs.update(before_created_at=None, before_id=None, report_type="INSPECTION_DAILY")
                assert await repository.page_current_reports(**kwargs) == []
        finally:
            engine.dispose()
    asyncio.run(run())


@pytest.mark.parametrize("query", [
    "?category=UNKNOWN", "?report_type=FAKE", "?template_ref=", "?template_ref=" + "x" * 129,
])
def test_internal_http_rejects_invalid_filter_contract(query):
    async def run():
        app = api(Store())
        app.include_router(routes.router)
        app.state.aiops_runtime_service = SimpleNamespace(list_reports=AsyncMock())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            response = await client.get("/internal/v1/aiops/reports" + query)
        assert response.status_code == 422
        app.state.aiops_runtime_service.list_reports.assert_not_awaited()
    asyncio.run(run())


def test_internal_http_and_client_preserve_exact_filters():
    async def run():
        runtime = SimpleNamespace(list_reports=AsyncMock(return_value={"items": [], "has_more": False}))
        app = api(Store())
        app.include_router(routes.router)
        app.state.aiops_runtime_service = runtime
        query = "?category=SPECIAL_ANALYSIS&report_type=POSTGRESQL_PGBADGER&template_ref=domain%3Aexact&cursor=signed&limit=7"
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://aiops.test") as client:
            response = await client.get("/internal/v1/aiops/reports" + query)
        assert response.status_code == 200, response.text
        kwargs = runtime.list_reports.await_args.kwargs
        assert kwargs["category"] == "SPECIAL_ANALYSIS"
        assert kwargs["template_ref"] == "domain:exact"
        upstream = SimpleNamespace(_json=AsyncMock(return_value={}))
        await AIOpsManagementClient.list_reports(upstream, target_id=None,
            report_type="POSTGRESQL_PGBADGER", category="SPECIAL_ANALYSIS", template_ref="domain:exact",
            cursor="signed", limit=7, auth_context=object())
        forwarded = parse_qs(urlsplit(upstream._json.await_args.args[1]).query)
        assert forwarded == {"limit": ["7"], "report_type": ["POSTGRESQL_PGBADGER"],
            "category": ["SPECIAL_ANALYSIS"], "template_ref": ["domain:exact"], "cursor": ["signed"]}
    asyncio.run(run())


def test_public_adapter_preserves_exact_filters():
    async def run():
        from main_api.api import ops
        client = SimpleNamespace(list_reports=AsyncMock(return_value={"items": [], "has_more": False}))
        authority = object()
        with patch.object(ops, "_client", return_value=client):
            await ops.list_reports(SimpleNamespace(state=SimpleNamespace(auth_context=authority)),
                target_id=None, report_type=ReportType.POSTGRESQL_PGBADGER,
                category=ReportCategory.SPECIAL_ANALYSIS, template_ref="domain:exact",
                cursor="signed", limit=7)
        client.list_reports.assert_awaited_once_with(target_id=None,
            report_type="POSTGRESQL_PGBADGER", category="SPECIAL_ANALYSIS",
            template_ref="domain:exact", cursor="signed", limit=7, auth_context=authority)
    asyncio.run(run())
