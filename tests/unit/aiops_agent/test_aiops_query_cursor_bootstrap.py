"""API/Worker 查询游标装配、签名边界和巡检多页查询回归。"""

import ast
import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
import hashlib
import importlib
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiops_agent.application.configuration.common import SignedCursorCodec
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.runtime.service import AIOpsRuntimeService
from aiops_agent.bootstrap.cursors import create_cursor_codec
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_report_queries import _context, _scope


def settings(*, production=False, worker_name="query-worker"):
    return SimpleNamespace(
        platform=SimpleNamespace(debug=False),
        api=SimpleNamespace(service_name="query-api", service_version="test"),
        worker=SimpleNamespace(service_name=worker_name, service_version="test"),
        management=SimpleNamespace(
            cursor_secret_env="TEST_AIOPS_QUERY_CURSOR_SECRET",
            cursor_ttl_seconds=900,
        ),
        is_production=lambda: production,
    )


def test_development_api_and_worker_replicas_accept_the_existing_api_cursor(monkeypatch):
    monkeypatch.delenv("TEST_AIOPS_QUERY_CURSOR_SECRET", raising=False)
    api_codec = create_cursor_codec(settings())
    worker_codec = create_cursor_codec(settings(worker_name="worker-replica-2"))
    previous_codec = SignedCursorCodec(
        secret=hashlib.sha256(b"query-api:development:cursor").hexdigest(),
        ttl_seconds=900,
    )
    scope, filters = _scope(), {"plan_id": None, "status": None}
    now, fire_id = datetime.now(UTC), uuid7()
    for issuer in (api_codec, worker_codec, previous_codec):
        token = issuer.encode(scope=scope, filters=filters, sort_at=now, resource_id=fire_id)
        for receiver in (api_codec, worker_codec, previous_codec):
            assert receiver.decode(token=token, scope=scope, filters=filters) == (now, fire_id)


def test_production_uses_configured_secret_and_ttl(monkeypatch):
    secret = "configured-query-cursor-key-" * 2
    monkeypatch.setenv("TEST_AIOPS_QUERY_CURSOR_SECRET", secret)
    codec = create_cursor_codec(settings(production=True))
    assert codec._ttl_seconds == 900
    scope, filters = _scope(), {"status": None}
    now, fire_id = datetime.now(UTC), uuid7()
    token = codec.encode(scope=scope, filters=filters, sort_at=now, resource_id=fire_id)
    expected = SignedCursorCodec(secret=secret, ttl_seconds=900)
    assert expected.decode(token=token, scope=scope, filters=filters) == (now, fire_id)


def test_missing_production_secret_never_falls_back_to_a_development_key(monkeypatch):
    monkeypatch.delenv("TEST_AIOPS_QUERY_CURSOR_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="生产环境缺少 AIOps Cursor"):
        create_cursor_codec(settings(production=True))


@pytest.mark.parametrize("production", [False, True])
def test_short_configured_secret_is_rejected(monkeypatch, production):
    monkeypatch.setenv("TEST_AIOPS_QUERY_CURSOR_SECRET", "too-short")
    with pytest.raises(ValueError, match="至少需要 32 字节"):
        create_cursor_codec(settings(production=production))


@pytest.mark.parametrize("process", ["api", "worker"])
def test_process_rejects_invalid_cursor_settings_before_opening_resources(monkeypatch, process):
    module = importlib.import_module(f"aiops_agent.bootstrap.{process}")
    factory = module.create_aiops_api if process == "api" else module.create_aiops_worker_probe
    database_factory = next(name for name in vars(module) if name.startswith("create_database_runtime"))
    monkeypatch.delenv("TEST_AIOPS_QUERY_CURSOR_SECRET", raising=False)

    async def run():
        with patch.object(module, "configure_process_logging"), patch.object(
            module, database_factory
        ) as database:
            app = factory(settings(production=True))
            with pytest.raises(RuntimeError, match="生产环境缺少 AIOps Cursor"):
                async with app.router.lifespan_context(app):
                    pytest.fail("无效签名配置不得启动后台进程")
            database.assert_not_called()

    asyncio.run(run())


@pytest.mark.parametrize("process", ["api", "worker"])
def test_both_processes_inject_the_shared_codec_into_runtime_and_change_services(process):
    module = importlib.import_module(f"aiops_agent.bootstrap.{process}")
    factory = module.create_aiops_api if process == "api" else module.create_aiops_worker_probe
    calls = [node for node in ast.walk(ast.parse(inspect.getsource(factory)))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    shared = [node for node in calls if node.func.id == "create_cursor_codec"]
    assert len(shared) == 1 and isinstance(shared[0].args[0], ast.Name)
    assert shared[0].args[0].id == "resolved"
    for service in ("AIOpsRuntimeService", "AIOpsChangeService"):
        constructors = [node for node in calls if node.func.id == service]
        assert constructors
        for node in constructors:
            injected = next((item.value for item in node.keywords if item.arg == "cursor_codec"), None)
            assert isinstance(injected, ast.Name) and injected.id == "cursor_codec"


def test_more_than_twenty_inspection_fires_generate_and_resume_a_signed_cursor(monkeypatch):
    monkeypatch.delenv("TEST_AIOPS_QUERY_CURSOR_SECRET", raising=False)
    now, plan_id = datetime.now(UTC), uuid7()
    fires = [
        SimpleNamespace(
            inspection_fire_id=uuid7(), inspection_plan_id=plan_id,
            plan_snapshot_json={"display_name": "每日巡检"},
            scheduled_for=now-timedelta(minutes=index),
            created_at=now-timedelta(minutes=index), status="COMPLETED",
            target_count=1, completed_count=1, failed_count=0,
            completed_at=now-timedelta(minutes=index), error_code=None, error_message=None,
        )
        for index in range(21)
    ]
    inspections = SimpleNamespace(page_fires=AsyncMock(side_effect=[fires, fires[-1:]]))
    runtime = AIOpsRuntimeService(
        uow_factory=lambda: _context(SimpleNamespace(inspections=inspections)),
        blueprint_registry=Mock(), handler_registry=Mock(),
        cursor_codec=create_cursor_codec(settings(worker_name="worker-replica-2")),
    )
    scope = _scope()

    async def run():
        first = await runtime.list_inspection_fires(
            scope=scope, plan_id=None, status=None, cursor=None, limit=20,
        )
        assert first.has_more and first.next_cursor and len(first.items) == 20
        assert [item.fire_id for item in first.items] == [fire.inspection_fire_id for fire in fires[:20]]
        # API 可验证 Worker 生成的游标，且不能换调用方或查询条件复用。
        api_codec = create_cursor_codec(settings())
        filters = {"plan_id": None, "status": None}
        anchor = api_codec.decode(token=first.next_cursor, scope=scope, filters=filters)
        assert anchor == (fires[19].created_at, fires[19].inspection_fire_id)
        for changed_scope, changed_filters in (
            (replace(scope, principal_id="another-principal"), filters),
            (scope, {"plan_id": None, "status": "FAILED"}),
        ):
            with pytest.raises(AIOpsApplicationError):
                api_codec.decode(token=first.next_cursor, scope=changed_scope, filters=changed_filters)
        second = await runtime.list_inspection_fires(
            scope=scope, plan_id=None, status=None, cursor=first.next_cursor, limit=20,
        )
        assert not second.has_more and second.next_cursor is None
        assert [item.fire_id for item in second.items] == [fires[20].inspection_fire_id]
        assert inspections.page_fires.await_args.kwargs == {
            "domain_id": scope.domain_id, "plan_id": None, "statuses": None,
            "before_created_at": fires[19].created_at,
            "before_id": fires[19].inspection_fire_id, "limit": 21,
        }

    asyncio.run(run())
