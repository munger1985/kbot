"""固定案例资源的部署补缺只影响指定空间，不重置任何业务配置。"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from platform_core.identity import uuid7
from scripts.deployment import provision_aiops_case_collection as provisioning


ROOT = Path(__file__).resolve().parents[3]
IS_TENANT = "tenant_id" in inspect.signature(provisioning.provision_case_collection).parameters


class Result:
    def __init__(self, *, scalar=None, rows=()):
        self.scalar, self.rows = scalar, rows

    def scalar_one_or_none(self):
        return self.scalar

    def mappings(self):
        return self

    def all(self):
        return self.rows


class Connection:
    def __init__(self, scope):
        self.scope, self.rows, self.calls, self.bindings = scope, [], [], []
        self.allowed = True
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.manual = {"collection_id": str(uuid7()), "models": {"embedding": str(uuid7())}}

    async def execute(self, statement, params=None):
        sql, params = str(statement), params or {}
        self.calls.append((sql, deepcopy(params)))
        if "INSERT INTO platform_resource_domain_collection" in sql:
            self.bindings.append({key: params[key] for key in ("tenant_id", "domain_id")}
                | {"access_mode": "manage"})
            return Result()
        if "FROM platform_resource_domain_collection WHERE" in sql:
            return Result(rows=deepcopy(self.bindings))
        if "SELECT domain." in sql:
            return Result(scalar=self.scope["domain_id"] if self.allowed else None)
        if "SELECT collection_id" in sql or 'COLLECTION_ID AS "collection_id"' in sql:
            return Result(rows=deepcopy(self.rows))
        if "INSERT INTO" in sql:
            row = {key: json.loads(params[key]) for key in
                ("models_json", "parse_policy_json", "metadata_json")}
            row.update(status="ACTIVE")
            if IS_TENANT:
                row.update(collection_id=params["collection_id"], tenant_id=params["tenant_id"],
                    source_system="aiops", external_key=params["external_key"])
            else:
                from uuid import UUID
                row.update(collection_id=UUID(hex=params["collection_hex"]),
                    domain_id=params["domain_id"], display_name="diagnosis-cases")
            self.rows.append(row)
            return Result()
        raise AssertionError(f"测试未允许的 SQL：{sql}")

    @asynccontextmanager
    async def connect(self):
        yield self


def fixture():
    scope = {"domain_id": uuid7() if IS_TENANT else 73,
        "actor_id": uuid7() if IS_TENANT else "deployment:case-resource"}
    if IS_TENANT:
        scope["tenant_id"] = uuid7()
    return Connection(scope), scope


def test_fresh_resource_is_text_only_empty_models_and_separate_from_manuals():
    async def run():
        connection, scope = fixture()
        manual = deepcopy(connection.manual)
        result = await provisioning.provision_case_collection(connection, **scope)
        assert result["created"] is True
        assert result["parse_strategy"] == "TEXT"
        assert result["models"] == {}
        assert result["configuration_status"] == "MODELS_REQUIRED"
        assert result["missing_model_roles"] == ["embedding", "parser_llm", "retrieval_llm"]
        assert result["retrieval_requires_publication"] is True
        from uuid import UUID
        assert UUID(result["collection_id"]).version == 7
        assert connection.manual == manual
        assert len(connection.calls) == (5 if IS_TENANT else 3)
        assert "FOR UPDATE" in connection.calls[0][0]
        assert connection.rows[0]["metadata_json"]["content_kind"] == "diagnosis_case"
        connection.commit.assert_not_awaited()
    asyncio.run(run())


@pytest.mark.parametrize("check_only", [False, True])
@pytest.mark.parametrize("strategy", ["TEXT", "AUTO", "VISUAL"])
def test_existing_resource_preserves_models_policy_metadata_and_identifier(check_only, strategy):
    async def run():
        connection, scope = fixture()
        await provisioning.provision_case_collection(connection, **scope)
        connection.rows[0]["models_json"] = {key: str(uuid7()) for key in
            ("parser_llm", "retrieval_llm", "embedding")}
        connection.rows[0]["parse_policy_json"] = {"parse_strategy": strategy, "custom": True}
        connection.rows[0]["metadata_json"]["operator_note"] = "管理员配置"
        original = deepcopy(connection.rows)
        connection.calls.clear()
        result = await provisioning.provision_case_collection(connection, **scope, check_only=check_only)
        assert result["created"] is False
        assert result["configuration_status"] == ("TEXT_POLICY_REQUIRED" if strategy == "VISUAL" else "CONFIGURED")
        assert connection.rows == original
        assert all(sql.lstrip().startswith("SELECT") for sql, _ in connection.calls)
        assert ("FOR UPDATE" in connection.calls[0][0]) is not check_only
    asyncio.run(run())


def test_check_only_missing_resource_fails_without_writes():
    async def run():
        connection, scope = fixture()
        with pytest.raises(provisioning.CaseResourceProvisioningError, match="缺失"):
            await provisioning.provision_case_collection(connection, **scope, check_only=True)
        assert connection.rows == []
        assert all(sql.lstrip().startswith("SELECT") and "FOR UPDATE" not in sql
            for sql, _ in connection.calls)
    asyncio.run(run())


@pytest.mark.parametrize("check_only", [False, True])
def test_wrong_or_inactive_domain_is_rejected_before_reading_any_collection(check_only):
    async def run():
        connection, scope = fixture()
        connection.allowed = False
        with pytest.raises(provisioning.CaseResourceProvisioningError, match="空间"):
            await provisioning.provision_case_collection(connection, **scope, check_only=check_only)
        assert len(connection.calls) == 1
        sql, params = connection.calls[0]
        assert params["domain_id"] == scope["domain_id"]
        if IS_TENANT:
            assert params["tenant_id"] == scope["tenant_id"]
            assert "domain.owner_app_id='aiops'" in sql
            assert "domain.domain_type='APP_OWNED'" in sql
            assert "installation.status='ACTIVE'" in sql
        else:
            assert "other_app.APP_ID<>'aiops'" in sql
            assert "app_domain.APP_ID='aiops'" in sql
            assert "app.STATUS='ACTIVE'" in sql
    asyncio.run(run())


@pytest.mark.parametrize("dimension", ["duplicate", "owner", "fixed", "kind", "scope", "inactive"])
def test_existing_conflict_does_not_reactivate_repair_or_merge(dimension):
    async def run():
        connection, scope = fixture()
        await provisioning.provision_case_collection(connection, **scope)
        row = connection.rows[0]
        if dimension == "duplicate":
            connection.rows.append(deepcopy(row))
        elif dimension == "owner":
            row["metadata_json"]["owner_app_id"] = "km"
        elif dimension == "fixed":
            row["metadata_json"]["fixed_resource"] = "true"
        elif dimension == "kind":
            row["metadata_json"]["content_kind"] = "operations_manual"
        elif dimension == "inactive":
            row["status"] = "DISABLED"
        elif IS_TENANT:
            row["metadata_json"]["domain_id"] = str(uuid7())
        else:
            row["domain_id"] += 1
        before = deepcopy(connection.rows)
        connection.calls.clear()
        with pytest.raises(provisioning.CaseResourceProvisioningError):
            await provisioning.provision_case_collection(connection, **scope)
        assert connection.rows == before
        assert all(sql.lstrip().startswith("SELECT") for sql, _ in connection.calls)
    asyncio.run(run())


@pytest.mark.parametrize("check_only", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_cli_transaction_owner_commits_only_successful_write_and_always_closes(check_only, failure):
    async def run():
        connection, scope = fixture()
        runtime = SimpleNamespace(engine=connection, close=AsyncMock())
        command = AsyncMock(side_effect=RuntimeError("数据库细节")) if failure else AsyncMock(return_value={"ok": True})
        settings_module = ("aiops_agent.config.get_aiops_settings" if IS_TENANT
            else "platform_core.config.get_settings")
        factory_module = ("platform_core.database.create_database_runtime_from_settings" if IS_TENANT
            else "platform_core.database.oracle.create_database_runtime")
        with patch(settings_module, return_value=object()), patch(factory_module, return_value=runtime), \
                patch.object(provisioning, "provision_case_collection", command):
            if failure:
                with pytest.raises(RuntimeError):
                    await provisioning.run(**scope, check_only=check_only)
            else:
                assert await provisioning.run(**scope, check_only=check_only) == {"ok": True}
        assert connection.commit.await_count == int(not failure and not check_only)
        connection.rollback.assert_awaited_once()
        runtime.close.assert_awaited_once()
    asyncio.run(run())


def test_canonical_initialization_uses_scoped_seed_or_preserves_existing_case_resource():
    if IS_TENANT:
        source = (ROOT / "scripts/deployment/init_postgresql.py").read_text()
        assert "await seed_enabled_case_collections(connection)" in source
        assert source.index("await _align_platform_boundary_schema(connection, schema=schema)") < source.index(
            "await seed_enabled_case_collections(connection)")
        seed = inspect.getsource(provisioning.seed_enabled_case_collections)
        assert "installation.status='ACTIVE'" in seed
        assert "await provision_case_collection(connection" in seed
    else:
        source = (ROOT / "database/oracle/bootstrap/aiops/initial_admin.sql").read_text()
        case_branch = source.split("ELSIF p_content_kind='diagnosis_case' THEN", 1)[1].split("\n        ELSE", 1)[0]
        assert "SELECT COUNT(*)" in case_branch
        assert "UPDATE " not in case_branch and "INSERT " not in case_branch
        assert "CASE WHEN p_content_kind='diagnosis_case' THEN '{\"parse_strategy\":\"TEXT\"}'" in source
    source = inspect.getsource(provisioning.provision_case_collection)
    assert not re.search(r"^\s*(UPDATE|DELETE)\s", source, re.MULTILINE)
    assert "USER_TABLES" not in source and "information_schema" not in source


@pytest.mark.parametrize("failure", [
    provisioning.CaseResourceProvisioningError("固定资源缺失"),
    RuntimeError("password=不可泄露"),
])
def test_cli_reports_safe_diagnostics_without_raw_driver_error(failure, capsys):
    args = ["case-resource", "--domain-id", str(uuid7() if IS_TENANT else 73),
        "--actor-id", str(uuid7()) if IS_TENANT else "deployment:case-resource"]
    if IS_TENANT:
        args += ["--tenant-id", str(uuid7())]
    with patch("sys.argv", args), patch.object(provisioning, "run", AsyncMock(side_effect=failure)):
        assert provisioning.main() == 1
    output = capsys.readouterr().out
    assert "不可泄露" not in output
    if isinstance(failure, provisioning.CaseResourceProvisioningError):
        assert "固定资源缺失" in output

@pytest.mark.skipif(not IS_TENANT, reason="仅 PostgreSQL 使用独立业务空间资源绑定")
@pytest.mark.parametrize("mode", ["missing", "cross_domain", "cross_tenant", "read_only", "duplicate"])
def test_domain_binding_is_required_and_never_silently_escalated(mode):
    async def run():
        connection, scope = fixture()
        await provisioning.provision_case_collection(connection, **scope)
        if mode == "missing":
            connection.bindings.clear()
        elif mode == "cross_domain":
            connection.bindings[0]["domain_id"] = uuid7()
        elif mode == "cross_tenant":
            connection.bindings[0]["tenant_id"] = uuid7()
        elif mode == "read_only":
            connection.bindings[0]["access_mode"] = "consume"
        else:
            connection.bindings.append(deepcopy(connection.bindings[0]))
        original = deepcopy(connection.bindings)
        connection.calls.clear()
        with pytest.raises(provisioning.CaseResourceProvisioningError):
            await provisioning.provision_case_collection(connection, **scope, check_only=True)
        assert connection.bindings == original
        assert all(sql.lstrip().startswith("SELECT") for sql, _ in connection.calls)
        connection.calls.clear()
        if mode == "missing":
            result = await provisioning.provision_case_collection(connection, **scope)
            assert result["created"] is False
            assert connection.bindings[0]["domain_id"] == scope["domain_id"]
            assert connection.bindings[0]["access_mode"] == "manage"
            assert len([sql for sql, _ in connection.calls if sql.lstrip().startswith("INSERT")]) == 1
        else:
            with pytest.raises(provisioning.CaseResourceProvisioningError):
                await provisioning.provision_case_collection(connection, **scope)
            assert connection.bindings == original
            assert all(sql.lstrip().startswith("SELECT") for sql, _ in connection.calls)
    asyncio.run(run())
