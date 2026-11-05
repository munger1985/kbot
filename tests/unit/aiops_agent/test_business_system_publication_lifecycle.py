"""业务系统发布与上下线的事务回归；SQLite 仅验证本地事务，不冒充 Oracle/PG 并发验收。"""

import asyncio
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.dialects import oracle, postgresql
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session

from aiops_agent.application.agent_boundaries import summarize_boundary_resources
from aiops_agent.application.configuration import AIOpsConfigurationService, ConfigurationScope
from aiops_agent.application.configuration.common import IdempotencyGuard
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import (
    BusinessSystemEntity, BusinessSystemResourceEntity, ComputeResourceEntity, DatabaseResourceEntity,
    DiagnosticSourceEntity, InboxEntity, ManagedResourceEntity, OutboxEntity,
    RelationParticipantEntity, ResourceRelationEntity, TargetEntity, ResourceSourceBindingEntity,
    AIOpsAgentVersionBusinessSystemEntity, BusinessSystemDraftEntity, BusinessSystemDraftChangeEntity,
    InspectionPlanEntity, InspectionFireEntity, OpsRunEntity, ResourceDiscoveryJobEntity,
)
from aiops_agent.repositories.business_system import BusinessSystemRepository
from aiops_agent.repositories.messaging import InboxRepository, OutboxRepository
from aiops_agent.repositories.monitoring import DiagnosticSourceRepository
from aiops_agent.repositories.target import TargetRepository
from platform_core.contracts.aiops import BusinessSystemDraftCreate, BusinessSystemLifecycleCommand, BusinessSystemResourceGraphCommit
from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialCipher

TABLES = (
    BusinessSystemEntity, BusinessSystemResourceEntity, ComputeResourceEntity, DatabaseResourceEntity, DiagnosticSourceEntity,
    InboxEntity, ManagedResourceEntity, OutboxEntity, RelationParticipantEntity,
    ResourceRelationEntity, TargetEntity, ResourceSourceBindingEntity,
    AIOpsAgentVersionBusinessSystemEntity, BusinessSystemDraftEntity, BusinessSystemDraftChangeEntity,
    InspectionPlanEntity, InspectionFireEntity, OpsRunEntity, ResourceDiscoveryJobEntity,
)


@compiles(postgresql.JSONB, "sqlite")
def compile_test_jsonb(_type, _compiler, **_kwargs):
    """仅测试适配 PostgreSQL JSONB，不修改生产模型或持久化合同。"""
    return "JSON"


class SessionBridge:
    """把同步内存事务交给现有异步仓储；不模拟锁或生产驱动。"""

    def __init__(self, session):
        self.sync = session
        self.add = session.add
        self.add_all = session.add_all

    async def execute(self, *args, **kwargs):
        return self.sync.execute(*args, **kwargs)

    async def flush(self):
        self.sync.flush()

    async def scalar(self, *args, **kwargs):
        return self.sync.scalar(*args, **kwargs)

    async def scalars(self, *args, **kwargs):
        return self.sync.scalars(*args, **kwargs)


class LocalUow:
    def __init__(self, engine):
        self.sync = Session(engine, expire_on_commit=False, autoflush=False)
        self.session = SessionBridge(self.sync)
        self.targets = TargetRepository(self.session)
        self.managed_credentials = object()
        self.business_systems = BusinessSystemRepository(self.session)
        self.diagnostic_sources = DiagnosticSourceRepository(self.session)
        self.inbox = InboxRepository(self.session)
        self.outbox = OutboxRepository(self.session)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.sync.rollback()
        self.sync.close()

    async def flush(self):
        self.sync.flush()

    async def commit(self):
        self.sync.commit()


@pytest.fixture
def lab():
    engine = create_engine("sqlite://")
    metadata = BusinessSystemEntity.metadata
    metadata.create_all(engine, tables=[entity.__table__ for entity in TABLES])
    has_tenant = "tenant_id" in ConfigurationScope.__dataclass_fields__
    scope_values = dict(domain_id=uuid7() if has_tenant else 87,
        principal_id="PORTAL:lifecycle-test", actor_id=str(uuid7()), request_id="publish-test", trace_id="publish-test")
    if has_tenant:
        scope_values["tenant_id"] = uuid7()
    scope = ConfigurationScope(**scope_values)
    owned = {"domain_id": scope.domain_id}
    if has_tenant:
        owned["tenant_id"] = scope.tenant_id
    source_id, group_id = uuid7(), uuid7()
    with Session(engine) as session:
        session.add(DiagnosticSourceEntity(**owned, diagnostic_source_id=source_id,
            display_name="OEM 测试源", source_type="OEM", adapter_id="oem", adapter_version="1.0.0",
            status="ENABLED", created_by=scope.actor_id, updated_by=scope.actor_id))
        session.commit()
    service = object.__new__(AIOpsConfigurationService)
    service._uow_factory = lambda: LocalUow(engine)
    service._now = lambda: datetime.now(UTC)
    service._idempotency = IdempotencyGuard()
    service._credential_cipher = ManagedCredentialCipher(key=b"l" * 32, key_version="test-v1")
    service._managed_credentials = SimpleNamespace(read=AsyncMock(return_value={"username": "reader"}), revoke=AsyncMock())
    service._validate_group = AsyncMock()

    async def create_target(*, uow, scope, request, now):
        target = TargetEntity(**owned, display_name=request.display_name, db_type=request.db_type,
            version_code=request.version_code, environment=request.environment,
            oracle_container_scope=request.oracle_container_scope,
            endpoint_json=request.endpoint.model_dump(mode="json"), readonly_connection_enabled=True,
            diagnostic_credential_id=uuid7(), status="DISABLED", connectivity_status="CONNECTED",
            last_connectivity_success_at=now, created_by=scope.actor_id, updated_by=scope.actor_id)
        return await uow.targets.add_target(target)

    async def bind(*, uow, target_id, **kwargs):
        resource = await uow.business_systems.get_resource_for_target(target_id=target_id, **owned)
        return await uow.targets._add(ResourceSourceBindingEntity(resource_id=resource.resource_id, target_id=target_id,
            diagnostic_source_id=source_id, source_locator_key="candidate-" + str(target_id),
            source_locator_json={"target_guid": str(target_id)}, created_by=scope.actor_id, updated_by=scope.actor_id))

    service._create_target_in_uow = AsyncMock(side_effect=create_target)
    service._create_discovered_source_binding_in_uow = AsyncMock(side_effect=bind)
    service._check_target_connection_now = AsyncMock()
    service._check_target_connectivity_now = AsyncMock()
    request = BusinessSystemResourceGraphCommit.model_validate({
        "system": {"display_name": "OEM 联调系统", "environment": "DEV", "responsibility_group_id": str(group_id)},
        "resources": [{"draft_resource_id": str(uuid7()), "mode": "NEW", "resource_type_key": "DATABASE",
            "display_name": "Oracle CDB", "owner_group_id": str(group_id), "architecture_layer_key": "DATA",
            "database": {"target": {"display_name": "Oracle CDB", "db_type": "ORACLE", "version_code": "19c",
                "environment": "DEV", "oracle_container_scope": "CDB_ROOT", "readonly_connection_enabled": True,
                "endpoint": {"host": "127.0.0.1", "port": 1521, "service": "ORCLCDB"},
                "diagnostic_credential": {"username": "reader", "password": "test-only-not-a-real-secret"}},
                "monitoring": {"source_id": str(source_id), "candidate_ref": "signed-test-candidate"}}}],
    })
    fixture = SimpleNamespace(engine=engine, service=service, scope=scope, owned=owned,
        source_id=source_id, request=request, key=str(uuid7()))
    yield fixture
    engine.dispose()


async def publish(f):
    return await f.service.commit_business_system(scope=f.scope, request=f.request, idempotency_key=f.key)


async def transition(f, detail, command, reason=None):
    return await f.service.command_business_system(scope=f.scope, business_system_id=detail.business_system_id,
        command=command, request=BusinessSystemLifecycleCommand(reason=reason),
        expected_version=detail.row_version, idempotency_key=str(uuid7()))


async def readiness(f, detail):
    async with f.service._uow_factory() as uow:
        system = await uow.business_systems.get_scoped(
            business_system_id=detail.business_system_id, **f.owned)
        return (await summarize_boundary_resources(uow, systems=[system], domain_id=f.scope.domain_id))["items"][0]


def test_publish_is_online_and_agent_monitoring_and_readonly_are_ready(lab):
    async def run():
        detail = await publish(lab)
        assert detail.status == "ENABLED"
        row = await readiness(lab, detail)
        assert row["target_enabled"] and row["monitoring_configured"] and row["readonly_configured"]
        assert not row["change_configured"]
        with Session(lab.engine) as session:
            target = session.get(TargetEntity, detail.resources[0].resource.database.target_id)
            assert target.status == "ENABLED" and not target.controlled_change_enabled
            assert detail.resources[0].resource.database.target_row_version == target.row_version
            assert session.scalar(select(func.count()).select_from(OutboxEntity).where(
                OutboxEntity.event_type == "TARGET_ENABLED")) == 1
        replay = await publish(lab)
        assert replay.business_system_id == detail.business_system_id
        lab.service._create_target_in_uow.assert_awaited_once()
    asyncio.run(run())


def test_offline_and_reonline_replace_separate_target_toggles(lab):
    async def run():
        detail = await publish(lab)
        target_id = detail.resources[0].resource.database.target_id
        with Session(lab.engine) as session:
            original = session.get(TargetEntity, target_id)
            credential_id, endpoint = original.diagnostic_credential_id, original.endpoint_json
        offline = await transition(lab, detail, "disable", "联调下线")
        row = await readiness(lab, offline)
        assert offline.status == "DISABLED"
        assert not row["target_enabled"] and not row["monitoring_configured"] and not row["readonly_configured"]
        with Session(lab.engine) as session:
            target = session.get(TargetEntity, target_id)
            assert target.status == "DISABLED"
            assert target.diagnostic_credential_id == credential_id and target.endpoint_json == endpoint
            assert session.scalar(select(func.count()).select_from(ResourceSourceBindingEntity)) == 1
        online = await transition(lab, offline, "enable")
        row = await readiness(lab, online)
        assert row["monitoring_configured"] and row["readonly_configured"]
        with Session(lab.engine) as session:
            events = session.scalars(select(OutboxEntity.event_type)).all()
            assert events.count("TARGET_ENABLED") == 2 and events.count("TARGET_DISABLED") == 1
    asyncio.run(run())


@pytest.mark.parametrize("failure", ["connection", "source_disabled", "no_mapping", "expired"])
def test_failed_publication_rolls_back_entire_resource_graph_and_events(lab, failure):
    async def fail_connection(**kwargs):
        kwargs["entity"].connectivity_status = "UNREACHABLE"
        kwargs["entity"].last_error_code = "TEST_CONNECTION_FAILED"
    async def expire(**kwargs):
        kwargs["entity"].last_connectivity_success_at = datetime.now(timezone.utc) - timedelta(hours=3)
    if failure == "connection":
        lab.service._check_target_connection_now.side_effect = fail_connection
        lab.service._check_target_connectivity_now.side_effect = fail_connection
    if failure == "expired":
        lab.service._check_target_connection_now.side_effect = expire
        lab.service._check_target_connectivity_now.side_effect = expire
    if failure == "source_disabled":
        with Session(lab.engine) as session:
            session.get(DiagnosticSourceEntity, lab.source_id).status = "DISABLED"
            session.commit()
    if failure == "no_mapping":
        lab.service._create_discovered_source_binding_in_uow.side_effect = None
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(publish(lab))
    with Session(lab.engine) as session:
        for entity in TABLES:
            if entity is not DiagnosticSourceEntity:
                assert session.scalar(select(func.count()).select_from(entity)) == 0


def test_reonline_validation_failure_preserves_offline_state(lab):
    async def run():
        offline = await transition(lab, await publish(lab), "disable", "测试")
        with Session(lab.engine) as session:
            session.get(DiagnosticSourceEntity, lab.source_id).status = "DISABLED"
            session.commit()
        with pytest.raises(AIOpsApplicationError, match="监控源"):
            await transition(lab, offline, "enable")
        with Session(lab.engine) as session:
            assert session.get(BusinessSystemEntity, offline.business_system_id).status == "DISABLED"
            assert session.get(TargetEntity, offline.resources[0].resource.database.target_id).status == "DISABLED"
    asyncio.run(run())


def test_shared_target_stays_online_until_last_system_goes_offline(lab):
    async def run():
        first = await publish(lab)
        with Session(lab.engine) as session:
            shared = BusinessSystemEntity(**lab.owned, display_name="共享范围", system_code="SYS-SHARED",
                environment="DEV", responsibility_group_id=uuid7(), status="ENABLED",
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id)
            session.add(shared)
            session.flush()
            shared_id = shared.business_system_id
            session.add(BusinessSystemResourceEntity(business_system_id=shared_id,
                resource_id=first.resources[0].resource.resource_id, architecture_layer_key="DATA",
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.commit()
        offline = await transition(lab, first, "disable", "当前系统下线")
        with Session(lab.engine) as session:
            assert session.get(TargetEntity, first.resources[0].resource.database.target_id).status == "ENABLED"
        row = await readiness(lab, offline)
        assert not row["scope_enabled"] and not row["monitoring_configured"]
        async with lab.service._uow_factory() as uow:
            system = await uow.business_systems.get_scoped(business_system_id=shared_id, **lab.owned)
            second = await lab.service._system_detail(uow=uow, scope=lab.scope, system=system)
        await transition(lab, second, "disable", "最后一个系统下线")
        with Session(lab.engine) as session:
            assert session.get(TargetEntity, first.resources[0].resource.database.target_id).status == "DISABLED"
    asyncio.run(run())


def test_managed_target_rejects_independent_enable_disable(lab):
    async def run():
        detail = await publish(lab)
        target_id = detail.resources[0].resource.database.target_id
        with Session(lab.engine) as session:
            version = int(session.get(TargetEntity, target_id).row_version)
        with pytest.raises(AIOpsApplicationError, match="所属业务系统"):
            await lab.service.command_target(scope=lab.scope, target_id=target_id, command="disable",
                expected_version=version, idempotency_key=str(uuid7()))
        with Session(lab.engine) as session:
            assert session.get(TargetEntity, target_id).status == "ENABLED"
    asyncio.run(run())


def test_later_database_failure_rolls_back_already_enabled_targets(lab):
    payload = lab.request.model_dump(mode="json")
    second = dict(payload["resources"][0])
    second["draft_resource_id"] = str(uuid7())
    second["display_name"] = "第二个数据库"
    second["database"] = {**second["database"], "target": {**second["database"]["target"], "display_name": "第二个数据库"}}
    payload["resources"].append(second)
    lab.request = BusinessSystemResourceGraphCommit.model_validate(payload)
    original = lab.service._validate_target_enablement
    validated = []

    async def validate(**kwargs):
        validated.append(kwargs["entity"].target_id)
        await original(**kwargs)
        if len(validated) == 2:
            raise AIOpsApplicationError(code="TEST_LATE_FAILURE", message="第二个端点校验失败", status_code=409)

    lab.service._validate_target_enablement = validate
    with pytest.raises(AIOpsApplicationError, match="第二个端点"):
        asyncio.run(publish(lab))
    assert len(validated) == 2
    with Session(lab.engine) as session:
        for entity in (BusinessSystemEntity, TargetEntity, ManagedResourceEntity, OutboxEntity, InboxEntity):
            assert session.scalar(select(func.count()).select_from(entity)) == 0


def test_initial_online_from_draft_and_stale_version_does_not_toggle_targets(lab):
    async def run():
        detail = await publish(lab)
        target_id = detail.resources[0].resource.database.target_id
        with Session(lab.engine) as session:
            system = session.get(BusinessSystemEntity, detail.business_system_id)
            system.status = "DRAFT"
            session.get(TargetEntity, target_id).status = "DISABLED"
            session.commit()
            draft = detail.model_copy(update={"status": "DRAFT", "row_version": int(system.row_version)})
        with pytest.raises(AIOpsApplicationError):
            await transition(lab, detail, "enable")
        online = await transition(lab, draft, "enable")
        assert online.status == "ENABLED"
        assert (await readiness(lab, online))["readonly_configured"]
    asyncio.run(run())


@pytest.mark.parametrize("foreign", ["offline_system", "inactive_link", "inactive_resource", "other_domain"])
def test_non_running_or_other_space_references_do_not_keep_target_online(lab, foreign):
    async def run():
        detail = await publish(lab)
        target_id = detail.resources[0].resource.database.target_id
        owned = dict(lab.owned)
        if foreign == "other_domain":
            owned["domain_id"] = uuid7() if "tenant_id" in owned else 99
        with Session(lab.engine) as session:
            session.get(BusinessSystemEntity, detail.business_system_id).status = "DISABLED"
            other = BusinessSystemEntity(**owned, display_name="其他系统", system_code="SYS-OTHER",
                environment="DEV", responsibility_group_id=uuid7(),
                status="DISABLED" if foreign == "offline_system" else "ENABLED",
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id)
            session.add(other)
            session.flush()
            resource_id = detail.resources[0].resource.resource_id
            if foreign == "inactive_resource":
                session.get(ManagedResourceEntity, resource_id).lifecycle_status = "INACTIVE"
            session.add(BusinessSystemResourceEntity(business_system_id=other.business_system_id,
                resource_id=resource_id, architecture_layer_key="DATA",
                status="INACTIVE" if foreign == "inactive_link" else "ACTIVE",
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.commit()
        async with lab.service._uow_factory() as uow:
            assert not await uow.business_systems.has_enabled_system_for_target(
                target_id=target_id, **lab.owned)
    asyncio.run(run())


def test_published_system_passes_agent_active_target_eligibility(lab):
    from aiops_agent.application.agents import AIOpsAgentService

    async def run():
        detail = await publish(lab)
        agent = object.__new__(AIOpsAgentService)
        agent._action_registry = None
        agent._validate_production_llm_providers = AsyncMock()
        values = {**lab.owned, "allowed_business_system_ids": [detail.business_system_id],
            "capabilities": ["MONITORING_QUERY"], "knowledge_policy": {"enabled": False},
            "models": {"planner_llm": str(uuid7()), "composer_llm": str(uuid7())}}
        async with lab.service._uow_factory() as uow:
            targets = await agent._validate_resources(uow, lab.scope.domain_id, "ACTIVE", values)
            assert len(targets) == 1 and targets[0].status == "ENABLED"
    asyncio.run(run())


@pytest.mark.parametrize("online", [True, False])
def test_adding_database_uses_system_state_without_implicitly_reonlining(lab, online):
    async def run():
        detail = await publish(lab)
        if not online:
            detail = await transition(lab, detail, "disable", "编辑资源")
        resource = lab.request.resources[0].model_copy(update={"display_name": "新增数据库"})
        result = await lab.service.add_system_resource(scope=lab.scope,
            business_system_id=detail.business_system_id, request=resource, idempotency_key=str(uuid7()))
        assert result.status == ("ENABLED" if online else "DISABLED")
        assert len(result.resources) == 2
        with Session(lab.engine) as session:
            states = session.scalars(select(TargetEntity.status)).all()
            assert states == ["ENABLED" if online else "DISABLED"] * 2
    asyncio.run(run())


@pytest.mark.parametrize("dialect", [oracle.dialect(), postgresql.dialect()])
def test_shared_reference_query_is_scoped_and_lock_refreshes_identity_map(lab, dialect):
    async def run():
        result = Mock()
        result.scalar_one_or_none.return_value = None
        session = SimpleNamespace(execute=AsyncMock(return_value=result))
        repository = BusinessSystemRepository(session)
        assert not await repository.has_enabled_system_for_target(target_id=uuid7(), **lab.owned)
        statement = session.execute.await_args.args[0]
        sql = str(statement.compile(dialect=dialect)).upper()
        for field in ("DOMAIN_ID", "LIFECYCLE_STATUS", "TARGET_ID", "STATUS"):
            assert field in sql
        assert {"ENABLED", "ACTIVE"} <= set(statement.compile(dialect=dialect).params.values())
        if "tenant_id" in lab.owned:
            assert "TENANT_ID" in sql
        await TargetRepository(session).get_scoped(target_id=uuid7(), domain_id=lab.scope.domain_id, lock=True, refresh=True)
        assert session.execute.await_args.args[0].get_execution_options()["populate_existing"]
    asyncio.run(run())


async def remove_system(f, detail):
    await f.service.delete_business_system(
        scope=f.scope, business_system_id=detail.business_system_id, expected_version=detail.row_version,
    )


def test_offline_deletion_preserves_history_and_retires_exclusive_resources(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "准备删除")
        resource_id = detail.resources[0].resource.resource_id
        target_id = detail.resources[0].resource.database.target_id
        await remove_system(lab, offline)
        with Session(lab.engine) as session:
            system = session.get(BusinessSystemEntity, detail.business_system_id)
            resource = session.get(ManagedResourceEntity, resource_id)
            target = session.get(TargetEntity, target_id)
            binding = session.scalar(select(ResourceSourceBindingEntity))
            link = session.scalar(select(BusinessSystemResourceEntity))
            assert system.status == "DELETED" and resource.lifecycle_status == "RETIRED"
            assert target.status == "DISABLED" and not target.readonly_connection_enabled
            assert binding.status == "DISABLED" and link.status == "INACTIVE"
            event = session.scalar(select(OutboxEntity).where(OutboxEntity.event_type == "BUSINESS_SYSTEM_DELETED"))
            assert event.payload_json["details"]["deletion_mode"] == "LOGICAL"
            assert event.payload_json["details"]["retired_resource_ids"] == [str(resource_id)]
            assert session.scalar(select(func.count()).select_from(OutboxEntity)) >= 4
        lab.service._managed_credentials.revoke.assert_awaited_once()
        assert not (await lab.service.list_business_systems(scope=lab.scope, status=None, limit=200)).items
        assert not await lab.service.list_managed_resources(scope=lab.scope, resource_type_key=None, limit=200)
        with pytest.raises(AIOpsApplicationError):
            await transition(lab, offline, "enable")
    asyncio.run(run())


@pytest.mark.parametrize("other_status", ["ENABLED", "DISABLED", "DRAFT"])
def test_deletion_preserves_resources_owned_by_any_undeleted_system(lab, other_status):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除共享系统")
        resource_id = detail.resources[0].resource.resource_id
        other_id = uuid7()
        with Session(lab.engine) as session:
            original = session.get(BusinessSystemEntity, detail.business_system_id)
            session.add(BusinessSystemEntity(**lab.owned, business_system_id=other_id,
                system_code="SYS-SHARED", display_name="共享归属系统", environment="DEV",
                responsibility_group_id=original.responsibility_group_id, status=other_status,
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(BusinessSystemResourceEntity(business_system_id=other_id, resource_id=resource_id,
                architecture_layer_key="DATA", status="ACTIVE",
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.commit()
        await remove_system(lab, offline)
        with Session(lab.engine) as session:
            assert session.get(ManagedResourceEntity, resource_id).lifecycle_status == "ACTIVE"
            assert session.scalar(select(ResourceSourceBindingEntity)).status == "ACTIVE"
            assert session.scalar(select(BusinessSystemResourceEntity).where(
                BusinessSystemResourceEntity.business_system_id == other_id)).status == "ACTIVE"
        lab.service._managed_credentials.revoke.assert_not_awaited()
    asyncio.run(run())


def test_last_shared_owner_deletion_retires_resource(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        resource_id = detail.resources[0].resource.resource_id
        other_id = uuid7()
        with Session(lab.engine) as session:
            original = session.get(BusinessSystemEntity, detail.business_system_id)
            session.add(BusinessSystemEntity(**lab.owned, business_system_id=other_id, system_code="SYS-LAST",
                display_name="最后引用", environment="DEV", responsibility_group_id=original.responsibility_group_id,
                status="DISABLED", created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(BusinessSystemResourceEntity(business_system_id=other_id, resource_id=resource_id,
                architecture_layer_key="DATA", status="ACTIVE", created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.commit()
        await remove_system(lab, offline)
        await lab.service.delete_business_system(scope=lab.scope, business_system_id=other_id, expected_version=1)
        with Session(lab.engine) as session:
            assert session.get(ManagedResourceEntity, resource_id).lifecycle_status == "RETIRED"
            assert session.scalar(select(ResourceSourceBindingEntity)).status == "DISABLED"
    asyncio.run(run())


def test_never_published_system_is_physically_deleted_but_resources_are_retired(lab):
    async def run():
        system_id, resource_id, relation_id, link_id = uuid7(), uuid7(), uuid7(), uuid7()
        with Session(lab.engine) as session:
            session.add(BusinessSystemEntity(**lab.owned, business_system_id=system_id,
                system_code="SYS-DRAFT", display_name="未发布系统", environment="DEV", status="DRAFT",
                responsibility_group_id=lab.request.system.responsibility_group_id,
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(ManagedResourceEntity(**lab.owned, resource_id=resource_id,
                resource_type_key="VIRTUAL_MACHINE", display_name="草稿主机", identity_kind="STABLE",
                owner_group_id=lab.request.system.responsibility_group_id,
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(BusinessSystemResourceEntity(system_resource_id=link_id,
                business_system_id=system_id, resource_id=resource_id, architecture_layer_key="INFRASTRUCTURE",
                created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(ResourceRelationEntity(relation_id=relation_id, business_system_id=system_id,
                relation_type_key="DEPENDS_ON", created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(RelationParticipantEntity(relation_id=relation_id, system_resource_id=link_id, participant_role="CONSUMER"))
            session.commit()
        await lab.service.delete_business_system(scope=lab.scope, business_system_id=system_id, expected_version=1)
        with Session(lab.engine) as session:
            assert session.get(BusinessSystemEntity, system_id) is None
            assert session.get(ManagedResourceEntity, resource_id).lifecycle_status == "RETIRED"
            assert session.get(BusinessSystemResourceEntity, link_id) is None
            assert session.get(ResourceRelationEntity, relation_id) is None
            assert not session.scalar(select(func.count()).select_from(RelationParticipantEntity))
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["CREATE", "EDIT"])
def test_unpublished_workspace_is_physically_deleted_without_touching_original(lab, mode):
    async def run():
        detail = await publish(lab)
        draft = await lab.service.create_business_system_draft(
            scope=lab.scope,
            request=BusinessSystemDraftCreate(
                mode=mode, business_system_id=detail.business_system_id if mode == "EDIT" else None),
        )
        with Session(lab.engine) as session:
            session.add(BusinessSystemDraftChangeEntity(draft_id=draft.draft_id, sequence_no=1,
                change_type="SAVE", payload_json={}, created_by=lab.scope.actor_id))
            session.add(ResourceDiscoveryJobEntity(draft_id=draft.draft_id, connector_type="OEM",
                status="SUCCEEDED", created_by=lab.scope.actor_id))
            session.commit()
        await lab.service.abandon_business_system_draft(scope=lab.scope, draft_id=draft.draft_id, expected_revision=1)
        with Session(lab.engine) as session:
            assert session.get(BusinessSystemDraftEntity, draft.draft_id) is None
            assert not session.scalar(select(func.count()).select_from(BusinessSystemDraftChangeEntity))
            assert not session.scalar(select(func.count()).select_from(ResourceDiscoveryJobEntity))
            assert session.get(BusinessSystemEntity, detail.business_system_id).status == "ENABLED"
            assert session.get(ManagedResourceEntity, detail.resources[0].resource.resource_id).lifecycle_status == "ACTIVE"
    asyncio.run(run())


def test_delete_rejects_stale_version_and_pending_discovery(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        with pytest.raises(AIOpsApplicationError):
            await remove_system(lab, detail)
        from platform_core.contracts.aiops import BusinessSystemDraftCreate
        draft = await lab.service.create_business_system_draft(scope=lab.scope,
            request=BusinessSystemDraftCreate(mode="EDIT", business_system_id=detail.business_system_id))
        with Session(lab.engine) as session:
            session.add(ResourceDiscoveryJobEntity(draft_id=draft.draft_id, connector_type="OEM",
                status="RUNNING", created_by=lab.scope.actor_id))
            session.commit()
        with pytest.raises(AIOpsApplicationError, match="资源发现任务"):
            await remove_system(lab, offline)
        with Session(lab.engine) as session:
            assert session.get(BusinessSystemEntity, detail.business_system_id).status == "DISABLED"
            assert session.get(ManagedResourceEntity, detail.resources[0].resource.resource_id).lifecycle_status == "ACTIVE"
    asyncio.run(run())


def test_deleted_system_frozen_authorization_cannot_access_a_shared_target(lab):
    async def run():
        from aiops_agent.repositories.agent import AIOpsAgentRepository
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        resource_id = detail.resources[0].resource.resource_id
        target_id = detail.resources[0].resource.database.target_id
        version_id, other_id = uuid7(), uuid7()
        with Session(lab.engine) as session:
            session.add(AIOpsAgentVersionBusinessSystemEntity(agent_version_id=version_id,
                business_system_id=detail.business_system_id))
            session.add(BusinessSystemEntity(**lab.owned, business_system_id=other_id, system_code="SYS-AUTH",
                display_name="共享在线系统", environment="DEV", responsibility_group_id=lab.request.system.responsibility_group_id,
                status="ENABLED", created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(BusinessSystemResourceEntity(business_system_id=other_id, resource_id=resource_id,
                architecture_layer_key="DATA", status="ACTIVE", created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.get(TargetEntity, target_id).status = "ENABLED"
            session.commit()
        await remove_system(lab, offline)
        async with LocalUow(lab.engine) as uow:
            agent_repo = AIOpsAgentRepository(uow.session, lambda: None)
            assert not await agent_repo.version_has_target(agent_version_id=version_id, target_id=target_id)
            assert not await agent_repo.version_has_business_system(agent_version_id=version_id,
                business_system_id=detail.business_system_id)
            assert not await agent_repo.version_source_ids(agent_version_id=version_id)
            assert await agent_repo.version_business_system_ids(agent_version_id=version_id) == [detail.business_system_id]
        with Session(lab.engine) as session:
            assert session.get(TargetEntity, target_id).status == "ENABLED"
            assert session.get(ManagedResourceEntity, resource_id).lifecycle_status == "ACTIVE"
    asyncio.run(run())


def test_deletion_disables_plans_cleans_edit_drafts_and_keeps_completed_history(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        plan_id, run_id = uuid7(), uuid7()
        draft = await lab.service.create_business_system_draft(scope=lab.scope,
            request=BusinessSystemDraftCreate(mode="EDIT", business_system_id=detail.business_system_id))
        with Session(lab.engine) as session:
            session.add(InspectionPlanEntity(**lab.owned, inspection_plan_id=plan_id,
                display_name="历史巡检", agent_id=uuid7(), business_system_id=detail.business_system_id,
                schedule_type="CRON", cron_expression="0 * * * *", timezone="Asia/Shanghai",
                inspection_template_id=uuid7(), inspection_template_version_id=uuid7(),
                timeout_seconds=300, overlap_policy="SKIP", misfire_policy="SKIP", status="ACTIVE",
                next_run_at=datetime.now(UTC), created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id))
            session.add(OpsRunEntity(**lab.owned, ops_run_id=run_id,
                target_id=detail.resources[0].resource.database.target_id, agent_id=uuid7(), agent_version_id=uuid7(),
                trigger_type="MANUAL", interaction_mode="INTERACTIVE", workflow_kind="DIAGNOSIS",
                actor_id=lab.scope.actor_id, idempotency_key=str(run_id), status="COMPLETED", trace_id="deletion-history"))
            session.commit()
        await remove_system(lab, offline)
        with Session(lab.engine) as session:
            assert session.get(InspectionPlanEntity, plan_id).status == "DISABLED"
            assert session.get(InspectionPlanEntity, plan_id).next_run_at is None
            assert session.get(BusinessSystemDraftEntity, draft.draft_id) is None
            assert session.get(OpsRunEntity, run_id).status == "COMPLETED"
    asyncio.run(run())


def test_running_target_blocks_deletion_and_rolls_back_retirement(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        run_id = uuid7()
        with Session(lab.engine) as session:
            session.add(OpsRunEntity(**lab.owned, ops_run_id=run_id,
                target_id=detail.resources[0].resource.database.target_id, agent_id=uuid7(), agent_version_id=uuid7(),
                trigger_type="MANUAL", interaction_mode="INTERACTIVE", workflow_kind="DIAGNOSIS",
                actor_id=lab.scope.actor_id, idempotency_key=str(run_id), status="WAITING_APPROVAL", trace_id="deletion-block"))
            session.commit()
        with pytest.raises(AIOpsApplicationError, match="未结束"):
            await remove_system(lab, offline)
        with Session(lab.engine) as session:
            assert session.get(BusinessSystemEntity, detail.business_system_id).status == "DISABLED"
            assert session.get(ManagedResourceEntity, detail.resources[0].resource.resource_id).lifecycle_status == "ACTIVE"
            assert session.scalar(select(ResourceSourceBindingEntity)).status == "ACTIVE"
            assert not session.scalar(select(func.count()).select_from(OutboxEntity).where(
                OutboxEntity.event_type == "BUSINESS_SYSTEM_DELETED"))
    asyncio.run(run())


def test_shared_monitor_source_credential_is_not_revoked(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        with Session(lab.engine) as session:
            target = session.get(TargetEntity, detail.resources[0].resource.database.target_id)
            session.get(DiagnosticSourceEntity, lab.source_id).auth_credential_id = target.diagnostic_credential_id
            session.commit()
        await remove_system(lab, offline)
        lab.service._managed_credentials.revoke.assert_not_awaited()
        with Session(lab.engine) as session:
            assert session.get(DiagnosticSourceEntity, lab.source_id).status == "ENABLED"
    asyncio.run(run())


def test_draft_publication_and_commit_marker_are_atomic(lab):
    async def run():
        draft = await lab.service.create_business_system_draft(scope=lab.scope,
            request=BusinessSystemDraftCreate(payload={"publish": lab.request.model_dump(mode="json")}))
        detail = await lab.service.commit_business_system_draft(scope=lab.scope,
            draft_id=draft.draft_id, expected_revision=1, idempotency_key=lab.key)
        with Session(lab.engine) as session:
            row = session.get(BusinessSystemDraftEntity, draft.draft_id)
            assert row.status == "COMMITTED" and row.business_system_id == detail.business_system_id
        with pytest.raises(AIOpsApplicationError):
            await lab.service.abandon_business_system_draft(scope=lab.scope, draft_id=draft.draft_id, expected_revision=1)
    asyncio.run(run())


def test_target_physical_delete_cannot_bypass_retired_resource_history(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        await remove_system(lab, offline)
        target_id = detail.resources[0].resource.database.target_id
        with Session(lab.engine) as session:
            version = session.get(TargetEntity, target_id).row_version
        with pytest.raises(AIOpsApplicationError, match="受管资源"):
            await lab.service.delete_target(scope=lab.scope, target_id=target_id,
                expected_version=int(version), idempotency_key=str(uuid7()))
        with Session(lab.engine) as session:
            assert session.get(TargetEntity, target_id) is not None
            assert session.get(ManagedResourceEntity, detail.resources[0].resource.resource_id).lifecycle_status == "RETIRED"
    asyncio.run(run())


def test_missing_system_cannot_be_deleted_from_another_scope(lab):
    async def run():
        detail = await publish(lab)
        offline = await transition(lab, detail, "disable", "删除")
        from dataclasses import replace
        other_scope = replace(lab.scope, domain_id=uuid7() if hasattr(lab.scope, "tenant_id") else 999)
        with pytest.raises(AIOpsApplicationError):
            await lab.service.delete_business_system(scope=other_scope,
                business_system_id=detail.business_system_id, expected_version=offline.row_version)
        with Session(lab.engine) as session:
            assert session.get(BusinessSystemEntity, detail.business_system_id).status == "DISABLED"
    asyncio.run(run())


def test_draft_status_with_publication_history_is_never_physically_deleted(lab):
    async def run():
        detail = await publish(lab)
        await transition(lab, detail, "disable", "删除")
        with Session(lab.engine) as session:
            system = session.get(BusinessSystemEntity, detail.business_system_id)
            system.status = "DRAFT"
            session.commit()
            version = int(system.row_version)
        await lab.service.delete_business_system(scope=lab.scope,
            business_system_id=detail.business_system_id, expected_version=version)
        with Session(lab.engine) as session:
            assert session.get(BusinessSystemEntity, detail.business_system_id).status == "DELETED"
            event = session.scalar(select(OutboxEntity).where(OutboxEntity.event_type == "BUSINESS_SYSTEM_DELETED"))
            assert event.payload_json["details"]["deletion_mode"] == "LOGICAL"
    asyncio.run(run())


def test_deleted_workspace_cannot_publish_an_orphan_system(lab):
    async def run():
        draft = await lab.service.create_business_system_draft(scope=lab.scope, request=BusinessSystemDraftCreate())
        await lab.service.abandon_business_system_draft(scope=lab.scope, draft_id=draft.draft_id, expected_revision=1)
        with pytest.raises(AIOpsApplicationError, match="草稿已删除"):
            await lab.service.commit_business_system(scope=lab.scope, request=lab.request,
                idempotency_key=lab.key, _draft_id=draft.draft_id, _draft_revision=1)
        with Session(lab.engine) as session:
            assert not session.scalar(select(func.count()).select_from(BusinessSystemEntity))
            assert not session.scalar(select(func.count()).select_from(ManagedResourceEntity))
            assert not session.scalar(select(func.count()).select_from(TargetEntity))
    asyncio.run(run())
