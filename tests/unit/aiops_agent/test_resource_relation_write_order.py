"""开启即时外键验证关系写入顺序；SQLite 不替代真实 PG/Oracle 并发验收。"""

import asyncio

import pytest
from sqlalchemy import ForeignKeyConstraint, MetaData, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import (
    BusinessSystemDraftEntity,
    BusinessSystemEntity,
    BusinessSystemResourceEntity,
    ComputeResourceEntity,
    DatabaseResourceEntity,
    InboxEntity,
    ManagedResourceEntity,
    OutboxEntity,
    RelationParticipantEntity,
    ResourceRelationEntity,
    ResourceSourceBindingEntity,
    ResponsibilityGroupEntity,
    TargetEntity,
)
from platform_core.contracts.aiops import (
    BusinessSystemDraftCreate,
    BusinessSystemResourceGraphCommit,
    ResourceRelationInputV2,
)
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab, publish
from tests.unit.aiops_agent.test_resource_monitoring import configure_oem_host


@pytest.fixture
def foreign_key_lab(lab):
    """仅在空的内存库补齐 Oracle 规范 DDL 的参与者外键，并开启校验。"""
    assert lab.engine.dialect.name == "sqlite" and lab.engine.url.database is None
    metadata = MetaData()
    for entity in (ResourceRelationEntity, BusinessSystemResourceEntity, RelationParticipantEntity):
        entity.__table__.to_metadata(metadata)
    participant = metadata.tables[RelationParticipantEntity.__tablename__]
    for column, parent, parent_column in (
        ("relation_id", ResourceRelationEntity, "relation_id"),
        ("system_resource_id", BusinessSystemResourceEntity, "system_resource_id"),
    ):
        if not participant.c[column].foreign_keys:
            participant.append_constraint(ForeignKeyConstraint(
                [column], [f"{parent.__tablename__}.{parent_column}"],
            ))
    with lab.engine.begin() as connection:
        RelationParticipantEntity.__table__.drop(connection)
        participant.create(connection)
        ResponsibilityGroupEntity.__table__.create(connection, checkfirst=True)
    with Session(lab.engine) as session:
        session.add(ResponsibilityGroupEntity(
            **lab.owned, responsibility_group_id=lab.request.system.responsibility_group_id,
            name="外键回归运维组", created_by=lab.scope.actor_id, updated_by=lab.scope.actor_id,
        ))
        session.commit()
    with lab.engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
    return lab


def test_missing_relation_is_rejected_by_real_foreign_key(foreign_key_lab):
    """保证测试库不会像默认 SQLite 配置那样静默接受孤儿参与者。"""
    with Session(foreign_key_lab.engine) as session:
        session.add(RelationParticipantEntity(
            participant_id=uuid7(), relation_id=uuid7(), system_resource_id=uuid7(),
            participant_role="HOST", display_order=0,
        ))
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            session.flush()


def assert_topology(session, detail):
    assert len(detail.resources) == 2 and len(detail.relations) == 1
    relation = detail.relations[0]
    assert relation.relation_type_key == "RUNS_ON"
    assert session.get(ResourceRelationEntity, relation.relation_id) is not None
    participants = session.scalars(select(RelationParticipantEntity).where(
        RelationParticipantEntity.relation_id == relation.relation_id,
    )).all()
    assert {item.participant_role for item in participants} == {"WORKLOAD", "HOST"}
    assert {item.system_resource_id for item in participants} == {
        item.system_resource_id for item in detail.resources
    }
    assert session.scalar(select(func.count()).select_from(ResourceSourceBindingEntity)) == 2


@pytest.mark.parametrize("resource_type", ["PHYSICAL_SERVER", "CONTAINER"])
def test_oracle_and_oem_host_draft_publish_with_foreign_keys(foreign_key_lab, resource_type):
    f = foreign_key_lab
    configure_oem_host(f)
    payload = f.request.model_dump(mode="json")
    payload["resources"][1]["resource_type_key"] = resource_type
    f.request = BusinessSystemResourceGraphCommit.model_validate(payload)

    async def run():
        draft = await f.service.create_business_system_draft(scope=f.scope,
            request=BusinessSystemDraftCreate(payload={"publish": f.request.model_dump(mode="json")}))
        detail = await f.service.commit_business_system_draft(scope=f.scope, draft_id=draft.draft_id,
            expected_revision=draft.revision, idempotency_key=f.key)
        assert detail.status == "ENABLED"
        with Session(f.engine) as session:
            assert_topology(session, detail)
            saved = session.get(BusinessSystemDraftEntity, draft.draft_id)
            assert saved.status == "COMMITTED" and saved.business_system_id == detail.business_system_id

    asyncio.run(run())


def test_incremental_relation_create_with_foreign_keys(foreign_key_lab):
    f = foreign_key_lab
    configure_oem_host(f)
    f.request = f.request.model_copy(update={"relations": ()})

    async def run():
        detail = await publish(f)
        request = ResourceRelationInputV2.model_validate({
            "relation_type_key": "RUNS_ON",
            "participants": [
                {"system_resource_id": str(detail.resources[0].system_resource_id), "participant_role": "WORKLOAD"},
                {"system_resource_id": str(detail.resources[1].system_resource_id), "participant_role": "HOST"},
            ],
        })
        key = str(uuid7())
        detail = await f.service.create_resource_relation(scope=f.scope,
            business_system_id=detail.business_system_id, request=request, idempotency_key=key)
        with Session(f.engine) as session:
            assert_topology(session, detail)
        replay = await f.service.create_resource_relation(scope=f.scope,
            business_system_id=detail.business_system_id, request=request, idempotency_key=key)
        assert replay.relations[0].relation_id == detail.relations[0].relation_id
        with Session(f.engine) as session:
            assert session.scalar(select(func.count()).select_from(RelationParticipantEntity)) == 2

    asyncio.run(run())


def test_topology_failure_rolls_back_and_keeps_draft_retryable(foreign_key_lab):
    f = foreign_key_lab
    configure_oem_host(f)
    original_detail = f.service._system_detail

    async def fail_after_topology_write(*, uow, **kwargs):
        assert uow.sync.scalar(select(func.count()).select_from(ResourceRelationEntity)) == 1
        assert uow.sync.scalar(select(func.count()).select_from(RelationParticipantEntity)) == 2
        raise AIOpsApplicationError(code="TEST_AFTER_TOPOLOGY_WRITE", message="隔离测试发布失败", status_code=409)

    async def run():
        draft = await f.service.create_business_system_draft(scope=f.scope,
            request=BusinessSystemDraftCreate(payload={"publish": f.request.model_dump(mode="json")}))
        f.service._system_detail = fail_after_topology_write
        with pytest.raises(AIOpsApplicationError, match="隔离测试发布失败"):
            await f.service.commit_business_system_draft(scope=f.scope, draft_id=draft.draft_id,
                expected_revision=draft.revision, idempotency_key=f.key)
        with Session(f.engine) as session:
            for entity in (BusinessSystemEntity, ManagedResourceEntity, BusinessSystemResourceEntity,
                    ComputeResourceEntity, DatabaseResourceEntity, TargetEntity, ResourceSourceBindingEntity,
                    ResourceRelationEntity, RelationParticipantEntity, InboxEntity, OutboxEntity):
                assert session.scalar(select(func.count()).select_from(entity)) == 0
            saved = session.get(BusinessSystemDraftEntity, draft.draft_id)
            assert saved.status == "ACTIVE" and saved.business_system_id is None
            assert saved.revision == draft.revision
        f.service._system_detail = original_detail
        detail = await f.service.commit_business_system_draft(scope=f.scope, draft_id=draft.draft_id,
            expected_revision=draft.revision, idempotency_key=f.key)
        with Session(f.engine) as session:
            assert_topology(session, detail)

    asyncio.run(run())
