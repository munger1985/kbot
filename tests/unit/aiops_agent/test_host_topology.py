"""真实仓储验证跨系统承载投影，SQLite 不冒充线上 Oracle/PG 验收。"""

import asyncio
from dataclasses import replace

import pytest
from sqlalchemy.orm import Session
from sqlalchemy import select

from aiops_agent.application.configuration.host_topology import read_host_topology
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.entities import (
    BusinessSystemEntity, BusinessSystemResourceEntity, ManagedResourceEntity,
    RelationParticipantEntity, ResourceRelationEntity,
)
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab


def seed(f):
    group = uuid7()
    authored = dict(created_by=f.scope.actor_id, updated_by=f.scope.actor_id)
    with Session(f.engine) as session:
        host = ManagedResourceEntity(**f.owned, **authored, resource_type_key="PHYSICAL_SERVER", display_name="共享主机", owner_group_id=group)
        session.add(host)
        session.flush()
        systems, workloads, links = [], [], []
        for index, kind in enumerate(("DATABASE", "APPLICATION_SERVICE")):
            system = BusinessSystemEntity(**f.owned, **authored, system_code=f"SYS-{index}", display_name=f"系统 {index}", environment="DEV", responsibility_group_id=group, status="ENABLED")
            work = ManagedResourceEntity(**f.owned, **authored, resource_type_key=kind, display_name=f"资源 {index}", owner_group_id=group)
            session.add_all([system, work])
            session.flush()
            host_link = BusinessSystemResourceEntity(**authored, business_system_id=system.business_system_id, resource_id=host.resource_id, architecture_layer_key="INFRASTRUCTURE")
            work_link = BusinessSystemResourceEntity(**authored, business_system_id=system.business_system_id, resource_id=work.resource_id, architecture_layer_key="DATA" if index == 0 else "APPLICATION")
            relation = ResourceRelationEntity(**authored, business_system_id=system.business_system_id, relation_type_key="RUNS_ON", display_name="承载")
            session.add_all([host_link, work_link, relation])
            session.flush()
            session.add_all([
                RelationParticipantEntity(relation_id=relation.relation_id, system_resource_id=host_link.system_resource_id, participant_role="HOST", display_order=1),
                RelationParticipantEntity(relation_id=relation.relation_id, system_resource_id=work_link.system_resource_id, participant_role="WORKLOAD", display_order=0),
            ])
            systems.append(system.business_system_id); workloads.append(work.resource_id); links.append(work_link.system_resource_id)
        session.commit()
        return host.resource_id, systems, workloads, links


def read(f, host, system, **kwargs):
    return asyncio.run(read_host_topology(f.service, scope=f.scope, business_system_id=system, resource_id=host, **kwargs))


def test_shared_host_has_one_identity_and_both_systems_without_connection_metadata(lab):
    host, systems, workloads, _ = seed(lab)
    result = read(lab, host, systems[0])
    assert len(result.resources) == 3 and len(result.relations) == 2
    assert len({item.resource_id for item in result.resources}) == 3
    root = next(item for item in result.resources if item.resource_id == host)
    assert {item.business_system_id for item in root.business_systems} == set(systems)
    assert {item.resource_id for item in result.resources} == {host, *workloads}
    for field in ("password", "endpoint", "diagnostic_username", "target_id", "credential_id"):
        assert field not in result.model_dump_json()
    assert not result.truncated


@pytest.mark.parametrize("exclusion", ["system_deleted", "resource_retired", "membership_inactive", "other_domain", "unconfirmed"])
def test_host_projection_excludes_out_of_scope_or_inactive_facts(lab, exclusion):
    host, systems, workloads, links = seed(lab)
    with Session(lab.engine) as session:
        if exclusion == "system_deleted":
            session.get(BusinessSystemEntity, systems[1]).status = "DELETED"
        elif exclusion == "resource_retired":
            session.get(ManagedResourceEntity, workloads[1]).lifecycle_status = "RETIRED"
        elif exclusion == "membership_inactive":
            session.get(BusinessSystemResourceEntity, links[1]).status = "REMOVED"
        elif exclusion == "other_domain":
            session.get(BusinessSystemEntity, systems[1]).domain_id = uuid7() if hasattr(lab.scope, "tenant_id") else 999
        else:
            from sqlalchemy import select
            session.scalar(select(ResourceRelationEntity).where(ResourceRelationEntity.business_system_id == systems[1])).confirmation_status = "PENDING"
        session.commit()
    result = read(lab, host, systems[0])
    assert {item.resource_id for item in result.resources} == {host, workloads[0]}
    assert len(result.relations) == 1
    assert all(item.business_system_id != systems[1] for resource in result.resources for item in resource.business_systems)


def test_host_projection_requires_current_system_membership_and_scope(lab):
    host, systems, workloads, _ = seed(lab)
    with pytest.raises(AIOpsApplicationError):
        read(lab, uuid7(), systems[0])
    with pytest.raises(AIOpsApplicationError):
        read(lab, host, uuid7())
    with pytest.raises(AIOpsApplicationError):
        read(lab, workloads[0], systems[0])
    other = replace(lab.scope, domain_id=uuid7() if hasattr(lab.scope, "tenant_id") else 999)
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(read_host_topology(lab.service, scope=other, business_system_id=systems[0], resource_id=host))
    if hasattr(lab.scope, "tenant_id"):
        other = replace(lab.scope, tenant_id=uuid7())
        with pytest.raises(AIOpsApplicationError):
            asyncio.run(read_host_topology(lab.service, scope=other, business_system_id=systems[0], resource_id=host))


def test_host_projection_marks_node_limit_as_incomplete(lab):
    host, systems, _, _ = seed(lab)
    result = read(lab, host, systems[0], max_nodes=2)
    assert result.truncated and len(result.resources) == 2
    assert all(p.resource_id in {r.resource_id for r in result.resources} for rel in result.relations for p in rel.participants)


def add_chain(f, host, system, *, cycle=False):
    """主机 -> 容器 -> 应用；循环仅用于验证异常关系不会无限展开。"""
    authored = dict(created_by=f.scope.actor_id, updated_by=f.scope.actor_id)
    with Session(f.engine) as session:
        root_link = session.scalar(select(BusinessSystemResourceEntity).where(
            BusinessSystemResourceEntity.business_system_id == system,
            BusinessSystemResourceEntity.resource_id == host,
        ))
        parent = root_link
        ids = []
        for index, kind in enumerate(("CONTAINER", "APPLICATION_SERVICE")):
            resource = ManagedResourceEntity(**f.owned, **authored, resource_type_key=kind, display_name=f"嵌套资源 {index}", owner_group_id=uuid7())
            session.add(resource); session.flush()
            link = BusinessSystemResourceEntity(**authored, business_system_id=system, resource_id=resource.resource_id, architecture_layer_key="APPLICATION")
            relation = ResourceRelationEntity(**authored, business_system_id=system, relation_type_key="RUNS_ON", display_name="嵌套承载")
            session.add_all([link, relation]); session.flush()
            session.add_all([
                RelationParticipantEntity(relation_id=relation.relation_id, system_resource_id=parent.system_resource_id, participant_role="HOST", display_order=1),
                RelationParticipantEntity(relation_id=relation.relation_id, system_resource_id=link.system_resource_id, participant_role="WORKLOAD", display_order=0),
            ])
            ids.append(resource.resource_id)
            if cycle and index == 0:
                loop = ResourceRelationEntity(**authored, business_system_id=system, relation_type_key="RUNS_ON", display_name="循环承载")
                session.add(loop); session.flush()
                session.add_all([
                    RelationParticipantEntity(relation_id=loop.relation_id, system_resource_id=link.system_resource_id, participant_role="HOST", display_order=1),
                    RelationParticipantEntity(relation_id=loop.relation_id, system_resource_id=root_link.system_resource_id, participant_role="WORKLOAD", display_order=0),
                ])
            parent = link
        session.commit()
        return ids


def test_host_projection_follows_nested_containers_and_stops_cycles(lab):
    host, systems, workloads, _ = seed(lab)
    container, app = add_chain(lab, host, systems[0], cycle=True)
    result = read(lab, host, systems[0])
    assert {item.resource_id for item in result.resources} == {host, *workloads, container, app}
    assert len(result.resources) == 5 and len(result.relations) == 5
    assert not result.truncated


def test_host_projection_marks_depth_limit_without_hiding_incompleteness(lab):
    host, systems, workloads, _ = seed(lab)
    container, app = add_chain(lab, host, systems[0])
    result = read(lab, host, systems[0], max_depth=1)
    assert {item.resource_id for item in result.resources} == {host, *workloads, container}
    assert result.truncated
    assert app not in {item.resource_id for item in result.resources}


def test_root_scope_is_verified_even_when_membership_is_corrupt(lab):
    host, systems, _, _ = seed(lab)
    with Session(lab.engine) as session:
        session.get(ManagedResourceEntity, host).domain_id = uuid7() if hasattr(lab.scope, "tenant_id") else 999
        session.commit()
    with pytest.raises(AIOpsApplicationError):
        read(lab, host, systems[0])


def test_workload_scope_is_verified_independently_of_system(lab):
    host, systems, workloads, _ = seed(lab)
    with Session(lab.engine) as session:
        work = session.get(ManagedResourceEntity, workloads[1])
        if hasattr(lab.scope, "tenant_id"):
            work.tenant_id = uuid7()
        else:
            work.domain_id = 999
        session.commit()
    result = read(lab, host, systems[0])
    assert {item.resource_id for item in result.resources} == {host, workloads[0]}
