"""业务系统、组件与资源拓扑配置用例。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from aiops_agent.application.configuration.common import ConfigurationScope, add_configuration_event
from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.entities import (
    BusinessSystemComponentEntity,
    BusinessSystemEntity,
    BusinessSystemSiteEntity,
    ComponentMembershipEntity,
    ManagedComponentEntity,
    ResourceRelationEntity,
)
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts.aiops import (
    BusinessSystemCommit,
    BusinessSystemCreate,
    BusinessSystemDetail,
    BusinessSystemPage,
    BusinessSystemPatch,
    BusinessSystemSiteView,
    BusinessSystemSummary,
    BusinessSystemTopologyPut,
    ComponentMemberView,
    ManagedComponentView,
    ResourceRelationView,
    TopologyValidationIssue,
)
from platform_core.contracts.aiops.configuration import validate_supported_database_version
from platform_core.identity import uuid7


def _scope_args(scope: ConfigurationScope) -> dict:
    values = {"domain_id": scope.domain_id}
    if hasattr(scope, "tenant_id"):
        values["tenant_id"] = scope.tenant_id
    return values


def _issue(
    severity: str,
    code: str,
    message: str,
    *,
    component_key: str | None = None,
    member_key: str | None = None,
) -> TopologyValidationIssue:
    return TopologyValidationIssue(
        severity=severity,
        code=code,
        message=message,
        component_key=component_key,
        member_key=member_key,
    )


def _summary(detail: BusinessSystemDetail) -> BusinessSystemSummary:
    return BusinessSystemSummary.model_validate(
        detail.model_dump(include=set(BusinessSystemSummary.model_fields))
    )


def _component_architecture_labels(*, members: list[ComponentMemberView], relations: list[ResourceRelationView]) -> tuple[str, ...]:
    labels: list[str] = []
    groups: dict[str, list[ComponentMemberView]] = {}
    for member in members:
        if member.cluster_group_key:
            groups.setdefault(member.cluster_group_key, []).append(member)
    if any(len(group_members) >= 2 for group_members in groups.values()):
        labels.append("RAC")
    relation_types = {relation.relation_type for relation in relations}
    if "DATA_GUARD" in relation_types:
        labels.append("DG")
    if "MYSQL_REPLICATION" in relation_types:
        labels.append("MYSQL_REPLICATION")
    if "POSTGRES_STREAMING" in relation_types:
        labels.append("POSTGRES_STREAMING")
    if "OGG" in relation_types:
        labels.append("OGG")
    versions = {(str(member.target_db_type), member.target_version_code) for member in members}
    if len(versions) > 1:
        labels.append("MIXED_VERSION")
    if len(members) == 1 and not labels:
        labels.append("SINGLE_INSTANCE")
    return tuple(labels)


class BusinessSystemConfigurationMixin:
    async def _system_detail(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        system: BusinessSystemEntity,
    ) -> BusinessSystemDetail:
        assert uow.business_systems is not None
        assert uow.targets is not None
        assert uow.recovery is not None
        sites = await uow.business_systems.list_sites(system.business_system_id)
        components = await uow.business_systems.list_components(system.business_system_id)
        component_links = await uow.business_systems.list_component_links(system.business_system_id)
        memberships = await uow.business_systems.list_members(system.business_system_id)
        relations = await uow.business_systems.list_relations(system.business_system_id)
        site_by_id = {site.site_id: site for site in sites}
        component_by_id = {component.component_id: component for component in components}
        membership_by_id = {member.membership_id: member for member in memberships}
        component_key_by_id = {component.component_id: component.component_key for component in components}

        issues: list[TopologyValidationIssue] = []
        member_views: list[ComponentMemberView] = []
        target_by_membership: dict[UUID, object] = {}
        for membership in memberships:
            component = component_by_id.get(membership.component_id)
            site = site_by_id.get(membership.site_id)
            if component is None or site is None:
                issues.append(_issue("BLOCKING", "MEMBERSHIP_PARENT_MISSING", "成员缺少组件或站点"))
                continue
            target = await uow.targets.get_scoped(target_id=membership.target_id, domain_id=scope.domain_id)
            if target is None:
                issues.append(_issue("BLOCKING", "TARGET_MISSING", "成员 Target 不存在", component_key=component.component_key, member_key=membership.member_key))
                continue
            target_by_membership[membership.membership_id] = target
            bindings = await uow.targets.list_source_bindings(target_id=target.target_id, domain_id=scope.domain_id)
            observed = target.db_role if target.db_role in {"PRIMARY", "STANDBY", "SOURCE", "REPLICA", "REPORTING"} else "UNKNOWN"
            member_views.append(ComponentMemberView(
                membership_id=membership.membership_id,
                component_id=component.component_id,
                component_key=component.component_key,
                member_key=membership.member_key,
                target_id=membership.target_id,
                target_display_name=target.display_name,
                target_db_type=target.db_type,
                target_version_code=target.version_code,
                target_environment=target.environment,
                target_row_version=int(target.row_version),
                site_code=site.site_code,
                cluster_group_key=membership.cluster_group_key,
                expected_role=membership.expected_role,
                observed_role=observed,
                connectivity_status=target.connectivity_status,
                monitoring_ready=bool(bindings),
                sort_order=int(membership.sort_order),
            ))
            if target.connectivity_status != "CONNECTED":
                issues.append(_issue("BLOCKING", "TARGET_NOT_CONNECTED", "成员连接验证尚未通过", component_key=component.component_key, member_key=membership.member_key))
            if not bindings:
                issues.append(_issue("BLOCKING", "MONITORING_NOT_READY", "成员监控映射尚未完成", component_key=component.component_key, member_key=membership.member_key))
            if observed != "UNKNOWN" and membership.expected_role != "INSTANCE" and observed != membership.expected_role:
                issues.append(_issue("WARNING", "ROLE_DEVIATION", "期望角色与探测角色不一致", component_key=component.component_key, member_key=membership.member_key))

        member_key_to_entity = {member.member_key: member for member in memberships}
        relation_views: list[ResourceRelationView] = []
        for relation in relations:
            source_component_key = component_key_by_id.get(relation.source_component_id)
            target_component_key = component_key_by_id.get(relation.target_component_id)
            if source_component_key is None or target_component_key is None:
                issues.append(_issue("BLOCKING", "RELATION_COMPONENT_MISSING", "资源关系缺少组件端点"))
                continue
            source_member = membership_by_id.get(relation.source_membership_id) if relation.source_membership_id else None
            target_member = membership_by_id.get(relation.target_membership_id) if relation.target_membership_id else None
            view = ResourceRelationView(
                relation_id=relation.relation_id,
                relation_key=relation.relation_key,
                source_component_key=source_component_key,
                target_component_key=target_component_key,
                source_member_key=source_member.member_key if source_member else None,
                target_member_key=target_member.member_key if target_member else None,
                relation_type=relation.relation_type,
                transport_mode=relation.transport_mode,
                responsibility_group_id=relation.responsibility_group_id,
                sort_order=int(relation.sort_order),
            )
            relation_views.append(view)
            if (source_member is None) != (target_member is None):
                issues.append(_issue("BLOCKING", "RELATION_ENDPOINT_MISMATCH", "资源关系端点层级不一致"))
            if source_member and target_member:
                source_target = target_by_membership.get(source_member.membership_id)
                target_target = target_by_membership.get(target_member.membership_id)
                if source_target is None or target_target is None:
                    continue
                engines = {str(source_target.db_type), str(target_target.db_type)}
                if relation.relation_type == "DATA_GUARD" and engines != {"ORACLE"}:
                    issues.append(_issue("BLOCKING", "DG_ENGINE_INVALID", "Data Guard 两端必须为 Oracle"))
                if relation.relation_type == "MYSQL_REPLICATION" and engines != {"MYSQL"}:
                    issues.append(_issue("BLOCKING", "MYSQL_REPLICATION_ENGINE_INVALID", "MySQL 复制两端必须为 MySQL"))
                if relation.relation_type == "POSTGRES_STREAMING" and engines != {"POSTGRESQL"}:
                    issues.append(_issue("BLOCKING", "POSTGRES_STREAMING_ENGINE_INVALID", "Streaming Replication 两端必须为 PostgreSQL"))
            if source_member and target_member and site_by_id[source_member.site_id].site_code != site_by_id[target_member.site_id].site_code and relation.transport_mode == "UNKNOWN":
                issues.append(_issue("WARNING", "TRANSPORT_MODE_UNKNOWN", "跨站点复制关系尚未填写同步模式"))

        component_views: list[ManagedComponentView] = []
        sort_order_by_component = {item.component_id: int(item.sort_order) for item in component_links}
        all_architecture_labels: list[str] = []
        configured_components = 0
        for component in components:
            component_members = [item for item in member_views if item.component_id == component.component_id]
            if component.component_type == "DATABASE" and not component_members:
                issues.append(_issue("BLOCKING", "DATABASE_COMPONENT_EMPTY", "数据库组件至少需要一个 Target", component_key=component.component_key))
            component_relations = [
                item for item in relation_views
                if item.source_component_key == component.component_key or item.target_component_key == component.component_key
            ]
            group_counts: dict[str, int] = {}
            for member in component_members:
                if member.cluster_group_key:
                    group_counts[member.cluster_group_key] = group_counts.get(member.cluster_group_key, 0) + 1
                    if str(member.target_db_type) != "ORACLE":
                        issues.append(_issue("BLOCKING", "RAC_ENGINE_INVALID", "RAC 集群成员必须为 Oracle", component_key=component.component_key, member_key=member.member_key))
            for group_key, count in group_counts.items():
                if count < 2:
                    issues.append(_issue("BLOCKING", "CLUSTER_GROUP_TOO_SMALL", f"集群组 {group_key} 至少需要两个成员", component_key=component.component_key))
            labels = _component_architecture_labels(members=component_members, relations=component_relations)
            all_architecture_labels.extend(labels)
            profile = await uow.recovery.get_active_profile(component_id=component.component_id, **_scope_args(scope))
            if profile is not None:
                configured_components += 1
            component_views.append(ManagedComponentView(
                component_id=component.component_id,
                component_key=component.component_key,
                display_name=component.display_name,
                component_type=component.component_type,
                responsibility_group_id=component.responsibility_group_id,
                description=component.description,
                sort_order=sort_order_by_component.get(component.component_id, 0),
                status=component.status,
                member_count=len(component_members),
                engine_types=tuple(sorted({item.target_db_type for item in component_members}, key=str)),
                versions=tuple(sorted({item.target_version_code for item in component_members})),
                architecture_labels=labels,
                protection_status="CONFIGURED" if profile else "UNCONFIGURED",
            ))

        database_components = [item for item in component_views if item.component_type == "DATABASE"]
        protection_status = "UNCONFIGURED"
        if database_components and configured_components == len(database_components):
            protection_status = "CONFIGURED"
        elif configured_components:
            protection_status = "PARTIAL"

        return BusinessSystemDetail(
            business_system_id=system.business_system_id,
            system_code=system.system_code,
            display_name=system.display_name,
            environment=system.environment,
            importance_level=int(system.importance_level),
            responsibility_group_id=system.responsibility_group_id,
            status=system.status,
            protection_status=protection_status,
            component_count=len(component_views),
            target_count=len(member_views),
            architecture_labels=tuple(dict.fromkeys(all_architecture_labels)),
            row_version=int(system.row_version),
            updated_at=system.updated_at,
            description=system.description,
            sites=tuple(BusinessSystemSiteView(
                site_id=item.site_id,
                site_code=item.site_code,
                display_name=item.display_name,
                site_role=item.site_role,
                sort_order=int(item.sort_order),
            ) for item in sites),
            components=tuple(component_views),
            members=tuple(member_views),
            relations=tuple(relation_views),
            validation_issues=tuple(issues),
            created_at=system.created_at,
        )

    async def _validate_group(self, *, uow: AIOpsUnitOfWork, scope: ConfigurationScope, group_id: UUID | None, label: str) -> None:
        assert uow.work_items is not None
        if group_id is None:
            raise validation_failed(f"{label}必须绑定责任组")
        group = await uow.work_items.get_group(group_id=group_id, **_scope_args(scope))
        if group is None or group.status != "ACTIVE":
            raise validation_failed(f"{label}责任组不存在或已停用")

    async def _add_topology(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        system: BusinessSystemEntity,
        components,
        sites,
        member_target_ids: dict[str, UUID],
        members,
        relations,
        now: datetime,
        existing_components: dict[str, ManagedComponentEntity] | None = None,
    ) -> None:
        assert uow.business_systems is not None
        existing_components = existing_components or {}
        component_entities: list[ManagedComponentEntity] = []
        component_ids: dict[str, UUID] = {}
        for item in components:
            await self._validate_group(uow=uow, scope=scope, group_id=item.responsibility_group_id, label=f"组件 {item.display_name}")
            if item.component_type != "DATABASE":
                raise validation_failed("一期只开放数据库组件；应用服务器、中间件和网络暂为占位")
            component = existing_components.get(item.component_key)
            if component is None:
                values = dict(
                    component_id=uuid7(), domain_id=scope.domain_id,
                    component_key=item.component_key, display_name=item.display_name,
                    component_type=item.component_type,
                    responsibility_group_id=item.responsibility_group_id,
                    description=item.description, status="ACTIVE",
                    created_by=scope.actor_id, updated_by=scope.actor_id,
                    created_at=now, updated_at=now,
                )
                if hasattr(scope, "tenant_id"):
                    values["tenant_id"] = scope.tenant_id
                component = ManagedComponentEntity(**values)
                component_entities.append(component)
            else:
                component.display_name = item.display_name
                component.component_type = item.component_type
                component.responsibility_group_id = item.responsibility_group_id
                component.description = item.description
                component.updated_by = scope.actor_id
                component.updated_at = now
            component_ids[item.component_key] = component.component_id
        await uow.business_systems.add_components(component_entities)
        existing_links = {item.component_id: item for item in await uow.business_systems.list_component_links(system.business_system_id)}
        for item in components:
            if component_ids[item.component_key] in existing_links:
                existing_links[component_ids[item.component_key]].sort_order = item.sort_order
        await uow.business_systems.add_component_links([
            BusinessSystemComponentEntity(
                system_component_id=uuid7(),
                business_system_id=system.business_system_id,
                component_id=component_ids[item.component_key],
                sort_order=item.sort_order,
            )
            for item in components if component_ids[item.component_key] not in existing_links
        ])
        site_entities = [BusinessSystemSiteEntity(
            site_id=uuid7(),
            business_system_id=system.business_system_id,
            site_code=item.site_code,
            display_name=item.display_name,
            site_role=item.site_role,
            sort_order=item.sort_order,
        ) for item in sites]
        await uow.business_systems.add_sites(site_entities)
        site_ids = {item.site_code: item.site_id for item in site_entities}
        membership_entities = [ComponentMembershipEntity(
            membership_id=uuid7(),
            component_id=component_ids[item.component_key],
            site_id=site_ids[item.site_code],
            target_id=member_target_ids[item.member_key],
            member_key=item.member_key,
            cluster_group_key=item.cluster_group_key,
            expected_role=item.expected_role,
            sort_order=item.sort_order,
        ) for item in members]
        await uow.business_systems.add_members(membership_entities)
        membership_ids = {item.member_key: item.membership_id for item in membership_entities}
        await uow.business_systems.add_relations([ResourceRelationEntity(
            relation_id=uuid7(),
            business_system_id=system.business_system_id,
            source_component_id=component_ids[item.source_component_key],
            target_component_id=component_ids[item.target_component_key],
            source_membership_id=membership_ids[item.source_member_key] if item.source_member_key else None,
            target_membership_id=membership_ids[item.target_member_key] if item.target_member_key else None,
            relation_key=item.relation_key,
            relation_type=item.relation_type,
            transport_mode=item.transport_mode,
            responsibility_group_id=item.responsibility_group_id,
            sort_order=item.sort_order,
        ) for item in relations])

    async def create_business_system(self, *, scope: ConfigurationScope, request: BusinessSystemCreate, idempotency_key: str) -> BusinessSystemDetail:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> BusinessSystemDetail:
            assert uow.business_systems is not None
            if await uow.business_systems.get_by_name(display_name=request.display_name, **_scope_args(scope)) is not None:
                raise state_conflict("当前 Domain 已存在同名业务系统")
            if await uow.business_systems.get_by_code(system_code=request.system_code, **_scope_args(scope)) is not None:
                raise state_conflict("当前 Domain 已存在相同系统编码")
            await self._validate_group(uow=uow, scope=scope, group_id=request.responsibility_group_id, label="业务系统")
            values = dict(
                business_system_id=uuid7(), domain_id=scope.domain_id, system_code=request.system_code,
                display_name=request.display_name, environment=request.environment,
                importance_level=request.importance_level,
                responsibility_group_id=request.responsibility_group_id,
                description=request.description, status="DRAFT", row_version=1,
                created_by=scope.actor_id, updated_by=scope.actor_id,
                created_at=now, updated_at=now,
            )
            if hasattr(scope, "tenant_id"):
                values["tenant_id"] = scope.tenant_id
            entity = BusinessSystemEntity(**values)
            await uow.business_systems.add_system(entity)
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="BUSINESS_SYSTEM", aggregate_id=entity.business_system_id, event_type="BUSINESS_SYSTEM_DRAFT_CREATED", row_version=1, details={})
            return await self._system_detail(uow=uow, scope=scope, system=entity)
        return await self._idempotent(scope=scope, operation="BUSINESS_SYSTEM_CREATE", parent_resource="business-systems", idempotency_key=idempotency_key, payload=request.model_dump(mode="json"), response_type=BusinessSystemDetail, handler=handler)

    async def commit_business_system(self, *, scope: ConfigurationScope, request: BusinessSystemCommit, idempotency_key: str) -> BusinessSystemDetail:
        """原子创建业务系统、组件、成员 Target、监控映射与关系。"""

        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> BusinessSystemDetail:
            assert uow.business_systems is not None and uow.targets is not None
            system_request = request.system
            if await uow.business_systems.get_by_name(display_name=system_request.display_name, **_scope_args(scope)) is not None:
                raise state_conflict("当前 Domain 已存在同名业务系统")
            if await uow.business_systems.get_by_code(system_code=system_request.system_code, **_scope_args(scope)) is not None:
                raise state_conflict("当前 Domain 已存在相同系统编码")
            await self._validate_group(uow=uow, scope=scope, group_id=system_request.responsibility_group_id, label="业务系统")
            values = dict(
                business_system_id=uuid7(), domain_id=scope.domain_id, system_code=system_request.system_code,
                display_name=system_request.display_name, environment=system_request.environment,
                importance_level=system_request.importance_level,
                responsibility_group_id=system_request.responsibility_group_id,
                description=system_request.description, status="DRAFT", row_version=1,
                created_by=scope.actor_id, updated_by=scope.actor_id,
                created_at=now, updated_at=now,
            )
            if hasattr(scope, "tenant_id"):
                values["tenant_id"] = scope.tenant_id
            system = BusinessSystemEntity(**values)
            await uow.business_systems.add_system(system)

            component_group = {item.component_key: item.responsibility_group_id for item in request.components}
            target_ids: dict[str, UUID] = {}
            for member in request.members:
                if member.mode == "NEW":
                    assert member.target is not None
                    if member.target.responsibility_group_id != component_group[member.component_key]:
                        raise validation_failed(f"成员 {member.member_key} 的 Target 责任组必须继承组件责任组")
                    target = await self._create_target_in_uow(uow=uow, scope=scope, request=member.target, now=now)
                    await uow.flush()
                    await self._check_target_connectivity_now(uow=uow, scope=scope, entity=target, checked_at=now)
                    if target.connectivity_status != "CONNECTED":
                        raise validation_failed(f"成员 {member.member_key} 数据库连接验证失败：{target.last_error_code or target.connectivity_status}")
                else:
                    assert member.target_id is not None
                    target = await uow.targets.get_scoped(target_id=member.target_id, domain_id=scope.domain_id, lock=True)
                    if target is None:
                        raise validation_failed(f"成员 {member.member_key} 引用的 Target 不存在")
                    if await uow.business_systems.target_is_assigned(target_id=target.target_id):
                        raise state_conflict(f"成员 {member.member_key} 已属于其他组件")
                    if target.connectivity_status != "CONNECTED":
                        raise validation_failed(f"成员 {member.member_key} 尚未通过数据库连接验证")
                existing_bindings = await uow.targets.list_source_bindings(target_id=target.target_id, domain_id=scope.domain_id)
                if member.monitoring is not None:
                    if existing_bindings:
                        raise state_conflict(f"成员 {member.member_key} 已有监控映射")
                    mapping = member.monitoring
                    if mapping.candidate_ref is not None:
                        await self._create_discovered_source_binding_in_uow(uow=uow, scope=scope, source_id=mapping.source_id, target_id=target.target_id, candidate_ref=mapping.candidate_ref, host_candidate_ref=mapping.host_candidate_ref, now=now)
                    else:
                        assert mapping.source_binding is not None
                        await self._create_source_binding_in_uow(uow=uow, scope=scope, target_id=target.target_id, request=mapping.source_binding, now=now)
                elif not existing_bindings:
                    raise validation_failed(f"成员 {member.member_key} 必须配置监控映射")
                target_ids[member.member_key] = target.target_id

            await self._add_topology(
                uow=uow, scope=scope, system=system,
                components=request.components, sites=request.sites,
                member_target_ids=target_ids, members=request.members,
                relations=request.relations, now=now,
            )
            await uow.flush()
            detail = await self._system_detail(uow=uow, scope=scope, system=system)
            blocking = [issue for issue in detail.validation_issues if issue.severity == "BLOCKING"]
            if blocking:
                raise validation_failed("；".join(issue.message for issue in blocking))
            system.status = "ENABLED"
            system.updated_at = now
            system.updated_by = scope.actor_id
            await uow.flush()
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="BUSINESS_SYSTEM", aggregate_id=system.business_system_id, event_type="BUSINESS_SYSTEM_COMMITTED", row_version=int(system.row_version), details={"component_count": len(request.components), "target_count": len(request.members)})
            return await self._system_detail(uow=uow, scope=scope, system=system)
        return await self._idempotent(scope=scope, operation="BUSINESS_SYSTEM_COMMIT", parent_resource="business-systems", idempotency_key=idempotency_key, payload=request.model_dump(mode="json"), response_type=BusinessSystemDetail, handler=handler)

    async def list_business_systems(self, *, scope: ConfigurationScope, status: str | None, limit: int) -> BusinessSystemPage:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            rows = await uow.business_systems.list_scoped(status=status, limit=limit, **_scope_args(scope))
            items = [
                _summary(
                    await self._system_detail(
                        uow=uow,
                        scope=scope,
                        system=row,
                    )
                )
                for row in rows
            ]
            return BusinessSystemPage(items=tuple(items))

    async def get_business_system(self, *, scope: ConfigurationScope, business_system_id: UUID) -> BusinessSystemDetail:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_scoped(business_system_id=business_system_id, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System")
            return await self._system_detail(uow=uow, scope=scope, system=entity)

    async def patch_business_system(self, *, scope: ConfigurationScope, business_system_id: UUID, request: BusinessSystemPatch, expected_version: int) -> BusinessSystemDetail:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_scoped(business_system_id=business_system_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System")
            self._check_version(entity.row_version, expected_version)
            changes = request.model_dump(exclude_unset=True, exclude={"schema_version"})
            if "responsibility_group_id" in changes:
                await self._validate_group(uow=uow, scope=scope, group_id=changes["responsibility_group_id"], label="业务系统")
            if "display_name" in changes and changes["display_name"] != entity.display_name:
                duplicate = await uow.business_systems.get_by_name(display_name=changes["display_name"], **_scope_args(scope))
                if duplicate is not None and duplicate.business_system_id != entity.business_system_id:
                    raise state_conflict("当前 Domain 已存在同名业务系统")
            if "system_code" in changes and changes["system_code"].casefold() != entity.system_code.casefold():
                duplicate = await uow.business_systems.get_by_code(system_code=changes["system_code"], **_scope_args(scope))
                if duplicate is not None and duplicate.business_system_id != entity.business_system_id:
                    raise state_conflict("当前 Domain 已存在相同系统编码")
            for key, value in changes.items():
                setattr(entity, key, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = self._now()
            await uow.flush()
            return await self._system_detail(uow=uow, scope=scope, system=entity)

    async def put_business_system_topology(self, *, scope: ConfigurationScope, business_system_id: UUID, request: BusinessSystemTopologyPut, expected_version: int) -> BusinessSystemDetail:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None and uow.targets is not None
            entity = await uow.business_systems.get_scoped(business_system_id=business_system_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System")
            self._check_version(entity.row_version, expected_version)
            component_group = {item.component_key: item.responsibility_group_id for item in request.components}
            target_ids: dict[str, UUID] = {}
            for member in request.members:
                if member.mode == "NEW":
                    assert member.target is not None
                    if member.target.responsibility_group_id != component_group[member.component_key]:
                        raise validation_failed(f"成员 {member.member_key} 的 Target 责任组必须继承组件责任组")
                    target = await self._create_target_in_uow(uow=uow, scope=scope, request=member.target, now=self._now())
                    await uow.flush()
                    await self._check_target_connectivity_now(uow=uow, scope=scope, entity=target, checked_at=self._now())
                    if target.connectivity_status != "CONNECTED":
                        raise validation_failed(f"成员 {member.member_key} 数据库连接验证失败：{target.last_error_code or target.connectivity_status}")
                else:
                    assert member.target_id is not None
                    target = await uow.targets.get_scoped(target_id=member.target_id, domain_id=scope.domain_id, lock=True)
                    if target is None:
                        raise validation_failed(f"成员 {member.member_key} 引用的 Target 不存在")
                    assigned_system = await uow.business_systems.find_system_for_target(target_id=target.target_id, **_scope_args(scope))
                    if assigned_system is not None and assigned_system.business_system_id != business_system_id:
                        raise state_conflict(f"成员 {member.member_key} 已属于其他业务系统")
                    if member.target_patch is not None:
                        assert member.target_row_version is not None
                        self._check_version(target.row_version, member.target_row_version)
                        try:
                            validate_supported_database_version(target.db_type, member.target_patch.version_code)
                        except ValueError as exc:
                            raise validation_failed(str(exc)) from exc
                        target.display_name = member.target_patch.display_name
                        target.version_code = member.target_patch.version_code
                        target.environment = member.target_patch.environment
                        target.db_role = member.target_patch.db_role
                        target.responsibility_group_id = component_group[member.component_key]
                        target.updated_by = scope.actor_id
                        target.updated_at = self._now()
                    if target.connectivity_status != "CONNECTED":
                        raise validation_failed(f"成员 {member.member_key} 尚未通过数据库连接验证")
                existing_bindings = await uow.targets.list_source_bindings(target_id=target.target_id, domain_id=scope.domain_id)
                if member.monitoring is not None:
                    if existing_bindings:
                        raise state_conflict(f"成员 {member.member_key} 已有监控映射")
                    mapping = member.monitoring
                    if mapping.candidate_ref is not None:
                        await self._create_discovered_source_binding_in_uow(uow=uow, scope=scope, source_id=mapping.source_id, target_id=target.target_id, candidate_ref=mapping.candidate_ref, host_candidate_ref=mapping.host_candidate_ref, now=self._now())
                    else:
                        assert mapping.source_binding is not None
                        await self._create_source_binding_in_uow(uow=uow, scope=scope, target_id=target.target_id, request=mapping.source_binding, now=self._now())
                elif not existing_bindings:
                    raise validation_failed(f"成员 {member.member_key} 必须配置监控映射")
                target_ids[member.member_key] = target.target_id
            current_components = {item.component_key: item for item in await uow.business_systems.list_components(business_system_id)}
            desired_keys = {item.component_key for item in request.components}
            removed = [item for key, item in current_components.items() if key not in desired_keys]
            for component in removed:
                if await uow.recovery.component_has_history(component_id=component.component_id):
                    raise state_conflict(f"组件 {component.display_name} 已有数据保护历史，不能直接移除")
            await uow.business_systems.clear_topology_edges(business_system_id)
            await uow.business_systems.remove_components(business_system_id=business_system_id, component_ids=[item.component_id for item in removed])
            remaining = {key: item for key, item in current_components.items() if key in desired_keys}
            await self._add_topology(uow=uow, scope=scope, system=entity, components=request.components, sites=request.sites, member_target_ids=target_ids, members=request.members, relations=request.relations, now=self._now(), existing_components=remaining)
            entity.updated_by = scope.actor_id
            entity.updated_at = self._now()
            await uow.flush()
            detail = await self._system_detail(uow=uow, scope=scope, system=entity)
            blocking = [issue for issue in detail.validation_issues if issue.severity == "BLOCKING"]
            if blocking:
                raise validation_failed("；".join(issue.message for issue in blocking))
            return detail

    async def command_business_system(self, *, scope: ConfigurationScope, business_system_id: UUID, command: str, expected_version: int, idempotency_key: str) -> BusinessSystemDetail:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> BusinessSystemDetail:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_scoped(business_system_id=business_system_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System")
            self._check_version(entity.row_version, expected_version)
            desired = "ENABLED" if command == "enable" else "DISABLED"
            entity.status = desired
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.flush()
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="BUSINESS_SYSTEM", aggregate_id=entity.business_system_id, event_type=f"BUSINESS_SYSTEM_{desired}", row_version=int(entity.row_version), details={"cascade": False})
            return await self._system_detail(uow=uow, scope=scope, system=entity)
        return await self._idempotent(scope=scope, operation=f"BUSINESS_SYSTEM_{command.upper()}", parent_resource=str(business_system_id), idempotency_key=idempotency_key, payload={"row_version": expected_version}, response_type=BusinessSystemDetail, handler=handler)
