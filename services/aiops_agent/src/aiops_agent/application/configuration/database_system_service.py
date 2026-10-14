"""数据库系统及拓扑配置用例。"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from aiops_agent.application.configuration.common import ConfigurationScope, add_configuration_event
from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.entities import DatabaseSystemEntity, DatabaseSystemMemberEntity, DatabaseSystemRelationEntity, DatabaseSystemSiteEntity
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts.aiops import (
    DatabaseSystemCommit,
    DatabaseSystemCreate,
    DatabaseSystemDetail,
    DatabaseSystemMemberView,
    DatabaseSystemPage,
    DatabaseSystemPatch,
    DatabaseSystemRelationView,
    DatabaseSystemSiteView,
    DatabaseSystemSummary,
    DatabaseSystemTopologyPut,
    TopologyValidationIssue,
)
from platform_core.identity import uuid7


def _scope_args(scope: ConfigurationScope) -> dict:
    values = {"domain_id": scope.domain_id}
    if hasattr(scope, "tenant_id"):
        values["tenant_id"] = scope.tenant_id
    return values


def _issue(severity: str, code: str, message: str, member_key: str | None = None) -> TopologyValidationIssue:
    return TopologyValidationIssue(severity=severity, code=code, message=message, member_key=member_key)


class DatabaseSystemConfigurationMixin:
    async def _system_detail(self, *, uow: AIOpsUnitOfWork, scope: ConfigurationScope, system: DatabaseSystemEntity) -> DatabaseSystemDetail:
        assert uow.database_systems is not None and uow.targets is not None and uow.recovery is not None
        sites = await uow.database_systems.list_sites(system.database_system_id)
        members = await uow.database_systems.list_members(system.database_system_id)
        relations = await uow.database_systems.list_relations(system.database_system_id)
        site_by_id = {site.site_id: site for site in sites}
        member_by_id = {member.member_id: member for member in members}
        member_views: list[DatabaseSystemMemberView] = []
        issues: list[TopologyValidationIssue] = []
        for member in members:
            target = await uow.targets.get_scoped(target_id=member.target_id, domain_id=scope.domain_id)
            if target is None:
                issues.append(_issue("BLOCKING", "TARGET_MISSING", "成员 Target 不存在", member.member_key))
                continue
            bindings = await uow.targets.list_source_bindings(target_id=target.target_id, domain_id=scope.domain_id)
            observed_role = target.db_role if target.db_role in {"PRIMARY", "STANDBY", "SOURCE", "REPLICA", "RAC_INSTANCE"} else "UNKNOWN"
            member_views.append(DatabaseSystemMemberView(
                member_id=member.member_id,
                member_key=member.member_key,
                target_id=member.target_id,
                target_display_name=target.display_name,
                site_code=site_by_id[member.site_id].site_code,
                group_key=member.group_key,
                member_type=member.member_type,
                expected_role=member.expected_role,
                observed_role=observed_role,
                connectivity_status=target.connectivity_status,
                monitoring_ready=bool(bindings),
                sort_order=int(member.sort_order),
            ))
            if target.db_type != system.db_type or target.version_code != system.version_code:
                issues.append(_issue("BLOCKING", "TARGET_ENGINE_MISMATCH", "成员数据库引擎或版本与系统不一致", member.member_key))
            if target.connectivity_status != "CONNECTED":
                issues.append(_issue("BLOCKING", "TARGET_NOT_CONNECTED", "成员连接验证尚未通过", member.member_key))
            if not bindings:
                issues.append(_issue("BLOCKING", "MONITORING_NOT_READY", "成员监控映射尚未完成", member.member_key))
            if observed_role != "UNKNOWN" and member.expected_role not in {"INSTANCE", "RAC_INSTANCE"} and observed_role != member.expected_role:
                issues.append(_issue("WARNING", "ROLE_DEVIATION", "期望角色与探测角色不一致", member.member_key))

        roles = [item.expected_role for item in member_views]
        site_count = len({item.site_code for item in member_views})
        caps = set(system.capabilities_json or [])
        if system.deployment_category == "SINGLE_INSTANCE":
            if len(member_views) != 1:
                issues.append(_issue("BLOCKING", "SINGLE_MEMBER_REQUIRED", "单实例系统必须且只能包含一个成员"))
        if "RAC" in caps:
            rac_members = [item for item in member_views if item.member_type == "RAC_INSTANCE"]
            rac_groups: dict[str, int] = {}
            rac_group_sites: dict[str, set[UUID]] = {}
            for item in rac_members:
                if item.group_key:
                    rac_groups[item.group_key] = rac_groups.get(item.group_key, 0) + 1
                    rac_group_sites.setdefault(item.group_key, set()).add(
                        member_by_id[item.member_id].site_id
                    )
            if (
                len(rac_members) < 2
                or len(rac_groups) == 0
                or sum(rac_groups.values()) != len(rac_members)
                or any(count < 2 for count in rac_groups.values())
                or any(len(site_ids) != 1 for site_ids in rac_group_sites.values())
            ):
                issues.append(_issue("BLOCKING", "RAC_GROUP_INVALID", "每个 RAC 成员组必须位于同一站点且至少包含两个 RAC 实例"))
        if "DATA_GUARD" in caps:
            primary_ready = roles.count("PRIMARY") == 1
            standby_ready = roles.count("STANDBY") >= 1
            if "RAC" in caps:
                primary_ready = primary_ready or any(
                    item.member_type == "RAC_INSTANCE"
                    and site_by_id[member_by_id[item.member_id].site_id].expected_role == "PRIMARY"
                    for item in member_views
                )
                standby_ready = standby_ready or any(
                    item.member_type == "RAC_INSTANCE"
                    and site_by_id[member_by_id[item.member_id].site_id].expected_role == "STANDBY"
                    for item in member_views
                )
            if not primary_ready or not standby_ready:
                issues.append(_issue("BLOCKING", "DATA_GUARD_ROLES_INVALID", "Data Guard 必须同时具备 Primary 端和 Standby 端；任一端均可为 RAC 组"))
        if "MYSQL_REPLICATION" in caps:
            if roles.count("SOURCE") != 1 or roles.count("REPLICA") < 1:
                issues.append(_issue("BLOCKING", "MYSQL_REPLICATION_ROLES_INVALID", "MySQL 复制必须有一个 Source 和至少一个 Replica"))
        if "POSTGRES_STREAMING" in caps:
            if roles.count("PRIMARY") != 1 or roles.count("STANDBY") < 1:
                issues.append(_issue("BLOCKING", "POSTGRES_STREAMING_ROLES_INVALID", "PostgreSQL Streaming 必须有一个 Primary 和至少一个 Standby"))
        relation_views: list[DatabaseSystemRelationView] = []
        for relation in relations:
            source = member_by_id.get(relation.source_member_id)
            target = member_by_id.get(relation.target_member_id)
            if source is None or target is None:
                issues.append(_issue("BLOCKING", "RELATION_ENDPOINT_MISSING", "复制关系缺少起点或终点"))
                continue
            relation_views.append(DatabaseSystemRelationView(
                relation_id=relation.relation_id,
                relation_key=relation.relation_key,
                source_member_key=source.member_key,
                target_member_key=target.member_key,
                relation_type=relation.relation_type,
                transport_mode=relation.transport_mode,
                sort_order=int(relation.sort_order),
            ))
            if site_by_id[source.site_id].site_code != site_by_id[target.site_id].site_code and relation.transport_mode == "UNKNOWN":
                issues.append(_issue("WARNING", "TRANSPORT_MODE_UNKNOWN", "跨站点复制关系尚未填写同步模式"))
        expected_relation = {
            "DATA_GUARD": "DATA_GUARD",
            "MYSQL_REPLICATION": "MYSQL_REPLICATION",
            "POSTGRES_STREAMING": "POSTGRES_STREAMING",
        }
        for capability, relation_type in expected_relation.items():
            if capability in caps and not any(item.relation_type == relation_type for item in relation_views):
                issues.append(_issue("BLOCKING", "REPLICATION_RELATION_REQUIRED", "高可用系统缺少有效复制关系"))
        if system.deployment_shape == "HIGH_AVAILABILITY" and member_views and site_count == 1:
            issues.append(_issue("WARNING", "SINGLE_SITE_HA", "高可用系统的全部成员位于同一站点"))
        profile = await uow.recovery.get_active_profile(database_system_id=system.database_system_id, **_scope_args(scope))
        if profile is None:
            issues.append(_issue("WARNING", "PROTECTION_UNCONFIGURED", "备份与恢复保护尚未配置"))
        return DatabaseSystemDetail(
            database_system_id=system.database_system_id,
            display_name=system.display_name,
            db_type=system.db_type,
            version_code=system.version_code,
            deployment_shape=system.deployment_shape,
            deployment_category=system.deployment_category,
            capabilities=tuple(system.capabilities_json or []),
            environment=system.environment,
            importance_level=int(system.importance_level),
            responsibility_group_id=system.responsibility_group_id,
            status=system.status,
            protection_status="CONFIGURED" if profile else "UNCONFIGURED",
            member_count=len(member_views),
            row_version=int(system.row_version),
            updated_at=system.updated_at,
            description=system.description,
            sites=tuple(DatabaseSystemSiteView(site_id=item.site_id, site_code=item.site_code, display_name=item.display_name, expected_role=item.expected_role, sort_order=int(item.sort_order)) for item in sites),
            members=tuple(member_views),
            relations=tuple(relation_views),
            validation_issues=tuple(issues),
            created_at=system.created_at,
        )

    async def create_database_system(self, *, scope: ConfigurationScope, request: DatabaseSystemCreate, idempotency_key: str) -> DatabaseSystemDetail:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> DatabaseSystemDetail:
            assert uow.database_systems is not None and uow.work_items is not None
            if await uow.database_systems.get_by_name(display_name=request.display_name, **_scope_args(scope)) is not None:
                raise state_conflict("当前 Domain 已存在同名数据库系统")
            if request.responsibility_group_id is not None:
                group = await uow.work_items.get_group(group_id=request.responsibility_group_id, **_scope_args(scope))
                if group is None or group.status != "ACTIVE":
                    raise validation_failed("责任组不存在或已停用")
            values = dict(
                database_system_id=uuid7(),
                domain_id=scope.domain_id,
                display_name=request.display_name,
                db_type=request.db_type.value,
                version_code=request.version_code,
                deployment_shape=request.deployment_shape,
                deployment_category=request.deployment_category,
                capabilities_json=list(request.capabilities),
                environment=request.environment,
                importance_level=request.importance_level,
                responsibility_group_id=request.responsibility_group_id,
                description=request.description,
                status="DRAFT",
                row_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            if hasattr(scope, "tenant_id"):
                values["tenant_id"] = scope.tenant_id
            entity = DatabaseSystemEntity(**values)
            await uow.database_systems.add_system(entity)
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="DATABASE_SYSTEM", aggregate_id=entity.database_system_id, event_type="DATABASE_SYSTEM_DRAFT_CREATED", row_version=1, details={"deployment_category": entity.deployment_category})
            return await self._system_detail(uow=uow, scope=scope, system=entity)
        return await self._idempotent(scope=scope, operation="DATABASE_SYSTEM_CREATE", parent_resource="database-systems", idempotency_key=idempotency_key, payload=request.model_dump(mode="json"), response_type=DatabaseSystemDetail, handler=handler)

    async def commit_database_system(self, *, scope: ConfigurationScope, request: DatabaseSystemCommit, idempotency_key: str) -> DatabaseSystemDetail:
        """原子创建并启用数据库系统，任何成员或映射失败都整体回滚。"""
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> DatabaseSystemDetail:
            assert uow.database_systems is not None and uow.targets is not None and uow.work_items is not None
            system_request = request.system
            if await uow.database_systems.get_by_name(display_name=system_request.display_name, **_scope_args(scope)) is not None: raise state_conflict("当前 Domain 已存在同名数据库系统")
            if system_request.responsibility_group_id is None: raise validation_failed("数据库系统必须绑定责任组")
            group = await uow.work_items.get_group(group_id=system_request.responsibility_group_id, **_scope_args(scope))
            if group is None or group.status != "ACTIVE": raise validation_failed("责任组不存在或已停用")
            values = dict(database_system_id=uuid7(), domain_id=scope.domain_id, display_name=system_request.display_name, db_type=system_request.db_type.value, version_code=system_request.version_code, deployment_shape=system_request.deployment_shape, deployment_category=system_request.deployment_category, capabilities_json=list(system_request.capabilities), environment=system_request.environment, importance_level=system_request.importance_level, responsibility_group_id=system_request.responsibility_group_id, description=system_request.description, status="DRAFT", row_version=1, created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now)
            if hasattr(scope, "tenant_id"): values["tenant_id"] = scope.tenant_id
            system = DatabaseSystemEntity(**values); await uow.database_systems.add_system(system)
            target_ids: dict[str, UUID] = {}
            for member in request.members:
                if member.mode == "NEW":
                    assert member.target is not None
                    if member.target.responsibility_group_id != system_request.responsibility_group_id: raise validation_failed(f"成员 {member.member_key} 的责任组必须与数据库系统一致")
                    target = await self._create_target_in_uow(uow=uow, scope=scope, request=member.target, now=now)
                    await uow.flush(); await self._check_target_connectivity_now(uow=uow, scope=scope, entity=target, checked_at=now)
                    if target.connectivity_status != "CONNECTED": raise validation_failed(f"成员 {member.member_key} 数据库连接验证失败：{target.last_error_code or target.connectivity_status}")
                else:
                    assert member.target_id is not None
                    target = await uow.targets.get_scoped(target_id=member.target_id, domain_id=scope.domain_id, lock=True)
                    if target is None: raise validation_failed(f"成员 {member.member_key} 引用的 Target 不存在")
                    if await uow.database_systems.target_is_assigned_elsewhere(target_id=target.target_id, database_system_id=system.database_system_id): raise state_conflict(f"成员 {member.member_key} 已属于其他数据库系统")
                    if target.connectivity_status != "CONNECTED": raise validation_failed(f"成员 {member.member_key} 尚未通过数据库连接验证")
                if target.db_type != system.db_type or target.version_code != system.version_code: raise validation_failed(f"成员 {member.member_key} 的数据库引擎或版本不一致")
                bindings = await uow.targets.list_source_bindings(target_id=target.target_id, domain_id=scope.domain_id)
                if member.monitoring is not None:
                    if bindings: raise state_conflict(f"成员 {member.member_key} 已有监控映射")
                    mapping = member.monitoring
                    if mapping.candidate_ref is not None:
                        await self._create_discovered_source_binding_in_uow(uow=uow, scope=scope, source_id=mapping.source_id, target_id=target.target_id, candidate_ref=mapping.candidate_ref, host_candidate_ref=mapping.host_candidate_ref, now=now)
                    else:
                        assert mapping.source_binding is not None
                        await self._create_source_binding_in_uow(uow=uow, scope=scope, target_id=target.target_id, request=mapping.source_binding, now=now)
                elif not bindings: raise validation_failed(f"成员 {member.member_key} 必须配置监控映射")
                target_ids[member.member_key] = target.target_id
            site_entities = [DatabaseSystemSiteEntity(site_id=uuid7(), database_system_id=system.database_system_id, site_code=item.site_code, display_name=item.display_name, expected_role=item.expected_role, sort_order=item.sort_order) for item in request.sites]
            await uow.database_systems.add_sites(site_entities); site_ids = {item.site_code: item.site_id for item in site_entities}
            member_entities = [DatabaseSystemMemberEntity(member_id=uuid7(), database_system_id=system.database_system_id, site_id=site_ids[item.site_code], target_id=target_ids[item.member_key], member_key=item.member_key, group_key=item.group_key, member_type=item.member_type, expected_role=item.expected_role, sort_order=item.sort_order) for item in request.members]
            await uow.database_systems.add_members(member_entities); member_ids = {item.member_key: item.member_id for item in member_entities}
            await uow.database_systems.add_relations([DatabaseSystemRelationEntity(relation_id=uuid7(), database_system_id=system.database_system_id, source_member_id=member_ids[item.source_member_key], target_member_id=member_ids[item.target_member_key], relation_key=item.relation_key, relation_type=item.relation_type, transport_mode=item.transport_mode, sort_order=item.sort_order) for item in request.relations])
            await uow.flush(); detail = await self._system_detail(uow=uow, scope=scope, system=system)
            blocking = [item.message for item in detail.validation_issues if item.severity == "BLOCKING"]
            if blocking: raise validation_failed("；".join(blocking))
            system.status = "ENABLED"
            for member in member_entities:
                target = await uow.targets.get_scoped(target_id=member.target_id, domain_id=scope.domain_id, lock=True); assert target is not None
                target.status = "ENABLED"; target.updated_by = scope.actor_id; target.updated_at = now
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="DATABASE_SYSTEM", aggregate_id=system.database_system_id, event_type="DATABASE_SYSTEM_COMMITTED", row_version=1, details={"member_count": len(member_entities)})
            await uow.flush(); return await self._system_detail(uow=uow, scope=scope, system=system)
        return await self._idempotent(scope=scope, operation="DATABASE_SYSTEM_COMMIT", parent_resource="database-systems", idempotency_key=idempotency_key, payload=request.model_dump(mode="json"), response_type=DatabaseSystemDetail, handler=handler)

    async def list_database_systems(self, *, scope: ConfigurationScope, status: str | None, limit: int) -> DatabaseSystemPage:
        async with self._uow_factory() as uow:
            assert uow.database_systems is not None
            rows = await uow.database_systems.list_scoped(status=status, limit=limit, **_scope_args(scope))
            details = [await self._system_detail(uow=uow, scope=scope, system=item) for item in rows]
            return DatabaseSystemPage(items=tuple(DatabaseSystemSummary.model_validate(item.model_dump()) for item in details))

    async def get_database_system(self, *, scope: ConfigurationScope, database_system_id: UUID) -> DatabaseSystemDetail:
        async with self._uow_factory() as uow:
            assert uow.database_systems is not None
            entity = await uow.database_systems.get_scoped(database_system_id=database_system_id, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Database System")
            return await self._system_detail(uow=uow, scope=scope, system=entity)

    async def patch_database_system(self, *, scope: ConfigurationScope, database_system_id: UUID, request: DatabaseSystemPatch, expected_version: int) -> DatabaseSystemDetail:
        async with self._uow_factory() as uow:
            assert uow.database_systems is not None
            entity = await uow.database_systems.get_scoped(database_system_id=database_system_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Database System")
            self._check_version(entity.row_version, expected_version)
            values = request.model_dump(exclude_unset=True, exclude={"schema_version"})
            if "responsibility_group_id" in values and values["responsibility_group_id"] is not None:
                assert uow.work_items is not None
                group = await uow.work_items.get_group(
                    group_id=values["responsibility_group_id"],
                    **_scope_args(scope),
                )
                if group is None or group.status != "ACTIVE":
                    raise validation_failed("责任组不存在或已停用")
            if "display_name" in values and values["display_name"] != entity.display_name:
                duplicate = await uow.database_systems.get_by_name(display_name=values["display_name"], **_scope_args(scope))
                if duplicate is not None and duplicate.database_system_id != entity.database_system_id:
                    raise state_conflict("当前 Domain 已存在同名数据库系统")
            for key, value in values.items():
                setattr(entity, key, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = datetime.now().astimezone()
            await uow.commit()
            return await self.get_database_system(scope=scope, database_system_id=database_system_id)

    async def put_database_system_topology(self, *, scope: ConfigurationScope, database_system_id: UUID, request: DatabaseSystemTopologyPut, expected_version: int) -> DatabaseSystemDetail:
        async with self._uow_factory() as uow:
            assert uow.database_systems is not None and uow.targets is not None
            system = await uow.database_systems.get_scoped(database_system_id=database_system_id, lock=True, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Database System")
            self._check_version(system.row_version, expected_version)
            if system.status == "ENABLED":
                raise state_conflict("启用中的数据库系统必须先停用再调整拓扑")
            for item in request.members:
                target = await uow.targets.get_scoped(target_id=item.target_id, domain_id=scope.domain_id, lock=True)
                if target is None:
                    raise validation_failed(f"成员 {item.member_key} 引用的 Target 不存在")
                if target.db_type != system.db_type or target.version_code != system.version_code:
                    raise validation_failed(f"成员 {item.member_key} 的数据库引擎或版本不一致")
                if await uow.database_systems.target_is_assigned_elsewhere(target_id=item.target_id, database_system_id=database_system_id):
                    raise state_conflict(f"成员 {item.member_key} 已属于其他数据库系统")
            await uow.database_systems.replace_topology(database_system_id)
            site_entities = [DatabaseSystemSiteEntity(site_id=uuid7(), database_system_id=database_system_id, site_code=item.site_code, display_name=item.display_name, expected_role=item.expected_role, sort_order=item.sort_order) for item in request.sites]
            await uow.database_systems.add_sites(site_entities)
            site_ids = {item.site_code: item.site_id for item in site_entities}
            member_entities = [DatabaseSystemMemberEntity(member_id=uuid7(), database_system_id=database_system_id, site_id=site_ids[item.site_code], target_id=item.target_id, member_key=item.member_key, group_key=item.group_key, member_type=item.member_type, expected_role=item.expected_role, sort_order=item.sort_order) for item in request.members]
            await uow.database_systems.add_members(member_entities)
            member_ids = {item.member_key: item.member_id for item in member_entities}
            relation_entities = [DatabaseSystemRelationEntity(relation_id=uuid7(), database_system_id=database_system_id, source_member_id=member_ids[item.source_member_key], target_member_id=member_ids[item.target_member_key], relation_key=item.relation_key, relation_type=item.relation_type, transport_mode=item.transport_mode, sort_order=item.sort_order) for item in request.relations]
            await uow.database_systems.add_relations(relation_entities)
            system.updated_by = scope.actor_id
            system.updated_at = datetime.now().astimezone()
            await uow.commit()
            return await self.get_database_system(scope=scope, database_system_id=database_system_id)

    async def command_database_system(self, *, scope: ConfigurationScope, database_system_id: UUID, command: str, expected_version: int, idempotency_key: str) -> DatabaseSystemDetail:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> DatabaseSystemDetail:
            assert uow.database_systems is not None and uow.targets is not None
            system = await uow.database_systems.get_scoped(database_system_id=database_system_id, lock=True, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Database System")
            self._check_version(system.row_version, expected_version)
            if command == "enable":
                detail = await self._system_detail(uow=uow, scope=scope, system=system)
                blocking = [item.message for item in detail.validation_issues if item.severity == "BLOCKING"]
                if blocking:
                    raise validation_failed("；".join(blocking))
                system.status = "ENABLED"
                for member in await uow.database_systems.list_members(database_system_id):
                    target = await uow.targets.get_scoped(target_id=member.target_id, domain_id=scope.domain_id, lock=True)
                    if target is not None:
                        target.status = "ENABLED"
                        target.updated_by = scope.actor_id
                        target.updated_at = now
            elif command == "disable":
                system.status = "DISABLED"
                for member in await uow.database_systems.list_members(database_system_id):
                    target = await uow.targets.get_scoped(target_id=member.target_id, domain_id=scope.domain_id, lock=True)
                    if target is not None:
                        target.status = "DISABLED"
                        target.updated_by = scope.actor_id
                        target.updated_at = now
            else:
                raise validation_failed("未知数据库系统命令")
            system.updated_by = scope.actor_id
            system.updated_at = now
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="DATABASE_SYSTEM", aggregate_id=database_system_id, event_type=f"DATABASE_SYSTEM_{system.status}", row_version=int(system.row_version), details={})
            await uow.flush()
            return await self._system_detail(uow=uow, scope=scope, system=system)
        return await self._idempotent(scope=scope, operation=f"DATABASE_SYSTEM_{command.upper()}", parent_resource=str(database_system_id), idempotency_key=idempotency_key, payload={"row_version": expected_version}, response_type=DatabaseSystemDetail, handler=handler)
