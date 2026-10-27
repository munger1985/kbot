"""业务系统资源图谱配置用例。"""

from __future__ import annotations

import base64
from collections import deque
from datetime import datetime, timedelta
from uuid import UUID

from pydantic import ValidationError

from aiops_agent.application.configuration.common import ConfigurationScope, add_configuration_event
from aiops_agent.application.errors import resource_not_found, state_conflict, validation_failed
from aiops_agent.application.managed_credentials import AIOpsManagedCredentialError
from aiops_agent.entities import (
    BusinessSystemDraftEntity,
    BusinessSystemEntity,
    BusinessSystemResourceEntity,
    ComputeResourceEntity,
    DatabaseResourceEntity,
    ManagedResourceEntity,
    RelationParticipantEntity,
    ResourceDiscoveryJobEntity,
    ResourceRelationEntity,
)
from aiops_agent.persistence import AIOpsUnitOfWork
from platform_core.contracts.aiops import (
    BusinessSystemDraftCreate,
    BusinessSystemDraftPage,
    BusinessSystemDraftSave,
    BusinessSystemDraftSummary,
    BusinessSystemDraftView,
    BusinessSystemInfoPatch,
    BusinessSystemLifecycleCommand,
    BusinessSystemResourcePatch,
    BusinessSystemResourceCommit,
    BusinessSystemResourceGraphCommit,
    BusinessSystemResourceGraphDetail,
    BusinessSystemResourceGraphPage,
    BusinessSystemResourceGraphSummary,
    BusinessSystemResourceView,
    DatabaseResourceView,
    ManagedResourcePatch,
    ManagedResourceView,
    RelationParticipantInput,
    RelationParticipantView,
    ResourceGraphValidationIssue,
    ResourceRelationInputV2,
    ResourceRelationPatchV2,
    ResourceRelationViewV2,
    ResourceTypeSummary,
    TopologyParticipantEdge,
    TopologyRelationNode,
    TopologyResourceNode,
    TopologySubgraph,
)
from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialPayload
from platform_core.contracts.aiops.configuration import (
    validate_database_deployment,
    validate_supported_database_version,
)


RESOURCE_TYPE_NAMES = {
    "DATABASE": "数据库",
    "PHYSICAL_SERVER": "物理服务器",
    "VIRTUAL_MACHINE": "虚拟机",
    "CONTAINER": "容器",
    "POD": "Pod",
    "KUBERNETES_CLUSTER": "Kubernetes 集群",
    "APPLICATION_SERVICE": "应用服务",
    "APPLICATION_SERVER": "应用服务器",
    "MESSAGE_QUEUE": "消息队列",
    "LOAD_BALANCER": "负载均衡",
    "NETWORK_DEVICE": "网络设备",
    "STORAGE": "存储",
}

RESOURCE_CODE_PREFIXES = {
    "DATABASE": "DB",
    "PHYSICAL_SERVER": "SRV",
    "VIRTUAL_MACHINE": "SRV",
    "CONTAINER": "CTR",
    "POD": "POD",
    "KUBERNETES_CLUSTER": "K8S",
    "APPLICATION_SERVICE": "APP",
    "APPLICATION_SERVER": "MID",
    "MESSAGE_QUEUE": "MQ",
    "LOAD_BALANCER": "LB",
    "NETWORK_DEVICE": "NET",
    "STORAGE": "STG",
}

RELATION_RULES: dict[str, dict[str, tuple[int, int | None]]] = {
    "ORACLE_RAC": {"MEMBER": (2, None)},
    "DATA_GUARD": {"PRIMARY": (1, None), "STANDBY": (1, None)},
    "OGG": {"SOURCE": (1, None), "TARGET": (1, None)},
    "MYSQL_REPLICATION": {"SOURCE": (1, 1), "REPLICA": (1, None)},
    "POSTGRES_STREAMING": {"PRIMARY": (1, 1), "STANDBY": (1, None)},
    "RUNS_ON": {"WORKLOAD": (1, 1), "HOST": (1, 1)},
    "USES_STORAGE": {"CONSUMER": (1, None), "PROVIDER": (1, None)},
    "DEPENDS_ON": {"CONSUMER": (1, None), "PROVIDER": (1, None)},
    "LOAD_BALANCES": {"BALANCER": (1, None), "BACKEND": (1, None)},
}


def _scope_args(scope: ConfigurationScope) -> dict:
    values = {"domain_id": scope.domain_id}
    if hasattr(scope, "tenant_id"):
        values["tenant_id"] = scope.tenant_id
    return values


def _code(prefix: str, value: UUID) -> str:
    return f"{prefix}-{value.hex[-6:].upper()}"


def _summary(detail: BusinessSystemResourceGraphDetail) -> BusinessSystemResourceGraphSummary:
    return BusinessSystemResourceGraphSummary.model_validate(
        detail.model_dump(include=set(BusinessSystemResourceGraphSummary.model_fields))
    )


def _issue(severity: str, code: str, message: str, *, resource_id: UUID | None = None, relation_id: UUID | None = None) -> ResourceGraphValidationIssue:
    return ResourceGraphValidationIssue(
        severity=severity,
        code=code,
        message=message,
        resource_id=resource_id,
        relation_id=relation_id,
    )


def _relation_rule_issues(
    relation: ResourceRelationEntity,
    participants: list[RelationParticipantEntity],
    resource_views: dict[UUID, BusinessSystemResourceView],
) -> list[ResourceGraphValidationIssue]:
    issues: list[ResourceGraphValidationIssue] = []
    rules = RELATION_RULES.get(relation.relation_type_key)
    if rules is None:
        return [_issue("BLOCKING", "RELATION_TYPE_UNKNOWN", f"不支持的关系类型：{relation.relation_type_key}", relation_id=relation.relation_id)]
    counts: dict[str, int] = {}
    for participant in participants:
        counts[participant.participant_role] = counts.get(participant.participant_role, 0) + 1
        if participant.participant_role not in rules:
            issues.append(_issue("BLOCKING", "RELATION_ROLE_INVALID", f"{relation.relation_type_key} 不允许角色 {participant.participant_role}", relation_id=relation.relation_id))
        if participant.system_resource_id not in resource_views:
            issues.append(_issue("BLOCKING", "RELATION_RESOURCE_MISSING", "关系参与者不属于当前业务系统", relation_id=relation.relation_id))
    for role, (minimum, maximum) in rules.items():
        count = counts.get(role, 0)
        if count < minimum or (maximum is not None and count > maximum):
            limit = f"{minimum}～{maximum}" if maximum is not None else f"至少 {minimum}"
            issues.append(_issue("BLOCKING", "RELATION_ROLE_COUNT_INVALID", f"{relation.relation_type_key} 的 {role} 角色需要{limit}个资源", relation_id=relation.relation_id))
    database_views = [resource_views[item.system_resource_id].resource.database for item in participants if item.system_resource_id in resource_views and resource_views[item.system_resource_id].resource.database is not None]
    engines = {str(item.db_type) for item in database_views}
    expected_engine = {
        "ORACLE_RAC": "ORACLE",
        "DATA_GUARD": "ORACLE",
        "MYSQL_REPLICATION": "MYSQL",
        "POSTGRES_STREAMING": "POSTGRESQL",
    }.get(relation.relation_type_key)
    if expected_engine is not None and engines != {expected_engine}:
        issues.append(_issue("BLOCKING", "RELATION_ENGINE_INVALID", f"{relation.relation_type_key} 的数据库资源必须全部为 {expected_engine}", relation_id=relation.relation_id))
    return issues


def _architecture_labels(resources: list[BusinessSystemResourceView], relations: list[ResourceRelationViewV2]) -> tuple[str, ...]:
    labels: list[str] = []
    relation_types = {item.relation_type_key for item in relations if item.confirmation_status == "CONFIRMED"}
    if "ORACLE_RAC" in relation_types:
        labels.append("RAC")
    if "DATA_GUARD" in relation_types:
        labels.append("DATA_GUARD")
    if "ORACLE_RAC" in relation_types and "DATA_GUARD" in relation_types:
        labels.insert(0, "RAC_DATA_GUARD")
    if "MYSQL_REPLICATION" in relation_types:
        labels.append("MYSQL_REPLICATION")
    if "POSTGRES_STREAMING" in relation_types:
        labels.append("POSTGRES_STREAMING")
    if "OGG" in relation_types:
        labels.append("OGG")
    databases = [item.resource.database for item in resources if item.resource.database is not None]
    if len(databases) == 1 and not relation_types:
        labels.append("SINGLE_INSTANCE")
    versions = {(str(item.db_type), item.version_code) for item in databases}
    if len(versions) > 1:
        labels.append("MIXED_VERSION")
    return tuple(dict.fromkeys(labels))


class BusinessSystemConfigurationMixin:
    _DRAFT_CREDENTIAL_NAMESPACE = "aiops-business-system-draft"

    def _seal_draft_payload(self, *, scope: ConfigurationScope, draft_id: UUID, payload: dict) -> dict:
        cipher_scope = dict(
            domain_id=scope.domain_id,
            namespace=self._DRAFT_CREDENTIAL_NAMESPACE,
            credential_kind="workspace",
            credential_id=draft_id,
        )
        if hasattr(scope, "tenant_id"):
            cipher_scope["tenant_id"] = scope.tenant_id
        encrypted = self._credential_cipher.encrypt(payload, **cipher_scope)
        return {
            "v": 1,
            "key_version": encrypted.key_version,
            "nonce": base64.b64encode(encrypted.nonce).decode("ascii"),
            "ciphertext": base64.b64encode(encrypted.ciphertext).decode("ascii"),
        }

    def _open_draft_payload(self, entity: BusinessSystemDraftEntity) -> dict:
        envelope = entity.system_payload_json
        try:
            cipher_scope = dict(
                domain_id=entity.domain_id,
                namespace=self._DRAFT_CREDENTIAL_NAMESPACE,
                credential_kind="workspace",
                credential_id=entity.draft_id,
            )
            if hasattr(entity, "tenant_id"):
                cipher_scope["tenant_id"] = entity.tenant_id
            return self._credential_cipher.decrypt(
                ManagedCredentialPayload(
                    ciphertext=base64.b64decode(envelope["ciphertext"]),
                    nonce=base64.b64decode(envelope["nonce"]),
                    key_version=str(envelope["key_version"]),
                ),
                **cipher_scope,
            )
        except Exception as exc:
            raise state_conflict("草稿内容无法解密，请联系管理员检查受管凭据密钥") from exc

    @staticmethod
    def _publish_validation_message(payload: dict, error: ValidationError) -> str:
        resources = payload.get("resources") if isinstance(payload.get("resources"), list) else []
        messages: list[str] = []
        for item in error.errors()[:10]:
            location = tuple(item.get("loc", ()))
            message = str(item.get("msg", "配置无效")).removeprefix("Value error, ")
            prefix = "发布配置"
            if len(location) >= 2 and location[0] == "resources" and isinstance(location[1], int):
                index = location[1]
                resource = resources[index] if index < len(resources) and isinstance(resources[index], dict) else {}
                prefix = str(resource.get("display_name") or f"资源 {index + 1}")
                path = ".".join(str(part) for part in location[2:])
                if path.endswith("monitoring.source_id"):
                    message = "未选择监控源"
                elif path.endswith("monitoring.candidate_ref"):
                    message = "未配置数据库监控 Target"
                elif path.endswith("target.endpoint.host"):
                    message = "未填写 Host"
                elif path.endswith("target.endpoint.service"):
                    message = "未填写 Service Name"
                elif path.endswith("target.diagnostic_credential.username"):
                    message = "未填写诊断用户名"
                elif path.endswith("target.diagnostic_credential.password"):
                    message = "未填写诊断密码"
            elif location and location[0] == "system":
                prefix = "业务信息"
            elif len(location) >= 2 and location[0] == "relations":
                prefix = f"关系 {int(location[1]) + 1}" if isinstance(location[1], int) else "资源关系"
            text = f"{prefix}：{message}"
            if text not in messages:
                messages.append(text)
        return "；".join(messages) or "草稿内容不满足发布要求"

    async def _validate_group(self, *, uow: AIOpsUnitOfWork, scope: ConfigurationScope, group_id: UUID | None, label: str) -> None:
        assert uow.work_items is not None
        if group_id is None:
            raise validation_failed(f"{label}必须绑定责任组")
        group = await uow.work_items.get_group(group_id=group_id, **_scope_args(scope))
        if group is None or group.status != "ACTIVE":
            raise validation_failed(f"{label}责任组不存在或已停用")

    async def _system_detail(
        self,
        *,
        uow: AIOpsUnitOfWork,
        scope: ConfigurationScope,
        system: BusinessSystemEntity,
        include_connection_metadata: bool = True,
    ) -> BusinessSystemResourceGraphDetail:
        assert uow.business_systems is not None and uow.targets is not None
        rows = await uow.business_systems.list_system_resource_rows(system.business_system_id)
        resource_views: list[BusinessSystemResourceView] = []
        issues: list[ResourceGraphValidationIssue] = []
        type_counts: dict[str, int] = {}
        for link, resource, database_profile, target in rows:
            database_view = None
            if database_profile is not None:
                if target is None:
                    issues.append(_issue("BLOCKING", "DATABASE_TARGET_MISSING", "数据库资源缺少内部连接端点", resource_id=resource.resource_id))
                else:
                    bindings = await uow.targets.list_source_bindings(target_id=target.target_id, domain_id=scope.domain_id)
                    diagnostic_username = None
                    if include_connection_metadata and target.diagnostic_credential_id is not None:
                        try:
                            credential = await self._managed_credentials.read(
                                uow=uow,
                                domain_id=scope.domain_id,
                                credential_id=target.diagnostic_credential_id,
                                credential_kind="target_diagnostic",
                                external_key=target.target_id,
                            )
                            username = credential.get("username")
                            if isinstance(username, str):
                                diagnostic_username = username
                        except AIOpsManagedCredentialError:
                            pass
                    database_view = DatabaseResourceView(
                        target_id=target.target_id,
                        db_type=target.db_type,
                        version_code=target.version_code or "UNKNOWN",
                        environment=target.environment,
                        deployment_type=target.deployment_type or "SINGLE_INSTANCE",
                        rac_node_number=target.rac_node_number,
                        db_role=target.db_role,
                        connectivity_status=target.connectivity_status,
                        monitoring_ready=bool(bindings),
                        endpoint=(dict(target.endpoint_json) if include_connection_metadata and target.endpoint_json else None),
                        oracle_container_scope=(target.oracle_container_scope if include_connection_metadata else None),
                        oracle_pdb_name=(target.oracle_pdb_name if include_connection_metadata else None),
                        diagnostic_username=diagnostic_username,
                    )
                    if target.connectivity_status != "CONNECTED":
                        issues.append(_issue("BLOCKING", "DATABASE_NOT_CONNECTED", "数据库连接验证尚未通过", resource_id=resource.resource_id))
                    if not bindings:
                        issues.append(_issue("BLOCKING", "MONITORING_NOT_READY", "数据库监控映射尚未完成", resource_id=resource.resource_id))
            view = BusinessSystemResourceView(
                system_resource_id=link.system_resource_id,
                business_system_id=link.business_system_id,
                architecture_layer_key=link.architecture_layer_key,
                responsibility_group_id=link.responsibility_group_id,
                effective_responsibility_group_id=link.responsibility_group_id or resource.owner_group_id,
                display_order=int(link.display_order),
                status=link.status,
                row_version=int(link.row_version),
                resource=ManagedResourceView(
                    resource_id=resource.resource_id,
                    resource_code=resource.resource_code,
                    resource_type_key=resource.resource_type_key,
                    display_name=resource.display_name,
                    identity_kind=resource.identity_kind,
                    lifecycle_status=resource.lifecycle_status,
                    owner_group_id=resource.owner_group_id,
                    location_name=resource.location_name,
                    row_version=int(resource.row_version),
                    database=database_view,
                ),
            )
            resource_views.append(view)
            if link.status == "ACTIVE":
                type_counts[resource.resource_type_key] = type_counts.get(resource.resource_type_key, 0) + 1

        relations = await uow.business_systems.list_relations(system.business_system_id)
        participants = await uow.business_systems.list_participants([item.relation_id for item in relations])
        participants_by_relation: dict[UUID, list[RelationParticipantEntity]] = {}
        for participant in participants:
            participants_by_relation.setdefault(participant.relation_id, []).append(participant)
        view_by_system_resource = {item.system_resource_id: item for item in resource_views}
        relation_views: list[ResourceRelationViewV2] = []
        for relation in relations:
            relation_participants = participants_by_relation.get(relation.relation_id, [])
            issues.extend(_relation_rule_issues(relation, relation_participants, view_by_system_resource))
            participant_views = tuple(
                RelationParticipantView(
                    participant_id=item.participant_id,
                    system_resource_id=item.system_resource_id,
                    resource_id=view_by_system_resource[item.system_resource_id].resource.resource_id,
                    participant_role=item.participant_role,
                    display_order=int(item.display_order),
                )
                for item in relation_participants
                if item.system_resource_id in view_by_system_resource
            )
            relation_views.append(ResourceRelationViewV2(
                relation_id=relation.relation_id,
                business_system_id=relation.business_system_id,
                relation_type_key=relation.relation_type_key,
                display_name=relation.display_name,
                responsibility_group_id=relation.responsibility_group_id,
                origin=relation.origin,
                confirmation_status=relation.confirmation_status,
                verification_status=relation.verification_status,
                attributes=dict(relation.attributes_json or {}),
                participants=participant_views,
                row_version=int(relation.row_version),
            ))

        if not resource_views:
            issues.append(_issue("BLOCKING", "BUSINESS_SYSTEM_EMPTY", "业务系统至少需要一个资源"))
        summary = tuple(
            ResourceTypeSummary(resource_type_key=key, display_name=RESOURCE_TYPE_NAMES.get(key, key), count=count)
            for key, count in sorted(type_counts.items())
        )
        return BusinessSystemResourceGraphDetail(
            business_system_id=system.business_system_id,
            system_code=system.system_code,
            display_name=system.display_name,
            environment=system.environment,
            importance_level=int(system.importance_level),
            responsibility_group_id=system.responsibility_group_id,
            description=system.description,
            status=system.status,
            protection_status="UNCONFIGURED",
            resource_count=sum(type_counts.values()),
            resource_type_summary=summary,
            architecture_labels=_architecture_labels(resource_views, relation_views),
            resources=tuple(resource_views),
            relations=tuple(relation_views),
            validation_issues=tuple(issues),
            row_version=int(system.row_version),
            created_at=system.created_at,
            updated_at=system.updated_at,
        )

    async def commit_business_system(self, *, scope: ConfigurationScope, request: BusinessSystemResourceGraphCommit, idempotency_key: str) -> BusinessSystemResourceGraphDetail:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> BusinessSystemResourceGraphDetail:
            assert uow.business_systems is not None and uow.targets is not None
            if await uow.business_systems.get_by_name(display_name=request.system.display_name, **_scope_args(scope)) is not None:
                raise state_conflict("当前 Domain 已存在同名业务系统")
            await self._validate_group(uow=uow, scope=scope, group_id=request.system.responsibility_group_id, label="业务系统")
            system_id = uuid7()
            system_values = dict(
                business_system_id=system_id,
                domain_id=scope.domain_id,
                system_code=_code("SYS", system_id),
                display_name=request.system.display_name,
                environment=request.system.environment,
                importance_level=request.system.importance_level,
                responsibility_group_id=request.system.responsibility_group_id,
                description=request.system.description,
                status="ENABLED",
                row_version=1,
                created_by=scope.actor_id,
                updated_by=scope.actor_id,
                created_at=now,
                updated_at=now,
            )
            if hasattr(scope, "tenant_id"):
                system_values["tenant_id"] = scope.tenant_id
            system = BusinessSystemEntity(**system_values)
            await uow.business_systems.add_system(system)

            system_resource_by_draft: dict[UUID, BusinessSystemResourceEntity] = {}
            used_resource_ids: set[UUID] = set()
            for item in request.resources:
                await self._validate_group(uow=uow, scope=scope, group_id=item.owner_group_id, label=f"资源 {item.display_name}")
                if item.responsibility_group_id is not None:
                    await self._validate_group(uow=uow, scope=scope, group_id=item.responsibility_group_id, label=f"资源 {item.display_name} 在本系统中的责任覆盖")
                if item.mode != "NEW":
                    raise validation_failed("当前版本只支持新接入资源")
                else:
                    resource_id = uuid7()
                    prefix = RESOURCE_CODE_PREFIXES.get(item.resource_type_key, "RES")
                    resource_values = dict(
                        resource_id=resource_id,
                        domain_id=scope.domain_id,
                        resource_code=None if item.identity_kind == "EPHEMERAL" else _code(prefix, resource_id),
                        resource_type_key=item.resource_type_key,
                        display_name=item.display_name,
                        identity_kind=item.identity_kind,
                        lifecycle_status="DISCOVERED" if item.identity_kind != "STABLE" else "ACTIVE",
                        owner_group_id=item.owner_group_id,
                        location_name=item.location_name,
                        identity_fingerprint=None,
                        attributes_json=dict(item.attributes),
                        row_version=1,
                        created_by=scope.actor_id,
                        updated_by=scope.actor_id,
                        created_at=now,
                        updated_at=now,
                    )
                    if hasattr(scope, "tenant_id"):
                        resource_values["tenant_id"] = scope.tenant_id
                    resource = ManagedResourceEntity(**resource_values)
                    await uow.business_systems.add_resource(resource)
                    if item.resource_type_key == "DATABASE":
                        assert item.database is not None
                        target_request = item.database.target.model_copy(update={"responsibility_group_id": item.owner_group_id})
                        target = await self._create_target_in_uow(uow=uow, scope=scope, request=target_request, now=now)
                        await uow.flush()
                        await self._check_target_connection_now(uow=uow, scope=scope, entity=target)
                        if target.connectivity_status != "CONNECTED":
                            raise validation_failed(f"数据库 {item.display_name} 连接验证失败：{target.last_error_code or target.connectivity_status}")
                        mapping = item.database.monitoring
                        if mapping is None:
                            raise validation_failed(f"数据库 {item.display_name} 必须配置监控映射")
                        if mapping.candidate_ref is not None:
                            await self._create_discovered_source_binding_in_uow(
                                uow=uow,
                                scope=scope,
                                source_id=mapping.source_id,
                                target_id=target.target_id,
                                candidate_ref=mapping.candidate_ref,
                                host_candidate_ref=mapping.host_candidate_ref,
                                now=now,
                                allow_locator_reuse=True,
                            )
                        else:
                            assert mapping.source_binding is not None
                            await self._create_source_binding_in_uow(uow=uow, scope=scope, target_id=target.target_id, request=mapping.source_binding, now=now)
                        await uow.business_systems.add_database_resource(DatabaseResourceEntity(resource_id=resource.resource_id, target_id=target.target_id))
                    elif item.resource_type_key in {"PHYSICAL_SERVER", "VIRTUAL_MACHINE", "CONTAINER", "POD", "KUBERNETES_CLUSTER"}:
                        await uow.business_systems.add_compute_resource(ComputeResourceEntity(
                            resource_id=resource.resource_id,
                            compute_kind=item.resource_type_key,
                            platform_name=str(item.attributes.get("platform_name")) if item.attributes.get("platform_name") else None,
                            platform_version=str(item.attributes.get("platform_version")) if item.attributes.get("platform_version") else None,
                        ))
                used_resource_ids.add(resource.resource_id)
                link = BusinessSystemResourceEntity(
                    system_resource_id=uuid7(),
                    business_system_id=system.business_system_id,
                    resource_id=resource.resource_id,
                    architecture_layer_key=item.architecture_layer_key,
                    responsibility_group_id=item.responsibility_group_id,
                    display_order=item.display_order,
                    status="ACTIVE",
                    row_version=1,
                    created_by=scope.actor_id,
                    updated_by=scope.actor_id,
                    created_at=now,
                    updated_at=now,
                )
                await uow.business_systems.add_system_resources([link])
                system_resource_by_draft[item.draft_resource_id] = link
            await uow.flush()

            relation_entities: list[ResourceRelationEntity] = []
            participant_entities: list[RelationParticipantEntity] = []
            for item in request.relations:
                if item.responsibility_group_id is not None:
                    await self._validate_group(uow=uow, scope=scope, group_id=item.responsibility_group_id, label=f"关系 {item.display_name or item.relation_type_key}")
                relation = ResourceRelationEntity(
                    relation_id=uuid7(),
                    business_system_id=system.business_system_id,
                    relation_type_key=item.relation_type_key,
                    display_name=item.display_name,
                    responsibility_group_id=item.responsibility_group_id,
                    origin=item.origin,
                    confirmation_status=item.confirmation_status,
                    verification_status="UNKNOWN",
                    attributes_json=dict(item.attributes),
                    row_version=1,
                    created_by=scope.actor_id,
                    updated_by=scope.actor_id,
                    created_at=now,
                    updated_at=now,
                )
                relation_entities.append(relation)
                participant_entities.extend(RelationParticipantEntity(
                    participant_id=uuid7(),
                    relation_id=relation.relation_id,
                    system_resource_id=system_resource_by_draft[participant.draft_resource_id].system_resource_id,
                    participant_role=participant.participant_role,
                    display_order=participant.display_order,
                ) for participant in item.participants)
            await uow.business_systems.add_relations(relation_entities)
            await uow.business_systems.add_participants(participant_entities)
            await uow.flush()
            detail = await self._system_detail(uow=uow, scope=scope, system=system)
            blockers = [item.message for item in detail.validation_issues if item.severity == "BLOCKING"]
            if blockers:
                raise validation_failed("；".join(blockers))
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="BUSINESS_SYSTEM",
                aggregate_id=system.business_system_id,
                event_type="BUSINESS_SYSTEM_RESOURCE_GRAPH_CREATED",
                row_version=int(system.row_version),
                details={"resource_count": detail.resource_count, "relation_count": len(detail.relations)},
            )
            return detail

        return await self._idempotent(
            scope=scope,
            operation="BUSINESS_SYSTEM_RESOURCE_GRAPH_COMMIT",
            parent_resource="business-systems",
            idempotency_key=idempotency_key,
            payload=request.model_dump(mode="json"),
            response_type=BusinessSystemResourceGraphDetail,
            handler=handler,
        )

    async def list_business_systems(self, *, scope: ConfigurationScope, status: str | None, limit: int) -> BusinessSystemResourceGraphPage:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            rows = await uow.business_systems.list_scoped(status=status, limit=limit, **_scope_args(scope))
            items: list[BusinessSystemResourceGraphSummary] = []
            for row in rows:
                detail = await self._system_detail(
                    uow=uow,
                    scope=scope,
                    system=row,
                    include_connection_metadata=False,
                )
                items.append(_summary(detail))
            return BusinessSystemResourceGraphPage(items=tuple(items))

    async def get_business_system(self, *, scope: ConfigurationScope, business_system_id: UUID) -> BusinessSystemResourceGraphDetail:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_scoped(business_system_id=business_system_id, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System")
            return await self._system_detail(uow=uow, scope=scope, system=entity)

    async def list_managed_resources(self, *, scope: ConfigurationScope, resource_type_key: str | None, limit: int) -> tuple[ManagedResourceView, ...]:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            rows = await uow.business_systems.list_resources_scoped(
                resource_type_key=resource_type_key,
                limit=limit,
                **_scope_args(scope),
            )
            return tuple(ManagedResourceView(
                resource_id=item.resource_id,
                resource_code=item.resource_code,
                resource_type_key=item.resource_type_key,
                display_name=item.display_name,
                identity_kind=item.identity_kind,
                lifecycle_status=item.lifecycle_status,
                owner_group_id=item.owner_group_id,
                location_name=item.location_name,
                row_version=int(item.row_version),
            ) for item in rows)

    async def add_system_resource(self, *, scope: ConfigurationScope, business_system_id: UUID, request: BusinessSystemResourceCommit, idempotency_key: str) -> BusinessSystemResourceGraphDetail:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> BusinessSystemResourceGraphDetail:
            assert uow.business_systems is not None and uow.targets is not None
            system = await uow.business_systems.get_scoped(business_system_id=business_system_id, lock=True, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Business System")
            await self._validate_group(uow=uow, scope=scope, group_id=request.owner_group_id, label=f"资源 {request.display_name}")
            if request.responsibility_group_id is not None:
                await self._validate_group(uow=uow, scope=scope, group_id=request.responsibility_group_id, label="资源责任覆盖")
            if request.mode != "NEW":
                raise validation_failed("当前版本只支持新接入资源")
            else:
                resource_id = uuid7()
                values = dict(
                    resource_id=resource_id, domain_id=scope.domain_id,
                    resource_code=None if request.identity_kind == "EPHEMERAL" else _code(RESOURCE_CODE_PREFIXES.get(request.resource_type_key, "RES"), resource_id),
                    resource_type_key=request.resource_type_key, display_name=request.display_name,
                    identity_kind=request.identity_kind,
                    lifecycle_status="DISCOVERED" if request.identity_kind != "STABLE" else "ACTIVE",
                    owner_group_id=request.owner_group_id, location_name=request.location_name,
                    identity_fingerprint=None, attributes_json=dict(request.attributes), row_version=1,
                    created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now,
                )
                if hasattr(scope, "tenant_id"):
                    values["tenant_id"] = scope.tenant_id
                resource = ManagedResourceEntity(**values)
                await uow.business_systems.add_resource(resource)
                if request.resource_type_key == "DATABASE":
                    assert request.database is not None
                    target_request = request.database.target.model_copy(update={"responsibility_group_id": request.owner_group_id})
                    target = await self._create_target_in_uow(uow=uow, scope=scope, request=target_request, now=now)
                    await uow.flush()
                    await self._check_target_connection_now(uow=uow, scope=scope, entity=target)
                    if target.connectivity_status != "CONNECTED":
                        raise validation_failed(f"数据库连接验证失败：{target.last_error_code or target.connectivity_status}")
                    mapping = request.database.monitoring
                    if mapping is None:
                        raise validation_failed("数据库必须配置监控映射")
                    if mapping.candidate_ref is not None:
                        await self._create_discovered_source_binding_in_uow(
                            uow=uow, scope=scope, source_id=mapping.source_id, target_id=target.target_id,
                            candidate_ref=mapping.candidate_ref, host_candidate_ref=mapping.host_candidate_ref, now=now,
                            allow_locator_reuse=True,
                        )
                    else:
                        await self._create_source_binding_in_uow(
                            uow=uow, scope=scope, target_id=target.target_id,
                            request=mapping.source_binding, now=now,
                        )
                    await uow.business_systems.add_database_resource(DatabaseResourceEntity(resource_id=resource.resource_id, target_id=target.target_id))
                elif request.resource_type_key in {"PHYSICAL_SERVER", "VIRTUAL_MACHINE", "CONTAINER", "POD", "KUBERNETES_CLUSTER"}:
                    await uow.business_systems.add_compute_resource(ComputeResourceEntity(
                        resource_id=resource.resource_id, compute_kind=request.resource_type_key,
                        platform_name=str(request.attributes.get("platform_name")) if request.attributes.get("platform_name") else None,
                        platform_version=str(request.attributes.get("platform_version")) if request.attributes.get("platform_version") else None,
                    ))
            existing = [row for row, _, _, _ in await uow.business_systems.list_system_resource_rows(business_system_id) if row.resource_id == resource.resource_id and row.status == "ACTIVE"]
            if existing:
                raise state_conflict("该资源已经加入当前业务系统")
            await uow.business_systems.add_system_resources([BusinessSystemResourceEntity(
                system_resource_id=uuid7(), business_system_id=business_system_id, resource_id=resource.resource_id,
                architecture_layer_key=request.architecture_layer_key,
                responsibility_group_id=request.responsibility_group_id, display_order=request.display_order,
                status="ACTIVE", row_version=1, created_by=scope.actor_id, updated_by=scope.actor_id,
                created_at=now, updated_at=now,
            )])
            await uow.flush()
            return await self._system_detail(uow=uow, scope=scope, system=system)
        return await self._idempotent(
            scope=scope, operation="BUSINESS_SYSTEM_RESOURCE_ADD", parent_resource=str(business_system_id),
            idempotency_key=idempotency_key, payload=request.model_dump(mode="json"),
            response_type=BusinessSystemResourceGraphDetail, handler=handler,
        )

    async def patch_managed_resource(self, *, scope: ConfigurationScope, resource_id: UUID, request: ManagedResourcePatch, expected_version: int) -> ManagedResourceView:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            assert uow.targets is not None
            if request.database is not None and request.database.workload_snapshot_policy is not None:
                system_id = request.database.business_system_id
                link_id = request.database.system_resource_id
                system = await uow.business_systems.get_scoped(business_system_id=system_id, lock=True, **_scope_args(scope))
                if system is None:
                    raise resource_not_found("Business System")
                if system.status == "ENABLED":
                    raise state_conflict("运行中的业务系统不能修改历史 SQL 采集策略")
                link = await uow.business_systems.get_system_resource_scoped(business_system_id=system_id, system_resource_id=link_id, lock=True)
                if link is None or link.resource_id != resource_id or link.status != "ACTIVE":
                    raise validation_failed("数据库资源不属于当前业务系统")
            entity = await uow.business_systems.get_resource_scoped(resource_id=resource_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Managed Resource")
            self._check_version(entity.row_version, expected_version)
            changes = request.model_dump(exclude_unset=True, exclude={"schema_version", "database"})
            owner_group_id = changes.get("owner_group_id")
            if owner_group_id is not None:
                await self._validate_group(uow=uow, scope=scope, group_id=owner_group_id, label="资源")
            if "attributes" in changes:
                changes["attributes_json"] = changes.pop("attributes")
            for key, value in changes.items():
                setattr(entity, key, value)
            database_view = None
            if request.database is not None:
                profile = await uow.business_systems.get_database_profile(resource_id)
                if profile is None:
                    raise validation_failed("只有数据库资源可以修改数据库版本、环境或角色")
                target = await uow.targets.get_scoped(
                    target_id=profile.target_id,
                    domain_id=scope.domain_id,
                    lock=True,
                )
                if target is None:
                    raise resource_not_found("Database Target")
                database_changes = request.database.model_dump(exclude_unset=True)
                policy = request.database.workload_snapshot_policy
                target_version = database_changes.pop("target_row_version", None)
                database_changes.pop("workload_snapshot_policy", None)
                database_changes.pop("business_system_id", None)
                database_changes.pop("system_resource_id", None)
                if policy is not None:
                    self._check_version(target.row_version, target_version)
                    target.workload_snapshot_policy_json = policy.model_dump(mode="json")
                    target.workload_next_run_at = self._now() if policy.enabled else None
                    target.workload_consecutive_failures = 0
                    target.workload_last_error_code = None
                    target.updated_by = scope.actor_id
                    target.updated_at = self._now()
                if "version_code" in database_changes:
                    try:
                        validate_supported_database_version(
                            target.db_type, database_changes["version_code"]
                        )
                    except ValueError as exc:
                        raise validation_failed(str(exc)) from exc
                effective_deployment = database_changes.get(
                    "deployment_type", target.deployment_type
                )
                effective_rac_node_number = database_changes.get(
                    "rac_node_number", target.rac_node_number
                )
                if (
                    "deployment_type" in database_changes
                    and effective_deployment == "SINGLE_INSTANCE"
                    and "rac_node_number" not in database_changes
                ):
                    effective_rac_node_number = None
                    database_changes["rac_node_number"] = None
                try:
                    validate_database_deployment(
                        target.db_type,
                        effective_deployment,
                        effective_rac_node_number,
                    )
                except ValueError as exc:
                    raise validation_failed(str(exc)) from exc
                for key, value in database_changes.items():
                    setattr(target, key, value)
                database_view = DatabaseResourceView(
                    target_id=target.target_id,
                    db_type=target.db_type,
                    version_code=target.version_code,
                    environment=target.environment,
                    deployment_type=target.deployment_type or "SINGLE_INSTANCE",
                    rac_node_number=target.rac_node_number,
                    db_role=target.db_role,
                    connectivity_status=target.connectivity_status,
                    monitoring_ready=bool(
                        await uow.targets.list_source_bindings(
                            target_id=target.target_id,
                            domain_id=scope.domain_id,
                        )
                    ),
                )
            entity.updated_by = scope.actor_id
            entity.updated_at = self._now()
            await uow.flush()
            result = ManagedResourceView(
                resource_id=entity.resource_id,
                resource_code=entity.resource_code,
                resource_type_key=entity.resource_type_key,
                display_name=entity.display_name,
                identity_kind=entity.identity_kind,
                lifecycle_status=entity.lifecycle_status,
                owner_group_id=entity.owner_group_id,
                location_name=entity.location_name,
                row_version=int(entity.row_version),
                database=database_view,
            )
            await uow.commit()
            return result

    async def patch_system_resource(self, *, scope: ConfigurationScope, business_system_id: UUID, system_resource_id: UUID, request: BusinessSystemResourcePatch, expected_version: int) -> BusinessSystemResourceGraphDetail:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            system = await uow.business_systems.get_scoped(business_system_id=business_system_id, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Business System")
            entity = await uow.business_systems.get_system_resource_scoped(
                business_system_id=business_system_id,
                system_resource_id=system_resource_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Business System Resource")
            self._check_version(entity.row_version, expected_version)
            changes = request.model_dump(exclude_unset=True, exclude={"schema_version"})
            if changes.get("responsibility_group_id") is not None:
                await self._validate_group(uow=uow, scope=scope, group_id=changes["responsibility_group_id"], label="资源责任覆盖")
            for key, value in changes.items():
                setattr(entity, key, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = self._now()
            await uow.flush()
            result = await self._system_detail(uow=uow, scope=scope, system=system)
            await uow.commit()
            return result

    async def remove_system_resource(self, *, scope: ConfigurationScope, business_system_id: UUID, system_resource_id: UUID, expected_version: int) -> None:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            system = await uow.business_systems.get_scoped(business_system_id=business_system_id, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Business System")
            entity = await uow.business_systems.get_system_resource_scoped(
                business_system_id=business_system_id,
                system_resource_id=system_resource_id,
                lock=True,
            )
            if entity is None:
                raise resource_not_found("Business System Resource")
            self._check_version(entity.row_version, expected_version)
            if await uow.business_systems.resource_count(business_system_id) <= 1:
                raise validation_failed("业务系统至少保留一个资源")
            await uow.business_systems.remove_system_resource(system_resource_id)
            await uow.commit()

    async def _validate_relation_participants(self, *, uow: AIOpsUnitOfWork, business_system_id: UUID, relation_type_key: str, participants: tuple[RelationParticipantInput, ...]) -> dict[UUID, BusinessSystemResourceView]:
        if relation_type_key not in RELATION_RULES:
            raise validation_failed(f"不支持的关系类型：{relation_type_key}")
        detail_rows = await uow.business_systems.list_system_resource_rows(business_system_id)
        known = {link.system_resource_id for link, _, _, _ in detail_rows if link.status == "ACTIVE"}
        missing = [item.system_resource_id for item in participants if item.system_resource_id not in known]
        if missing:
            raise validation_failed("资源关系包含不属于当前业务系统的资源")
        rules = RELATION_RULES[relation_type_key]
        counts: dict[str, int] = {}
        for item in participants:
            if item.participant_role not in rules:
                raise validation_failed(f"{relation_type_key} 不允许角色 {item.participant_role}")
            counts[item.participant_role] = counts.get(item.participant_role, 0) + 1
        for role, (minimum, maximum) in rules.items():
            count = counts.get(role, 0)
            if count < minimum or (maximum is not None and count > maximum):
                raise validation_failed(f"{relation_type_key} 的 {role} 参与者数量无效")
        return {}

    async def create_resource_relation(self, *, scope: ConfigurationScope, business_system_id: UUID, request: ResourceRelationInputV2, idempotency_key: str) -> BusinessSystemResourceGraphDetail:
        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> BusinessSystemResourceGraphDetail:
            assert uow.business_systems is not None
            system = await uow.business_systems.get_scoped(business_system_id=business_system_id, lock=True, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Business System")
            await self._validate_relation_participants(uow=uow, business_system_id=business_system_id, relation_type_key=request.relation_type_key, participants=request.participants)
            if request.responsibility_group_id is not None:
                await self._validate_group(uow=uow, scope=scope, group_id=request.responsibility_group_id, label="资源关系")
            relation = ResourceRelationEntity(
                relation_id=uuid7(), business_system_id=business_system_id,
                relation_type_key=request.relation_type_key, display_name=request.display_name,
                responsibility_group_id=request.responsibility_group_id, origin=request.origin,
                confirmation_status=request.confirmation_status, verification_status="UNKNOWN",
                attributes_json=dict(request.attributes), row_version=1,
                created_by=scope.actor_id, updated_by=scope.actor_id, created_at=now, updated_at=now,
            )
            await uow.business_systems.add_relations([relation])
            await uow.business_systems.add_participants([RelationParticipantEntity(
                participant_id=uuid7(), relation_id=relation.relation_id,
                system_resource_id=item.system_resource_id, participant_role=item.participant_role,
                display_order=item.display_order,
            ) for item in request.participants])
            await uow.flush()
            return await self._system_detail(uow=uow, scope=scope, system=system)
        return await self._idempotent(
            scope=scope, operation="RESOURCE_RELATION_CREATE", parent_resource=str(business_system_id),
            idempotency_key=idempotency_key, payload=request.model_dump(mode="json"),
            response_type=BusinessSystemResourceGraphDetail, handler=handler,
        )

    async def patch_resource_relation(self, *, scope: ConfigurationScope, business_system_id: UUID, relation_id: UUID, request: ResourceRelationPatchV2, expected_version: int) -> BusinessSystemResourceGraphDetail:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            system = await uow.business_systems.get_scoped(business_system_id=business_system_id, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Business System")
            relation = await uow.business_systems.get_relation_scoped(business_system_id=business_system_id, relation_id=relation_id, lock=True)
            if relation is None:
                raise resource_not_found("Resource Relation")
            self._check_version(relation.row_version, expected_version)
            changes = request.model_dump(exclude_unset=True, exclude={"schema_version", "participants"})
            if changes.get("responsibility_group_id") is not None:
                await self._validate_group(uow=uow, scope=scope, group_id=changes["responsibility_group_id"], label="资源关系")
            if "attributes" in changes:
                changes["attributes_json"] = changes.pop("attributes")
            for key, value in changes.items():
                setattr(relation, key, value)
            if request.participants is not None:
                await self._validate_relation_participants(uow=uow, business_system_id=business_system_id, relation_type_key=relation.relation_type_key, participants=request.participants)
                await uow.business_systems.delete_relation_participants(relation_id)
                await uow.business_systems.add_participants([RelationParticipantEntity(
                    participant_id=uuid7(), relation_id=relation_id,
                    system_resource_id=item.system_resource_id, participant_role=item.participant_role,
                    display_order=item.display_order,
                ) for item in request.participants])
            relation.updated_by = scope.actor_id
            relation.updated_at = self._now()
            await uow.flush()
            result = await self._system_detail(uow=uow, scope=scope, system=system)
            await uow.commit()
            return result

    async def delete_resource_relation(self, *, scope: ConfigurationScope, business_system_id: UUID, relation_id: UUID, expected_version: int) -> None:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            system = await uow.business_systems.get_scoped(business_system_id=business_system_id, **_scope_args(scope))
            if system is None:
                raise resource_not_found("Business System")
            relation = await uow.business_systems.get_relation_scoped(business_system_id=business_system_id, relation_id=relation_id, lock=True)
            if relation is None:
                raise resource_not_found("Resource Relation")
            self._check_version(relation.row_version, expected_version)
            await uow.business_systems.delete_relation(relation_id)
            await uow.commit()

    async def get_topology_subgraph(self, *, scope: ConfigurationScope, business_system_id: UUID, focus_resource_id: UUID | None, max_depth: int, max_nodes: int) -> TopologySubgraph:
        detail = await self.get_business_system(scope=scope, business_system_id=business_system_id)
        resources = [item for item in detail.resources if item.status == "ACTIVE"]
        resource_by_system = {item.system_resource_id: item for item in resources}
        if focus_resource_id is None:
            selected_system_ids = set(resource_by_system)
            truncated = len(selected_system_ids) > max_nodes
            selected_system_ids = set(list(selected_system_ids)[:max_nodes])
        else:
            focus = next((item for item in resources if item.resource.resource_id == focus_resource_id), None)
            if focus is None:
                raise resource_not_found("Managed Resource")
            relation_by_system: dict[UUID, set[UUID]] = {}
            relation_map = {item.relation_id: item for item in detail.relations}
            for relation in detail.relations:
                ids = {item.system_resource_id for item in relation.participants}
                for system_id in ids:
                    relation_by_system.setdefault(system_id, set()).update(ids - {system_id})
            selected_system_ids = {focus.system_resource_id}
            queue = deque([(focus.system_resource_id, 0)])
            truncated = False
            while queue:
                current, depth = queue.popleft()
                if depth >= max_depth:
                    continue
                for neighbor in relation_by_system.get(current, set()):
                    if neighbor in selected_system_ids:
                        continue
                    if len(selected_system_ids) >= max_nodes:
                        truncated = True
                        queue.clear()
                        break
                    selected_system_ids.add(neighbor)
                    queue.append((neighbor, depth + 1))
        selected_relations = [item for item in detail.relations if any(p.system_resource_id in selected_system_ids for p in item.participants)]
        selected_relation_ids = {item.relation_id for item in selected_relations}
        return TopologySubgraph(
            business_system_id=business_system_id,
            resources=tuple(TopologyResourceNode(
                system_resource_id=item.system_resource_id,
                resource_id=item.resource.resource_id,
                resource_code=item.resource.resource_code,
                display_name=item.resource.display_name,
                resource_type_key=item.resource.resource_type_key,
                architecture_layer_key=item.architecture_layer_key,
                lifecycle_status=item.resource.lifecycle_status,
            ) for item in resources if item.system_resource_id in selected_system_ids),
            relations=tuple(TopologyRelationNode(
                relation_id=item.relation_id, relation_type_key=item.relation_type_key,
                display_name=item.display_name, verification_status=item.verification_status,
            ) for item in selected_relations),
            participants=tuple(TopologyParticipantEdge(
                participant_id=item.participant_id, relation_id=relation.relation_id,
                system_resource_id=item.system_resource_id, participant_role=item.participant_role,
            ) for relation in selected_relations for item in relation.participants
              if relation.relation_id in selected_relation_ids and item.system_resource_id in selected_system_ids),
            truncated=truncated,
        )

    async def _draft_view(self, *, uow: AIOpsUnitOfWork, entity: BusinessSystemDraftEntity) -> BusinessSystemDraftView:
        return BusinessSystemDraftView(
            draft_id=entity.draft_id,
            mode=entity.mode,
            business_system_id=entity.business_system_id,
            current_step=entity.current_step,
            revision=int(entity.revision),
            status=entity.status,
            payload=self._open_draft_payload(entity),
            expires_at=entity.expires_at,
            created_at=entity.created_at,
            updated_at=entity.updated_at,
        )

    async def create_business_system_draft(self, *, scope: ConfigurationScope, request: BusinessSystemDraftCreate) -> BusinessSystemDraftView:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            base_row_version = None
            if request.business_system_id is not None:
                system = await uow.business_systems.get_scoped(business_system_id=request.business_system_id, **_scope_args(scope))
                if system is None:
                    raise resource_not_found("Business System")
                base_row_version = int(system.row_version)
            now = self._now()
            draft_id = uuid7()
            values = dict(
                draft_id=draft_id, domain_id=scope.domain_id, business_system_id=request.business_system_id,
                mode=request.mode, current_step=request.current_step, revision=1, status="ACTIVE",
                base_row_version=base_row_version, system_payload_json=self._seal_draft_payload(scope=scope, draft_id=draft_id, payload=dict(request.payload)),
                expires_at=now + timedelta(days=90), created_by=scope.actor_id, updated_by=scope.actor_id,
                created_at=now, updated_at=now,
            )
            if hasattr(scope, "tenant_id"):
                values["tenant_id"] = scope.tenant_id
            entity = BusinessSystemDraftEntity(**values)
            await uow.business_systems.add_draft(entity)
            await uow.flush()
            result = await self._draft_view(uow=uow, entity=entity)
            await uow.commit()
            return result

    async def list_business_system_drafts(self, *, scope: ConfigurationScope, limit: int) -> BusinessSystemDraftPage:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entities = await uow.business_systems.list_active_drafts(now=self._now(), limit=limit, **_scope_args(scope))
            items: list[BusinessSystemDraftSummary] = []
            for entity in entities:
                payload = self._open_draft_payload(entity)
                system = payload.get("system") if isinstance(payload.get("system"), dict) else {}
                resources = payload.get("resources") if isinstance(payload.get("resources"), list) else []
                try:
                    importance_level = int(system["importance_level"]) if system.get("importance_level") is not None else None
                except (TypeError, ValueError):
                    importance_level = None
                try:
                    responsibility_group_id = UUID(str(system["responsibility_group_id"])) if system.get("responsibility_group_id") else None
                except (TypeError, ValueError):
                    responsibility_group_id = None
                items.append(BusinessSystemDraftSummary(
                    draft_id=entity.draft_id,
                    mode=entity.mode,
                    business_system_id=entity.business_system_id,
                    current_step=entity.current_step,
                    revision=int(entity.revision),
                    status=entity.status,
                    display_name=str(system.get("display_name") or "未命名业务系统草稿"),
                    environment=str(system["environment"]) if system.get("environment") else None,
                    importance_level=importance_level,
                    responsibility_group_id=responsibility_group_id,
                    resource_count=len(resources),
                    updated_at=entity.updated_at,
                ))
            return BusinessSystemDraftPage(items=tuple(items), total=len(items))

    async def get_business_system_draft(self, *, scope: ConfigurationScope, draft_id: UUID) -> BusinessSystemDraftView:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_draft_scoped(draft_id=draft_id, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System Draft")
            if entity.status == "ACTIVE" and entity.expires_at <= self._now():
                entity.status = "EXPIRED"
                entity.updated_at = self._now()
                await uow.flush()
                result = await self._draft_view(uow=uow, entity=entity)
                await uow.commit()
                return result
            return await self._draft_view(uow=uow, entity=entity)

    async def save_business_system_draft(self, *, scope: ConfigurationScope, draft_id: UUID, request: BusinessSystemDraftSave) -> BusinessSystemDraftView:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_draft_scoped(draft_id=draft_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System Draft")
            if entity.status != "ACTIVE" or entity.expires_at <= self._now():
                raise state_conflict("草稿已提交、放弃或过期")
            if int(entity.revision) != request.revision:
                raise state_conflict("草稿已被其他会话修改，请刷新后继续")
            next_revision = int(entity.revision) + 1
            now = self._now()
            entity.system_payload_json = self._seal_draft_payload(scope=scope, draft_id=draft_id, payload=dict(request.payload))
            entity.current_step = request.current_step
            entity.revision = next_revision
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            entity.expires_at = now + timedelta(days=90)
            await uow.flush()
            result = await self._draft_view(uow=uow, entity=entity)
            await uow.commit()
            return result

    async def abandon_business_system_draft(self, *, scope: ConfigurationScope, draft_id: UUID, expected_revision: int) -> None:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_draft_scoped(draft_id=draft_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System Draft")
            if int(entity.revision) != expected_revision:
                raise state_conflict("草稿已被其他会话修改，请刷新后继续")
            if entity.status != "ACTIVE":
                raise state_conflict("草稿已结束")
            entity.status = "ABANDONED"
            entity.updated_by = scope.actor_id
            entity.updated_at = self._now()
            await uow.commit()

    async def commit_business_system_draft(self, *, scope: ConfigurationScope, draft_id: UUID, expected_revision: int, idempotency_key: str) -> BusinessSystemResourceGraphDetail:
        draft = await self.get_business_system_draft(scope=scope, draft_id=draft_id)
        if draft.status != "ACTIVE" or draft.revision != expected_revision:
            raise state_conflict("草稿状态或版本已变化")
        if draft.mode != "CREATE":
            raise validation_failed("编辑既有业务系统请使用资源和关系的增量接口")
        publish_payload = draft.payload.get("publish")
        if not isinstance(publish_payload, dict):
            raise validation_failed("草稿缺少可发布资源快照，请重新保存草稿后发布")
        try:
            request = BusinessSystemResourceGraphCommit.model_validate(publish_payload)
        except ValidationError as exc:
            raise validation_failed(self._publish_validation_message(publish_payload, exc)) from exc
        result = await self.commit_business_system(scope=scope, request=request, idempotency_key=idempotency_key)
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_draft_scoped(draft_id=draft_id, lock=True, **_scope_args(scope))
            if entity is not None and entity.status == "ACTIVE" and int(entity.revision) == expected_revision:
                entity.status = "COMMITTED"
                entity.business_system_id = result.business_system_id
                entity.updated_by = scope.actor_id
                entity.updated_at = self._now()
                await uow.commit()
        return result

    async def patch_business_system(self, *, scope: ConfigurationScope, business_system_id: UUID, request: BusinessSystemInfoPatch, expected_version: int) -> BusinessSystemResourceGraphDetail:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_scoped(business_system_id=business_system_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System")
            self._check_version(entity.row_version, expected_version)
            changes = request.model_dump(exclude_unset=True, exclude={"schema_version"})
            if changes.get("responsibility_group_id") is not None:
                await self._validate_group(uow=uow, scope=scope, group_id=changes["responsibility_group_id"], label="业务系统")
            if "display_name" in changes and changes["display_name"] != entity.display_name:
                duplicate = await uow.business_systems.get_by_name(display_name=changes["display_name"], **_scope_args(scope))
                if duplicate is not None and duplicate.business_system_id != entity.business_system_id:
                    raise state_conflict("当前 Domain 已存在同名业务系统")
            for key, value in changes.items():
                setattr(entity, key, value)
            entity.updated_by = scope.actor_id
            entity.updated_at = self._now()
            await uow.flush()
            result = await self._system_detail(uow=uow, scope=scope, system=entity)
            await uow.commit()
            return result

    async def command_business_system(self, *, scope: ConfigurationScope, business_system_id: UUID, command: str, request: BusinessSystemLifecycleCommand, expected_version: int, idempotency_key: str) -> BusinessSystemResourceGraphDetail:
        reason = request.reason.strip() if request.reason else None
        if command == "disable" and not reason:
            raise validation_failed("请填写下线原因")

        async def handler(uow: AIOpsUnitOfWork, now: datetime) -> BusinessSystemResourceGraphDetail:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_scoped(business_system_id=business_system_id, lock=True, **_scope_args(scope))
            if entity is None:
                raise resource_not_found("Business System")
            self._check_version(entity.row_version, expected_version)
            desired = "ENABLED" if command == "enable" else "DISABLED"
            allowed_current = {"DRAFT", "DISABLED"} if desired == "ENABLED" else {"ENABLED"}
            if entity.status not in allowed_current:
                action = "上线" if desired == "ENABLED" else "下线"
                raise state_conflict(f"当前状态不允许{action}业务系统")
            detail = await self._system_detail(uow=uow, scope=scope, system=entity)
            if desired == "ENABLED":
                blockers = [item.message for item in detail.validation_issues if item.severity == "BLOCKING"]
                if blockers:
                    raise validation_failed("；".join(blockers))
            entity.status = desired
            entity.updated_by = scope.actor_id
            entity.updated_at = now
            await uow.flush()
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="BUSINESS_SYSTEM", aggregate_id=entity.business_system_id, event_type=f"BUSINESS_SYSTEM_{desired}", row_version=int(entity.row_version), details={"cascade": False, "reason": reason})
            return await self._system_detail(uow=uow, scope=scope, system=entity)

        return await self._idempotent(
            scope=scope,
            operation=f"BUSINESS_SYSTEM_{command.upper()}",
            parent_resource=str(business_system_id),
            idempotency_key=idempotency_key,
            payload={"row_version": expected_version, "reason": reason},
            response_type=BusinessSystemResourceGraphDetail,
            handler=handler,
        )

    async def delete_business_system_draft(self, *, scope: ConfigurationScope, business_system_id: UUID, expected_version: int) -> None:
        async with self._uow_factory() as uow:
            assert uow.business_systems is not None
            entity = await uow.business_systems.get_scoped(
                business_system_id=business_system_id,
                lock=True,
                include_deleted=True,
                **_scope_args(scope),
            )
            if entity is None or entity.status == "DELETED":
                raise resource_not_found("Business System")
            self._check_version(entity.row_version, expected_version)
            if entity.status != "DRAFT":
                raise state_conflict("只有草稿业务系统可以删除")
            entity.status = "DELETED"
            entity.updated_by = scope.actor_id
            entity.updated_at = self._now()
            await uow.flush()
            await add_configuration_event(
                uow=uow,
                scope=scope,
                aggregate_type="BUSINESS_SYSTEM",
                aggregate_id=entity.business_system_id,
                event_type="BUSINESS_SYSTEM_DRAFT_DELETED",
                row_version=int(entity.row_version),
                details={"cascade": False},
            )
            await uow.commit()
