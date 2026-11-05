"""删除系统时保留发布历史，按全局引用退役独占资源。"""

from uuid import UUID

from aiops_agent.application.configuration.common import ConfigurationScope, add_configuration_event
from aiops_agent.application.errors import resource_not_found, state_conflict


def deletion_scope(scope: ConfigurationScope) -> dict:
    values = {"domain_id": scope.domain_id}
    if hasattr(scope, "tenant_id"):
        values["tenant_id"] = scope.tenant_id
    return values


class BusinessSystemDeletionMixin:
    async def delete_business_system(
        self, *, scope: ConfigurationScope, business_system_id: UUID, expected_version: int,
    ) -> None:
        """删除当前系统配置并收缩运行权限，绝不级联删除资源历史。"""
        async with self._uow_factory() as uow:
            repo = uow.business_systems
            assert repo is not None and uow.targets is not None
            system = await repo.get_scoped(
                business_system_id=business_system_id, lock=True, **deletion_scope(scope),
            )
            if system is None:
                raise resource_not_found("Business System")
            self._check_version(system.row_version, expected_version)
            if system.status not in {"DRAFT", "DISABLED"}:
                raise state_conflict("业务系统必须先下线才能删除")
            # DRAFT 没有回退入口；若存在发布、冻结授权或巡检引用，仍禁止物理删除。
            published = system.status != "DRAFT" or await repo.system_has_publication_history(business_system_id)
            now = self._now()
            plans = await repo.list_deletion_plans(business_system_id)
            if any(plan.lease_until is not None and plan.lease_until > now for plan in plans):
                raise state_conflict("巡检计划正在调度，请等待完成后再删除")
            rows = await repo.list_system_resource_rows(business_system_id)
            resources = {}
            # 锁定共享资源后重新判断引用，连续删除两个共享系统时只会退役一次。
            for resource_id in sorted({row.resource_id for _, row, _, _ in rows}, key=str):
                resource = await repo.get_resource_scoped(
                    resource_id=resource_id, lock=True, **deletion_scope(scope),
                )
                if resource is None:
                    raise resource_not_found("Managed Resource")
                resources[resource_id] = resource
            target_ids = {database.target_id for _, _, database, _ in rows if database is not None}
            # 与上下线使用相同的稳定 Target 锁顺序；锁内重新读取，避免共享端点误停用。
            targets = {}
            for target_id in sorted(target_ids, key=str):
                target = await uow.targets.get_scoped(
                    target_id=target_id, domain_id=scope.domain_id, lock=True, refresh=True,
                )
                if target is None or (hasattr(scope, "tenant_id") and target.tenant_id != scope.tenant_id):
                    raise resource_not_found("Target")
                targets[target_id] = target
            if await repo.system_has_unfinished_work(system_id=business_system_id, target_ids=target_ids):
                raise state_conflict("业务系统仍有未结束的运行或巡检，请先完成或取消后再删除")
            drafts = await repo.list_uncommitted_system_drafts(business_system_id)
            for draft in drafts:
                if await repo.draft_has_pending_discovery(draft.draft_id):
                    raise state_conflict("草稿仍有资源发现任务，请等待完成后再删除")
            retired, shared = [], []
            for resource_id, resource in resources.items():
                if await repo.resource_has_other_system(resource_id=resource_id, system=system):
                    shared.append(str(resource_id))
                    continue
                resource.lifecycle_status = "RETIRED"
                resource.updated_by, resource.updated_at = scope.actor_id, now
                retired.append(str(resource_id))
                for binding in await repo.list_resource_source_bindings(resource_id, active_only=True):
                    binding.status = "DISABLED"
                    binding.updated_by, binding.updated_at = scope.actor_id, now
            # 只撤销本系统关联；共享资源在其他系统中的关联完全不变。
            for link, _, _, _ in rows:
                link.status = "INACTIVE"
                link.updated_by, link.updated_at = scope.actor_id, now
            for plan in plans:
                plan.status = "DISABLED"
                plan.next_run_at = None
                plan.lease_owner = plan.lease_token = plan.lease_until = None
                plan.updated_by, plan.updated_at = scope.actor_id, now
            await uow.flush()
            disabled_targets = []
            for target_id, target in targets.items():
                if await repo.target_has_unretired_resource(target_id):
                    continue
                if await uow.targets.has_active_runs(target_id=target_id):
                    raise state_conflict("独占数据库仍有未结束的运行，请先完成或取消后再删除")
                target.status = "DISABLED"
                target.readonly_connection_enabled = False
                target.controlled_change_enabled = False
                target.workload_next_run_at = target.activity_next_sample_at = None
                target.activity_sampler_status = "DISABLED"
                target.updated_by, target.updated_at = scope.actor_id, now
                for credential_id, kind in (
                    (target.diagnostic_credential_id, "target_diagnostic"),
                    (target.execution_credential_id, "target_execution"),
                ):
                    if credential_id is not None and not await repo.credential_has_other_owner(
                        credential_id=credential_id, target_id=target_id,
                    ):
                        await self._managed_credentials.revoke(
                            uow=uow, credential_id=credential_id, credential_kind=kind,
                            actor_id=scope.actor_id, **deletion_scope(scope),
                        )
                disabled_targets.append(str(target_id))
            for draft in drafts:
                await repo.delete_draft_workspace(draft.draft_id)
            if published:
                system.status = "DELETED"
                system.updated_by, system.updated_at = scope.actor_id, now
                await uow.flush()
            await add_configuration_event(
                uow=uow, scope=scope, aggregate_type="BUSINESS_SYSTEM",
                aggregate_id=business_system_id,
                event_type="BUSINESS_SYSTEM_DELETED" if published else "BUSINESS_SYSTEM_DRAFT_DELETED",
                row_version=int(system.row_version),
                details={
                    "deletion_mode": "LOGICAL" if published else "PHYSICAL",
                    "system_code": system.system_code, "display_name": system.display_name,
                    "retired_resource_ids": retired, "preserved_shared_resource_ids": shared,
                    "disabled_target_ids": disabled_targets,
                    "disabled_plan_ids": [str(plan.inspection_plan_id) for plan in plans],
                    "deleted_draft_ids": [str(draft.draft_id) for draft in drafts],
                    "history_preserved": True,
                },
            )
            if not published:
                await repo.delete_unpublished_system(business_system_id)
            await uow.commit()
