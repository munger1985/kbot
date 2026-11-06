"""外发和点击前重新核对平台授权与业务资源；不使用历史用户 JWT。"""

from uuid import UUID

from platform_core.contracts.aiops.notification_projection import NotificationResourceRequest, NotificationResourceSnapshot
from platform_core.security import create_service_auth_context


class NotificationDeliveryAuthority:
    async def domain_active(self, *, scope):
        if scope.tenant_id is not None:
            return False
        async with self._uow_factory() as uow:
            return await uow.domains.exists_active(domain_id=int(scope.domain_id))

    async def delivery_available(self, *, scope):
        if not await self.domain_active(scope=scope):
            return False
        async with self._uow_factory() as uow:
            app = await uow.access.get_application("aiops")
            return app is not None and app.status == "ACTIVE" and any(
                int(row.domain_id) == int(scope.domain_id) and row.status == "ACTIVE"
                for row in await uow.access.list_app_domains(app_id="aiops"))

    def __init__(self, *, uow_factory, client, caller_service):
        self._uow_factory, self.client, self.caller = uow_factory, client, caller_service

    async def actor_active(self, *, scope, actor_id):
        if scope.domain_id is None:
            if scope.tenant_id is not None:
                return False
            async with self._uow_factory() as uow:
                user = await uow.access.get_user(actor_id)
                return user is not None and user.status == "ACTIVE"
        return "aiops:use" in await self.permissions(scope=scope, actor_id=actor_id)

    async def check(self, *, scope, actor_id, entry, actionable):
        try:
            if not await self.delivery_available(scope=scope):
                return "NOTIFICATION_DOMAIN_UNAVAILABLE"
            request = NotificationResourceRequest(event_type=entry["event_type"],
                domain_id=scope.domain_id, tenant_id=scope.tenant_id,
                route_key=entry["route_key"], route_params=entry["route_params"])
            if actor_id is not None:
                permissions = await self.permissions(scope=scope, actor_id=actor_id)
                required = {"aiops:use"}
                if entry["event_type"] == "aiops.proposal.review_required":
                    required.add("aiops:proposal:approve")
                if not required <= permissions:
                    return "NOTIFICATION_ACCESS_REVOKED"
            elif entry["audience"] != "GROUP":
                return "NOTIFICATION_ACTOR_UNAVAILABLE"
            payload = await self.client.resolve_notification_resource(payload=request.model_dump(mode="json"),
                auth_context=create_service_auth_context(caller_service=self.caller,
                    trace_id="notification-resource-check"))
            snapshot = NotificationResourceSnapshot.model_validate(payload)
            if snapshot.request != request:
                return "NOTIFICATION_RESOURCE_SCOPE_INVALID"
            if actionable and not snapshot.actionable:
                return "NOTIFICATION_TASK_CLOSED"
            if entry["event_type"] == "aiops.proposal.review_required":
                if actor_id in {snapshot.initiator_actor_id, snapshot.run_actor_id}:
                    return "NOTIFICATION_APPLICANT_NOT_APPROVER"
                if snapshot.assignee_actor_id is not None and actor_id != snapshot.assignee_actor_id:
                    return "NOTIFICATION_NOT_ASSIGNEE"
            if entry["event_type"] == "aiops.diagnostic.input_required":
                if actor_id != snapshot.assignee_actor_id:
                    return "NOTIFICATION_NOT_ASSIGNEE"
            return None
        except Exception:
            return "NOTIFICATION_AUTHORITY_UNAVAILABLE"

    async def permissions(self, *, scope, actor_id):
        from platform_core.authorization import BUILT_IN_APP_ROLE_CODES, built_in_role_permissions
        if scope.tenant_id is not None:
            return set()
        async with self._uow_factory() as uow:
            user = await uow.access.get_user(actor_id)
            if user is None or user.status != "ACTIVE" or not await uow.domains.exists_active(domain_id=int(scope.domain_id)):
                return set()
            result = set(await uow.access.permissions_for(app_id="aiops", domain_id=int(scope.domain_id), user_id=actor_id))
            for role in await uow.access.list_roles(app_id="aiops", domain_id=int(scope.domain_id), user_id=actor_id):
                if role in BUILT_IN_APP_ROLE_CODES:
                    result.update(built_in_role_permissions("aiops", role))
            return result
