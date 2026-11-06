"""渠道治理权限不得扩大为绑定、重发或机器主体权限。"""
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
import pytest
from fastapi import HTTPException
from main_api.api.notification_delivery_auth import external_context, external_entry_actor
from main_api.application.access_control import AccessDeniedError

def request(kind="PORTAL",domain="12",allowed=True):
    access=NS(require_platform=AsyncMock(side_effect=None if allowed else AccessDeniedError("denied")))
    return NS(state=NS(auth_context=NS(principal_kind=kind,domain_id=domain,asserted_user_id="B")),
        app=NS(state=NS(access_control_service=access)))

@pytest.mark.asyncio
@pytest.mark.parametrize("kind",["SERVICE","APP_API_CLIENT","API_CLIENT"])
async def test_machine_cannot_manage_or_resolve_personal_entry(kind):
    req=request(kind=kind)
    with pytest.raises(HTTPException): await external_context(req,permission="channel")
    with pytest.raises(HTTPException): await external_entry_actor(req)
    req.app.state.access_control_service.require_platform.assert_not_awaited()

@pytest.mark.asyncio
async def test_platform_configuration_needs_no_domain():
    req=request(domain=None)
    scope, actor = await external_context(req,permission="binding")
    assert scope.domain_id is None and scope.tenant_id is None and actor == "B"
    req.app.state.access_control_service.require_platform.assert_awaited_once()

@pytest.mark.asyncio
@pytest.mark.parametrize("kind,code",[("channel","channel_manage"),("binding","binding_manage"),
    ("read","delivery_read"),("retry","delivery_retry")])
async def test_governance_capabilities_are_independent(kind,code):
    req=request(allowed=False)
    with pytest.raises(HTTPException): await external_context(req,permission=kind)
    assert req.app.state.access_control_service.require_platform.await_args.kwargs["permission_code"]=="platform:notification_"+code
