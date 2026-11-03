"""两端模板删除的公开权限与转发合同，不访问真实服务。"""

import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from platform_core.identity import uuid7

IS_KBOT = Path(__file__).resolve().parents[2].name == "kbot4"


def test_public_template_delete_permission_and_revision_forwarding():
    if IS_KBOT:
        from main_api.api import aiops_app as api
    else:
        from main_api.api import ops as api

    async def case():
        route = next(route for route in api.router.routes if route.endpoint.__name__ == "delete_inspection_template")
        assert route.methods == {"DELETE"}
        context = object()
        request = SimpleNamespace(state=SimpleNamespace(auth_context=context))
        template_id = uuid7()
        body = api.InspectionTemplateRevisionPayload(expected_row_version=3)
        client = SimpleNamespace(inspection_template_request=AsyncMock(return_value={"deleted":True}))
        with patch.object(api, "_client", return_value=client):
            if IS_KBOT:
                with patch.object(api, "_require", new_callable=AsyncMock) as require:
                    result = await api.delete_inspection_template(template_id, request, body)
                    require.assert_awaited_once_with(request, "aiops:plan_manage")
                client.inspection_template_request.assert_awaited_once_with(
                    "DELETE", f"/{template_id}", payload={"expected_row_version":3}, auth_context=context,
                )
                client.inspection_template_request.reset_mock()
                with patch.object(api, "_require", new_callable=AsyncMock, side_effect=HTTPException(403)):
                    with pytest.raises(HTTPException):
                        await api.delete_inspection_template(template_id, request, body)
                client.inspection_template_request.assert_not_awaited()
            else:
                from main_api.iam import dependencies
                from platform_core.authorization import get_app_authorization_policy
                assert get_app_authorization_policy("aiops").route_permission(
                    path=route.path, method="DELETE",
                ) == "aiops:plan_manage"
                guard = next(item.call for item in route.dependant.dependencies
                             if inspect.getclosurevars(item.call).nonlocals.get("permission") == "aiops:plan_manage")
                with patch.object(dependencies, "authorize_app_request", new_callable=AsyncMock, side_effect=HTTPException(403)) as authorize:
                    with pytest.raises(HTTPException):
                        await guard(request, object())
                    assert authorize.await_args.kwargs["permission"] == "aiops:plan_manage"
                client.inspection_template_request.assert_not_awaited()
                result = await api.delete_inspection_template(template_id, request, body)
                client.inspection_template_request.assert_awaited_once_with(
                    "DELETE", f"/{template_id}", payload={"expected_row_version":3}, auth_context=context,
                )
            assert result == {"deleted":True}
    asyncio.run(case())
