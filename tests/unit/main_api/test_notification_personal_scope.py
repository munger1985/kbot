"""个人通知跨空间聚合、撤权过滤和确切资源定位回归。"""

from types import SimpleNamespace as NS
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from main_api.application.notification_center import NotificationCenterError, NotificationCenterService


def context():
    repo = NS(inbox_domain_for_actor=AsyncMock(return_value=99))
    uow = NS(access=NS(list_active_domain_ids=AsyncMock(return_value=(12, 99, 77))),
        domains=NS(exists_active=AsyncMock(side_effect=lambda domain_id: domain_id != 77)), notifications=repo)
    manager = AsyncMock()
    manager.__aenter__.return_value = uow
    return repo, uow, manager


@pytest.mark.asyncio
async def test_personal_page_merges_authorized_domains_before_pagination(monkeypatch):
    _, _, manager = context()
    service = NotificationCenterService(uow_factory=lambda:manager)
    async def snapshot(*, app_id, domain_id, user_id):
        return NS(permissions={"aiops:use", "aiops:proposal:approve"} if domain_id == 12 else {"aiops:use"})
    monkeypatch.setattr("main_api.application.notification_center.AccessControlService.snapshot", snapshot_method(snapshot))
    async def page(**kwargs):
        domain = kwargs["domain_id"]
        if domain == 12:
            assert "aiops.proposal.review_required" in kwargs["allowed_event_types"]
            return [{"event_sequence":3,"domain_id":12},{"event_sequence":1,"domain_id":12}]
        assert "aiops.proposal.review_required" not in kwargs["allowed_event_types"]
        return [{"event_sequence":4,"domain_id":99},{"event_sequence":2,"domain_id":99}]
    service.list_notifications = AsyncMock(side_effect=page)
    result = await service.personal("list_notifications", actor_id="B", limit=2, before_sequence=None)
    assert [row["event_sequence"] for row in result] == [4,3]
    assert [row["domain_id"] for row in result] == [99,12]
    assert service.list_notifications.await_count == 2


def snapshot_method(fn):
    async def invoke(self, **kwargs):
        return await fn(**kwargs)
    return invoke


@pytest.mark.asyncio
async def test_detail_uses_record_domain_and_rejects_revoked_scope(monkeypatch):
    repo, _, manager = context()
    service = NotificationCenterService(uow_factory=lambda:manager)
    monkeypatch.setattr("main_api.application.notification_center.AccessControlService.snapshot",
        snapshot_method(AsyncMock(return_value=NS(permissions={"aiops:use"}))))
    service.get_notification = AsyncMock(return_value={"domain_id":99})
    inbox_id=uuid4()
    assert (await service.personal("get_notification", actor_id="B", inbox_id=inbox_id))["domain_id"]==99
    assert service.get_notification.await_args.kwargs["domain_id"]==99
    repo.inbox_domain_for_actor.return_value=77
    with pytest.raises(NotificationCenterError):
        await service.personal("get_notification", actor_id="B", inbox_id=inbox_id)
    assert service.get_notification.await_count==1


def test_managed_global_credentials_cannot_be_read_in_resource_domain():
    from cryptography.exceptions import InvalidTag
    from platform_core.managed_credentials.cipher import ManagedCredentialCipher
    cipher=ManagedCredentialCipher(key=b"f"*32,key_version="test")
    key=uuid4()
    kwargs={"namespace":"notifications","credential_kind":"feishu","credential_id":key}
    encrypted=cipher.encrypt({"app_secret":"fixture"},domain_id=None,**kwargs)
    assert cipher.decrypt(encrypted,domain_id=None,**kwargs)=={"app_secret":"fixture"}
    with pytest.raises(InvalidTag):
        cipher.decrypt(encrypted,domain_id=12,**kwargs)
