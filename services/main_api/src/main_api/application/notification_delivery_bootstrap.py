"""外部投递组件装配；装配不发送消息，也不检查自身 Schema。"""

from platform_clients.aiops import AIOpsClientAuth, AIOpsManagementClient
from platform_core.security import create_auth_context_codec, create_service_identity_codec
from platform_core.notifications.feishu import FeishuSender
from main_api.application.notification_credentials import NotificationCredentialVault
from main_api.application.notification_delivery import ExternalNotificationService
from main_api.application.notification_delivery_authority import NotificationDeliveryAuthority


def create_external_notifications(*, settings, uow_factory, runtime, client=None):
    settings.external_delivery.validate_runtime(production=settings.is_production())
    authority_client = client or AIOpsManagementClient(
        base_url=settings.aiops.base_url, timeout_seconds=min(settings.aiops.timeout_seconds, 10),
        auth=AIOpsClientAuth(caller_service=settings.api.service_name, audience=settings.aiops.audience,
            scopes=("aiops.notification.reconcile",), auth_context_codec=create_auth_context_codec(),
            service_identity_codec=create_service_identity_codec()))
    sender = FeishuSender(config=settings.external_delivery)
    authority = NotificationDeliveryAuthority(uow_factory=uow_factory, client=authority_client, caller_service=settings.api.service_name)
    vault = NotificationCredentialVault(uow_factory=uow_factory)
    return ExternalNotificationService(uow_factory=uow_factory, config=settings.external_delivery,
        vault=vault, sender=sender, authority=authority)
