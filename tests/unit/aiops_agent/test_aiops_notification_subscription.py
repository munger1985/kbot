"""AIOps 站内主动分享订阅与路由测试。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.configuration.service import AIOpsConfigurationService
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.repositories.notification import (
    NotificationSubscriptionRepository,
)
from aiops_agent.repositories.platform_notification import (
    PlatformNotificationRepository,
)
from platform_core.contracts.aiops import NotificationSubscriptionUpsert
from platform_core.identity import uuid7


class _UowContext:
    def __init__(self, uow) -> None:
        self.uow = uow

    async def __aenter__(self):
        return self.uow

    async def __aexit__(self, *_):
        return None


class NotificationSubscriptionConfigurationTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.target_id = uuid7()
        self.scope = ConfigurationScope(
            domain_id=100,
            principal_id="PORTAL:km_portal",
            actor_id="portal-user-1",
            request_id="request-1",
            trace_id="trace-1",
        )

    def _service(self, *, target_status: str = "ENABLED"):
        subscriptions = SimpleNamespace(
            get_for_actor=AsyncMock(return_value=None),
            add=AsyncMock(),
            list_for_actor=AsyncMock(return_value=[]),
        )
        uow = SimpleNamespace(
            targets=SimpleNamespace(
                get_scoped=AsyncMock(
                    return_value=SimpleNamespace(status=target_status)
                )
            ),
            notification_subscriptions=subscriptions,
            commit=AsyncMock(),
        )
        service = object.__new__(AIOpsConfigurationService)
        service._uow_factory = lambda: _UowContext(uow)
        return service, uow

    def test_subscription_stage_must_be_nonempty_and_unique(self) -> None:
        with self.assertRaises(ValidationError):
            NotificationSubscriptionUpsert(stages=())
        with self.assertRaises(ValidationError):
            NotificationSubscriptionUpsert(
                stages=("REPORT_READY", "REPORT_READY")
            )

    async def test_create_subscription_is_owned_by_trusted_actor(self) -> None:
        service, uow = self._service()
        result = await service.upsert_notification_subscription(
            scope=self.scope,
            target_id=self.target_id,
            request=NotificationSubscriptionUpsert(
                minimum_severity="CRITICAL",
                stages=("SITUATION_DETECTED", "REPORT_READY"),
            ),
            expected_version=None,
        )

        row = uow.notification_subscriptions.add.await_args.args[0]
        self.assertEqual(self.scope.actor_id, row.recipient_actor_id)
        self.assertEqual("IN_APP", row.channel)
        self.assertEqual("CRITICAL", result.minimum_severity)
        self.assertEqual(
            ("SITUATION_DETECTED", "REPORT_READY"), result.stages
        )
        uow.commit.assert_awaited_once()

    async def test_disabled_target_cannot_receive_new_subscription(self) -> None:
        service, _ = self._service(target_status="DISABLED")
        with self.assertRaises(AIOpsApplicationError) as caught:
            await service.upsert_notification_subscription(
                scope=self.scope,
                target_id=self.target_id,
                request=NotificationSubscriptionUpsert(),
                expected_version=None,
            )
        self.assertEqual(404, caught.exception.status_code)


class NotificationSubscriptionRoutingTest(unittest.IsolatedAsyncioTestCase):
    async def test_repository_filters_stage_and_minimum_severity(self) -> None:
        rows = [
            SimpleNamespace(
                recipient_actor_id="critical-only",
                minimum_severity="CRITICAL",
                stages_json=["SITUATION_DETECTED"],
            ),
            SimpleNamespace(
                recipient_actor_id="high-user",
                minimum_severity="HIGH",
                stages_json=["SITUATION_DETECTED"],
            ),
            SimpleNamespace(
                recipient_actor_id="report-only",
                minimum_severity="INFO",
                stages_json=["REPORT_READY"],
            ),
        ]
        session = SimpleNamespace(scalars=AsyncMock(return_value=rows))
        repository = NotificationSubscriptionRepository(session)

        recipients = await repository.recipient_actor_ids(
            domain_id=100,
            target_id=uuid7(),
            stage="SITUATION_DETECTED",
            severity="HIGH",
        )

        self.assertEqual(("high-user",), recipients)

    async def test_situation_notification_uses_explicit_subscribers(self) -> None:
        target_id = uuid7()
        situation_id = uuid7()
        subscriptions = SimpleNamespace(
            recipient_actor_ids=AsyncMock(
                return_value=("portal-user-1", "portal-user-2")
            )
        )
        uow = SimpleNamespace(notification_subscriptions=subscriptions)
        repository = PlatformNotificationRepository(uow)
        target = SimpleNamespace(
            domain_id=100,
            target_id=target_id,
            display_name="oracle-prod-01",
        )
        situation = SimpleNamespace(
            situation_id=situation_id,
            severity="CRITICAL",
        )

        with patch(
            "aiops_agent.repositories.platform_notification.publish_notification",
            new=AsyncMock(),
        ) as publish:
            await repository.emit_situation_event(
                target=target,
                situation=situation,
                event_type="aiops.situation.detected",
                stage="SITUATION_DETECTED",
                summary="oracle-prod-01 数据库不可用",
                trace_id="trace-1",
            )

        envelope = publish.await_args.kwargs["envelope"]
        self.assertEqual(
            ["portal-user-1", "portal-user-2"],
            envelope.recipient_actor_ids,
        )
        self.assertEqual(str(target_id), envelope.safe_data["target_id"])
        self.assertNotIn("raw_log", envelope.safe_data)

    async def test_system_report_is_delivered_only_to_subscribers(self) -> None:
        run_id = uuid7()
        report_id = uuid7()
        target_id = uuid7()
        situation_id = uuid7()
        uow = SimpleNamespace(
            notification_subscriptions=SimpleNamespace(
                recipient_actor_ids=AsyncMock(
                    return_value=("portal-user-1",)
                )
            ),
            situations=SimpleNamespace(
                get_situation=AsyncMock(
                    return_value=SimpleNamespace(severity="CRITICAL")
                )
            ),
        )
        repository = PlatformNotificationRepository(uow)
        run = SimpleNamespace(
            ops_run_id=run_id,
            domain_id=100,
            target_id=target_id,
            situation_id=situation_id,
            actor_id="system:signal-intake",
            trace_id="trace-2",
        )
        report = SimpleNamespace(
            report_id=report_id,
            summary="根因诊断报告已生成",
        )

        with patch(
            "aiops_agent.repositories.platform_notification.publish_notification",
            new=AsyncMock(),
        ) as publish:
            await repository.emit_report_ready(
                run=run,
                report=report,
                actor_id=run.actor_id,
            )

        envelope = publish.await_args.kwargs["envelope"]
        self.assertEqual(["portal-user-1"], envelope.recipient_actor_ids)
        self.assertNotIn("system:signal-intake", envelope.recipient_actor_ids)

    async def test_system_report_without_subscriber_creates_no_notification(
        self,
    ) -> None:
        uow = SimpleNamespace(
            notification_subscriptions=SimpleNamespace(
                recipient_actor_ids=AsyncMock(return_value=())
            ),
            situations=SimpleNamespace(
                get_situation=AsyncMock(
                    return_value=SimpleNamespace(severity="CRITICAL")
                )
            ),
        )
        repository = PlatformNotificationRepository(uow)
        run = SimpleNamespace(
            ops_run_id=uuid7(),
            domain_id=100,
            target_id=uuid7(),
            situation_id=uuid7(),
            actor_id="system:signal-intake",
            trace_id="trace-3",
        )
        report = SimpleNamespace(report_id=uuid7(), summary="诊断完成")

        with patch(
            "aiops_agent.repositories.platform_notification.publish_notification",
            new=AsyncMock(),
        ) as publish:
            result = await repository.emit_report_ready(
                run=run,
                report=report,
                actor_id=run.actor_id,
            )

        self.assertIsNone(result)
        publish.assert_not_awaited()

    async def test_large_subscription_set_is_split_by_envelope_limit(self) -> None:
        recipients = tuple(f"portal-user-{index:03d}" for index in range(51))
        uow = SimpleNamespace(
            notification_subscriptions=SimpleNamespace(
                recipient_actor_ids=AsyncMock(return_value=recipients)
            )
        )
        repository = PlatformNotificationRepository(uow)
        target = SimpleNamespace(
            domain_id=100,
            target_id=uuid7(),
            display_name="oracle-prod-01",
        )
        situation = SimpleNamespace(
            situation_id=uuid7(),
            severity="CRITICAL",
        )

        with patch(
            "aiops_agent.repositories.platform_notification.publish_notification",
            new=AsyncMock(),
        ) as publish:
            await repository.emit_situation_event(
                target=target,
                situation=situation,
                event_type="aiops.situation.detected",
                stage="SITUATION_DETECTED",
                summary="oracle-prod-01 数据库不可用",
                trace_id="trace-4",
            )

        self.assertEqual(2, publish.await_count)
        first = publish.await_args_list[0].kwargs
        second = publish.await_args_list[1].kwargs
        self.assertEqual(50, len(first["envelope"].recipient_actor_ids))
        self.assertEqual(1, len(second["envelope"].recipient_actor_ids))
        self.assertTrue(first["event_key"].endswith(":part:1"))
        self.assertTrue(second["event_key"].endswith(":part:2"))


if __name__ == "__main__":
    unittest.main()
