"""AIOps 步骤 5 监控契约与确定性编排测试。"""

import hashlib
import hmac
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from aiops_agent.adapters.monitoring import MonitorProviderRegistry
from aiops_agent.adapters.monitoring.base import MonitorAdapterError
from aiops_agent.adapters.monitoring.payload_store import (
    LocalMonitorPayloadStore,
)
from aiops_agent.adapters.secret_store import ConfiguredSecretStore
from aiops_agent.contracts.monitoring import (
    MetricPoint,
    MetricSeries,
)
from aiops_agent.adapters.monitoring.catalog import load_metric_catalog
from aiops_agent.domain.monitoring import summarize_points
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_monitor_observe_blueprint,
)
from aiops_agent.ports.monitor import (
    MonitorProviderContext,
    RawWebhookRequest,
)
from aiops_agent.workers import (
    AIOpsDomainOutboxSink,
    TaskExecutionContext,
)
from aiops_agent.workers.monitoring_handlers import MonitorReportHandler
from platform_core.security import create_public_auth_middleware


class MetricCatalogTest(unittest.TestCase):
    def test_baseline_catalog_is_versioned_and_complete(self) -> None:
        catalog = load_metric_catalog()
        self.assertEqual("baseline.v1", catalog.version)
        self.assertEqual(64, len(catalog.manifest_hash))
        selected = catalog.select(
            (
                "db.availability",
                "db.cpu.utilization",
                "db.connection.active",
                "db.connection.utilization",
                "db.transaction.throughput",
                "db.response.latency",
                "db.storage.utilization",
                "db.error.rate",
            ),
            db_type="ORACLE",
        )
        self.assertEqual(8, len(selected))

    def test_invalid_sample_is_not_converted_to_zero(self) -> None:
        now = datetime.now(UTC)
        series = (
            MetricSeries(
                points=(
                    MetricPoint(observed_at=now, value=None, quality="INVALID"),
                    MetricPoint(observed_at=now, value=4.0, quality="GOOD"),
                )
            ),
        )
        summary = summarize_points(series)
        self.assertEqual(1, summary["count"])
        self.assertEqual(4.0, summary["last"])
        self.assertNotEqual(0, summary["avg"])


class MonitorBlueprintTest(unittest.TestCase):
    def test_binding_order_does_not_change_blueprint(self) -> None:
        first = build_monitor_observe_blueprint(("b", "a"))
        second = build_monitor_observe_blueprint(("a", "b"))
        self.assertEqual(first, second)
        BlueprintRegistry.validate(first, max_tasks=8)
        self.assertEqual(
            ["scope", "observe:a", "observe:b", "report"],
            [item.task_key for item in first.tasks],
        )

    def test_no_monitor_source_still_produces_partial_report_task(self) -> None:
        blueprint = build_monitor_observe_blueprint(())
        BlueprintRegistry.validate(blueprint, max_tasks=8)
        self.assertEqual(
            ["scope", "report"],
            [item.task_key for item in blueprint.tasks],
        )


class PrometheusWebhookTest(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, secret: str):
        return MonitorProviderRegistry(
            session=Mock(),
            webhook_replay_seconds=300,
        ).create(
            MonitorProviderContext(
                source_id="source-1",
                source_type="PROMETHEUS",
                source_version=1,
                endpoint="https://prometheus.example.com",
                credentials={"webhook_secret": secret},
                capabilities={"external_target_label": "instance"},
            )
        )

    @staticmethod
    def _request(body: bytes, secret: str, now: datetime):
        timestamp = str(int(now.timestamp()))
        signature = hmac.new(
            secret.encode(),
            timestamp.encode() + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        return RawWebhookRequest(
            headers={
                "x-kbot-timestamp": timestamp,
                "x-kbot-signature": f"sha256={signature}",
            },
            body=body,
            received_at=now,
        )

    async def test_verified_alert_is_normalized_without_target_id(self) -> None:
        now = datetime.now(UTC).replace(microsecond=0)
        payload = {
            "status": "firing",
            "alerts": [
                {
                    "status": "firing",
                    "labels": {
                        "instance": "db-prod-1",
                        "alertname": "DatabaseDown",
                        "severity": "critical",
                        "target_id": "untrusted",
                    },
                    "annotations": {"summary": "数据库探针不可用"},
                    "startsAt": now.isoformat(),
                    "fingerprint": "provider-fingerprint",
                }
            ],
        }
        body = json.dumps(payload).encode()
        batch = await self._adapter("secret").verify_and_parse_webhook(
            self._request(body, "secret", now)
        )
        self.assertEqual(1, len(batch.events))
        event = batch.events[0]
        self.assertEqual("db-prod-1", event.external_target_key)
        self.assertEqual("CRITICAL", event.severity)
        self.assertNotIn("target_id", event.provider_attributes)

    async def test_invalid_signature_is_rejected_before_parsing(self) -> None:
        now = datetime.now(UTC).replace(microsecond=0)
        with self.assertRaises(MonitorAdapterError) as caught:
            await self._adapter("correct").verify_and_parse_webhook(
                self._request(b"{}", "wrong", now)
            )
        self.assertEqual("MONITOR_AUTH_FAILED", caught.exception.code)


class MonitoringHandlerTest(unittest.IsolatedAsyncioTestCase):
    async def test_partial_report_only_states_observed_facts(self) -> None:
        now = datetime.now(UTC)
        context = TaskExecutionContext(
            run_id="run-1",
            task_id="task-1",
            task_key="report",
            target_id="target-1",
            agent_id="agent-1",
            trigger_type="ALERT",
            trace_id="trace-1",
            attempt=1,
            deadline_at=None,
            plan_snapshot={
                "monitoring": {
                    "window": {
                        "start": (now - timedelta(hours=1)).isoformat(),
                        "end": now.isoformat(),
                    },
                    "catalog_version": "baseline.v1",
                    "catalog_hash": "a" * 64,
                    "bindings": [],
                    "initial_gaps": [
                        {
                            "binding_id": "binding-1",
                            "source_id": "source-1",
                            "code": "MONITOR_SOURCE_INACTIVE",
                            "detail": "监控源未激活",
                        }
                    ],
                }
            },
            policy_snapshot={},
            input_artifacts=(),
        )
        result = await MonitorReportHandler().execute(context)
        self.assertEqual("PARTIAL", result.status)
        self.assertEqual("INCONCLUSIVE", result.root_cause_level)
        serialized = result.model_dump_json()
        self.assertNotIn("root cause", serialized.lower())
        self.assertNotIn("sql", serialized.lower())


class SecretAndOutboxTest(unittest.IsolatedAsyncioTestCase):
    async def test_environment_secret_is_resolved_without_repr_value(self) -> None:
        store = ConfiguredSecretStore(
            provider="environment", allowed_schemes=("env",)
        )
        with patch.dict(
            os.environ,
            {"AIOPS_TEST_MONITOR": '{"token":"top-secret"}'},
        ):
            secret = await store.resolve("env://AIOPS_TEST_MONITOR")
        self.assertEqual("top-secret", secret.values["token"])
        self.assertNotIn("top-secret", repr(secret))

    async def test_alert_outbox_builds_typed_observe_run(self) -> None:
        runtime = Mock()
        runtime.create_run = AsyncMock()
        fallback = Mock()
        fallback.publish = AsyncMock()
        sink = AIOpsDomainOutboxSink(
            runtime_service=runtime, fallback=fallback
        )
        await sink.publish(
            "OPS_ALERT_AUTO_RUN_REQUESTED",
            {
                "app_id": 100,
                "domain_id": 2,
                "agent_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11",
                "target_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a12",
                "alert_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a13",
                "event_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a14",
                "trace_id": "trace-1",
            },
        )
        command = runtime.create_run.await_args.args[0]
        self.assertEqual("monitor.observe-report", command.blueprint_id)
        self.assertEqual("ALERT", command.trigger_type)
        fallback.publish.assert_not_awaited()

    async def test_verified_payload_store_is_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = LocalMonitorPayloadStore(Path(directory))
            body = b'{"status":"firing"}'
            digest = hashlib.sha256(body).hexdigest()
            first = await store.store_verified(
                source_id="019c03b5-4b88-7ab2-8c19-7b6ea34f2a11",
                body=body,
                content_hash=digest,
            )
            second = await store.store_verified(
                source_id="019c03b5-4b88-7ab2-8c19-7b6ea34f2a11",
                body=body,
                content_hash=digest,
            )
            self.assertEqual(first.uri, second.uri)
            self.assertTrue(Path(first.uri.removeprefix("file://")).exists())


class IntegrationAuthBoundaryTest(unittest.TestCase):
    def test_monitoring_integration_bypasses_only_portal_auth(self) -> None:
        app = FastAPI()
        verifier = Mock()
        validator = AsyncMock(return_value=True)
        app.middleware("http")(
            create_public_auth_middleware(
                verifier=verifier,
                domain_validator=validator,
                public_prefixes={
                    "/api/v1/integrations/monitoring/"
                },
            )
        )

        @app.post("/api/v1/integrations/monitoring/{key}/events")
        async def intake(key: str):
            return {"accepted": True}

        response = TestClient(app).post(
            "/api/v1/integrations/monitoring/secret/events"
        )
        self.assertEqual(200, response.status_code)
        verifier.verify_authorization.assert_not_called()


if __name__ == "__main__":
    unittest.main()
