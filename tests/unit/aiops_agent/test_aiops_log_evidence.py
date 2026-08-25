"""Loki 日志证据、受控定位和 Evidence Index 测试。"""

import json
import unittest
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from aiops_agent.adapters.diagnostic_sources.loki import LokiAdapter
from aiops_agent.application.configuration import AIOpsConfigurationService
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.domain.diagnosis.evidence import normalize_evidence_artifacts
from aiops_agent.orchestration import (
    BlueprintRegistry,
    build_diagnosis_blueprint,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_LOG_QUERY,
    DiagnosticSourceContext,
    LogEvidenceRequest,
    LogSourceLocator,
)


class _LokiResponse:
    def __init__(self, payload: dict):
        self.status = 200
        self._raw = json.dumps(payload).encode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def read(self) -> bytes:
        return self._raw


class _LokiSession:
    def __init__(self, payload: dict):
        self.response = _LokiResponse(payload)
        self.params = {}

    def get(self, _url, **kwargs):
        self.params = kwargs.get("params", {})
        return self.response


class LokiEvidenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_exact_selector_query_redacts_credentials(self) -> None:
        now = datetime.now(UTC).replace(microsecond=0)
        session = _LokiSession(
            {
                "status": "success",
                "data": {
                    "resultType": "streams",
                    "result": [
                        {
                            "stream": {
                                "job": "oracle-alert",
                                "instance": "oracle-dev-01",
                            },
                            "values": [
                                [
                                    str(int(now.timestamp() * 1_000_000_000)),
                                    "ORA-01034 password=plain-secret",
                                ]
                            ],
                        }
                    ],
                },
            }
        )
        adapter = LokiAdapter(
            context=DiagnosticSourceContext(
                source_id="source-1",
                source_type="LOKI",
                adapter_id="loki",
                adapter_version="1.0.0",
                config_version=1,
                endpoint="http://loki.example.com",
                declared_capabilities={CAPABILITY_LOG_QUERY: {}},
            ),
            session=session,  # type: ignore[arg-type]
            request_timeout_seconds=10,
            webhook_replay_seconds=300,
        )

        result = await adapter.query_logs(
            LogEvidenceRequest(
                target_id="target-1",
                binding_id="binding-1",
                source_locator_key="oracle-dev-01",
                selector_labels={
                    "job": "oracle-alert",
                    "instance": 'oracle-dev-01"}',
                },
                window_start=now - timedelta(minutes=5),
                window_end=now,
                max_entries=100,
                max_response_bytes=10240,
                trace_id="trace-1",
            )
        )

        self.assertEqual(1, len(result.entries))
        self.assertIn("password=[已脱敏]", result.entries[0].line)
        self.assertNotIn("plain-secret", result.model_dump_json())
        self.assertIn(r'instance="oracle-dev-01\"}"', session.params["query"])

    async def test_invalid_label_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LogSourceLocator.model_validate(
                {"labels": {"job|=": "oracle-alert"}}
            )

    async def test_log_binding_rejects_uncontrolled_locator(self) -> None:
        service = object.__new__(AIOpsConfigurationService)
        source = SimpleNamespace(
            declared_capabilities_json={CAPABILITY_LOG_QUERY: {}}
        )
        with self.assertRaises(AIOpsApplicationError):
            service._validate_source_binding_locator(
                source=source,
                source_locator={"query": '{job=~".*"}'},
            )


class LogEvidenceIndexTest(unittest.TestCase):
    def test_log_artifact_becomes_source_verified_fact(self) -> None:
        now = datetime.now(UTC)
        evidence = normalize_evidence_artifacts(
            (
                {
                    "artifact_id": "artifact-log-1",
                    "schema_version": "LOG_EVIDENCE_SET.v1",
                    "payload": {
                        "source_id": "source-1",
                        "binding_id": "binding-1",
                        "window_start": (now - timedelta(minutes=5)).isoformat(),
                        "window_end": now.isoformat(),
                        "entries": [
                            {
                                "observed_at": now.isoformat(),
                                "line": "ORA-01034: ORACLE not available",
                                "labels": {"job": "oracle-alert"},
                                "structured_fields": {},
                                "entry_fingerprint": "a" * 64,
                            }
                        ],
                        "gaps": [],
                        "truncated": False,
                    },
                },
            ),
            target_id="target-1",
        )

        self.assertEqual(1, evidence.fact_count)
        self.assertEqual("LOG_ENTRY", evidence.facts[0].source_type)
        self.assertEqual("SOURCE_VERIFIED", evidence.facts[0].trust_level)

    def test_diagnosis_blueprint_includes_log_before_evidence_index(self) -> None:
        blueprint = build_diagnosis_blueprint(
            binding_ids=("metrics-1",),
            log_binding_ids=("logs-1",),
            tool_ids=(),
        )
        BlueprintRegistry.validate(blueprint, max_tasks=64)
        tasks = {item.task_key: item for item in blueprint.tasks}
        self.assertEqual(
            "LOG_EVIDENCE_SET.v1", tasks["log:logs-1"].output_schema_version
        )
        self.assertIn(
            "log:logs-1", tasks["diagnosis:evidence:r0"].depends_on
        )


if __name__ == "__main__":
    unittest.main()
