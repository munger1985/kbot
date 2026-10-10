"""统一实时监控 Profile、Readiness、View 与 Zabbix 合同测试。"""

import json
import unittest
from datetime import UTC, datetime, timedelta

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.adapters.diagnostic_sources.zabbix import ZabbixAdapter
from aiops_agent.domain.evidence import DEFAULT_BASELINE_METRICS
from platform_core.contracts.aiops.monitoring import (
    MonitoringGap,
    MonitoringInstanceSummary,
)
from aiops_agent.monitoring import (
    GrafanaLinkSecurity,
    MonitoringProfileDefinition,
    MonitoringQueryError,
    MonitoringViewBuilder,
    build_monitoring_cache_key,
    load_monitoring_profile_catalog,
    project_source_readiness,
    resolve_metric_definitions,
    resolve_grafana_integration,
    resolve_monitoring_window,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_QUERY,
    CAPABILITY_METRIC_QUERY_RANGE,
    DiagnosticSourceContext,
    MetricsEvidenceRequest,
    SourceHealthRequest,
)
from platform_core.identity import uuid7


class _Response:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    async def read(self):
        return json.dumps(self._payload).encode()


class _ResponseContext:
    def __init__(self, payload):
        self._response = _Response(payload)

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, traceback):
        return None


class _Session:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.requests = []

    def post(self, endpoint, *, json, timeout):
        self.requests.append((endpoint, json, timeout))
        return _ResponseContext(self.payloads.pop(0))


class MonitoringProfileContractTest(unittest.TestCase):
    def setUp(self):
        self.metrics = load_metric_catalog()
        self.profiles = load_monitoring_profile_catalog(self.metrics)

    def test_profiles_are_stable_and_provider_aligned(self):
        mysql = self.profiles.list_for_db_types(("MYSQL",))
        self.assertEqual(
            [
                "mysql-overview",
                "database-capacity",
                "host-overview",
            ],
            [item.profile_id for item in mysql],
        )
        self.assertEqual(
            "oracle-overview",
            self.profiles.list_for_db_types(("ORACLE",))[0].profile_id,
        )
        for profile in self.profiles._profiles.values():
            self.assertLessEqual(len(profile.metric_codes), 8)
            for code in profile.metric_codes:
                definition = self.metrics.get(code)
                self.assertIn("PROMETHEUS", definition.providers)
                provider = definition.providers["ZABBIX"]
                self.assertTrue(provider.exact_item_key)
                self.assertTrue(provider.value_type)
                self.assertEqual(definition.unit, provider.unit)
        self.assertEqual(
            "kbot-oracle-overview",
            self.profiles.get("oracle-overview").grafana_dashboard_uid,
        )
        self.assertIn(
            "host.memory.utilization",
            self.profiles.get("oracle-overview").metric_codes,
        )

    def test_default_database_profiles_use_authorized_exporter_metrics(self):
        expected_profiles = {
            "ORACLE": "oracle-overview",
            "MYSQL": "mysql-overview",
            "POSTGRESQL": "postgresql-overview",
        }
        authorized = set(DEFAULT_BASELINE_METRICS)
        for db_type, profile_id in expected_profiles.items():
            profile = self.profiles.list_for_db_types((db_type,))[0]
            with self.subTest(db_type=db_type):
                self.assertEqual(profile_id, profile.profile_id)
                self.assertTrue(set(profile.metric_codes).issubset(authorized))

        mysql_codes = self.profiles.get("mysql-overview").metric_codes
        self.assertTrue(all(code.startswith("mysql.") for code in mysql_codes))
        postgresql_codes = self.profiles.get(
            "postgresql-overview"
        ).metric_codes
        self.assertNotIn(
            "postgresql.replication.lag_bytes", postgresql_codes
        )
        self.assertNotIn(
            "postgresql.replication.slot_retained_bytes", postgresql_codes
        )

    def test_oracle_overview_prometheus_queries_have_controlled_fallbacks(self):
        expected_raw_metrics = {
            "db.cpu.utilization": "oracledb_kbot_cpu_utilization_percent",
            "db.connection.active": "oracledb_sessions_value",
            "db.connection.utilization": (
                "oracledb_kbot_connection_current_sessions"
            ),
            "db.transaction.throughput": "oracledb_activity_user_commits",
            "db.response.latency": (
                "oracledb_exporter_last_scrape_duration_seconds"
            ),
            "db.storage.utilization": "oracledb_tablespace_used_percent",
        }
        for code, raw_metric in expected_raw_metrics.items():
            provider = self.metrics.get(code).providers["PROMETHEUS"]
            fallbacks = provider.fallback_query_templates
            with self.subTest(metric_code=code):
                self.assertTrue(
                    any(
                        'target_key="${external_target}"' in query
                        for query in fallbacks
                    )
                )
                self.assertTrue(
                    any(raw_metric in query for query in fallbacks)
                )

    def test_grafana_requires_fixed_uid_and_complete_link_security_gate(self):
        disabled = resolve_grafana_integration(
            source_type="PROMETHEUS",
            profile_dashboard_uid="kbot-oracle-overview",
            instance_count=1,
        )
        self.assertEqual("DISABLED", disabled.integration_mode)
        self.assertEqual(
            "GRAFANA_SECURE_GATEWAY_UNAVAILABLE", disabled.reason_code
        )
        security = GrafanaLinkSecurity(
            non_admin_identity=True,
            target_locked=True,
            domain_isolated=True,
            audit_enabled=True,
            https_upstream=True,
        )
        unsafe = resolve_grafana_integration(
            source_type="PROMETHEUS",
            profile_dashboard_uid="kbot-oracle-overview",
            instance_count=1,
            launch_url="https://grafana.invalid/d/kbot-oracle-overview?var-target_key=secret",
            security=security,
        )
        self.assertEqual("DISABLED", unsafe.integration_mode)
        linked = resolve_grafana_integration(
            source_type="PROMETHEUS",
            profile_dashboard_uid="kbot-oracle-overview",
            instance_count=2,
            launch_url="/api/v1/apps/aiops/grafana/launch/session-1",
            security=security,
        )
        self.assertEqual("LINK", linked.integration_mode)
        self.assertEqual("kbot-database-fleet", linked.dashboard_uid)

    def test_oem_is_not_a_first_phase_monitoring_source(self):
        with self.assertRaises(MonitoringQueryError) as raised:
            project_source_readiness(
                source_id=str(uuid7()),
                display_name="OEM",
                source_type="OEM",
                status="ENABLED",
                connectivity_status="CONNECTED",
                capabilities={CAPABILITY_METRIC_QUERY_RANGE},
                active_binding_count=1,
                metrics_aligned=True,
                alert_ingress_ready=False,
            )
        self.assertEqual(
            "MONITORING_SOURCE_TYPE_UNSUPPORTED", raised.exception.code
        )

    def test_readiness_distinguishes_no_mapping_and_partial_diagnosis(self):
        no_mapping = project_source_readiness(
            source_id=uuid7(),
            display_name="Prometheus",
            source_type="PROMETHEUS",
            status="ENABLED",
            connectivity_status="CONNECTED",
            capabilities={CAPABILITY_METRIC_QUERY_RANGE},
            active_binding_count=0,
            metrics_aligned=True,
            alert_ingress_ready=False,
        )
        self.assertEqual("NO_MAPPED_INSTANCE", no_mapping.monitoring_readiness)
        ready = project_source_readiness(
            source_id=uuid7(),
            display_name="Zabbix",
            source_type="ZABBIX",
            status="ENABLED",
            connectivity_status="CONNECTED",
            capabilities={CAPABILITY_METRIC_QUERY_RANGE},
            active_binding_count=1,
            metrics_aligned=True,
            alert_ingress_ready=False,
        )
        self.assertEqual("READY", ready.monitoring_readiness)
        self.assertEqual("PARTIAL", ready.diagnostic_readiness)
        self.assertEqual((), ready.capability_gaps)
        self.assertEqual(
            {"SOURCE_EVENT_QUERY_MISSING", "SOURCE_ALERT_INGRESS_MISSING"},
            {item.code for item in ready.diagnostic_gaps},
        )

    def test_cache_key_contains_domain_and_configuration_versions(self):
        now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
        source_id, target_id = uuid7(), uuid7()
        common = {
            "source_id": source_id,
            "instance_ids": (target_id,),
            "profile_id": "mysql-overview",
            "window": "1h",
            "end_time": now,
            "binding_versions": (3,),
            "source_config_version": 4,
        }
        first = build_monitoring_cache_key(domain_id="domain-a", **common)
        second = build_monitoring_cache_key(domain_id="domain-b", **common)
        changed = build_monitoring_cache_key(
            domain_id="domain-a", **{**common, "binding_versions": (4,)}
        )
        compared = build_monitoring_cache_key(
            domain_id="domain-a",
            compare_source_id=uuid7(),
            **common,
        )
        self.assertNotEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertNotEqual(first, compared)

    def test_view_keeps_no_data_as_gap_instead_of_zero(self):
        now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
        source_id, target_id, binding_id = uuid7(), uuid7(), uuid7()
        source = project_source_readiness(
            source_id=source_id,
            display_name="Prometheus",
            source_type="PROMETHEUS",
            status="ENABLED",
            connectivity_status="CONNECTED",
            capabilities={
                CAPABILITY_METRIC_QUERY_RANGE,
                CAPABILITY_EVENT_QUERY,
            },
            active_binding_count=1,
            metrics_aligned=True,
            alert_ingress_ready=True,
        )
        instance = MonitoringInstanceSummary(
            instance_id=target_id,
            display_name="核心 MySQL",
            db_type="MYSQL",
            status="PARTIAL",
            monitoring_readiness="READY",
            diagnostic_readiness="READY",
        )
        definition = self.metrics.get("mysql.availability")
        observation = ZabbixAdapter(
            context=DiagnosticSourceContext(
                source_id=str(source_id),
                source_type="ZABBIX",
                adapter_id="zabbix",
                adapter_version="1.0.0",
                config_version=1,
                endpoint="https://zabbix.example.com/api_jsonrpc.php",
                credentials={"token": "test"},
            ),
            session=_Session([]),
            request_timeout_seconds=5,
            webhook_replay_seconds=300,
        )._observation(
            request=MetricsEvidenceRequest(
                target_id=str(target_id),
                binding_id=str(binding_id),
                source_locator_key="mysql-01",
                metric_definitions=(definition,),
                window_start=now - timedelta(hours=1),
                window_end=now,
                requested_step_seconds=60,
                max_response_bytes=1024,
                trace_id="trace-1",
            ),
            definition=definition,
            raw_series=[
                (
                    {},
                    [
                        (
                            now,
                            1,
                        )
                    ],
                )
            ],
            provider_response_hash="a" * 64,
            effective_step=60,
            truncated=False,
        )
        profile = MonitoringProfileDefinition(
            profile_id="mysql-state",
            version="1.0.0",
            display_name="MySQL 状态",
            description="MySQL 可用性状态。",
            supported_db_types=("MYSQL",),
            metric_codes=("mysql.availability", "mysql.sql.slow_query_rate"),
        )
        gap = MonitoringGap(
            scope="METRIC",
            instance_id=target_id,
            metric_code="mysql.sql.slow_query_rate",
            code="SOURCE_NO_DATA",
            detail="没有历史采样",
        )
        view = MonitoringViewBuilder(metric_catalog=self.metrics).build(
            source=source,
            profile=profile,
            window=resolve_monitoring_window("1h", now=now),
            instances=(instance,),
            observations=(observation,),
            gaps=(gap,),
            generated_at=now,
        )
        self.assertTrue(view.partial)
        missing = next(
            item
            for item in view.panels
            if item.metric_code == "mysql.sql.slow_query_rate"
        )
        self.assertEqual("NO_DATA", missing.quality)
        self.assertEqual((), missing.series)
        available = next(
            item
            for item in view.panels
            if item.metric_code == "mysql.availability"
        )
        self.assertEqual(source_id, available.series[0].source_id)
        self.assertEqual("Prometheus", available.series[0].source_display_name)
        self.assertEqual("PROMETHEUS", available.series[0].source_type)
        dumped = view.model_dump_json()
        self.assertNotIn("source_locator", dumped)
        self.assertNotIn("query_template", dumped)

    def test_zabbix_item_key_override_is_exact(self):
        definition = self.metrics.get("mysql.availability")
        resolved = resolve_metric_definitions(
            {
                "binding_version": 7,
                "metrics": [definition.model_dump(mode="json")],
                "mapping_overrides": {
                    "zabbix_item_keys": {
                        "mysql.availability": "customer.mysql.ping"
                    }
                },
            }
        )[0]
        provider = resolved.providers["ZABBIX"]
        self.assertEqual("customer.mysql.ping", provider.exact_item_key)
        self.assertEqual("binding.mysql.availability", provider.template_id)


class ZabbixAdapterContractTest(unittest.IsolatedAsyncioTestCase):
    def _adapter(self, session):
        return ZabbixAdapter(
            context=DiagnosticSourceContext(
                source_id=str(uuid7()),
                source_type="ZABBIX",
                adapter_id="zabbix",
                adapter_version="1.0.0",
                config_version=1,
                endpoint="https://zabbix.example.com/api_jsonrpc.php",
                credentials={"token": "test-token"},
                config={"auth_mode": "API_TOKEN"},
            ),
            session=session,
            request_timeout_seconds=5,
            webhook_replay_seconds=300,
        )

    async def test_health_check_uses_json_rpc_and_authenticates(self):
        session = _Session(
            [
                {"jsonrpc": "2.0", "result": "7.0.0", "id": 1},
                {"jsonrpc": "2.0", "result": [], "id": 1},
            ]
        )
        adapter = self._adapter(session)
        result = await adapter.health_check(SourceHealthRequest(trace_id="t"))
        self.assertTrue(result.healthy)
        self.assertEqual(
            ["apiinfo.version", "host.get"],
            [item[1]["method"] for item in session.requests],
        )
        self.assertNotIn("auth", session.requests[0][1])
        self.assertEqual("test-token", session.requests[1][1]["auth"])

    async def test_metric_query_uses_declared_zabbix_history_type(self):
        now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
        session = _Session(
            [
                {
                    "jsonrpc": "2.0",
                    "result": [{"hostid": "7", "host": "mysql-01"}],
                    "id": 1,
                },
                {
                    "jsonrpc": "2.0",
                    "result": [
                        {
                            "itemid": "9",
                            "key_": "kbot.mysql.availability",
                            "value_type": "3",
                            "units": "state",
                        }
                    ],
                    "id": 1,
                },
                {
                    "jsonrpc": "2.0",
                    "result": [{"clock": str(int(now.timestamp())), "value": "1"}],
                    "id": 1,
                },
            ]
        )
        adapter = self._adapter(session)
        definition = load_metric_catalog().get("mysql.availability")
        result = await adapter.query_metrics(
            MetricsEvidenceRequest(
                target_id=str(uuid7()),
                binding_id=str(uuid7()),
                source_locator_key="mysql-01",
                metric_definitions=(definition,),
                window_start=now - timedelta(minutes=15),
                window_end=now,
                requested_step_seconds=60,
                max_response_bytes=1024 * 1024,
                trace_id="trace-zabbix",
            )
        )
        self.assertEqual(1, len(result.observations))
        history_request = session.requests[2][1]
        self.assertEqual("history.get", history_request["method"])
        self.assertEqual(3, history_request["params"]["history"])
