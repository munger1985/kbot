"""实时监控应用服务的权限、门禁、缓存与脱敏合同。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from aiops_agent.adapters.diagnostic_sources.catalog import load_metric_catalog
from aiops_agent.application.errors import AIOpsApplicationError, dependency_unavailable
from aiops_agent.application.monitoring_views import MonitoringApplicationService
from aiops_agent.contracts.evidence import (
    MetricObservation,
    MetricPoint,
    MetricSeries,
)
from aiops_agent.monitoring import (
    load_monitoring_profile_catalog,
    project_source_readiness,
)
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_EVENT_QUERY,
    CAPABILITY_EVENT_RECEIVE,
    CAPABILITY_METRIC_QUERY_RANGE,
    MetricsEvidenceResult,
)
from platform_core.contracts.aiops.monitoring import (
    MonitoringGap,
    MonitoringViewRequest,
)


class _DiagnosticSources:
    def __init__(self, source):
        self.source = source

    async def page_scoped(self, **_kwargs):
        return [self.source] if self.source is not None else []

    async def get_scoped(self, *, diagnostic_source_id, domain_id, **_kwargs):
        if (
            self.source is None
            or self.source.diagnostic_source_id != diagnostic_source_id
            or self.source.domain_id != domain_id
        ):
            return None
        return self.source


class _Targets:
    def __init__(self, target, binding):
        self.target = target
        self.binding = binding

    async def list_source_bindings_by_source(
        self, *, diagnostic_source_id, domain_id
    ):
        if (
            self.binding.diagnostic_source_id != diagnostic_source_id
            or self.target.domain_id != domain_id
        ):
            return []
        return [self.binding]

    async def get_scoped(self, *, target_id, domain_id, **_kwargs):
        if self.target.target_id != target_id or self.target.domain_id != domain_id:
            return None
        return self.target

    async def update_observed_status(self, **kwargs):
        self.target.observed_status = kwargs["observed_status"]
        self.target.last_observed_at = kwargs["checked_at"]
        self.target.last_error_code = kwargs["last_error_code"]
        return True


class _Uow:
    def __init__(self, source, target, binding):
        self.diagnostic_sources = _DiagnosticSources(source)
        self.targets = _Targets(target, binding)
        self.commits = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def commit(self):
        self.commits += 1


class _SnapshotBuilder:
    def __init__(self, metric_catalog, source, binding, *, secret_ref=None):
        self.metric_catalog = metric_catalog
        self.source = source
        self.binding = binding
        self.secret_ref = secret_ref
        self.calls = 0

    async def build(
        self,
        *,
        uow,
        domain_id,
        target,
        now,
        allowed_source_ids=None,
        window_start=None,
        window_end=None,
        requested_metric_codes=None,
    ):
        self.calls += 1
        codes = tuple(requested_metric_codes or ())
        return {
            "initial_gaps": (),
            "bindings": (
                {
                    "binding_id": str(self.binding.target_source_binding_id),
                    "binding_version": self.binding.row_version,
                    "source_locator_key": self.binding.source_locator_key,
                    "source_locator": {"instance": "db-01"},
                    "mapping_overrides": {},
                    "metrics": tuple(
                        self.metric_catalog.get(code).model_dump(mode="json")
                        for code in codes
                    ),
                    "source": {
                        "source_id": str(self.source.diagnostic_source_id),
                        "source_type": self.source.source_type,
                        "adapter_id": "prometheus",
                        "adapter_version": "1.0.0",
                        "config_version": self.source.row_version,
                        "endpoint": "https://monitor.invalid",
                        "secret_ref": self.secret_ref,
                        "declared_capabilities": self.source.declared_capabilities_json,
                        "config": {},
                    },
                },
            ),
        }


class _Adapter:
    def __init__(self):
        self.calls = 0
        self.result = MetricsEvidenceResult()

    async def query_metrics(self, _request):
        self.calls += 1
        return self.result


class _Registry:
    def __init__(self):
        self.adapter = _Adapter()
        self.create_calls = 0

    def create(self, *_args, **_kwargs):
        self.create_calls += 1
        return self.adapter


class _SecretStore:
    async def resolve(self, _reference):
        return SimpleNamespace(values={"token": "never-returned"})


class _FailingSecretStore:
    async def resolve(self, _reference):
        raise dependency_unavailable("凭据不可用")


def _fixture(*, connected=True, secret_ref=None):
    domain_id = uuid4()
    source_id = uuid4()
    target_id = uuid4()
    source = SimpleNamespace(
        diagnostic_source_id=source_id,
        domain_id=domain_id,
        display_name="生产监控",
        source_type="PROMETHEUS",
        status="ENABLED",
        connectivity_status="CONNECTED" if connected else "FAILED",
        declared_capabilities_json={
            CAPABILITY_METRIC_QUERY_RANGE: {},
            CAPABILITY_EVENT_QUERY: {},
        },
        config_json={"alert_ingress_ready": True},
        row_version=3,
    )
    target = SimpleNamespace(
        target_id=target_id,
        domain_id=domain_id,
        display_name="核心库",
        db_type="ORACLE",
        status="ENABLED",
        observed_status="UNKNOWN",
        last_observed_at=None,
        last_error_code=None,
    )
    binding = SimpleNamespace(
        target_source_binding_id=uuid4(),
        target_id=target_id,
        diagnostic_source_id=source_id,
        status="ACTIVE",
        health_status="READY",
        source_locator_key="prometheus:db-01",
        row_version=5,
    )
    metrics = load_metric_catalog()
    profiles = load_monitoring_profile_catalog(metrics)
    registry = _Registry()
    uow = _Uow(source, target, binding)
    service = MonitoringApplicationService(
        uow_factory=lambda: uow,
        metric_catalog=metrics,
        profile_catalog=profiles,
        diagnostic_source_registry=registry,
        secret_store=_FailingSecretStore() if secret_ref else _SecretStore(),
        max_response_bytes=1024 * 1024,
    )
    snapshot = _SnapshotBuilder(
        metrics, source, binding, secret_ref=secret_ref
    )
    service._snapshot_builder = snapshot
    scope = SimpleNamespace(domain_id=domain_id, trace_id="trace-monitoring")
    request = MonitoringViewRequest(
        profile_id="oracle-overview",
        instance_ids=(target_id,),
        window="1h",
    )
    return service, scope, source, target, binding, registry, snapshot, request


def test_not_ready_source_does_not_call_provider():
    service, scope, source, *_rest, registry, snapshot, request = _fixture(
        connected=False
    )
    with pytest.raises(AIOpsApplicationError) as caught:
        asyncio.run(
            service.get_view(
                scope=scope,
                source_id=source.diagnostic_source_id,
                request=request,
            )
        )
    assert caught.value.code == "MONITORING_SOURCE_NOT_READY"
    assert registry.create_calls == 0
    assert snapshot.calls == 0


def test_alert_ingress_readiness_uses_bound_alertmanager_webhook():
    service, scope, _source, target, *_rest = _fixture()
    alertmanager_id = uuid4()
    alertmanager = SimpleNamespace(
        diagnostic_source_id=alertmanager_id,
        source_type="ALERTMANAGER",
        status="ENABLED",
        declared_capabilities_json={CAPABILITY_EVENT_RECEIVE: {}},
        webhook_key_hash="hash",
        webhook_credential_id=uuid4(),
    )
    binding = SimpleNamespace(
        diagnostic_source_id=alertmanager_id,
        target_id=target.target_id,
        status="ACTIVE",
    )
    uow = SimpleNamespace(
        diagnostic_sources=SimpleNamespace(
            page_scoped=AsyncMock(return_value=[alertmanager])
        ),
        targets=SimpleNamespace(
            list_source_bindings_by_source=AsyncMock(return_value=[binding])
        ),
    )

    target_ids = asyncio.run(
        service._alert_ingress_target_ids(
            uow=uow,
            domain_id=scope.domain_id,
        )
    )

    assert target_ids == {target.target_id}


def test_unmapped_instance_is_hidden_as_not_found():
    service, scope, source, *_rest, registry, _snapshot, request = _fixture()
    tampered = request.model_copy(update={"instance_ids": (uuid4(),)})
    with pytest.raises(AIOpsApplicationError) as caught:
        asyncio.run(
            service.get_view(
                scope=scope,
                source_id=source.diagnostic_source_id,
                request=tampered,
            )
        )
    assert caught.value.status_code == 404
    assert registry.create_calls == 0


def test_cache_is_invalidated_by_binding_version():
    service, scope, source, _target, binding, registry, _snapshot, request = _fixture()
    now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
    first = asyncio.run(service.get_view(
        scope=scope,
        source_id=source.diagnostic_source_id,
        request=request,
        now=now,
    ))
    second = asyncio.run(service.get_view(
        scope=scope,
        source_id=source.diagnostic_source_id,
        request=request,
        now=now + timedelta(seconds=1),
    ))
    assert second is first
    assert registry.adapter.calls == 1
    binding.row_version += 1
    asyncio.run(service.get_view(
        scope=scope,
        source_id=source.diagnostic_source_id,
        request=request,
        now=now + timedelta(seconds=2),
    ))
    assert registry.adapter.calls == 2


def test_credential_failure_becomes_sanitized_gap():
    service, scope, source, *_rest, request = _fixture(
        secret_ref="managed://hidden"
    )
    view = asyncio.run(service.get_view(
        scope=scope, source_id=source.diagnostic_source_id, request=request
    ))
    assert view.partial is True
    assert {gap.code for gap in view.gaps} == {"SOURCE_CREDENTIAL_UNAVAILABLE"}
    encoded = view.model_dump_json()
    for forbidden in (
        "endpoint",
        "locator",
        "query_template",
        "hidden",
        "never-returned",
    ):
        assert forbidden not in encoded.lower()


def test_live_availability_sample_updates_target_observed_status():
    service, scope, source, target, binding, registry, _snapshot, request = (
        _fixture()
    )
    observed_at = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
    registry.adapter.result = MetricsEvidenceResult(
        observations=(
            MetricObservation(
                metric_code="db.availability",
                semantic_version="1.0.0",
                unit="state",
                value_kind="STATE",
                window_start=observed_at - timedelta(minutes=5),
                window_end=observed_at,
                requested_step_seconds=60,
                effective_step_seconds=60,
                source_id=str(source.diagnostic_source_id),
                source_type="PROMETHEUS",
                source_version=source.row_version,
                target_id=str(target.target_id),
                binding_id=str(binding.target_source_binding_id),
                external_target_fingerprint="a" * 64,
                series=(
                    MetricSeries(
                        points=(
                            MetricPoint(
                                observed_at=observed_at,
                                value=1,
                                quality="GOOD",
                            ),
                        )
                    ),
                ),
                summary={"last": 1.0},
                expected_points=1,
                actual_points=1,
                coverage_ratio=1.0,
            ),
        )
    )

    asyncio.run(
        service.get_view(
            scope=scope,
            source_id=source.diagnostic_source_id,
            request=request,
            now=observed_at,
        )
    )

    assert target.observed_status == "UP"
    assert target.last_observed_at == observed_at
    assert service._uow_factory().commits == 1


def test_target_view_can_compare_prometheus_and_zabbix_independently():
    service, scope, source, target, *_rest, request = _fixture()
    now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
    primary = asyncio.run(
        service.get_view(
            scope=scope,
            source_id=source.diagnostic_source_id,
            request=request,
            now=now,
        )
    )
    compare_source_id = uuid4()
    compare_source = project_source_readiness(
        source_id=compare_source_id,
        display_name="生产 Zabbix",
        source_type="ZABBIX",
        status="ENABLED",
        connectivity_status="CONNECTED",
        capabilities={CAPABILITY_METRIC_QUERY_RANGE, CAPABILITY_EVENT_QUERY},
        active_binding_count=1,
        metrics_aligned=True,
        alert_ingress_ready=True,
    )
    comparison = primary.model_copy(
        update={
            "source": compare_source,
            "gaps": (
                MonitoringGap(
                    scope="SOURCE",
                    source_id=compare_source_id,
                    code="SOURCE_UNREACHABLE",
                    detail="监控源查询失败",
                    retryable=True,
                ),
            ),
            "partial": True,
        }
    )
    service.get_view = AsyncMock(side_effect=(primary, comparison))

    result = asyncio.run(
        service.get_target_view(
            scope=scope,
            target_id=target.target_id,
            source_id=source.diagnostic_source_id,
            compare_source_id=compare_source_id,
            profile_id="oracle-overview",
            window="1h",
        )
    )

    assert service.get_view.await_count == 2
    assert result.source.source_type == "PROMETHEUS"
    assert result.compare_source == compare_source
    assert result.partial is True
    assert result.gaps[0].source_id == compare_source_id


def test_target_view_rejects_comparing_source_with_itself_before_query():
    service, scope, source, target, *_rest = _fixture()
    service.get_view = AsyncMock()
    with pytest.raises(AIOpsApplicationError) as caught:
        asyncio.run(
            service.get_target_view(
                scope=scope,
                target_id=target.target_id,
                source_id=source.diagnostic_source_id,
                compare_source_id=source.diagnostic_source_id,
                profile_id="oracle-overview",
                window="1h",
            )
        )
    assert caught.value.code == "MONITORING_COMPARE_SOURCE_INVALID"
    service.get_view.assert_not_awaited()
