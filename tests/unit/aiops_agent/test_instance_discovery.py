"""实例发现引用、白名单 Provider 查询和批量映射合同测试。"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
from aiops_agent.adapters.diagnostic_sources.registry import (
    DiagnosticSourceAdapterRegistry,
)
from aiops_agent.adapters.diagnostic_sources.zabbix import ZabbixAdapter
from aiops_agent.application.configuration.instance_discovery import (
    InstanceCandidateRefCodec,
)
from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.configuration.service import (
    AIOpsConfigurationService,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.ports.diagnostic_source import (
    CAPABILITY_INSTANCE_DISCOVERY,
    CAPABILITY_METRIC_QUERY_RANGE,
    DiagnosticSourceContext,
    HostDiscoveryCandidate as ProviderHostDiscoveryCandidate,
    InstanceDiscoveryCandidate as ProviderInstanceDiscoveryCandidate,
    InstanceDiscoveryRequest,
    InstanceDiscoveryResult,
)
from platform_core.contracts.aiops import InstanceMappingRequest, SourceBindingPatch
from platform_core.identity import uuid7
from platform_core.managed_credentials import ManagedCredentialCipher


class _Response:
    status = 200

    def __init__(self, payload):
        self._raw = json.dumps(payload).encode("utf-8")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def read(self):
        return self._raw


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.url = None
        self.params = None

    def get(self, url, **kwargs):
        self.url = url
        self.params = kwargs.get("params")
        return _Response(self.payload)


def _codec():
    return InstanceCandidateRefCodec(
        cipher=ManagedCredentialCipher(key=b"k" * 32, key_version="test-v1")
    )


def test_candidate_ref_is_opaque_scoped_and_expires():
    domain_id = 100
    source_id = uuid7()
    actor_id = "user-1"
    now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
    codec = _codec()
    token = codec.encode(
        domain_id=domain_id,
        source_id=source_id,
        actor_id=actor_id,
        purpose="candidate",
        value={"source_locator_key": "db-secret-01"},
        now=now,
    )
    assert "db-secret-01" not in token
    assert codec.decode(
        token,
        domain_id=domain_id,
        source_id=source_id,
        actor_id=actor_id,
        purpose="candidate",
        now=now,
    )["source_locator_key"] == "db-secret-01"
    with pytest.raises(AIOpsApplicationError):
        codec.decode(
            token,
                domain_id=101,
            source_id=source_id,
            actor_id=actor_id,
            purpose="candidate",
            now=now,
        )
    with pytest.raises(AIOpsApplicationError):
        codec.decode(
            token,
                domain_id=domain_id,
            source_id=source_id,
            actor_id=actor_id,
            purpose="candidate",
            now=now + timedelta(minutes=11),
        )
    with pytest.raises(AIOpsApplicationError):
        codec.decode(
            token,
            domain_id=domain_id,
            source_id=source_id,
            actor_id="user-2",
            purpose="candidate",
            now=now,
        )


def test_registry_allows_management_discovery_without_legacy_declaration():
    registry = DiagnosticSourceAdapterRegistry(session=_Session({}))
    adapter = registry.create(
        DiagnosticSourceContext(
            source_id=str(uuid7()),
            source_type="PROMETHEUS",
            adapter_id="prometheus",
            adapter_version="1.0.0",
            config_version=1,
            endpoint="https://prom.example.com",
            declared_capabilities={CAPABILITY_METRIC_QUERY_RANGE: {}},
        ),
        capability=CAPABILITY_INSTANCE_DISCOVERY,
    )
    assert isinstance(adapter, PrometheusAdapter)


def test_prometheus_discovery_uses_fixed_series_and_projects_locator():
    session = _Session(
        {
            "status": "success",
            "data": [
                {
                    "__name__": "mysql_up",
                    "job": "mysql",
                    "target_key": "mysql-prod-01",
                    "untrusted": "must-not-leak",
                },
                {
                    "__name__": "up",
                    "job": "node",
                    "target_key": "host-prod-01",
                }
            ],
        }
    )
    adapter = DiagnosticSourceAdapterRegistry(session=session).create(
        DiagnosticSourceContext(
            source_id=str(uuid7()),
            source_type="PROMETHEUS",
            adapter_id="prometheus",
            adapter_version="1.0.0",
            config_version=1,
            endpoint="https://prom.example.com",
        ),
        capability=CAPABILITY_INSTANCE_DISCOVERY,
    )
    result = asyncio.run(
        adapter.discover_instances(
            InstanceDiscoveryRequest(
                db_types=("MYSQL",),
                include_hosts=True,
                limit=10,
                trace_id="trace",
            )
        )
    )
    assert str(session.url).endswith("/api/v1/series")
    assert ("match[]", 'mysql_up{job="mysql"}') in session.params
    assert ("match[]", 'up{job="node"}') in session.params
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.source_locator_key == "mysql-prod-01"
    assert candidate.source_locator == {
        "target_key": "mysql-prod-01",
        "instance": "mysql-prod-01",
        "database_type": "MYSQL",
    }
    assert "untrusted" not in candidate.model_dump_json()
    assert result.host_candidates == (
        ProviderHostDiscoveryCandidate(
            source_locator_key="host-prod-01",
            display_name="Node Exporter 主机",
        ),
    )


def test_zabbix_discovery_requires_controlled_database_tag_or_template():
    assert ZabbixAdapter._database_type(
        {"tags": [{"tag": "database_type", "value": "postgres"}]}
    ) == "POSTGRESQL"
    assert ZabbixAdapter._database_type(
        {"parentTemplates": [{"name": "Linux by Zabbix agent"}]}
    ) is None
    assert ZabbixAdapter._database_type(
        {"parentTemplates": [{"name": "Oracle database monitoring"}]}
    ) == "ORACLE"


def test_batch_mapping_contract_rejects_duplicate_targets():
    target_id = uuid7()
    with pytest.raises(ValueError):
        InstanceMappingRequest.model_validate(
            {
                "mappings": [
                    {"candidate_ref": "candidate-a", "target_id": target_id},
                    {"candidate_ref": "candidate-b", "target_id": target_id},
                ]
            }
        )


def _mapping_service(*, decoded, verified, targets):
    now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
    source = SimpleNamespace(source_type="PROMETHEUS")
    target_repository = AsyncMock()
    target_repository.get_scoped.side_effect = targets
    target_repository.list_source_bindings.return_value = ()
    target_repository.get_source_binding_by_locator_any_status.return_value = None
    diagnostic_sources = AsyncMock()
    diagnostic_sources.get_scoped.return_value = source
    uow = SimpleNamespace(
        targets=target_repository,
        diagnostic_sources=diagnostic_sources,
    )
    adapter = AsyncMock()
    adapter.discover_instances.return_value = InstanceDiscoveryResult(
        candidates=tuple(verified),
        host_candidates=(
            ProviderHostDiscoveryCandidate(
                source_locator_key="host-1",
                display_name="主机 1",
            ),
            ProviderHostDiscoveryCandidate(
                source_locator_key="host-2",
                display_name="主机 2",
            ),
        ),
    )
    service = object.__new__(AIOpsConfigurationService)
    service._instance_candidate_refs = Mock()
    service._instance_candidate_refs.decode.side_effect = decoded
    service._instance_discovery_adapter = AsyncMock(return_value=adapter)
    service._validate_source_binding_locator = Mock()

    async def execute_handler(**kwargs):
        return await kwargs["handler"](uow, now)

    service._idempotent = AsyncMock(side_effect=execute_handler)
    scope = ConfigurationScope(
        domain_id=100,
        principal_id="PORTAL:aiops",
        actor_id="user-1",
        request_id="request-map",
        trace_id="trace-map",
    )
    return service, scope, target_repository


def _candidate(locator_key, db_type, locator_value):
    return ProviderInstanceDiscoveryCandidate(
        source_locator_key=locator_key,
        source_locator={"target_key": locator_value},
        db_type=db_type,
        display_name=locator_key,
    )


def test_batch_mapping_revalidates_changed_candidate_before_writes():
    source_id = uuid7()
    target_id = uuid7()
    service, scope, repository = _mapping_service(
        decoded=[
            {
                "candidate_kind": "DATABASE",
                "source_locator_key": "db-1",
                "source_locator": {"target_key": "old"},
                "db_type": "ORACLE",
            },
            {"candidate_kind": "HOST", "host_target_key": "host-1"},
        ],
        verified=[_candidate("db-1", "ORACLE", "new")],
        targets=[SimpleNamespace(db_type="ORACLE")],
    )
    with pytest.raises(AIOpsApplicationError) as caught:
        asyncio.run(
            service.map_source_instances(
                scope=scope,
                source_id=source_id,
                request=InstanceMappingRequest.model_validate(
                    {
                        "mappings": [
                            {
                                "candidate_ref": "candidate-a",
                                "host_candidate_ref": "host-a",
                                "target_id": target_id,
                            }
                        ]
                    }
                ),
                idempotency_key="map-1",
            )
        )
    assert caught.value.status_code == 409
    repository.add_source_binding.assert_not_awaited()


def test_batch_mapping_preflights_all_target_types_before_writes():
    source_id = uuid7()
    target_ids = (uuid7(), uuid7())
    decoded = [
        {
            "candidate_kind": "DATABASE",
            "source_locator_key": "db-1",
            "source_locator": {"target_key": "db-1"},
            "db_type": "ORACLE",
        },
        {"candidate_kind": "HOST", "host_target_key": "host-1"},
        {
            "candidate_kind": "DATABASE",
            "source_locator_key": "db-2",
            "source_locator": {"target_key": "db-2"},
            "db_type": "MYSQL",
        },
        {"candidate_kind": "HOST", "host_target_key": "host-2"},
    ]
    service, scope, repository = _mapping_service(
        decoded=decoded,
        verified=[
            _candidate("db-1", "ORACLE", "db-1"),
            _candidate("db-2", "MYSQL", "db-2"),
        ],
        targets=[
            SimpleNamespace(db_type="ORACLE"),
            SimpleNamespace(db_type="POSTGRESQL"),
        ],
    )
    with pytest.raises(AIOpsApplicationError) as caught:
        asyncio.run(
            service.map_source_instances(
                scope=scope,
                source_id=source_id,
                request=InstanceMappingRequest.model_validate(
                    {
                        "mappings": [
                        {
                            "candidate_ref": f"candidate-{index}",
                            "host_candidate_ref": f"host-{index}",
                            "target_id": target_id,
                        }
                            for index, target_id in enumerate(target_ids)
                        ]
                    }
                ),
                idempotency_key="map-2",
            )
        )
    assert caught.value.status_code == 422
    repository.add_source_binding.assert_not_awaited()


def test_prometheus_mapping_persists_reverified_host_label():
    source_id = uuid7()
    target_id = uuid7()
    binding_id = uuid7()
    service, scope, _repository = _mapping_service(
        decoded=[
            {
                "candidate_kind": "DATABASE",
                "source_locator_key": "db-1",
                "source_locator": {"target_key": "db-1"},
                "db_type": "ORACLE",
            },
            {"candidate_kind": "HOST", "host_target_key": "host-1"},
        ],
        verified=[_candidate("db-1", "ORACLE", "db-1")],
        targets=[],
    )
    entity = SimpleNamespace(
        target_source_binding_id=binding_id,
        target_id=target_id,
        diagnostic_source_id=source_id,
        source_locator_key="db-1",
        status="ACTIVE",
        health_status="UNKNOWN",
        row_version=1,
    )
    service._prepare_source_binding_in_uow = AsyncMock(return_value=entity)
    service._persist_source_binding_in_uow = AsyncMock()

    result = asyncio.run(
        service.map_source_instances(
            scope=scope,
            source_id=source_id,
            request=InstanceMappingRequest.model_validate(
                {
                    "mappings": [
                        {
                            "candidate_ref": "candidate-a",
                            "host_candidate_ref": "host-a",
                            "target_id": target_id,
                        }
                    ]
                }
            ),
            idempotency_key="map-host-1",
        )
    )

    prepared = service._prepare_source_binding_in_uow.await_args.kwargs[
        "request"
    ]
    assert prepared.source_locator == {
        "target_key": "db-1",
        "host_target_key": "host-1",
    }
    assert result.items[0].binding_id == binding_id


def test_patch_binding_reverifies_host_candidate_before_persisting():
    now = datetime(2026, 10, 10, 8, 0, tzinfo=UTC)
    source_id = uuid7()
    target_id = uuid7()
    binding_id = uuid7()
    entity = SimpleNamespace(
        target_source_binding_id=binding_id,
        target_id=target_id,
        diagnostic_source_id=source_id,
        source_locator_key="db-1",
        source_locator_json={"target_key": "db-1"},
        role="PRIMARY",
        priority=100,
        capability_scope_json=None,
        mapping_overrides_json=None,
        query_budget_json=None,
        status="ACTIVE",
        health_status="UNKNOWN",
        row_version=3,
        created_at=now,
        updated_at=now,
        updated_by="old-user",
    )
    source = SimpleNamespace(
        source_type="PROMETHEUS",
        declared_capabilities_json={},
    )
    uow = SimpleNamespace(
        targets=AsyncMock(),
        diagnostic_sources=AsyncMock(),
        session=AsyncMock(),
        outbox=AsyncMock(),
        commit=AsyncMock(),
    )
    uow.targets.get_source_binding_scoped.return_value = entity
    uow.diagnostic_sources.get_scoped.return_value = source

    class _UowContext:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, *_args):
            return None

    adapter = AsyncMock()
    adapter.discover_instances.return_value = InstanceDiscoveryResult(
        host_candidates=(
            ProviderHostDiscoveryCandidate(
                source_locator_key="host-1",
                display_name="主机 1",
            ),
        )
    )
    service = object.__new__(AIOpsConfigurationService)
    service._uow_factory = Mock(return_value=_UowContext())
    service._instance_discovery_adapter = AsyncMock(return_value=adapter)
    service._instance_candidate_refs = Mock()
    service._instance_candidate_refs.decode.return_value = {
        "candidate_kind": "HOST",
        "host_target_key": "host-1",
    }
    scope = ConfigurationScope(
        domain_id=100,
        principal_id="PORTAL:aiops",
        actor_id="user-1",
        request_id="request-host-patch",
        trace_id="trace-host-patch",
    )

    result = asyncio.run(
        service.patch_source_binding(
            scope=scope,
            target_id=target_id,
            binding_id=binding_id,
            request=SourceBindingPatch(host_candidate_ref="host-ref"),
            expected_version=3,
        )
    )

    assert entity.source_locator_json == {
        "target_key": "db-1",
        "host_target_key": "host-1",
    }
    assert result.source_locator["host_target_key"] == "host-1"
    uow.commit.assert_awaited_once()
