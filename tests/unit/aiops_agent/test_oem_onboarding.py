"""OEM 服务级接入与业务系统数据库候选绑定回归。"""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem import OEMAdapter
from aiops_agent.adapters.diagnostic_sources.oem_target_identity import normalize_target
from aiops_agent.adapters.diagnostic_sources.registry import DiagnosticSourceAdapterCatalog, DiagnosticSourceAdapterRegistry
from aiops_agent.application.configuration.oem_binding import validate_oem_database_binding
from aiops_agent.application.configuration.service import AIOpsConfigurationService
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.ports.diagnostic_source import CAPABILITY_INSTANCE_DISCOVERY, DiagnosticSourceContext, InstanceDiscoveryRequest
from platform_core.contracts.aiops import SourceBindingCreate
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection, metadata, request

CDB = "1" * 32
PDB = "2" * 32
ROOT = "3" * 32
HOST = "4" * 32


def target(guid=CDB, name="cdb-01", kind="oracle_database", **extra):
    return {"id": guid, "name": name, "typeName": kind, **extra}


def properties(**values):
    return collection([{"id": key, "value": value} for key, value in values.items()])


def discover(value, **changes):
    return asyncio.run(value.discover_instances(InstanceDiscoveryRequest(trace_id="test", **changes)))


def cdb_payloads(guid=CDB, name="cdb-01"):
    return [target(guid, name, hostName="dbhost"), properties(CDB="YES", DBName="CDB01", Version="19.3", RACOption="NO")]


def test_registry_supports_oem_management_discovery_without_metric_configuration():
    registry = DiagnosticSourceAdapterRegistry(session=SimpleNamespace())
    value = registry.create(DiagnosticSourceContext(
        source_id=str(uuid7()), source_type="OEM", adapter_id="oem", adapter_version="1.0.0",
        config_version=1, endpoint="https://oem.example/em/api",
    ), capability=CAPABILITY_INSTANCE_DISCOVERY)
    assert isinstance(value, OEMAdapter)
    assert DiagnosticSourceAdapterCatalog.normalize_config(source_type="OEM", config={}) == {}


@pytest.mark.parametrize("config", [
    {"target_id": CDB}, {"target_name": "cdb"}, {"database_type": "ORACLE"},
    {"standard_metric_mappings": {}}, {"query": "arbitrary"},
])
def test_source_configuration_cannot_bind_database_or_request_metric_json(config):
    with pytest.raises(ValueError):
        DiagnosticSourceAdapterCatalog.normalize_config(source_type="OEM", config=config)


def test_discovery_uses_real_page_tokens_filters_host_and_excludes_root_auxiliary_target():
    listing = collection([target(CDB), target(HOST, "host", "host")],
                         links={"next": {"href": "/em/api/targets?page=page2"}})
    value, calls = adapter(
        listing, collection([target(PDB, "pdb", "oracle_pdb"), target(ROOT, "root", "oracle_pdb")]),
        *cdb_payloads(),
        target(PDB, "pdb", "oracle_pdb", hostName="dbhost"), properties(IS_ROOT="NO", Name="APP1", PDBGuid="a"*32),
        target(ROOT, "root", "oracle_pdb"), properties(IS_ROOT="YES", Name="CDB$ROOT"),
    )
    result = discover(value, db_types=("ORACLE",))
    assert [item.source_locator_key for item in result.candidates] == [CDB, PDB]
    assert not result.host_candidates
    cdb, pdb = result.candidates
    assert cdb.source_locator["oracle_container_scope"] == "CDB_ROOT"
    assert "dbhost" in cdb.display_name and "19.3" in cdb.display_name
    assert pdb.source_locator["oracle_pdb_name"] == "APP1"
    assert calls[1][1]["params"] == {"page": "page2"}
    assert all(not kwargs["allow_redirects"] for _, kwargs in calls)
    assert not any("metric" in url for url, _ in calls)


def test_guid_case_is_canonical_and_after_locator_filters_before_detail_requests():
    value, calls = adapter(collection([target(CDB.upper()), target(PDB.upper(), "pdb", "oracle_pdb")]),
        target(PDB, "pdb", "oracle_pdb"), properties(IS_ROOT="NO", Name="APP1"))
    result = discover(value, after_locator_key=CDB)
    assert [item.source_locator_key for item in result.candidates] == [PDB]
    assert len(calls) == 3


def test_engine_filter_does_not_guess_database_type_from_display_name():
    value, calls = adapter(collection([target(), target(HOST, "mysql-oracle-prod", "host"),
                                      target(PDB, "postgres", "unregistered_plugin")]))
    assert not discover(value, db_types=("MYSQL",)).candidates
    assert len(calls) == 1


def test_mysql_plugin_candidate_is_not_an_oracle_target():
    mysql = target(PDB, "mysql-01", "mysql_database")
    value, _ = adapter(collection([mysql]), mysql, properties(Version="8.4"))
    candidate, = discover(value, db_types=("MYSQL",)).candidates
    assert candidate.db_type == "MYSQL"
    assert "oracle_container_scope" not in candidate.source_locator


def test_oem_host_discovery_uses_host_guid_and_skips_database_properties():
    host = target(HOST, "oracle19c.lab", "host")
    value, calls = adapter(collection([target(), host]), host)
    result = discover(value, include_hosts=True, host_only=True)
    assert not result.candidates
    candidate, = result.host_candidates
    assert candidate.source_locator_key == HOST
    assert candidate.display_name == "oracle19c.lab"
    assert candidate.source_locator == {
        "resource_scope": "HOST", "target_id": HOST, "target_type": "host",
        "target_name": "oracle19c.lab", "host_name": "oracle19c.lab",
    }
    assert len(calls) == 2 and not any("properties" in url for url, _ in calls)


def test_exact_oem_host_revalidation_never_accepts_a_database_guid():
    value, calls = adapter(target())
    result = discover(value, include_hosts=True, host_only=True, host_locator_keys=(CDB,))
    assert not result.host_candidates and not result.candidates and len(calls) == 1


def test_oem_host_pagination_applies_cursor_before_detail_and_respects_limit():
    first, second = target(HOST, "host-1", "host"), target("5" * 32, "host-2", "host")
    value, calls = adapter(collection([first, second]), second)
    result = discover(value, include_hosts=True, host_only=True, after_locator_key=HOST, limit=1)
    assert [item.source_locator_key for item in result.host_candidates] == ["5" * 32]
    assert len(calls) == 2


def test_oem_host_detail_identity_change_is_rejected():
    value, _ = adapter(collection([target(HOST, "host-1", "host")]), target(HOST, "host-2", "host"))
    with pytest.raises(DiagnosticSourceAdapterError):
        discover(value, include_hosts=True, host_only=True)


def test_exact_guid_revalidation_reads_only_requested_identity():
    value, calls = adapter(target(), *cdb_payloads())
    candidate, = discover(value, db_types=("ORACLE",), locator_keys=(CDB,)).candidates
    assert candidate.source_locator_key == CDB
    assert not any(url.endswith("/targets") for url, _ in calls)


@pytest.mark.parametrize("guid", ["database-name", "../x", "https://other.invalid", "f"*33])
def test_invalid_guid_never_becomes_a_network_path(guid):
    value, calls = adapter()
    with pytest.raises(DiagnosticSourceAdapterError):
        discover(value, locator_keys=(guid,))
    assert not calls


@pytest.mark.parametrize("href", [
    "https://other.invalid/em/api/targets?page=x", "//other.invalid/em/api/targets?page=x",
    "/em/api/targets/other?page=x", "/em/api/targets?page=x&targetName=other",
    "/em/api/targets?page=x&page=y", "/em/api/targets?page=", "/em/api/targets?page=x#fragment",
])
def test_provider_pagination_cannot_redirect_or_change_query(href):
    value, calls = adapter(collection([], links={"next": {"href": href}}))
    with pytest.raises(DiagnosticSourceAdapterError):
        discover(value)
    assert len(calls) == 1


def test_repeated_native_cursor_is_rejected_instead_of_looping():
    page = collection([], links={"next": {"href": "/em/api/targets?page=repeated"}})
    value, calls = adapter(page, page)
    with pytest.raises(DiagnosticSourceAdapterError) as exc:
        discover(value)
    assert exc.value.code == "SOURCE_PAGINATION_INVALID" and len(calls) == 2


def test_duplicate_guid_across_pages_fails_before_binding():
    value, _ = adapter(collection([target()], links={"next": {"href": "/em/api/targets?page=second"}}),
                       collection([target()]))
    with pytest.raises(DiagnosticSourceAdapterError):
        discover(value)


def test_unknown_oracle_scope_is_not_guessed_from_target_name():
    value, _ = adapter(collection([target()]), target(), properties(DBName="CDB01"))
    with pytest.raises(DiagnosticSourceAdapterError) as exc:
        discover(value)
    assert exc.value.code == "SOURCE_TARGET_IDENTITY_UNVERIFIED"


def test_changed_detail_identity_is_rejected():
    value, _ = adapter(collection([target()]), target(name="replaced"))
    with pytest.raises(DiagnosticSourceAdapterError):
        discover(value)


def test_modern_target_identity_is_also_used_by_existing_native_metric_path():
    value, _ = adapter(target("a"*32), collection([metadata()]))
    result = asyncio.run(value.discover_metrics(request(True, source_locator_key="a"*32)))
    assert result.entries


@pytest.mark.parametrize("payload", [
    target("x"*32), target(name="bad\nname"), target(kind=""), {"id": CDB, "name": "x"},
    target(targetName="other"), target(targetId=PDB),
])
def test_target_identity_rejects_malformed_or_conflicting_fields(payload):
    with pytest.raises(DiagnosticSourceAdapterError):
        normalize_target(payload, expected_guid=CDB)


def binding_target(**changes):
    return SimpleNamespace(db_type="ORACLE", oracle_container_scope="PDB", oracle_pdb_name="APP1",
                           observed_oracle_container_scope="PDB", observed_oracle_container_name="APP1",
                           observed_oracle_database_name="CDB01", deployment_type="SINGLE_INSTANCE",
                           **changes)


def binding_locator():
    return {"database_type": "ORACLE", "oracle_container_scope": "PDB",
            "oracle_pdb_name": "APP1", "database_name": "CDB01"}


def test_oem_binding_accepts_matching_database_identity():
    validate_oem_database_binding(target=binding_target(), locator=binding_locator())


@pytest.mark.parametrize("changes", [
    {"database_type": "MYSQL"}, {"oracle_container_scope": "CDB"},
    {"oracle_pdb_name": "OTHER"}, {"database_name": "OTHER"},
])
def test_oem_binding_rejects_wrong_engine_container_and_observed_database(changes):
    with pytest.raises(AIOpsApplicationError):
        validate_oem_database_binding(target=binding_target(), locator={**binding_locator(), **changes})


def test_oem_binding_rejects_wrong_rac_node():
    database = binding_target()
    database.deployment_type = "RAC"
    database.rac_node_number = 2
    with pytest.raises(AIOpsApplicationError):
        validate_oem_database_binding(target=database,
                                     locator={**binding_locator(), "rac_option": "YES", "rac_instance_number": "1"})


def test_raw_guid_binding_cannot_bypass_candidate_verification():
    service = object.__new__(AIOpsConfigurationService)
    repository = AsyncMock()
    repository.get_scoped.return_value = binding_target()
    source = SimpleNamespace(source_type="OEM")
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service._prepare_source_binding_in_uow(
            uow=SimpleNamespace(targets=repository, diagnostic_sources=AsyncMock()), scope=SimpleNamespace(domain_id=1),
            target_id=uuid7(), request=SourceBindingCreate(
                source_id=uuid7(), source_locator_key=CDB, source_locator=binding_locator(),
            ), source=source, now=datetime.now(UTC),
        ))
    repository.add_source_binding.assert_not_awaited()


def test_revalidation_rejects_changed_guid_identity_before_creating_any_binding():
    service = object.__new__(AIOpsConfigurationService)
    service._instance_candidate_refs = Mock()
    service._instance_candidate_refs.decode.return_value = {
        "candidate_kind": "DATABASE", "source_locator_key": CDB,
        "source_locator": {"database_type": "ORACLE", "target_name": "old"}, "db_type": "ORACLE",
    }
    from aiops_agent.ports.diagnostic_source import InstanceDiscoveryCandidate, InstanceDiscoveryResult
    current = InstanceDiscoveryCandidate(source_locator_key=CDB,
        source_locator={"database_type": "ORACLE", "target_name": "new"}, db_type="ORACLE", display_name="new")
    provider = AsyncMock()
    provider.discover_instances.return_value = InstanceDiscoveryResult(candidates=(current,))
    service._instance_discovery_adapter = AsyncMock(return_value=provider)
    service._create_source_binding_in_uow = AsyncMock()
    source_id = uuid7()
    uow = SimpleNamespace(diagnostic_sources=AsyncMock())
    uow.diagnostic_sources.get_scoped.return_value = SimpleNamespace(source_type="OEM")
    with pytest.raises(AIOpsApplicationError):
        asyncio.run(service._create_discovered_source_binding_in_uow(
            uow=uow, scope=SimpleNamespace(tenant_id=uuid7(), domain_id=1, actor_id=uuid7(), trace_id="test"),
            source_id=source_id, target_id=uuid7(), candidate_ref="opaque", now=datetime.now(UTC),
        ))
    service._create_source_binding_in_uow.assert_not_awaited()
