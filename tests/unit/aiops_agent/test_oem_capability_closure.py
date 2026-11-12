"""审计缺口的精确身份、独立通道、跨来源服务语义与分析回归。"""

import asyncio
import copy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem_exadata_components import confirm_components
from aiops_agent.adapters.diagnostic_sources.oem_native_emcli import query_native_emcli
from aiops_agent.adapters.diagnostic_sources.oracle_services import OracleServiceMetrics, validate_locator
from aiops_agent.adapters.diagnostic_sources.registry import DiagnosticSourceAdapterCatalog
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.application.oracle_service_catalog import service_metrics, mappings_for
from aiops_agent.application.oracle_service_chat import (
    OracleServiceChatQuery, OracleServiceChatReader, select_oracle_service_systems, analyze_service_samples,
)
from aiops_agent.application.oracle_service_monitoring import OracleServiceMonitoring
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.ports.diagnostic_source import NativeMetricQueryRequest
from platform_core.contracts.aiops.exadata import ExadataDiscoveryRequest
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeQuery
from platform_core.contracts.aiops.oracle_services import OracleServiceMetricMapping
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, metadata, request


NOW = datetime(2026, 11, 12, 12, tzinfo=UTC)
PARENT = {'target_id': 'a'*32, 'target_type': 'oracle_si_pdu', 'display_name': 'PDU-1'}
COMPONENT = {'target_id': 'b'*32, 'target_type': 'oracle_si_hw_component_pdu',
    'display_name': 'Module 0 Phase 1', 'parent_target_id': PARENT['target_id'], 'evidence_ref': 'OEM asset export SHA256:reviewed'}


def test_imported_component_keeps_admin_provenance_and_online_guid_type_name():
    value, calls = adapter({'id': 'b'*32, 'name': COMPONENT['display_name'], 'typeName': COMPONENT['target_type']})
    rows = asyncio.run(confirm_components(value, [PARENT], [COMPONENT]))
    assert rows == [PARENT, {**COMPONENT, 'membership_origin': 'IMPORT_CONFIRMED'}]
    assert calls[0][0].endswith('/targets/' + 'b'*32)
    assert ExadataDiscoveryRequest(root_id='c'*32, component_manifest=(COMPONENT,)).component_manifest[0].evidence_ref


@pytest.mark.parametrize('changes', [
    {'parent_target_id': 'c'*32}, {'target_type': 'oracle_si_network_port_switch'},
    {'target_id': 'a'*32}, {'target_id': '../escape'}, {'evidence_ref': '\nunsafe'},
])
def test_component_import_rejects_unknown_parent_wrong_type_duplicate_or_unsafe_identity(changes):
    value, calls = adapter()
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(confirm_components(value, [PARENT], [{**COMPONENT, **changes}]))
    assert not calls


def test_component_import_is_not_name_based_and_rechecks_each_time():
    for changed in [{'id': 'd'*32}, {'name': 'different'}, {'typeName': 'oracle_si_hw_component_switch'}]:
        value, _ = adapter({'id': 'b'*32, 'name': COMPONENT['display_name'], 'typeName': COMPONENT['target_type'], **changed})
        with pytest.raises(DiagnosticSourceAdapterError) as caught:
            asyncio.run(confirm_components(value, [PARENT], [COMPONENT]))
        assert caught.value.code == 'SOURCE_TARGET_MISMATCH'


def test_extended_catalog_55_fields_have_audited_types_dimensions_and_non_gauge_exclusions():
    _, _, metrics, profiles, diagnoses = load_catalog()
    pdu = [item for item in metrics.values() if item['target_type'] == 'oracle_si_hw_component_pdu']
    assert len(metrics) == 347 and len(pdu) == 13
    assert all(item['keys'] == ['ComponentId'] for item in pdu)
    for item in pdu:
        if 'Number' in item['selector'] or 'Threshold' in item['selector']:
            assert item['semantics'] == 'PROPERTY' and not item['baseline_supported']
        if item['selector'].endswith(('TotalInputEnergy', 'ModulePhaseEnergy')):
            assert item['semantics'] == 'CUMULATIVE' and not item['baseline_supported']
        assert any(item['code'] in profile['metric_codes'] for profile in profiles)
        assert any(item['code'] in profile['metric_codes'] for profile in diagnoses)


def native_body(value='12.50'):
    return {'Entity Type': 'oracle_database', 'Entity Name': 'mes-db', 'Metric Group Name': 'Filesystems',
        'isKeyBased': True, 'keyColumn_1_name': 'mountPoint', 'data_columns': [{'Metric Column Name': 'size',
            'keyValues': [{'key1': '/', 'values': [{'First Collection Time': '2026-11-12 11:50:00',
                'Last Collection Time': '2026-11-12 11:55:00', 'Value': value}]}]}]}


def native_emcli_adapter():
    value, _ = adapter({'targetName': 'mes-db', 'targetType': 'oracle_database'}, metadata())
    value.context = value.context.model_copy(update={'config': {'native_metric_transport': 'EMCLI', 'native_emcli_timezone': 'UTC'}})
    value._exadata_emcli_client = NS(read_native=AsyncMock(return_value=(native_body(), 'a'*64, 4096)))
    return value


@pytest.mark.parametrize('days', [1, 7])
def test_native_emcli_opt_in_reads_real_dimensions_precision_and_seven_day_boundary(days):
    value = native_emcli_adapter()
    with patch('aiops_agent.adapters.diagnostic_sources.oem_native_emcli.datetime') as clock:
        clock.now.return_value = NOW+timedelta(milliseconds=2)
        result = asyncio.run(value.query_native_metric(request(window_start=NOW-timedelta(days=days), window_end=NOW), metric_name='Filesystems/size', limit=1000))
    assert result.points == ((NOW-timedelta(minutes=5), '12.50'),)
    assert result.sample_dimensions == ({'mountPoint': '/'},)
    assert value._exadata_emcli_client.read_native.call_args.kwargs['days'] == min(7, days+1)


@pytest.mark.parametrize('days', [8, 30])
def test_native_emcli_never_relabels_short_history_as_month(days):
    value = native_emcli_adapter()
    with patch('aiops_agent.adapters.diagnostic_sources.oem_native_emcli.datetime') as clock:
        clock.now.return_value = NOW
        with pytest.raises(DiagnosticSourceAdapterError) as caught:
            asyncio.run(value.query_native_metric(request(window_start=NOW-timedelta(days=days), window_end=NOW), metric_name='Filesystems/size', limit=1000))
    assert caught.value.code == 'SOURCE_HISTORY_WINDOW_UNSUPPORTED'
    value._exadata_emcli_client.read_native.assert_not_awaited()


def test_native_emcli_has_independent_configuration_and_required_certificate_pin():
    catalog = DiagnosticSourceAdapterCatalog()
    exadata = {'exadata_metric_transport': 'EMCLI', 'exadata_emcli_timezone': 'UTC', 'exadata_emcli_certificate_sha256': 'a'*64}
    with pytest.raises(ValueError):
        catalog.normalize_config(source_type='OEM', config={**exadata, 'native_metric_transport': 'EMCLI'})
    assert catalog.normalize_config(source_type='OEM', config={**exadata, 'native_metric_transport': 'OEM_REST'})['native_metric_transport'] == 'OEM_REST'


def test_service_catalog_is_shared_and_never_borrows_database_or_other_provider_defaults():
    definitions = service_metrics()
    assert len(definitions) == 33
    assert len(mappings_for('ORACLE_ASM', 'OEM', 'osm_cluster', {})) == 16
    for source in ['PROMETHEUS', 'ZABBIX']:
        assert not any(mappings_for('ORACLE_ASM', source, None, {}).values())
    rac = mappings_for('ORACLE_RAC', 'OEM', 'rac_database', {})
    assert not rac['rac.clusterware'] and not rac['rac.instances']
    assert not any(mappings_for('ORACLE_ASM', 'OEM', 'oracle_database', {}).values())
    assert definitions['listener.established_per_minute']['unit'] == 'operations/min'
    assert definitions['listener.registered_services']['semantics'] == 'PROPERTY'


@pytest.mark.parametrize('overrides', [
    {'unknown': {}}, {'oracle_service_metrics': {'rac.clusterware': []}},
    {'oracle_service_metrics': {'asm.free_mb': [{'metric_name': 'free', 'dimensions': {}}]}},
    {'oracle_service_metrics': {'asm.free_mb': [{'metric_name': 'free', 'native_value_type': 'STRING', 'dimensions': {'dg': 'diskgroup'}}]}},
])
def test_service_mapping_rejects_unknown_cross_resource_and_wrong_complete_dimensions(overrides):
    with pytest.raises(ValueError):
        mappings_for('ORACLE_ASM', 'PROMETHEUS', None, overrides)


@pytest.mark.parametrize('source,key,locator', [
    ('OEM', 'a'*32, {'resource_scope': 'ORACLE_ASM', 'target_id': 'a'*32, 'target_type': 'oracle_database'}),
    ('PROMETHEUS', 'service', {'resource_scope': 'ORACLE_ASM', 'labels': {'job': 'oracle'}}),
    ('PROMETHEUS', 'service', {'resource_scope': 'ORACLE_RAC', 'labels': {'target_key': 'service'}}),
    ('ZABBIX', 'asm-host', {'resource_scope': 'ORACLE_ASM', 'host': 'other'}),
])
def test_service_locator_cannot_expand_frozen_identity(source,key,locator):
    with pytest.raises(DiagnosticSourceAdapterError):
        validate_locator(source, 'ORACLE_ASM', key, locator)


class ServiceAdapter(OracleServiceMetrics):
    def __init__(self, source_type):
        self.context = NS(source_type=source_type)
        self.verify_oracle_service = AsyncMock()
        self._service_prometheus_samples = AsyncMock(return_value=([({'__name__': 'asm_free_mb', 'target_key': 'asm-1', 'dg': 'DATA'}, [(NOW-timedelta(minutes=1), '12.125')])], 'a'*64))
        self._metric_host = AsyncMock(return_value=('12', 'b'*64))


def service_request():
    return NativeMetricQueryRequest(action_id='a0', target_id=str(uuid7()), binding_id=str(uuid7()), source_locator_key='asm-1',
        window_start=NOW-timedelta(hours=1), window_end=NOW, trace_id='test')


def test_prometheus_service_query_is_exact_and_preserves_each_dimension_without_affecting_old_profiles():
    value = ServiceAdapter('PROMETHEUS')
    mapping = OracleServiceMetricMapping(metric_name='asm_free_mb', native_unit='MB', dimensions={'dg': 'diskgroup'}, scale='1024')
    result = asyncio.run(value.query_oracle_service_metric(service_request(), resource_type='ORACLE_ASM',
        locator={'resource_scope': 'ORACLE_ASM', 'labels': {'target_key': 'asm-1'}}, mapping=mapping))
    assert value._service_prometheus_samples.call_args.args[1] == 'asm_free_mb{target_key="asm-1"}'
    assert result['points'][0][1] == '12416.000' and result['dimensions'] == ({'diskgroup': 'DATA'},)


@pytest.mark.parametrize('bad', ['sum(asm_free_mb)', 'asm_free_mb{job=~".*"}', 'asm_free_mb or up'])
def test_prometheus_service_mapping_does_not_allow_queries(bad):
    value = ServiceAdapter('PROMETHEUS')
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.query_oracle_service_metric(service_request(), resource_type='ORACLE_ASM',
            locator={'resource_scope': 'ORACLE_ASM', 'labels': {'target_key': 'asm-1'}}, mapping=OracleServiceMetricMapping(metric_name=bad)))
    value._service_prometheus_samples.assert_not_awaited()


@pytest.mark.parametrize('labels', [
    {'__name__': 'asm_free_mb', 'target_key': 'other', 'dg': 'DATA'},
    {'__name__': 'other', 'target_key': 'asm-1', 'dg': 'DATA'},
    {'__name__': 'asm_free_mb', 'target_key': 'asm-1', 'dg': 'DATA', 'instance': 'unscoped'},
])
def test_prometheus_service_rejects_foreign_series_and_unreviewed_labels(labels):
    value = ServiceAdapter('PROMETHEUS')
    value._service_prometheus_samples.return_value = ([(labels, [(NOW-timedelta(minutes=1), '1')])], 'a'*64)
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.query_oracle_service_metric(service_request(), resource_type='ORACLE_ASM',
            locator={'resource_scope': 'ORACLE_ASM', 'labels': {'target_key': 'asm-1'}},
            mapping=OracleServiceMetricMapping(metric_name='asm_free_mb', dimensions={'dg': 'diskgroup'})))


def test_zabbix_service_uses_precise_item_and_explicit_fixed_diskgroup_not_host_guessing():
    value = ServiceAdapter('ZABBIX')
    value.query_native_metric = AsyncMock(return_value=NS(points=((NOW-timedelta(minutes=1), '512'),), unit='MB',
        native_value_type='3', captured_at=NOW, truncated=False, response_sha256='c'*64))
    result = asyncio.run(value.query_oracle_service_metric(service_request(), resource_type='ORACLE_ASM',
        locator={'resource_scope': 'ORACLE_ASM', 'host': 'asm-1'},
        mapping=OracleServiceMetricMapping(metric_name='oracle.asm.free[DATA]', native_unit='MB', fixed_dimensions={'diskgroup': 'DATA'})))
    assert result['dimensions'] == ({'diskgroup': 'DATA'},)
    assert value.query_native_metric.call_args.kwargs['metric_name'] == 'oracle.asm.free[DATA]'


def frozen(source='OEM'):
    return {'scope': 'ORACLE_ASM', 'target_type': 'osm_cluster', 'source': {'source_type': source},
        'mapping_overrides': {}, 'binding_id': uuid7(), 'version': 1, 'locator': 'a'*32,
        'source_locator': {'resource_scope': 'ORACLE_ASM', 'target_type': 'osm_cluster', 'target_id': 'a'*32},
        'max_response_bytes': 262144}


def test_service_baseline_is_by_complete_dimensions_and_stale_current_cannot_be_analyzed():
    f = frozen()
    now = datetime.now(UTC)
    read = AsyncMock(return_value=([(now-timedelta(hours=2), '10')], [{'diskgroup': 'DATA'}], now, False, 'a'*64))
    native = NS(_adapter=AsyncMock(return_value=NS()), _unchanged=AsyncMock(), _response=lambda result, _: result)
    service = OracleServiceMonitoring(native); service.read = read
    service.read_baseline = AsyncMock(return_value=([
        {'dimensions': {'diskgroup': 'DATA'}, 'total': '30', 'sample_count': 2},
        {'dimensions': {'diskgroup': 'RECO'}, 'total': '90', 'sample_count': 1}], False, 'b'*64))
    result = asyncio.run(service.samples(scope=NS(trace_id='test'), source_id=uuid7(), instance_id=uuid7(), frozen=f,
        query=MonitoringNativeQuery(metric_name='asm.free_mb', baseline='1w')))
    assert result.points[0].current_eligible is False
    assert {tuple(row.dimensions.items()): Decimal(row.average) for row in result.baseline.series} == {
        (('diskgroup', 'DATA'),): Decimal(15), (('diskgroup', 'RECO'),): Decimal(90)}
    assert '不作为当前诊断证据' in analyze_service_samples(result)[0]


def test_counter_and_state_never_accept_mean_baseline():
    service = OracleServiceMonitoring(NS())
    for code in ['asm.reads_total', 'asm.disk_mode', 'asm.rebalancing']:
        with pytest.raises(AIOpsApplicationError):
            asyncio.run(service.samples(scope=NS(trace_id='test'), source_id=uuid7(), instance_id=uuid7(), frozen=frozen(),
                query=MonitoringNativeQuery(metric_name=code, baseline='1w', mode='LATEST')))


def test_service_agent_preserves_explicit_unauthorized_resource_and_never_queries_database():
    query = {'resource_type_key': 'ORACLE_ASM', 'query_kind': 'METRIC', 'metric_codes': ['asm.free_mb'], 'resource_names': ['unauthorized']}
    systems = [{'display_name': 'test', 'resources': [{'resource_type_key': 'ORACLE_ASM', 'display_name': 'authorized', 'resource_code': 'ASM-1'}]}]
    assert not select_oracle_service_systems(systems, query)
    reader = OracleServiceChatReader(NS())
    result = asyncio.run(reader.read(parameters=query, plan=NS(), resources=[], scope=NS(), analyze=True))
    assert result.gap_code == 'ORACLE_SERVICE_SCOPE_EMPTY'
    with pytest.raises(ValueError):
        OracleServiceChatQuery(resource_type_key='ORACLE_ASM', metric_codes=['rac.clusterware'])


def test_numeric_service_states_compare_exact_decimal_and_reject_equivalent_dictionary_keys():
    from platform_core.contracts.aiops.oracle_services import state_key
    assert state_key('1.00', 'NUMBER') == state_key('1', 'NUMBER')
    assert state_key('1.00000000000000000001', 'NUMBER') != state_key('1', 'NUMBER')
    assert state_key('Up', 'STRING') != state_key('UP', 'STRING')
    with pytest.raises(ValueError):
        OracleServiceMetricMapping(metric_name='status', state_values={
            '1': {'label': '可用', 'tone': 'success'}, '1.0': {'label': '其他', 'tone': 'danger'}})


def test_prometheus_service_reads_original_scrape_timestamp_instead_of_evaluation_time():
    from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
    from tests.unit.aiops_agent.test_aiops_oem_native_metrics import Response
    payloads = [{'status': 'success', 'data': [{'__name__': 'asm_free_mb', 'target_key': 'asm-1'}]},
        {'status': 'success', 'data': {'resultType': 'matrix', 'result': [
            {'metric': {'__name__': 'asm_free_mb', 'target_key': 'asm-1', 'dg': 'DATA'},
             'values': [[(NOW-timedelta(minutes=35)).timestamp(), '10']]}]}}]
    calls = []
    def get(url, **kwargs):
        calls.append((url, kwargs))
        return Response(payloads.pop(0))
    value = PrometheusAdapter(context=NS(endpoint='http://prom.test', credentials={}, source_type='PROMETHEUS'),
        session=NS(get=get), request_timeout_seconds=5, webhook_replay_seconds=60)
    result = asyncio.run(value.query_oracle_service_metric(service_request(), resource_type='ORACLE_ASM',
        locator={'resource_scope': 'ORACLE_ASM', 'labels': {'target_key': 'asm-1'}},
        mapping=OracleServiceMetricMapping(metric_name='asm_free_mb', native_unit='MB', dimensions={'dg': 'diskgroup'})))
    assert result['points'] == ((NOW-timedelta(minutes=35), '10'),)
    assert calls[1][0].endswith('/api/v1/query')
    assert calls[1][1]['params']['query'] == 'asm_free_mb{target_key="asm-1"}[3601s]'


def test_imported_pdu_component_latest_uses_guid_rest_not_unsupported_emcli_target_name():
    from aiops_agent.adapters.diagnostic_sources.oem_exadata import OEMExadata
    definition = load_catalog()[2]['oracle_si_hw_component_pdu.PduModulePhaseStatus.ModulePhaseVoltage']
    value = NS(context=NS(config={'exadata_metric_transport': 'EMCLI'}, declared_capabilities={'metric.query_range': {}}))
    with patch('aiops_agent.adapters.diagnostic_sources.oem_exadata.query_latest', new=AsyncMock(return_value={'points': ()})) as latest:
        asyncio.run(OEMExadata.exadata_latest.__wrapped__.__wrapped__(value, NS(), definition, 'oracle_si_hw_component_pdu'))
    assert latest.call_args.kwargs['expected_target_type'] == 'oracle_si_hw_component_pdu'


@pytest.mark.parametrize('source,name', [('PROMETHEUS', 'sum(asm_free_mb)'), ('OEM', 'bad-selector'), ('OEM', 'DiskGroup/free mb')])
def test_invalid_mapping_is_rejected_when_configuration_is_saved(source, name):
    with pytest.raises(ValueError):
        mappings_for('ORACLE_ASM', source, 'osm_cluster', {'oracle_service_metrics': {
            'asm.free_mb': [{'metric_name': name, 'dimensions': {'dg': 'diskgroup'}}]}})


def test_zabbix_service_half_open_projection_never_counts_end_boundary():
    value = ServiceAdapter('ZABBIX')
    value.query_native_metric = AsyncMock(return_value=NS(points=((NOW, '999'), (NOW-timedelta(seconds=1), '2')),
        unit='MB', native_value_type='3', captured_at=NOW, truncated=False, response_sha256='c'*64))
    result = asyncio.run(value.query_oracle_service_metric(service_request(), resource_type='ORACLE_ASM',
        locator={'resource_scope': 'ORACLE_ASM', 'host': 'asm-1'},
        mapping=OracleServiceMetricMapping(metric_name='asm.free', native_unit='MB', fixed_dimensions={'diskgroup': 'DATA'})))
    assert result['points'] == ((NOW-timedelta(seconds=1), '2'),)


def test_prometheus_baseline_aggregates_all_samples_without_thousand_point_limit():
    from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
    from tests.unit.aiops_agent.test_aiops_oem_native_metrics import Response
    payloads = [{'status': 'success', 'data': {'resultType': 'vector', 'result': [
        {'metric': {'target_key': 'asm-1', 'dg': 'DATA'}, 'value': [NOW.timestamp()-0.000001, number]}]}}
        for number in ['20160', '10080']]
    calls = []
    def get(url, **kwargs):
        calls.append(kwargs['params'])
        return Response(payloads.pop(0))
    value = PrometheusAdapter(context=NS(endpoint='http://prom.test', credentials={}, source_type='PROMETHEUS'),
        session=NS(get=get), request_timeout_seconds=5, webhook_replay_seconds=60)
    value.verify_oracle_service = AsyncMock()
    result = asyncio.run(value.summarize_oracle_service_metric(service_request().model_copy(update={'window_start': NOW-timedelta(days=7)}),
        resource_type='ORACLE_ASM', locator={'resource_scope': 'ORACLE_ASM', 'labels': {'target_key': 'asm-1'}},
        mapping=OracleServiceMetricMapping(metric_name='asm_free_mb', native_unit='MB', dimensions={'dg': 'diskgroup'}, scale='1024')))
    assert Decimal(result['series'][0]['total']) == Decimal(20160)*1024
    assert result['series'][0]['sample_count'] == 10080
    assert result['series'][0]['dimensions'] == {'diskgroup': 'DATA'}
    assert calls[0]['query'] == 'sum_over_time(asm_free_mb{target_key="asm-1"}[604800s])'
    assert calls[1]['query'].startswith('count_over_time(')


def test_zabbix_service_baseline_streams_overlap_and_excludes_end_without_changing_ordinary_history():
    from aiops_agent.adapters.diagnostic_sources.zabbix import ZabbixAdapter
    value = object.__new__(ZabbixAdapter)
    value.context = NS()
    value._metric_host = AsyncMock(return_value=('12', 'a'*64))
    item = {'itemid': '1', 'hostid': '12', 'key_': 'asm.free', 'value_type': '0', 'status': '0', 'state': '0', 'units': 'MB'}
    start = NOW-timedelta(hours=1)
    rows = [{'itemid': '1', 'clock': str(int(start.timestamp())+i), 'ns': '0', 'value': '2'} for i in range(1001)]
    end = {'itemid': '1', 'clock': str(int(NOW.timestamp())), 'ns': '0', 'value': '999'}
    value._rpc = AsyncMock(side_effect=[([item], 'b'*64), (rows[:1000], 'c'*64), (rows[999:]+[end], 'd'*64)])
    result = asyncio.run(value.query_native_metric(service_request(), metric_name='asm.free', limit=1000, _service_summary_only=True))
    assert not result['truncated'] and result['series'][0]['sample_count'] == 1001
    assert Decimal(result['series'][0]['total']) == 2002
    assert value._rpc.call_args_list[-1].args[1]['time_from'] == int(start.timestamp())+999


def test_component_history_uses_guid_rest_even_if_parent_transport_is_emcli():
    from aiops_agent.adapters.diagnostic_sources.oem_exadata import OEMExadata
    definition = load_catalog()[2]['oracle_si_hw_component_pdu.PduModulePhaseStatus.ModulePhaseVoltage']
    value = NS(context=NS(config={'exadata_metric_transport': 'EMCLI'}, declared_capabilities={'metric.query_range': {}}),
        query_native_metric=AsyncMock(return_value={'points': ()}))
    asyncio.run(OEMExadata.exadata_history.__wrapped__.__wrapped__(value, NS(), definition, 'oracle_si_hw_component_pdu'))
    assert value.query_native_metric.call_args.kwargs['_expected_target_type'] == 'oracle_si_hw_component_pdu'
