"""服务资源发布、持久化绑定、实时读取及授权 Agent 的贯通回归。"""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.orm import Session

from aiops_agent.application.oracle_service_chat import OracleServiceChatReader
from aiops_agent.entities import DiagnosticSourceEntity
from platform_core.contracts.aiops import BusinessSystemResourceGraphCommit
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeQuery
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab, publish
from tests.unit.aiops_agent.test_monitoring_business_system_catalog import service_for


@pytest.mark.parametrize('source_type', ['OEM', 'PROMETHEUS', 'ZABBIX'])
@pytest.mark.parametrize('resource_type,code,metric,dimensions', [
    ('ORACLE_ASM', 'asm.free_mb', 'asm_free_mb', {'dg': 'diskgroup'}),
    ('ORACLE_RAC', 'rac.response_ms', 'rac_response_ms', {'db': 'database', 'svc': 'service'}),
    ('ORACLE_LISTENER', 'listener.tns_ping_ms', 'listener_tns_ping_ms', {}),
])
def test_service_publication_catalog_samples_and_agent_share_one_binding(lab, source_type, resource_type, code, metric, dimensions):
    key = 'a'*32 if source_type == 'OEM' else 'service-01'
    target_type = {'ORACLE_ASM': 'osm_cluster', 'ORACLE_RAC': 'rac_database', 'ORACLE_LISTENER': 'oracle_listener'}[resource_type]
    locator = {'resource_scope': resource_type} | ({'target_id': key, 'target_type': target_type} if source_type == 'OEM'
        else {'labels': {'target_key': key}} if source_type == 'PROMETHEUS' else {'host': key})
    native_mapping = {'metric_name': 'Service/' + metric if source_type == 'OEM' else metric, 'native_unit': 'MB' if resource_type == 'ORACLE_ASM' else 'ms',
        'dimensions': dimensions if source_type != 'ZABBIX' else {},
        'fixed_dimensions': {} if source_type != 'ZABBIX' else {value: 'SIM-' + value for value in dimensions.values()}}
    with Session(lab.engine) as session:
        source = session.get(DiagnosticSourceEntity, lab.source_id)
        source.source_type, source.adapter_id = source_type, source_type.lower()
        source.connectivity_status = 'CONNECTED'
        source.endpoint = 'https://source.test'
        source.declared_capabilities_json = {'metric.query_range': {}}
        session.commit()
    raw = lab.request.model_dump(mode='json')
    raw['resources'] = [{'draft_resource_id': str(uuid7()), 'resource_type_key': resource_type,
        'display_name': 'SIM 基础服务', 'owner_group_id': str(lab.request.system.responsibility_group_id),
        'architecture_layer_key': 'DATA', 'monitoring': {'source_id': str(lab.source_id),
            'source_binding': {'source_id': str(lab.source_id), 'source_locator_key': key, 'source_locator': locator,
                'mapping_overrides': {'oracle_service_metrics': {code: [native_mapping]}}}}}]
    lab.request = BusinessSystemResourceGraphCommit.model_validate(raw)
    now = datetime.now(UTC)
    labels = {value: 'SIM-' + value for value in dimensions.values()}
    adapter = NS(context=NS(declared_capabilities={'metric.query_range': {}}),
        verify_oracle_service=AsyncMock(), query_oracle_service_metric=AsyncMock(return_value={
            'points': ((now-timedelta(minutes=1), '12.50'),), 'dimensions': (labels,), 'unit': native_mapping['native_unit'],
            'native_value_type': 'NUMBER', 'captured_at': now, 'truncated': False, 'response_sha256': 'b'*64}))
    lab.service._instance_discovery_adapter = AsyncMock(return_value=adapter)
    monitoring = service_for(lab)
    monitoring.native._adapter = AsyncMock(return_value=adapter)

    async def run():
        detail = await publish(lab)
        saved = detail.resources[0].resource
        assert saved.database is None and saved.monitoring_bindings[0].source_locator == locator
        systems = await monitoring.list_business_systems(scope=lab.scope)
        resource = systems[0].resources[0]
        assert resource.resource_type_key == resource_type and resource.sources[0].monitoring_readiness == 'READY'
        catalog = await monitoring.native.catalog(scope=lab.scope, source_id=lab.source_id, instance_id=resource.instance_id)
        assert next(p for p in catalog.overview_panels if p.panel_id == code).metric.name == code
        samples = await monitoring.native.samples(scope=lab.scope, source_id=lab.source_id, instance_id=resource.instance_id,
            query=MonitoringNativeQuery(metric_name=code))
        assert samples.points[0].value == '12.50' and samples.points[0].current_eligible
        plan = NS(time_range=None, budget=NS(max_live_queries=4, timeout_seconds=10))
        result = await OracleServiceChatReader(monitoring).read(parameters={'resource_type_key': resource_type,
            'metric_codes': [code], 'baseline_period': None}, plan=plan, scope=lab.scope, analyze=True,
            resources=[{'business_system_id': str(detail.business_system_id), 'resource_id': str(saved.resource_id),
                'resource_type_key': resource_type, 'display_name': saved.display_name}])
        assert result.gap_code is None and len(result.evidence) == 1
        assert result.evidence[0]['source_id'] == str(lab.source_id)
        assert '没有可用的同来源同维度均值' in result.markdown
    asyncio.run(run())
