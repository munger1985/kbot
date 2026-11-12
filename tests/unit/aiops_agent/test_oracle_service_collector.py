"""可选采集器仅解释受控原厂输出，未知和失败不产生健康值。"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

path = Path(__file__).resolve().parents[3] / 'scripts/deployment/aiops_oracle_services/collector.py'
spec = importlib.util.spec_from_file_location('oracle_service_collector', path)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def test_listener_dynamic_ready_is_not_static_unknown():
    text = '''Service "APP" has 1 instance(s).
  Instance "DB1", status READY, has 1 handler(s) for this service...
Service "STATIC" has 1 instance(s).
  Instance "DB1", status UNKNOWN, has 1 handler(s) for this service...
The command completed successfully'''
    result = collector.parse_listener_services(text, ['APP', 'STATIC', 'MISSING'])
    assert [(item['dimensions']['service'], item['value']) for item in result] == [('APP', '1'), ('MISSING', '0')]
    with pytest.raises(ValueError):
        collector.parse_listener_services('unknown partial output', ['APP'])


@pytest.mark.parametrize('code,text,expected', [(0, 'OK (12 msec)\n', [('listener.availability', '1'), ('listener.tns_ping_ms', '12')]),
    (1, 'TNS-12541: TNS:no listener', [('listener.availability', '0')])])
def test_tns_response_only_uses_verified_protocol_tokens(code, text, expected):
    replies = iter([(code, text), (1, 'unknown')])
    result = collector.collect_listener({'tns_alias': 'APP', 'listener_name': 'LISTENER', 'oracle_home': '/oracle'}, command=lambda *_: next(replies))
    assert [(item['code'], item['value']) for item in result] == expected


def test_missing_tns_alias_is_not_a_service_down_value():
    with pytest.raises(ValueError):
        collector.collect_listener({'tns_alias': 'APP', 'listener_name': 'LISTENER', 'oracle_home': '/oracle'}, command=lambda *_: (1, 'TNS-03505: Failed to resolve name'))


def test_metrics_preserve_precise_values_dimensions_and_counter_kind():
    text = collector.prometheus({'target_key': 'asm-1', 'resource_type': 'ORACLE_ASM', 'samples': [
        collector.row('asm.reads_total', '90071992547409931', {'diskgroup': 'DATA', 'inst_id': '1'})]})
    assert ' counter\n' in text and '90071992547409931' in text and 'target_key="asm-1"' in text
    assert 'inst_id="1"' in text and 'diskgroup="DATA"' in text
    with pytest.raises(ValueError):
        collector.row('asm.free_mb', 'NaN')


def test_config_rejects_public_secrets_and_symlinks(tmp_path):
    path = tmp_path / 'private.json'
    path.write_text('{"resource_type":"ORACLE_ASM","target_key":"asm-1"}')
    path.chmod(0o644)
    with pytest.raises(ValueError):
        collector.config(path)
    path.chmod(0o600)
    assert collector.config(path)['target_key'] == 'asm-1'
    link = tmp_path / 'link.json'; link.symlink_to(path)
    with pytest.raises(ValueError):
        collector.config(link)


def test_prometheus_label_escaping_is_protocol_not_json_control_escapes():
    document = {'target_key': 'asm-1', 'resource_type': 'ORACLE_ASM', 'samples': [
        collector.row('asm.free_mb', 1, {'diskgroup': '中文"\\\nDATA'})]}
    assert '中文\\"\\\\\\nDATA' in collector.prometheus(document)
    document['samples'][0]['dimensions']['diskgroup'] = 'DATA\tunsafe'
    with pytest.raises(ValueError):
        collector.prometheus(document)


def test_database_with_same_name_is_not_accepted_as_asm(monkeypatch):
    connection = NS(call_timeout=0)
    class Connection:
        def __enter__(self): return connection
        def __exit__(self, *_): return False
    replies = iter([[('+ASM',)], [('RDBMS',)]])
    monkeypatch.setattr(collector, 'sql_rows', lambda *_: next(replies))
    with pytest.raises(ValueError, match='不是 ASM'):
        collector.collect_sql({'resource_type': 'ORACLE_ASM', 'instance_name': '+ASM',
            'username': 'test', 'password': 'test', 'dsn': 'test'}, connect=lambda **_: Connection())
