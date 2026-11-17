"""既有数据库容器的显式日志白名单不依赖重建容器或cAdvisor。"""

import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path
import re
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
LOADER = SourceFileLoader("aiops_stack_log_containers", str(ROOT / "scripts/aiops-stack"))
SPEC = importlib.util.spec_from_loader(LOADER.name, LOADER)
STACK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = STACK
SPEC.loader.exec_module(STACK)


def settings(tmp_path, name="db.mysql-57", logs=True, duplicate=False):
    path = tmp_path / "aiops-stack.ini"
    text = (
        "[deployment]\ndeployment_id = test-logs\nrole = all-in-one\nlocal_access = true\n"
        "[metrics]\nenabled = true\n"
        f"[logs]\n{'enabled = true' if logs else '# enabled = true'}\n"
        "[mysql:mysql-test]\nenabled = true\nversion = 5.7\nhost = mysql-test\n"
        f"username = monitor\npassword = test-only\nlog_container = {name}\n"
        "[postgres:pg-test]\nenabled = true\nversion = 16\nuri = pg-test:5432/test\n"
        f"username = monitor\npassword = test-only\nlog_container = {name if duplicate else 'db-postgres-16'}\n"
    )
    path.write_text(text)
    path.chmod(0o600)
    return STACK._load_settings(path)


def test_explicit_whitelist_collects_without_cadvisor_or_container_changes(tmp_path):
    config = settings(tmp_path)
    STACK._prepare_runtime(config)
    generated = json.loads((config.runtime_dir / "compose.generated.yaml").read_text())
    assert "/var/run/docker.sock:/var/run/docker.sock:ro" in generated["services"]["alloy"]["volumes"]
    assert "cadvisor" not in STACK._selected_services(config)
    assert not {"mysql57", "postgres16"}.intersection(generated["services"])
    alloy = (config.runtime_dir / "alloy/config.alloy").read_text()
    assert 'replacement = "mysql-test"' in alloy
    assert 'replacement = "pg-test"' in alloy
    assert 'replacement = "mysql_error"' in alloy
    assert 'replacement = "postgresql_log"' in alloy
    assert json.dumps('/db\\.mysql\\-57') in alloy
    assert 'action = "keep"' in alloy
    assert '__EXPLICIT_CONTAINER_RULES__' not in alloy
    expression = re.search(r'expression = `([^`]+)`', STACK._docker_log_alloy_config(config))[1]
    assert re.search(expression, '2026-11-16T12:00:00Z 0 [ERROR] test')["severity"] == "ERROR"


@pytest.mark.parametrize("name", ["../mysql", "mysql|other", "*", "mysql other"])
def test_rejects_non_exact_container_names(tmp_path, name):
    with pytest.raises(ValueError, match="log_container"):
        settings(tmp_path, name=name)


def test_explicit_container_requires_logs(tmp_path):
    with pytest.raises(ValueError, match="启用"):
        settings(tmp_path, logs=False)


def test_container_cannot_belong_to_multiple_targets(tmp_path):
    with pytest.raises(ValueError, match="多个"):
        settings(tmp_path, duplicate=True)
