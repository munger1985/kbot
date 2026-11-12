"""2026-11-10 真实 OEM 归档发现的配置、PDB、状态和时效回归。"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from aiops_agent.adapters.diagnostic_sources.registry import DiagnosticSourceAdapterCatalog
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.application.exadata_chat import ExadataChatQuery, exadata_parameter_contract
from aiops_agent.application.monitoring_native import public_points
from aiops_agent.application.monitoring_oem_overview import overview_panels, freshness_seconds
from platform_core.contracts.aiops.monitoring_native import MonitoringNativeMetric
from tests.unit.aiops_agent.test_aiops_oem_standard_metrics import configured, mapping, request


def test_shared_oem_configuration_does_not_invalidate_standard_metrics():
    value, calls = configured()
    value.context = value.context.model_copy(update={"config": {**value.context.config,
        "exadata_metric_transport": "OEM_REST"}})
    result = asyncio.run(value.query_metrics(request()))
    assert result.observations and not result.gaps and len(calls) == 3
    normalized = DiagnosticSourceAdapterCatalog.normalize_config(source_type="OEM", config=value.context.config)
    assert normalized["standard_metric_mappings"]["db.storage.used"] == {**mapping(), "filters": {}}
    with pytest.raises(ValueError):
        DiagnosticSourceAdapterCatalog.normalize_config(source_type="OEM", config={**normalized, "unknown": True})


def test_pdb_profile_uses_its_own_guid_metadata_not_cdb_metric_names():
    # 归档去重定义：保留类型、单位编码所对应的显示单位和全部维度；无目标名或客户数值。
    rows = [("Response/State", "STRING", None, ()),
        ("waitcnt/cpu_active_sessions", "NUMBER", "n/a", ()),
        ("waitcnt/io_active_sessions", "NUMBER", "n/a", ()),
        ("waitcnt/other_active_sessions", "NUMBER", "n/a", ()),
        ("problemTbsp/pctUsed", "NUMBER", "%", ("name",)),
        ("problemTbspTemp/pctUsed", "NUMBER", "%", ("name",)),
        ("tbspAllocation/spaceAllocated", "NUMBER", "MB", ("name",)),
        ("tbspAllocation/spaceUsed", "NUMBER", "MB", ("name",))]
    metrics = [MonitoringNativeMetric(name=name, display_name=name, group=name.split("/")[0],
        native_value_type=kind, unit=unit, dimensions=keys, query_supported=kind == "NUMBER")
        for name, kind, unit, keys in rows]
    panels = overview_panels("DATABASE", metrics, target_type="oracle_pdb")
    assert len(panels) == 8 and all(panel.metric for panel in panels)
    assert not panels[0].baseline_supported
    assert all(panel.metric is None for panel in overview_panels("DATABASE", metrics, target_type="oracle_database"))
    assert freshness_seconds("DATABASE", metrics[1], target_type="oracle_pdb") == 1800
    assert freshness_seconds("DATABASE", metrics[1], target_type="oracle_database") is None


def test_unprofiled_metrics_are_not_advertised_or_accepted_by_agent():
    _, _, metrics, profiles, _ = load_catalog()
    active = {code for profile in profiles for code in profile["metric_codes"]}
    assert {item["code"] for item in exadata_parameter_contract()["metric_catalog"]} == active
    for code in metrics.keys() - active:
        with pytest.raises(ValidationError):
            ExadataChatQuery(resource_type_key="EXADATA", metric_codes=(code,))


@pytest.mark.parametrize("latest", [True, False])
@pytest.mark.parametrize("truncated", [True, False])
def test_old_and_truncated_samples_never_become_current_comparisons(latest, truncated):
    now = datetime(2026, 11, 12, 10, tzinfo=UTC)
    points = [(now - timedelta(hours=3), "20"), (now - timedelta(minutes=5), "21"), (now, None)]
    result = public_points(points, ({}, {}, {}), captured_at=now, max_age=1800, truncated=truncated, latest=latest)
    assert [point.current_eligible for point in result] == [False, not truncated, False]
    assert result[0].value == "20"
    assert result[0].quality == ("TRUNCATED" if truncated else "STALE" if latest else "OBSERVED")
    assert result[-1].quality == ("TRUNCATED" if truncated else "EMPTY")
    unreviewed = public_points(points, ({}, {}, {}), captured_at=now, max_age=None, truncated=False, latest=latest)
    assert not any(point.current_eligible for point in unreviewed)


def test_cumulative_metrics_do_not_support_baselines_or_implicit_rates():
    groups = {"FlashCache_Metric": {"all_bytes", "all_reqs", "fc_io_rq_w_first", "fc_io_rq_w_overwrite",
        "fc_io_rq_w_skip", "keep_bytes", "keep_miss_reqs", "keep_reqs", "miss_bytes", "miss_reqs"},
        "FlashLog_Metric": {"flashlog_disk_io_errs", "flashlog_flash_io_errs"},
        "Flash_Disk_IORM_DB_Metric": {"db_fd_io_tm"}, "IORMDB_Metric": {"db_fd_io_tm", "db_io_tm_lg", "db_io_tm_sm"}}
    audited = [metric for metric in load_catalog()[2].values()
        if metric["selector"].split("/")[1] in groups.get(metric["selector"].split("/")[0], set())]
    assert len(audited) == 16
    assert all(metric["semantics"] == "CUMULATIVE" and not metric["baseline_supported"] for metric in audited)
