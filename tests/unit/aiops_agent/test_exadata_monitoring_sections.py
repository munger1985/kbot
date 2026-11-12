"""整机分区与设备详情共用审核目录；不扩大已确认成员和诊断边界。"""

import asyncio

from platform_core.contracts.aiops.exadata import ExadataQuery
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from tests.unit.aiops_agent.test_exadata_oem import exadata_lab, publish_exadata, args
from tests.unit.aiops_agent.test_business_system_publication_lifecycle import lab


def test_db_server_profile_is_exact_type_scoped_and_matches_diagnosis():
    _, _, metrics, monitoring, diagnosis = load_catalog()
    profile = next(item for item in monitoring if item["id"] == "dbserver")
    assert profile == next(item for item in diagnosis if item["id"] == "dbserver")
    assert {metrics[code]["target_type"] for code in profile["metric_codes"]} == {"host", "oracle_si_server_map"}
    assert {"host.CPUUsageSummary.OverAllUsagePct", "host.Load.memUsedPct"} <= set(profile["metric_codes"])
    assert ExadataQuery(scene="dbserver").scene == "dbserver"


def test_cell_first_page_includes_services_resources_and_temperature():
    _, _, _, monitoring, _ = load_catalog()
    codes = next(item for item in monitoring if item["id"] == "cell")["metric_codes"][:12]
    assert {"oracle_exadata.CellSrv_Status_Metric.Status", "oracle_exadata.CellSrv_Status_Metric.msStatus",
        "oracle_exadata.CellSrv_Status_Metric.rsStatus", "oracle_exadata.Cell_Metric.temperature",
        "oracle_exadata.Cell_Metric.cpu_utilization", "oracle_exadata.Cell_Metric.memory_utilization"} <= set(codes)


def test_db_server_scene_does_not_read_cell_or_guess_members(exadata_lab):
    f = exadata_lab
    async def run():
        await publish_exadata(f)
        view = await f.exadata.collect(**args(f, scene="dbserver"))
        assert not view.samples and view.total_queries == 0
        assert any(gap.code == "NO_APPLICABLE_METRICS" for gap in view.gaps)
        f.adapter.exadata_latest.assert_not_awaited()
    asyncio.run(run())


def test_original_unsupported_code_is_not_rewritten_as_empty_or_health(exadata_lab):
    f = exadata_lab
    f.adapter.exadata_latest.side_effect = DiagnosticSourceAdapterError("SOURCE_QUERY_UNSUPPORTED", "原厂查询不支持")
    async def run():
        await publish_exadata(f)
        view = await f.exadata.collect(**args(f, scene="overview", limit=1))
        assert not view.samples
        gap = next(item for item in view.gaps if item.metric_code)
        assert gap.code == "SOURCE_QUERY_UNSUPPORTED"
        assert "不支持" in gap.detail
        assert not any(item.code == "NO_DATA" for item in view.gaps)
    asyncio.run(run())
