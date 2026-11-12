"""可用性、管理状态和事件标志的审核目录，不把文本列统统当状态。"""

from aiops_agent.application.exadata_catalog import load_catalog
from platform_core.contracts.aiops.exadata import ExadataMetricDefinition


def test_all_reviewed_status_metrics_have_typed_dictionary_and_provenance():
    _, _, metrics, _, _ = load_catalog()
    statuses = {code: item for code, item in metrics.items() if item.get("status")}
    assert len(statuses) == 16
    for item in metrics.values():
        definition = ExadataMetricDefinition.model_validate(item)
        if definition.status:
            assert definition.status.provenance
            assert definition.status.values
    for name in ("Status", "msStatus", "rsStatus"):
        status = statuses[f"oracle_exadata.CellSrv_Status_Metric.{name}"]["status"]
        assert status["values"]["1"] == {"label": "可用", "tone": "success"}
        assert status["values"]["0"] == {"label": "不可用", "tone": "danger"}
        assert status["values"]["2"] == {"label": "未知", "tone": "unknown"}
        assert "oracle_exadata_cellsrv_status.pl" in status["provenance"]


def test_boolean_down_event_is_not_current_availability_and_native_type_is_unchanged():
    metrics = load_catalog()[2]
    for target in ("server", "switch"):
        event = metrics[f"oracle_si_network_port_{target}.NetworkPortPerformance.OperStatusDown"]
        assert event["native_value_type"] == "NUMBER"
        assert event["status"]["category"] == "EVENT"
        assert event["status"]["values"]["0"] == {"label": "未触发", "tone": "neutral"}
        assert event["status"]["values"]["1"]["label"] == "已触发"
        admin = metrics[f"oracle_si_network_port_{target}.NetworkPortPerformance.AdministrationState"]["status"]
        assert admin["category"] == "ADMINISTRATIVE"
        assert admin["values"]["down"] == {"label": "已禁用", "tone": "neutral"}
    led = metrics["oracle_exadata.Cell_Metric.led_status"]["status"]
    assert led["category"] == "INDICATOR"
    assert led["values"]["off"] == {"label": "未点亮", "tone": "neutral"}


def test_non_status_strings_remain_native_text_including_numeric_looking_capacity():
    metrics = load_catalog()[2]
    for selector in ("Cell_Metric.cell_name", "Aggregated_Exadata_Diskgroup_Capacity_Metric.size",
                     "Aggregated_Exadata_Diskgroup_Capacity_Metric.count", "CellDisk_Metric.flash_type"):
        item = metrics[f"oracle_exadata.{selector}"]
        assert item["native_value_type"] == "STRING"
        assert ExadataMetricDefinition.model_validate(item).status is None
    for target in ("server", "switch"):
        detail = metrics[f"oracle_si_hw_component_{target}.ComponentState.ErrorDetail"]
        assert detail["native_value_type"] == "STRING" and not detail.get("status")
    unavailable_buffer = metrics["oracle_exadata.FlashLog_Metric.flashlog_io_w_skip_no_buffer"]
    assert not unavailable_buffer.get("status")


def test_state_definitions_are_frozen_in_view_and_unknown_tokens_are_not_invented():
    metrics = load_catalog()[2]
    disk = ExadataMetricDefinition.model_validate(metrics["oracle_exadata.Exadata_Disk_Status_Metric.status"])
    assert disk.model_dump(mode="json")["status"]["values"]["normal"]["label"] == "正常"
    assert "0" not in disk.status.values and "1" not in disk.status.values
    assert "future-failure-code" not in disk.status.values
