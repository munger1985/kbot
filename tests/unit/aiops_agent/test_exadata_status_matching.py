"""真实 OEM NUMBER 状态可返回 1.0；精确等值匹配不能污染 STRING 或原始证据。"""

from datetime import UTC, datetime
from types import SimpleNamespace as NS

import pytest

from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.application.exadata_analysis import analyze_view
from aiops_agent.application.exadata_status import match_status
from aiops_agent.application.exadata_chat import ExadataChatReader
from platform_core.contracts.aiops.exadata import ExadataMetricDefinition, ExadataSample


def definition(code="oracle_si_pdu.PduHttpResponse.Status"):
    return ExadataMetricDefinition.model_validate(load_catalog()[2][code])


@pytest.mark.parametrize("raw", ["1", "1.0", "01.000", "+1.00", "1e0", "10e-1", " .1e1 "])
def test_number_status_uses_exact_decimal_equality(raw):
    item = match_status(definition(), raw)
    assert item.label == "可达" and item.tone == "success"


@pytest.mark.parametrize("raw", ["0.0", "-0", "0e99", "+.0"])
def test_zero_and_signed_zero_are_unreachable_not_missing(raw):
    item = match_status(definition(), raw)
    assert item.label == "不可达" and item.tone == "danger"


@pytest.mark.parametrize("raw", [None, "", " ", "1.0000000000000000001", "0.9999999999999999999",
    "NaN", "Infinity", "-Infinity", "1_0", "0x1", "true", "１", "1e9999999", "unreviewed"])
def test_unknown_nonfinite_or_rounded_only_values_remain_unknown(raw):
    assert match_status(definition(), raw) is None


def test_string_status_keeps_exact_text_semantics_and_case_insensitive_native_tokens():
    item = definition().model_copy(update={"native_value_type": "STRING"})
    assert match_status(item, "1").label == "可达"
    assert match_status(item, "1.0") is None
    ap = definition("oracle_si_netswitch.AccessPointResponse.statusMsg")
    assert match_status(ap, " up ").label == "可达"
    assert match_status(ap, "1.0") is None


@pytest.mark.parametrize("quality", ["OBSERVED", "STALE", "EMPTY", "TRUNCATED"])
def test_analysis_matches_numeric_status_but_never_upgrades_invalid_quality(quality):
    item = definition()
    sample = ExadataSample(component_id="a"*32, metric_code=item.code, selector=item.selector,
        observed_at=datetime.now(UTC), value="1.0", dimensions={}, unit="n/a", native_value_type="NUMBER",
        quality=quality, response_sha256="b"*64)
    view = NS(definitions=[item], members=[NS(target_id="a"*32, display_name="SIM PDU")], samples=[sample], query_mode="LATEST_SNAPSHOT")
    findings = analyze_view(view)
    assert sample.value == "1.0"
    if quality == "OBSERVED":
        assert findings[0].code == "COMPONENT_STATE_OBSERVED" and "可达" in findings[0].detail
        assert "不据此判定整机" in findings[0].detail
    else:
        assert findings[0].code == "CURRENT_EVIDENCE_INSUFFICIENT"
    rendered = ExadataChatReader._render(view, [item.code])
    assert ("：可达" in rendered) is (quality == "OBSERVED")
