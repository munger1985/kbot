"""Main API 监控映射响应不得暴露内部 Locator。"""

from main_api.api.ops import _safe_source_binding
from platform_core.identity import uuid7


def test_public_source_binding_projection_only_returns_locator_hint():
    raw_locator = "oracle-prod-secret-instance"
    result = _safe_source_binding(
        {
            "binding_id": uuid7(),
            "target_id": uuid7(),
            "source_id": uuid7(),
            "source_locator_key": raw_locator,
            "source_locator": {"target_key": raw_locator, "password": "secret"},
            "status": "ACTIVE",
            "health_status": "HEALTHY",
            "row_version": 3,
        }
    )

    payload = result.model_dump(mode="json")
    assert payload["locator_hint"] == "or***ce"
    assert raw_locator not in result.model_dump_json()
    assert "source_locator" not in payload
