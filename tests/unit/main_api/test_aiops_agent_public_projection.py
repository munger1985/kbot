"""AIOps Agent 公开投影不得泄露内部 Target 与连接状态。"""

from copy import deepcopy

from main_api.api.aiops_app import (
    _aiops_agent_public_view,
    _aiops_agent_use_view,
)


def _internal_agent():
    return {
        "agent_id": "agent-1",
        "display_name": "MES Agent",
        "status": "ACTIVE",
        "target_candidates": [
            {
                "target_id": "target-1",
                "resource_id": "resource-1",
                "display_name": "MES RAC",
                "db_type": "ORACLE",
                "status": "ENABLED",
                "importance_level": 1,
                "connectivity_status": "CONNECTED",
                "readonly_connection_enabled": True,
                "controlled_change_enabled": False,
            },
            {
                "target_id": "orphan-target",
                "resource_id": None,
                "display_name": "Orphan",
                "db_type": "ORACLE",
            },
        ],
        "controlled_action_execution": [
            {
                "target_id": "target-1",
                "resource_id": "resource-1",
                "enabled": True,
                "allowed_action_ids": ["db.session.terminate"],
                "object_scopes": {"schemas": ["MES"]},
                "max_daily_executions": 2,
                "internal_policy_id": "policy-1",
            }
        ],
    }


def test_public_agent_view_exposes_only_resource_level_candidates() -> None:
    internal = _internal_agent()
    original = deepcopy(internal)

    public = _aiops_agent_public_view(internal)

    assert public["target_candidates"] == [
        {
            "resource_id": "resource-1",
            "display_name": "MES RAC",
            "db_type": "ORACLE",
            "importance_level": 1,
        }
    ]
    assert public["controlled_action_execution"] == [
        {
            "resource_id": "resource-1",
            "enabled": True,
            "allowed_action_ids": ["db.session.terminate"],
            "object_scopes": {"schemas": ["MES"]},
            "max_daily_executions": 2,
        }
    ]
    assert internal == original


def test_use_view_applies_same_target_redaction() -> None:
    public = _aiops_agent_use_view(_internal_agent())

    assert "controlled_action_execution" not in public
    assert public["target_candidates"][0]["resource_id"] == "resource-1"
    assert "target_id" not in public["target_candidates"][0]
    assert "connectivity_status" not in public["target_candidates"][0]
