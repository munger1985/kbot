"""校验 KBot4 与 Ammolite 的 AIOps 统一助手关键语义保持一致。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SHARED_CONTRACTS = (
    Path("packages/platform_core/src/platform_core/contracts/aiops/orchestration.py"),
    Path("packages/platform_core/src/platform_core/contracts/aiops/conversation.py"),
    Path("services/aiops_agent/src/aiops_agent/contracts/investigation.py"),
    Path("services/aiops_agent/src/aiops_agent/application/diagnosis/investigation_record.py"),
    Path("services/aiops_agent/src/aiops_agent/contracts/diagnosis_state.py"),
    Path("services/aiops_agent/src/aiops_agent/contracts/measurements.py"),
    Path("services/aiops_agent/src/aiops_agent/domain/measurements.py"),
    Path("services/aiops_agent/src/aiops_agent/application/diagnosis/provenance.py"),
    Path("services/aiops_agent/src/aiops_agent/application/diagnosis/revisions.py"),
    Path("services/aiops_agent/src/aiops_agent/application/implementation/diagnostic_runbook.py"),
    Path("services/aiops_agent/src/aiops_agent/application/case_redaction.py"),
    Path("services/aiops_agent/src/aiops_agent/application/source_authorization.py"),
    Path("services/aiops_agent/src/aiops_agent/domain/evidence/observations.py"),
    Path("packages/platform_core/src/platform_core/contracts/aiops/knowledge.py"),
    Path("tests/evaluation/evaluate_aiops_dba_chat.py"),
    Path("services/aiops_agent/src/aiops_agent/executor/diagnostic_deadline.py"),
)
AGENT_APPLICATION = Path(
    "services/aiops_agent/src/aiops_agent/application/agents.py"
)
AGENT_ENTITY = Path("services/aiops_agent/src/aiops_agent/entities/agent.py")
CONVERSATION_ENTITY = Path(
    "services/aiops_agent/src/aiops_agent/entities/conversation.py"
)
TURN_PLANNER = Path(
    "services/aiops_agent/src/aiops_agent/application/turn_planner.py"
)
INVESTIGATION_CONTRACT = Path(
    "packages/platform_core/src/platform_core/contracts/aiops/investigation.py"
)
RUNTIME_SERVICE = Path(
    "services/aiops_agent/src/aiops_agent/application/runtime/service.py"
)
PUBLIC_CONTRACT_ROOT = Path(
    "packages/platform_core/src/platform_core/contracts/aiops"
)
CAPABILITY_CATALOG = Path(
    "services/aiops_agent/src/aiops_agent/application/"
    "conversation_starters/catalog.json"
)
PROMPT_CATALOG = Path(
    "packages/platform_core/src/platform_core/resources/prompts.toml"
)
ALIGNED_AIOPS_PROMPTS = {
    "aiops_agent.capability_resolver",
    "aiops_agent.turn_router",
    "aiops_agent.compact_planner",
    "aiops_agent.investigation_planner",
    "aiops_agent.diagnosis_answer_compose",
    "aiops_agent.sql_diagnosis",
}

AGENT_OPENAPI_FIELDS = {
    "required": {
        "allowed_business_system_ids",
        "capabilities",
        "knowledge_policy",
    },
    "forbidden": {"target_ids"},
}

CONVERSATION_OPENAPI_SCHEMAS = {
    "ConversationCreate": {
        "required": {"agent_id", "context_hint", "source"},
        "forbidden": {"target_id"},
    },
    "ConversationStarterCatalogView": {
        "required": {
            "catalog_version", "agent_id", "context_hint",
            "business_systems", "starters",
        },
        "forbidden": {"target_id", "db_type"},
    },
    "ConversationStarterView": {
        "required": {"starter_id", "required_capabilities"},
        "forbidden": set(),
    },
    "ConversationSummary": {
        "required": {
            "conversation_id",
            "agent_id",
            "agent_version_id",
            "context_hint",
        },
        "forbidden": {"target_id"},
    },
    "TurnSummary": {
        "required": {
            "turn_id",
            "capability_plan",
            "scope_resolution",
            "clarification",
        },
        "forbidden": {"resolved_target_id"},
    },
    "TurnView": {
        "required": {
            "turn_id",
            "capability_plan",
            "scope_resolution",
            "clarification",
        },
        "forbidden": {"resolved_target_id"},
    },
}

OPENAPI_PROFILES = {
    "kbot4": (
        {
            "path": Path("docs/openapi/aiops_internal_v1.json"),
            "schemas": CONVERSATION_OPENAPI_SCHEMAS,
            "starter_path": "/internal/v1/aiops/conversation-starters",
            "conversations_path": "/internal/v1/aiops/conversations",
        },
        {
            "path": Path("docs/openapi/main_api_public_v1.json"),
            "schemas": {
                "AIOpsAgentCreatePayload": AGENT_OPENAPI_FIELDS,
                "AIOpsAgentUpdatePayload": AGENT_OPENAPI_FIELDS,
                **{
                    (
                        "ConversationStartPayload"
                        if name == "ConversationCreate"
                        else name
                    ): fields
                    for name, fields in CONVERSATION_OPENAPI_SCHEMAS.items()
                },
                "ConversationStartPayload": {
                    "required": {"agent_id", "context_hint"},
                    "forbidden": {"target_id"},
                },
            },
            "starter_path": "/api/v1/apps/aiops/conversation-starters",
            "conversations_path": "/api/v1/apps/aiops/conversations",
        },
    ),
    "ammolite_cube": (
        {
            "path": Path("docs/openapi/aiops_internal_v1.json"),
            "schemas": CONVERSATION_OPENAPI_SCHEMAS,
            "starter_path": "/internal/v1/aiops/conversation-starters",
            "conversations_path": "/internal/v1/aiops/conversations",
        },
        {
            "path": Path("docs/openapi/aiops_public_v1.json"),
            "schemas": CONVERSATION_OPENAPI_SCHEMAS,
            "starter_path": (
                "/api/v1/apps/aiops/domains/{domain_code}"
                "/ops/conversation-starters"
            ),
            "conversations_path": (
                "/api/v1/apps/aiops/domains/{domain_code}/ops/conversations"
            ),
        },
        {
            "path": Path("docs/openapi/ammolite_main_api_public_v1.json"),
            "schemas": {
                "AIOpsAgentCreateRequest": AGENT_OPENAPI_FIELDS,
                "AIOpsAgentUpdateRequest": AGENT_OPENAPI_FIELDS,
                **CONVERSATION_OPENAPI_SCHEMAS,
            },
            "starter_path": (
                "/api/v1/apps/aiops/domains/{domain_code}"
                "/ops/conversation-starters"
            ),
            "conversations_path": (
                "/api/v1/apps/aiops/domains/{domain_code}/ops/conversations"
            ),
        },
    ),
}

KEY_FIELDS = {
    INVESTIGATION_CONTRACT: {
        "TaskFrame": {
            "requested_window_seconds",
            "visualization_profile_id",
            "visualization_window_seconds",
            "temporal_analysis_mode",
        },
    },
    AGENT_APPLICATION: {
        "ResourceControlledActionExecution": {
            "resource_id",
            "enabled",
            "allowed_action_ids",
            "object_scopes",
            "max_daily_executions",
        },
        "CreateAIOpsAgentCommand": {
            "allowed_business_system_ids",
            "capabilities",
            "knowledge_policy",
            "controlled_action_execution",
        },
        "UpdateAIOpsAgentCommand": {
            "allowed_business_system_ids",
            "capabilities",
            "knowledge_policy",
            "controlled_action_execution",
        },
    },
    AGENT_ENTITY: {
        "AIOpsAgentVersionBusinessSystemEntity": {
            "agent_version_id",
            "business_system_id",
        },
        "AIOpsAgentVersionResourceActionPolicyEntity": {
            "agent_version_id",
            "resource_id",
            "controlled_action_policy_json",
        },
    },
    CONVERSATION_ENTITY: {
        "OpsConversationEntity": {
            "agent_id",
            "agent_version_id",
            "context_hint_json",
        },
        "OpsConversationTurnEntity": {
            "focused_target_id",
            "capability_plan_json",
            "scope_resolution_json",
            "clarification_json",
        },
    },
}

CONTRACT_KEY_FIELDS = {
    "ConversationCreate": {"agent_id", "context_hint", "source"},
    "ConversationReceipt": {
        "conversation_id",
        "agent_id",
        "agent_version_id",
        "context_hint",
    },
    "ConversationSummary": {
        "conversation_id",
        "agent_id",
        "agent_version_id",
        "context_hint",
    },
    "TurnSummary": {
        "turn_id",
        "conversation_id",
        "capability_plan",
        "scope_resolution",
        "clarification",
    },
}


def _default_peer(root: Path) -> Path:
    peer_name = "kbot4" if root.name == "ammolite_cube" else "ammolite_cube"
    return root.parent / peer_name


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _classes(path: Path) -> dict[str, ast.ClassDef]:
    return {
        node.name: node
        for node in _tree(path).body
        if isinstance(node, ast.ClassDef)
    }


def _fields(node: ast.ClassDef) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in node.body:
        if not isinstance(item, ast.AnnAssign) or not isinstance(
            item.target, ast.Name
        ):
            continue
        result[item.target.id] = ast.dump(
            item.annotation,
            annotate_fields=True,
            include_attributes=False,
        )
    return result


def _contract_alignment(root: Path, peer: Path) -> tuple[dict[str, Any], list[str]]:
    results: dict[str, Any] = {}
    violations: list[str] = []
    for relative in SHARED_CONTRACTS:
        local_path = root / relative
        peer_path = peer / relative
        local_bytes = local_path.read_bytes()
        peer_bytes = peer_path.read_bytes()
        byte_equal = local_bytes == peer_bytes
        ast_equal = ast.dump(
            _tree(local_path), annotate_fields=True, include_attributes=False
        ) == ast.dump(
            _tree(peer_path), annotate_fields=True, include_attributes=False
        )
        results[str(relative)] = {
            "byte_equal": byte_equal,
            "ast_equal": ast_equal,
            "local_sha256": hashlib.sha256(local_bytes).hexdigest(),
            "peer_sha256": hashlib.sha256(peer_bytes).hexdigest(),
        }
        if not byte_equal:
            violations.append(f"共享合同字节不一致: {relative}")
        if not ast_equal:
            violations.append(f"共享合同 AST 不一致: {relative}")
    return results, violations


def _semantic_alignment(root: Path, peer: Path) -> tuple[dict[str, Any], list[str]]:
    results: dict[str, Any] = {}
    violations: list[str] = []
    for relative, class_specs in KEY_FIELDS.items():
        local_classes = _classes(root / relative)
        peer_classes = _classes(peer / relative)
        for class_name, expected_fields in class_specs.items():
            key = f"{relative}:{class_name}"
            local_node = local_classes.get(class_name)
            peer_node = peer_classes.get(class_name)
            local_fields = _fields(local_node) if local_node else {}
            peer_fields = _fields(peer_node) if peer_node else {}
            local_projection = {
                name: local_fields.get(name) for name in sorted(expected_fields)
            }
            peer_projection = {
                name: peer_fields.get(name) for name in sorted(expected_fields)
            }
            missing_local = sorted(expected_fields - local_fields.keys())
            missing_peer = sorted(expected_fields - peer_fields.keys())
            equal = (
                local_node is not None
                and peer_node is not None
                and not missing_local
                and not missing_peer
                and local_projection == peer_projection
            )
            results[key] = {
                "equal": equal,
                "missing_local": missing_local,
                "missing_peer": missing_peer,
            }
            if not equal:
                violations.append(f"关键语义字段不一致: {key}")

    conversation_path = SHARED_CONTRACTS[1]
    local_contracts = _classes(root / conversation_path)
    peer_contracts = _classes(peer / conversation_path)
    for class_name, expected_fields in CONTRACT_KEY_FIELDS.items():
        key = f"{conversation_path}:{class_name}"
        local_node = local_contracts.get(class_name)
        peer_node = peer_contracts.get(class_name)
        local_fields = _fields(local_node) if local_node else {}
        peer_fields = _fields(peer_node) if peer_node else {}
        missing_local = sorted(expected_fields - local_fields.keys())
        missing_peer = sorted(expected_fields - peer_fields.keys())
        equal = (
            local_node is not None
            and peer_node is not None
            and not missing_local
            and not missing_peer
        )
        results[key] = {
            "equal": equal,
            "missing_local": missing_local,
            "missing_peer": missing_peer,
        }
        if not equal:
            violations.append(f"共享合同缺少关键语义字段: {key}")
    return results, violations


def _capability_alignment(
    root: Path, peer: Path
) -> tuple[dict[str, Any], list[str]]:
    local_catalog = json.loads((root / CAPABILITY_CATALOG).read_text("utf-8"))
    peer_catalog = json.loads((peer / CAPABILITY_CATALOG).read_text("utf-8"))
    local_prompts = {
        item["prompt_key"]: item["content"]
        for item in tomllib.loads((root / PROMPT_CATALOG).read_text("utf-8"))[
            "prompts"
        ]
    }
    peer_prompts = {
        item["prompt_key"]: item["content"]
        for item in tomllib.loads((peer / PROMPT_CATALOG).read_text("utf-8"))[
            "prompts"
        ]
    }
    prompt_alignment = {
        key: local_prompts.get(key) == peer_prompts.get(key)
        for key in sorted(ALIGNED_AIOPS_PROMPTS)
    }
    violations = []
    if local_catalog != peer_catalog:
        violations.append("版本化运维能力目录不一致")
    if local_catalog.get("catalog_version") != "2.1.0":
        violations.append("运维能力目录必须使用2.1.0语义合同")
    if len(local_catalog.get("starters", ())) != 26:
        violations.append("运维能力目录必须完整包含26项能力")
    for key, equal in prompt_alignment.items():
        if not equal:
            violations.append(f"AIOps关键Prompt语义不一致: {key}")
    return {
        "catalog_equal": local_catalog == peer_catalog,
        "catalog_version": local_catalog.get("catalog_version"),
        "capability_count": len(local_catalog.get("starters", ())),
        "prompt_alignment": prompt_alignment,
    }, violations


def _forbidden_regressions(root: Path) -> list[str]:
    violations: list[str] = []

    agent_classes = _classes(root / AGENT_ENTITY)
    for class_name, node in agent_classes.items():
        if "AgentVersionTarget" in class_name:
            violations.append(f"禁止恢复 AgentVersionTarget 实体: {class_name}")
        if class_name == "AIOpsAgentVersionEntity" and "target_ids" in _fields(node):
            violations.append("Agent Version 禁止恢复 target_ids 字段")

    application_classes = _classes(root / AGENT_APPLICATION)
    for class_name in ("CreateAIOpsAgentCommand", "UpdateAIOpsAgentCommand"):
        node = application_classes.get(class_name)
        if node is not None and "target_ids" in _fields(node):
            violations.append(f"{class_name} 禁止恢复 target_ids 字段")

    conversation_classes = _classes(root / CONVERSATION_ENTITY)
    conversation = conversation_classes.get("OpsConversationEntity")
    if conversation is None:
        violations.append("缺少 OpsConversationEntity")
    elif "target_id" in _fields(conversation):
        violations.append("Conversation 禁止恢复 target_id 字段")

    for path in sorted((root / PUBLIC_CONTRACT_ROOT).glob("*.py")):
        for class_name, node in _classes(path).items():
            if "resolved_target_id" in _fields(node):
                violations.append(
                    "公开合同禁止恢复 resolved_target_id: "
                    f"{path.relative_to(root)}:{class_name}"
                )
    required_runtime_fragments = {
        TURN_PLANNER: (
            "aggregate_multi_resource_diagnosis",
            "MULTI_RESOURCE_BUDGET_EXCEEDED",
        ),
        RUNTIME_SERVICE: (
            'return "RESOURCE_DIAGNOSIS_CHILD"',
            'event_type="OPS_RUN_TERMINATED"',
            'purpose=f"RESOURCE_{resource_sequence:03d}"',
        ),
    }
    for relative, fragments in required_runtime_fragments.items():
        source = (root / relative).read_text(encoding="utf-8")
        for fragment in fragments:
            if fragment not in source:
                violations.append(
                    f"多资源诊断协调缺少关键语义: {relative}:{fragment}"
                )
    planner_source = (root / TURN_PLANNER).read_text(encoding="utf-8")
    if "MULTI_RESOURCE_DIAGNOSIS_ADAPTER_UNAVAILABLE" in planner_source:
        violations.append("禁止恢复多资源诊断立即降级 PARTIAL 的旧路径")
    return violations


def _operation_parameter_names(
    document: dict[str, Any], path: str, method: str
) -> set[str] | None:
    path_item = document.get("paths", {}).get(path)
    if not isinstance(path_item, dict):
        return None
    operation = path_item.get(method.lower())
    if not isinstance(operation, dict):
        return None
    parameters = [
        *path_item.get("parameters", ()),
        *operation.get("parameters", ()),
    ]
    return {
        str(item["name"])
        for item in parameters
        if isinstance(item, dict) and item.get("name") is not None
    }


def _openapi_guard(root: Path) -> tuple[dict[str, Any], list[str]]:
    results: dict[str, Any] = {}
    violations: list[str] = []
    profiles = OPENAPI_PROFILES.get(root.name)
    if profiles is None:
        return results, [f"没有为仓库配置 OpenAPI 门禁: {root.name}"]

    for profile in profiles:
        relative = profile["path"]
        path = root / relative
        key = str(relative)
        if not path.is_file():
            results[key] = {"exists": False}
            violations.append(f"缺少 AIOps OpenAPI: {relative}")
            continue
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            results[key] = {"exists": True, "valid_json": False}
            violations.append(f"AIOps OpenAPI 无法读取: {relative}: {exc}")
            continue

        schema_results: dict[str, Any] = {}
        schemas = document.get("components", {}).get("schemas", {})
        for schema_name, fields in profile["schemas"].items():
            schema = schemas.get(schema_name)
            properties = (
                set(schema.get("properties", {}))
                if isinstance(schema, dict)
                else set()
            )
            missing = sorted(fields["required"] - properties)
            forbidden = sorted(fields["forbidden"] & properties)
            passed = isinstance(schema, dict) and not missing and not forbidden
            schema_results[schema_name] = {
                "passed": passed,
                "missing": missing,
                "forbidden": forbidden,
            }
            if not isinstance(schema, dict):
                violations.append(f"OpenAPI 缺少 Schema: {relative}:{schema_name}")
            elif missing:
                violations.append(
                    f"OpenAPI Schema 缺少统一助手字段: {relative}:"
                    f"{schema_name}: {','.join(missing)}"
                )
            if forbidden:
                violations.append(
                    f"OpenAPI Schema 仍暴露会话 Target: {relative}:"
                    f"{schema_name}: {','.join(forbidden)}"
                )

        path_results: dict[str, Any] = {}
        for path_name, required, forbidden in (
            (
                profile["starter_path"],
                {"agent_id", "context_hint"},
                {"target_id"},
            ),
            (profile["conversations_path"], set(), {"target_id"}),
        ):
            names = _operation_parameter_names(document, path_name, "get")
            missing = sorted(required - (names or set()))
            exposed = sorted(forbidden & (names or set()))
            passed = names is not None and not missing and not exposed
            path_results[path_name] = {
                "passed": passed,
                "missing": missing,
                "forbidden": exposed,
            }
            if names is None:
                violations.append(f"OpenAPI 缺少 GET 路径: {relative}:{path_name}")
            elif missing:
                violations.append(
                    f"OpenAPI 路径缺少统一助手参数: {relative}:"
                    f"{path_name}: {','.join(missing)}"
                )
            if exposed:
                violations.append(
                    f"OpenAPI 路径仍要求会话 Target: {relative}:"
                    f"{path_name}: {','.join(exposed)}"
                )

        results[key] = {
            "exists": True,
            "valid_json": True,
            "schemas": schema_results,
            "paths": path_results,
        }
    return results, violations


def audit(root: Path, peer: Path) -> dict[str, Any]:
    contracts, contract_violations = _contract_alignment(root, peer)
    semantics, semantic_violations = _semantic_alignment(root, peer)
    capabilities, capability_violations = _capability_alignment(root, peer)
    local_openapi, local_openapi_violations = _openapi_guard(root)
    peer_openapi, peer_openapi_violations = _openapi_guard(peer)
    local_regressions = _forbidden_regressions(root)
    peer_regressions = _forbidden_regressions(peer)
    violations = [
        *contract_violations,
        *semantic_violations,
        *capability_violations,
        *(f"本仓: {item}" for item in local_openapi_violations),
        *(f"对端仓: {item}" for item in peer_openapi_violations),
        *(f"本仓: {item}" for item in local_regressions),
        *(f"对端仓: {item}" for item in peer_regressions),
    ]
    return {
        "schema_version": "aiops.cross-project-alignment.v2",
        "local_root": str(root),
        "peer_root": str(peer),
        "passed": not violations,
        "shared_contracts": contracts,
        "semantic_fields": semantics,
        "capability_semantics": capabilities,
        "openapi": {
            "local": local_openapi,
            "peer": peer_openapi,
        },
        "violations": violations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-root", type=Path, default=_default_peer(ROOT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    peer = args.peer_root.resolve()
    result = audit(ROOT, peer)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif result["passed"]:
        print("PASS AIOps 跨项目合同与关键语义一致")
    else:
        for violation in result["violations"]:
            print(f"FAIL {violation}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
