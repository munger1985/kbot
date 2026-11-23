"""离线校验阶段 0 的目标合同样例，不导入服务、不连接数据库。"""

import json
from pathlib import Path
import re
import unittest
from uuid import UUID


FIXTURES = Path(__file__).with_name("fixtures") / "aiops_knowledge_alignment_v1.json"
STATES = {"NOT_ENABLED", "PENDING_VERIFICATION", "AVAILABLE", "DEGRADED"}
DIRECTIONS = {"TEXT_TO_TEXT", "TEXT_TO_IMAGE", "IMAGE_TO_TEXT", "IMAGE_TO_IMAGE"}
USAGE_REASONS = {
    "assets": "ASSET_RECORDS",
    "current_agent_versions": "CURRENT_AGENT_VERSION",
    "historical_agent_versions": "HISTORICAL_AGENT_VERSION",
}
PRIVATE_KEYS = {"query_ticket", "ticket_id", "scope_ref", "scope_hash", "cursor", "continuation"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_write(payload, phase):
    allowed = {"display_name", "description", "tagging_model_id"}
    require(phase in {"create", "update"}, "unknown write phase")
    if phase == "update":
        allowed.add("expected_row_version")
        version = payload.get("expected_row_version")
        require(type(version) is int and version > 0, "edit requires a row version")
    require(set(payload) <= allowed, "unknown write field")
    name = payload.get("display_name")
    require(isinstance(name, str) and 0 < len(name.strip()) <= 256, "invalid name")
    description = payload.get("description")
    require(description is None or isinstance(description, str) and len(description) <= 1000,
            "invalid description")
    if "tagging_model_id" not in payload:
        return "UNCHANGED" if phase == "update" else "UNCONFIGURED"
    model_id = payload["tagging_model_id"]
    if model_id is None:
        return "CLEAR" if phase == "update" else "UNCONFIGURED"
    UUID(model_id)
    return "SET"


def validate_capability(payload):
    capabilities = payload["capabilities"]
    require(set(capabilities) == {
        "native_parse", "ocr", "visual_enrichment", "lexical", "semantic_directions"
    }, "unexpected capability fields")
    for key in ("native_parse", "ocr", "visual_enrichment", "lexical"):
        require(capabilities[key] in STATES, "invalid capability state")
    for semantic in capabilities["semantic_directions"]:
        require(semantic["direction"] in DIRECTIONS, "invalid semantic direction")
        require(semantic["state"] in STATES - {"NOT_ENABLED"}, "unconfigured semantic direction")
    require(payload["index"]["state"] in {"EMPTY", "NOT_READY", "PARTIAL", "READY", "ERROR"},
            "invalid index state")
    tagging = payload["tagging"]
    if tagging["state"] == "NOT_CONFIGURED":
        require(tagging["model_id"] is None, "unconfigured tagging cannot have a binding")
    else:
        require(tagging["state"] in STATES - {"NOT_ENABLED"}, "invalid tagging state")
        require(isinstance(tagging["model_id"], str), "tagging binding is missing")
        UUID(tagging["model_id"])


def validate_public_fields(value):
    if isinstance(value, dict):
        require(not PRIVATE_KEYS.intersection(value), "internal retrieval field")
        for child in value.values():
            validate_public_fields(child)
    elif isinstance(value, list):
        for child in value:
            validate_public_fields(child)


def validate_search(payload):
    validate_public_fields(payload)
    conclusion = payload["retrieval_conclusion"]
    coverage = payload["coverage"]
    results = payload["results"]
    require(conclusion in {"EVIDENCE_FOUND", "NO_ELIGIBLE_RESOURCES", "NO_MATCH", "NOT_VERIFIED"},
            "invalid retrieval conclusion")
    require(coverage["scope"] in {"COMPLETE", "PARTIAL", "NOT_VERIFIED"}, "invalid scope coverage")
    require(coverage["ranking"] in {"GLOBAL", "LIMITED", "NOT_APPLICABLE"}, "invalid ranking status")
    require(bool(results) == (conclusion == "EVIDENCE_FOUND"), "conclusion and primary results disagree")
    complete = coverage["scope"] == "COMPLETE" and not coverage["gaps"] and not coverage["channel_failures"]
    if conclusion in {"NO_ELIGIBLE_RESOURCES", "NO_MATCH"}:
        require(complete, "empty conclusion requires complete coverage")
    if coverage["ranking"] == "GLOBAL":
        require(complete and bool(results), "global ranking requires complete coverage")
    for failure in coverage["channel_failures"]:
        require(bool(failure["channel"] and failure["reason"]), "missing channel failure reason")
    for item in results:
        require(bool(item["evidence_groups"] and item["citation_pack"]), "missing primary citation")
        for group in item["evidence_groups"]:
            primary = group["primary"]
            require(type(primary["rank"]) is int and primary["rank"] > 0, "invalid primary rank")
            require(bool(primary["evidence_key"] and primary["quote"] and primary["locator"]),
                    "primary evidence is not traceable")
            UUID(primary["document_version_id"])
            for related in group["related"]:
                require("rank" not in related, "related evidence cannot rank")
                require(type(related["readable"]) is bool, "missing readability")
            require(isinstance(group["relation_gaps"], list), "missing relation gaps")


def validate_usage(payload):
    usage = payload["usage"]
    require(set(USAGE_REASONS) <= set(usage), "missing protected dependency count")
    for name in USAGE_REASONS:
        require(type(usage[name]) is int and usage[name] >= 0, "invalid usage count")
    expected = {reason for name, reason in USAGE_REASONS.items() if usage[name]}
    require(set(payload["blockers"]).intersection(USAGE_REASONS.values()) == expected,
            "usage and blockers disagree")


def deletion_outcome(if_match, row_version, blockers):
    """核权并取得目标库之后的错误优先级示例；不模拟数据库锁。"""
    if if_match is None:
        return {"status": 428, "code": "PRECONDITION_REQUIRED"}
    match = re.fullmatch(r'"rv-([1-9][0-9]*)"', if_match)
    if not match:
        return {"status": 400}
    if int(match[1]) != row_version:
        return {"status": 412, "code": "STATE_VERSION_CONFLICT"}
    if blockers:
        return {"status": 409, "code": "AIOPS_KNOWLEDGE_BASE_IN_USE", "blockers": blockers}
    return {"status": 204}


class TestAIOpsKnowledgeAlignmentStage0(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURES.read_text(encoding="utf-8"))

    def check_cases(self, contract, validator):
        cases = self.cases[contract]
        self.assertTrue(any("error" in case for case in cases))
        self.assertTrue(any("error" not in case for case in cases))
        for case in cases:
            with self.subTest(case=case["name"]):
                if "error" in case:
                    with self.assertRaisesRegex(ValueError, re.escape(case["error"])):
                        validator(case)
                else:
                    validator(case)

    def test_write_scope_and_nullable_tagging(self):
        def validate(case):
            operation = validate_write(case["payload"], case["phase"])
            self.assertEqual(operation, case["tagging_operation"])

        self.check_cases("write", validate)

    def test_capability_index_and_tagging_states(self):
        self.check_cases("capability", lambda case: validate_capability(case["payload"]))

    def test_primary_related_coverage_and_no_match(self):
        self.check_cases("search", lambda case: validate_search(case["payload"]))

    def test_current_history_and_asset_blockers(self):
        self.check_cases("usage", lambda case: validate_usage(case["payload"]))

    def test_delete_status_and_precondition_priority(self):
        for case in self.cases["deletion"]:
            with self.subTest(case=case["name"]):
                response = deletion_outcome(case["if_match"], case["row_version"], case["blockers"])
                self.assertEqual(response, case["expected"])


if __name__ == "__main__":
    unittest.main()
