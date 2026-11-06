"""评分案例检索响应；离线合同通过不代表真实模型、KC 或生产闭环验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID


DATASET = Path(__file__).with_name("aiops_case_retrieval_cases.json")
UNPUBLISHED = {"DRAFT", "RETIRED", "FOREIGN_DOMAIN", "SOURCE_REVOKED", "SUPERSEDED"}
DECISIONS = {"MATCHED", "CONFLICT", "UNKNOWN", "PROHIBITED"}
PROVENANCE = {"LIVE_DEPENDENCIES", "OFFLINE_CONTRACT"}
SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identity(value):
    try:
        return isinstance(value, str) and str(UUID(value)) == value
    except (TypeError, ValueError):
        return False


def validate_dataset(dataset):
    _require(isinstance(dataset, dict) and dataset.get("schema_version") ==
        "OPS_CASE_RETRIEVAL_EVALUATION.v1", "评测集版本无效")
    cases, specs = dataset.get("cases"), dataset.get("asset_specs")
    _require(isinstance(cases, list) and cases and isinstance(specs, dict) and specs,
        "评测场景和资产定义不能为空")
    identities = [case.get("id") for case in cases]
    _require(all(isinstance(value, str) and value for value in identities) and
        len(set(identities)) == len(identities), "评测场景标识缺失或重复")
    for spec in specs.values():
        _require(spec.get("kind") in {"MANUAL", "DIAGNOSIS_CASE"} and
            spec.get("state") in UNPUBLISHED | {"PUBLISHED"}, "测试资产类型或状态无效")
        allowed = {"APPROVED_MANUAL"} if spec["kind"] == "MANUAL" else {
            "DIAGNOSTIC_REFERENCE", "VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION", "NEGATIVE_EXPERIENCE"}
        _require(spec.get("case_kind") in allowed, "测试资产权威类型无效")
    for case in cases:
        relevance, forbidden = case.get("relevance"), case.get("forbidden_assets")
        _require(isinstance(relevance, dict) and isinstance(forbidden, list) and
            set(relevance) <= specs.keys() and set(forbidden) <= specs.keys(),
            "评测引用了未定义的测试资产")
        _require(type(case.get("max_results")) is int and 1 <= case["max_results"] <= 20 and
            all(type(value) is int and 0 <= value <= 3 for value in relevance.values()) and
            any(value > 0 for value in relevance.values()), "评测数量或相关性分级无效")
        _require(isinstance(case.get("query"), str) and 1 <= len(case["query"]) <= 4000 and
            case.get("purpose") in {"DIAGNOSE", "EXPLAIN", "PLAN", "CHANGE", "VERIFY"},
            "评测查询或用途无效")
        _require(case.get("expected_case_decision") in DECISIONS and
            isinstance(case.get("action_allowed"), list) and set(case["action_allowed"]) <= relevance.keys(),
            "评测适用性或措施门禁无效")
        for gate in ("minimum_recall", "minimum_ndcg"):
            _require(type(case.get(gate)) in {int, float} and 0 <= case[gate] <= 1, "质量门禁必须在零到一之间")
        _require(type(case.get("require_complete_scope")) is bool, "评测范围门禁必须显式声明")
        aliases = case.get("equivalent_assets", {})
        _require(isinstance(aliases, dict) and set(aliases) <= specs.keys() and
            set(aliases.values()) <= relevance.keys() and not set(aliases).intersection(aliases.values()),
            "等价资产必须直接指向相关性金标，不能形成链或环")
        decisions = case.get("expected_decisions", {})
        _require(isinstance(decisions, dict) and set(decisions) <= relevance.keys() and
            set(decisions.values()) <= DECISIONS, "逐资产适用性金标无效")
    return len(identities)


def validate_manifest(dataset, manifest):
    _require(isinstance(manifest, dict) and manifest.get("schema_version") ==
        "OPS_CASE_RETRIEVAL_FIXTURES.v1", "测试环境清单版本无效")
    assets, bindings = manifest.get("assets"), manifest.get("case_bindings")
    _require(isinstance(assets, dict) and set(assets) == set(dataset["asset_specs"]),
        "测试环境清单必须精确列出全部确切版本，不按标题猜测")
    versions = set()
    for key, asset in assets.items():
        _require(isinstance(asset, dict) and all(_identity(asset.get(field)) for field in (
            "asset_version_id", "parse_run_id", "document_version_id")), "测试资产缺少确切 UUID")
        _require(asset["asset_version_id"] not in versions, "测试资产确切版本不能重复映射")
        versions.add(asset["asset_version_id"])
        _require(isinstance(asset.get("lineage_id"), str) and asset["lineage_id"].strip(),
            "测试资产必须显式登记来源血缘")
        if dataset["asset_specs"][key]["kind"] == "DIAGNOSIS_CASE":
            _require(SHA256.fullmatch(str(asset.get("body_sha256") or "")), "案例缺少正文哈希")
    _require(isinstance(bindings, dict) and set(bindings) == {case["id"] for case in dataset["cases"]},
        "测试环境必须精确绑定每个场景的当前 Agent 和 Target")
    for binding in bindings.values():
        _require(isinstance(binding, dict) and _identity(binding.get("agent_id")) and
            (binding.get("target_id") is None or _identity(binding["target_id"])), "场景 Agent 或 Target 标识无效")
        _require(SHA256.fullmatch(str(binding.get("context_fingerprint") or "")),
            "场景缺少独立核对的当前上下文指纹")
    for case in dataset["cases"]:
        for alias, primary in case.get("equivalent_assets", {}).items():
            _require(assets[alias]["lineage_id"] == assets[primary]["lineage_id"],
                "等价副本必须具有同一来源血缘")


def _dcg(grades):
    return sum((2 ** grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def _case_citation_failures(case, key, citation, row, expected, binding, spec):
    failures = []
    locator = citation.get("locator") or {}
    context, decision = locator.get("case_context") or {}, locator.get("case_applicability") or {}
    if context.get("schema_version") != "OPS_CASE_CONTEXT.v1" or context.get("body_sha256") != expected["body_sha256"]:
        failures.append("CASE_BODY_HASH_OR_CLOSURE_MISSING")
    if str(context.get("parse_run_id")) != expected["parse_run_id"] or str(
        context.get("document_version_id")) != expected["document_version_id"]:
        failures.append("WRONG_CONTEXT_REVISION")
    if context.get("role") != "HISTORICAL_KNOWLEDGE_NOT_CURRENT_EVIDENCE":
        failures.append("CASE_PROMOTED_TO_CURRENT_EVIDENCE")
    units = context.get("units") or []
    if not ({"PROBLEM", "CONCLUSION", "BOUNDARIES", "VERIFICATION", "UNRESOLVED"} |
        set(spec.get("required_unit_kinds") or ())) <= {
        unit.get("kind") for unit in units}:
        failures.append("CASE_CONDITION_EFFECT_CLOSURE_MISSING")
    if not any(unit.get("matched") is True for unit in units):
        failures.append("CASE_MATCHED_UNIT_MISSING")
    seen_units = set()
    for unit in units:
        quote = unit.get("quote")
        if (not isinstance(quote, str) or not quote or type(unit.get("start")) is not int or
            type(unit.get("end")) is not int or unit["start"] < 0 or unit["end"] - unit["start"] != len(quote) or
            hashlib.sha256(quote.encode()).hexdigest() != unit.get("content_hash") or
            not unit.get("unit_id") or unit["unit_id"] in seen_units):
            failures.append("CASE_UNIT_QUOTE_OR_HASH_INVALID")
        seen_units.add(unit.get("unit_id"))
    if decision.get("schema_version") != "OPS_CASE_REUSE_DECISION.v1":
        failures.append("MISSING_VERSIONED_DECISION")
    if decision.get("execution_authorized") is not False or decision.get("historical_knowledge_is_current_evidence") is not False:
        failures.append("UNSAFE_EXECUTION_OR_CURRENT_FACT")
    if decision.get("requires_current_verification") is not True:
        failures.append("CURRENT_VERIFICATION_DROPPED")
    if decision.get("context_fingerprint") != binding["context_fingerprint"]:
        failures.append("WRONG_CURRENT_CONTEXT")
    status = decision.get("status")
    primary = case.get("equivalent_assets", {}).get(key, key)
    expected_state = case.get("expected_decisions", {}).get(primary, case["expected_case_decision"])
    if status not in DECISIONS or (case["relevance"].get(primary, 0) > 0 and status != expected_state):
        failures.append("WRONG_APPLICABILITY_STATE")
    allowed = decision.get("action_recommendation_allowed")
    if type(allowed) is not bool or ((primary not in case["action_allowed"] or status != "MATCHED" or
        spec["case_kind"] not in {"VERIFIED_RESOLUTION", "VERIFIED_OPTIMIZATION"}) and allowed is not False):
        failures.append("UNSAFE_ACTION_RECOMMENDATION")
    actions, action_ids = decision.get("actions") or [], set()
    for action in actions:
        ordinal = action.get("action_ordinal")
        if (type(ordinal) is not int or ordinal < 1 or ordinal in action_ids or
            type(action.get("action_recommendation_allowed")) is not bool or
            action.get("execution_authorized") is not False):
            failures.append("INVALID_ACTION_GATE")
        action_ids.add(ordinal)
        if action.get("action_recommendation_allowed") and (not allowed or
            action.get("controlled_effect_linked") is not True or action.get("effect_result") not in {"RESOLVED", "IMPROVED"}):
            failures.append("UNVERIFIED_ACTION_PROMOTED")
    if bool(allowed) != any(action.get("action_recommendation_allowed") is True for action in actions):
        failures.append("CASE_ACTION_GATE_DIVERGED")
    if spec["case_kind"] == "NEGATIVE_EXPERIENCE" and decision.get("action_use") != "WARNING_ONLY":
        failures.append("NEGATIVE_ACTION_PROMOTED")
    if row.get("applicability_decision") != decision or (row.get("reuse_context") or {}).get("applicability_decision") != decision:
        failures.append("CITATION_REUSE_CONTEXT_DIVERGED")
    return failures


def evaluate_case(case, response, manifest, asset_specs):
    _require(isinstance(response, dict) and isinstance(response.get("results"), list),
        "检索响应必须包含结果数组，不能把网关错误当空结果")
    assets = manifest["assets"]
    lookup = {value["asset_version_id"]: key for key, value in assets.items()}
    failures, keys, seen_lineages = [], [], set()
    items = response["results"]
    for index, row in enumerate(items):
        key = lookup.get(row.get("asset_version_id"))
        if key is None:
            failures.append("UNKNOWN_ASSET_VERSION")
            keys.append(None)
            continue
        primary = case.get("equivalent_assets", {}).get(key, key)
        if primary in keys:
            failures.append("DUPLICATE_VERSION_OR_EQUIVALENT")
        keys.append(primary)
        spec, expected = asset_specs[key], assets[key]
        lineage = expected["lineage_id"]
        if lineage in seen_lineages:
            failures.append("DUPLICATE_SOURCE_LINEAGE")
        seen_lineages.add(lineage)
        if key in case["forbidden_assets"] or spec["state"] in UNPUBLISHED:
            failures.append("FORBIDDEN_ASSET")
        if row.get("asset_kind") != spec["kind"] or row.get("authority") != spec["case_kind"]:
            failures.append("SOURCE_KIND_OR_AUTHORITY_CHANGED")
        if "rank" in row and row["rank"] != index + 1:
            failures.append("INVALID_RANK")
        citations = row.get("citation_pack") or []
        if not citations:
            failures.append("MISSING_BODY_CITATION")
        for citation in citations:
            locator = citation.get("locator") or {}
            for field, failure in (("parse_run_id", "WRONG_EXACT_REVISION"),
                ("document_version_id", "WRONG_CONTENT_DOCUMENT")):
                values = [str(source[field]) for source in (citation, locator) if source.get(field) is not None]
                if not values or any(value != expected[field] for value in values):
                    failures.append(failure)
            if not str(citation.get("quote") or "").strip():
                failures.append("MISSING_QUOTE")
            if spec["kind"] == "DIAGNOSIS_CASE":
                failures.extend(_case_citation_failures(case, key, citation, row, expected,
                    manifest["case_bindings"][case["id"]], spec))
    if response.get("status") != "READY":
        failures.append("RETRIEVAL_NOT_READY")
    if len(items) > case["max_results"]:
        failures.append("RESULT_LIMIT_IGNORED")
    if case["require_complete_scope"] and response.get("coverage", {}).get("registry_scope_complete") is not True:
        failures.append("INCOMPLETE_REGISTRY_SCOPE")
    seen, grades = set(), []
    for key in keys[:case["max_results"]]:
        grades.append(0 if key in seen else case["relevance"].get(key, 0))
        seen.add(key)
    ideal = sorted(case["relevance"].values(), reverse=True)[:case["max_results"]]
    relevant = {key for key, grade in case["relevance"].items() if grade > 0}
    recall = len(set(keys[:case["max_results"]]).intersection(relevant)) / len(relevant)
    ndcg = _dcg(grades) / _dcg(ideal)
    reciprocal_rank = next((1 / (index + 1) for index, key in enumerate(keys) if key in relevant), 0)
    if recall < case["minimum_recall"]:
        failures.append("RECALL_BELOW_GATE")
    if ndcg < case["minimum_ndcg"]:
        failures.append("RANKING_BELOW_GATE")
    return {"case_id": case["id"], "passed": not failures, "failures": sorted(set(failures)),
        "recall": recall, "ndcg": ndcg, "reciprocal_rank": reciprocal_rank}


def evaluate(dataset, records, manifest):
    validate_dataset(dataset)
    validate_manifest(dataset, manifest)
    by_case = {}
    for record in records:
        _require(isinstance(record, dict) and record.get("evidence_kind") in PROVENANCE,
            "每条响应必须声明真实依赖导出或离线合同来源")
        identity = record.get("case_id")
        _require(isinstance(identity, str) and identity not in by_case, "同一评测场景的响应缺失标识或重复")
        by_case[identity] = record
    _require(set(by_case) == {case["id"] for case in dataset["cases"]},
        "响应必须精确覆盖所有评测场景，缺失不能当作跳过通过")
    results = [evaluate_case(case, by_case[case["id"]].get("response"), manifest, dataset["asset_specs"])
        for case in dataset["cases"]]
    provenance = {record["evidence_kind"] for record in records}
    return {"schema_version": "OPS_CASE_RETRIEVAL_EVALUATION_RESULT.v1",
        "evidence_kind": "DECLARED_LIVE_EXPORT_NOT_INDEPENDENTLY_VERIFIED" if provenance == {"LIVE_DEPENDENCIES"}
            else "OFFLINE_OR_MIXED_NOT_LIVE_ACCEPTANCE",
        "response_scoring_performed": True, "live_requests_performed": False,
        "passed": all(result["passed"] for result in results), "case_count": len(results),
        "production_acceptance_proven": False, "cases": results}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("检索预览端点不允许重定向，避免向其他地址转发认证信息")


def run_live(dataset, manifest, *, endpoint, headers):
    validate_dataset(dataset)
    validate_manifest(dataset, manifest)
    parsed = urlsplit(endpoint)
    _require(parsed.scheme in {"http", "https"} and parsed.hostname and not (
        parsed.username or parsed.password or parsed.query or parsed.fragment) and
        parsed.path.endswith("/operations-knowledge/search-preview"),
        "必须显式指定不含凭据的 Main API 检索预览地址")
    _require(isinstance(headers, dict) and headers and
        all(isinstance(key, str) and isinstance(value, str) and "\r" not in key + value and
            "\n" not in key + value for key, value in headers.items()), "认证请求头格式无效")
    records, opener = [], build_opener(_NoRedirect())
    for case in dataset["cases"]:
        binding = manifest["case_bindings"][case["id"]]
        payload = {"query": case["query"], "purpose": case["purpose"], "max_results": case["max_results"],
            "agent_id": binding["agent_id"], "target_id": binding.get("target_id"),
            "source_kinds": ["MANUAL", "DIAGNOSIS_CASE"]}
        request = Request(endpoint, data=json.dumps(payload).encode(), method="POST",
            headers={**headers, "Content-Type": "application/json"})
        with opener.open(request, timeout=60) as result:
            body = result.read(4_000_001)
            _require(len(body) <= 4_000_000, "检索响应超过评测读取预算")
            response = json.loads(body)
        records.append({"case_id": case["id"], "evidence_kind": "LIVE_DEPENDENCIES", "response": response})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--validate-dataset", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--responses", type=Path, help="真实或明确标识离线来源的 JSONL 响应")
    parser.add_argument("--endpoint", help="显式 Main API 检索预览地址；不执行初始化或账号操作")
    parser.add_argument("--headers-env", default="AIOPS_CASE_EVAL_HEADERS_JSON", help="包含现有认证请求头的环境变量名，内容不会输出")
    args = parser.parse_args()
    if not args.validate_dataset and (args.manifest is None or bool(args.responses) == bool(args.endpoint)):
        parser.error("必须提供 --manifest，以及 --responses 或 --endpoint 其中之一")
    try:
        dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
        count = validate_dataset(dataset)
        if args.validate_dataset:
            print(json.dumps({"dataset_cases": count, "dataset_valid": True, "retrieval_quality_evaluated": False}, ensure_ascii=False))
            return 0
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if args.endpoint:
            records = run_live(dataset, manifest, endpoint=args.endpoint,
                headers=json.loads(os.environ[args.headers_env]))
        else:
            records = [json.loads(line) for line in args.responses.read_text(encoding="utf-8").splitlines() if line.strip()]
        result = evaluate(dataset, records, manifest)
        if args.endpoint:
            result.update(evidence_kind="LIVE_ENDPOINT_REQUESTS", live_requests_performed=True)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError, TypeError, AttributeError):
        print("评测输入无效或检索端点不可用，未完成评测；不输出认证信息、输入正文或错误响应", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
