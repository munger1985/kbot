"""真实评测结果的离线审计器；本工具不读取业务正文、不调用模型或数据库。"""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path


def require(value, message):
    if not value:
        raise ValueError(message)


def assess(manifest, results):
    require(manifest.get("schema_version") == "knowledge.quality_manifest.v1", "评测清单版本无效")
    require(manifest.get("owner") in {"knowledge_retrieval", "aiops"}, "评测必须固定单个 App")
    require(manifest.get("split") in {"development", "validation", "holdout"}, "数据分割无效")
    require(manifest.get("dataset_revision") and manifest.get("configuration_hash"), "缺少冻结数据或配置")
    cases = manifest.get("cases", [])
    require(cases and len(cases) <= 100000, "评测查询数量无效")
    expected = {case["query_id"]: case for case in cases}
    require(len(expected) == len(cases), "评测查询身份重复")
    observed = {}
    for row in results:
        require(row["query_id"] in expected and row["query_id"] not in observed, "结果身份重复或未登记")
        observed[row["query_id"]] = row
    require(set(observed) == set(expected), "不能删除失败或缺失的查询后计算效果")
    rows, durations, violation_count = [], [], 0
    for identity, case in expected.items():
        row = observed[identity]
        require(row.get("status") in {"COMPLETE", "PARTIAL", "EMPTY", "FAILED", "CANCELLED"}, "结果状态无效")
        duration = row.get("duration_ms")
        require(type(duration) in {int, float} and math.isfinite(duration) and duration >= 0, "耗时无效")
        durations.append(duration)
        hits, evidence, delivered = row.get("primary_keys", []), row.get("evidence_keys", []), row.get("delivered_keys", [])
        require(len(set(hits)) == len(hits) and len(set(evidence)) == len(evidence)
                and len(set(delivered)) == len(delivered), "结果身份重复")
        forbidden = set(case.get("forbidden_keys", []))
        violations = bool(forbidden & (set(hits) | set(evidence) | set(delivered)))
        violations |= not (set(hits) | set(delivered)) <= set(evidence)
        violations |= row.get("owner") != manifest["owner"]
        sent = row.get("delivery_status") == "SENT"
        violations |= sent and not delivered
        violation_count += int(violations)
        relevant = set(case["relevant_keys"])
        require(bool(relevant) or case.get("expect_empty") is True, "无命中任务必须明确 expect_empty")
        k = case["requested_count"]
        require(type(k) is int and k > 0, "结果数量无效")
        ranked = hits[:k]
        gains = [int(key in relevant) for key in ranked]
        ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
        ndcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains)) / ideal if ideal else float(not hits)
        expected_links = {tuple(pair) for pair in case.get("required_links", [])}
        links = {tuple(pair) for pair in row.get("returned_links", [])}
        success = row["status"] in {"COMPLETE", "PARTIAL"}
        can_deliver = bool(row.get("readable_within_budget"))
        silent_zero = can_deliver and case.get("answer_required", False) and not delivered
        rows.append({"query_id": identity, "task": case["task"], "status": row["status"],
            "recall_at_k": len(set(ranked) & relevant) / len(relevant) if relevant else float(not hits),
            "precision_at_k": sum(gains) / k, "ndcg_at_k": ndcg,
            "link_recall": len(links & expected_links)/len(expected_links) if expected_links else 1.,
            "forbidden_or_identity_violation": bool(violations), "silent_zero_delivery": bool(silent_zero),
            "successful": success, "duration_ms": duration})
    groups = {}
    for task in sorted({row["task"] for row in rows}):
        group = [row for row in rows if row["task"] == task]
        groups[task] = {field: sum(row[field] for row in group)/len(group)
                       for field in ("recall_at_k", "precision_at_k", "ndcg_at_k", "link_recall", "successful")}
        groups[task]["queries"] = len(group)
    durations.sort()
    p95 = durations[max(0, math.ceil(.95 * len(durations))-1)]
    report = {"schema_version": "knowledge.quality_report.v1", "owner": manifest["owner"],
        "dataset_revision": manifest["dataset_revision"], "configuration_hash": manifest["configuration_hash"],
        "split": manifest["split"], "query_count": len(rows), "by_task": groups, "queries": rows,
        "p95_ms": p95, "identity_or_access_violations": violation_count,
        "silent_zero_delivery": sum(row["silent_zero_delivery"] for row in rows),
        "production_quality_claim": False, "gate": "NOT_EVALUATED"}
    thresholds = manifest.get("thresholds", {})
    require(set(groups) <= set(thresholds.get("by_task", {})), "每个任务必须提前登记独立质量门槛")
    passed = violation_count == 0 and report["silent_zero_delivery"] == 0
    max_p95 = thresholds.get("max_p95_ms")
    require(type(max_p95) in {int, float} and math.isfinite(max_p95) and max_p95 > 0, "必须预登记 P95 预算")
    passed &= p95 <= max_p95
    for task, metrics in groups.items():
        required = thresholds["by_task"][task]
        for field in ("recall_at_k", "ndcg_at_k", "link_recall", "successful"):
            minimum = required.get(field)
            require(type(minimum) in {int, float} and math.isfinite(minimum) and 0 <= minimum <= 1,
                    "任务门槛缺失或不合法")
            passed &= metrics[field] >= minimum
    report["gate"] = "PASS" if passed else "FAIL"
    # 离线分数不能单独证明真实数据授权、模型能力、数据库竞态和发布通过。
    return report


def read_json(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, "评测工件超过预算")
    return raw


def main():
    parser = argparse.ArgumentParser(description="只审计已取得的真实/合成评测结果，不发起真实测试")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    manifest_raw = read_json(args.manifest, 32 * 1024 * 1024)
    result_raw = read_json(args.results, 128 * 1024 * 1024)
    report = assess(json.loads(manifest_raw), [json.loads(line) for line in result_raw.splitlines() if line.strip()])
    report["input_hashes"] = {"manifest": sha256(manifest_raw).hexdigest(), "results": sha256(result_raw).hexdigest()}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
