"""逐 App 的 G-A 合成基准；不调用真实数据库、模型或业务文件。"""

import asyncio
from hashlib import sha256
import json
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/knowledge_algorithms/src"))
sys.path.insert(0, str(ROOT / "tests/unit/knowledge_algorithms"))

from knowledge_algorithm_fixtures import (FakeBackend, FakeModel, candidate, parse_profile,
    retrieval_profile, run_for, source, spec)
from knowledge_algorithms.contracts.content import ContentKind
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.identity import SourceSpan
from knowledge_algorithms.contracts.query import QueryInput, QueryMode
from knowledge_algorithms.contracts.retrieval import ChannelSpec
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
from knowledge_algorithms.references.relations import RelationType, ResolvedRelation
from knowledge_algorithms.retrieval.engine import RetrievalEngine


async def evaluate_case(dataset, case):
    owner, kind = dataset["owner"], case["kind"]
    backend, model = FakeBackend(owner), FakeModel()
    if kind == "PARSE":
        reader, profile = source(case["text"], owner), parse_profile()
        parsed = await ParseEngine(backend.binding, (PlainTextAdapter(),)).parse(reader, profile, run_for(reader, profile))
        assert parsed.quality_status == case["expected_status"], "解析质量状态不一致"
        assert "".join(n.text for n in parsed.nodes) == case["text"], "原文丢失"
        assert all(n.key.owner_binding_ref == owner and n.locators for n in parsed.nodes), "解析身份/定位不完整"
        return {"status": parsed.quality_status, "nodes": len(parsed.nodes), "owner": owner}
    p = backend.add("paragraph")
    f = backend.add("figure", ContentKind.FIGURE)
    raw_p = backend.add("paragraph", projected=False)
    raw_f = backend.add("figure", ContentKind.FIGURE, projected=False)
    image = backend.add("image", ContentKind.IMAGE, projected=False)
    backend.relations = (
        ResolvedRelation("ref-1", RelationType.REFERS_TO, raw_p, raw_f, "SYNTHETIC_LABEL",
            "occ-1", SourceSpan("paragraph", 0, 4)),
        ResolvedRelation("asset-1", RelationType.HAS_ASSET, raw_f, image, "SYNTHETIC_ASSET"))
    query = QueryInput(QueryMode.RELEVANCE, 1, ("MIXED",), text=case["text"])
    profile = retrieval_profile()
    backend.rows = {"lexical": ((p, 1),)}
    if kind in {"IMAGE_TO_IMAGE", "TEXT_TO_IMAGE"}:
        channel = ChannelSpec("image", "image", "DENSE", kind, 1, 10, 1, True,
            spec(input_kind="IMAGE", direction=kind))
        profile = retrieval_profile(channels=(channel,))
        backend.rows = {"image": ((f, 1),)}
        if kind == "IMAGE_TO_IMAGE":
            query = QueryInput(QueryMode.RELEVANCE, 1, ("FIGURE",), image_handles=("synthetic-image-query",))
    if kind == "OPTIONAL_FAILURE":
        channel = ChannelSpec("dense", "dense", "DENSE", "TEXT_TO_TEXT", 1, 10, 1, False, spec())
        profile = retrieval_profile(channels=profile.channels + (channel,))
        backend.rows["dense"] = RuntimeError("合成通道失败")
    if kind == "FINAL_REVOCATION":
        backend.final_revoked = {raw_f}
    if kind == "NO_ACCESS":
        backend.rows = {"lexical": ((candidate("missing", owner), 1),)}
    if kind == "REQUIRED_FAILURE":
        backend.rows = {"lexical": RuntimeError("合成必需通道失败")}
    if kind == "CROSS_APP":
        other = "aiops" if owner == "knowledge_retrieval" else "knowledge_retrieval"
        backend.rows = {"lexical": ((candidate("foreign", other), 1),)}
    if kind == "EMPTY":
        backend.rows = {}
    try:
        result = await RetrievalEngine(backend, model).retrieve(query, backend.scope, profile)
    except AlgorithmError as error:
        assert error.code.value == case.get("expected_error"), "错误分类不一致"
        assert all(call not in backend.calls for call in case.get("forbidden_backend_calls", [])), "已触发越权读取"
        return {"error": error.code.value, "owner": owner}
    assert "expected_error" not in case, "未拒绝越权"
    assert [h.evidence_key.object_key for h in result.primary_hits] == case.get("expected_primary", []), "主命中不一致"
    if "expected_status" in case:
        assert result.status == case["expected_status"], "状态不一致"
    if "expected_reason" in case:
        assert result.reason == case["expected_reason"], "缺口原因不一致"
    actual = {item.key.object_key for item in result.evidence}
    assert not (actual & set(case.get("forbidden", []))), "撤权证据泄漏"
    assert all(item.key.owner_binding_ref == owner for item in result.evidence), "跨 App 结果"
    if "expected_linked" in case:
        assert {k.object_key for k in result.groups[0].linked_keys} == set(case["expected_linked"]), "双向关联不完整"
    if result.primary_hits:
        assert result.context_selection.selected_evidence_keys, "合法有界候选静默零交付清单"
    for record in result.trace:
        assert len(set(record.input_keys)) == len(record.input_keys), "输入身份重复"
        assert set(record.output_keys) | {key for key, _ in record.excluded} == set(record.input_keys), "阶段身份不守恒"
    return {"status": result.status, "reason": result.reason, "primary_count": len(result.primary_hits),
            "context_count": len(result.context_selection.selected_evidence_keys), "owner": owner}


async def evaluate():
    started = time.monotonic()
    config = json.loads((ROOT / "tests/contract/knowledge_algorithms/experiments-v1.json").read_text())
    inventory = json.loads((ROOT / "tests/contract/knowledge_algorithms/inventory-v1.json").read_text())
    results = []
    for owner, app in inventory["apps"].items():
        dataset = json.loads((ROOT / app["fixture"]).read_text())
        assert dataset["owner"] == owner and dataset["real_data"] is False
        for case in dataset["cases"]:
            result = await evaluate_case(dataset, case)
            results.append({"app": owner, "case": case["id"], "split": case["split"], **result})
    assert time.monotonic() - started < config["runner_budget"]["timeout_seconds"], "合成评测超过时间预算"
    return {"schema_version": "knowledge.algorithm_evaluation.v1", "gate": "G_A",
            "production_quality_claim": False, "results": results,
            "input_manifest": {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in sorted(
                list((ROOT / "packages/knowledge_algorithms/src").rglob("*.py"))
                + [ROOT / "tests/unit/knowledge_algorithms/knowledge_algorithm_fixtures.py",
                   ROOT / "tests/contract/knowledge_algorithms/experiments-v1.json",
                   ROOT / "tests/contract/knowledge_algorithms/inventory-v1.json"]
                + [ROOT / app["fixture"] for app in inventory["apps"].values()])}}


def main():
    tracemalloc.start()
    report = asyncio.run(asyncio.wait_for(evaluate(), timeout=30))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    config = json.loads((ROOT / "tests/contract/knowledge_algorithms/experiments-v1.json").read_text())
    assert peak <= config["runner_budget"]["max_traced_memory_bytes"], "合成评测超过内存预算"
    report["peak_traced_memory_bytes"] = peak
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
