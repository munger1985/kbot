"""将已授权的检索回执投影为主命中及关联证据。"""

from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.wire import encode


def retrieval_coverage(*, gaps=(), channels=(), shards=0, has_results=False):
    gaps = list(dict.fromkeys(gaps))
    return {
        "scope": ("PARTIAL" if has_results else "NOT_VERIFIED") if gaps or channels else "COMPLETE",
        "ranking": (
            "GLOBAL" if shards == 1 and not gaps and not channels
            else "LIMITED"
        ) if has_results else "NOT_APPLICABLE",
        "gaps": gaps,
        "channel_failures": list(channels),
    }


def coverage_summary(result):
    coverage = result.get("coverage") or {}
    return {"retrieval_conclusion": result.get("retrieval_conclusion", "NOT_VERIFIED"),
        "coverage": {key: coverage[key] for key in ("scope", "ranking", "gaps", "channel_failures")
            if key in coverage}}


def coverage_note(summary):
    conclusion = summary.get("retrieval_conclusion")
    coverage = summary.get("coverage") or {}
    if conclusion == "NOT_VERIFIED":
        return "本次未能充分核验运维知识，不表示没有匹配资料。"
    if conclusion == "NO_ELIGIBLE_RESOURCES":
        return "当前授权和筛选范围内没有符合资格的已发布运维资料。"
    if conclusion == "NO_MATCH":
        return "已完成本次范围检索，没有匹配的正文证据。"
    if coverage.get("scope") in {"PARTIAL", "NOT_VERIFIED"}:
        return "当前回答仅使用已核验的部分知识证据，不代表完整覆盖。"
    if coverage.get("ranking") == "LIMITED":
        return "当前知识结果排序受限，不代表全局相关性排名。"
    return None


def coverage_gap_code(summary):
    coverage = summary.get("coverage") or {}
    if summary.get("retrieval_conclusion") == "NOT_VERIFIED":
        return "KNOWLEDGE_NOT_VERIFIED"
    if coverage.get("scope") in {"PARTIAL", "NOT_VERIFIED"}:
        return "KNOWLEDGE_COVERAGE_PARTIAL"
    if coverage.get("ranking") == "LIMITED":
        return "KNOWLEDGE_RANKING_LIMITED"
    return None


async def project_receipt(retrieval, access, receipt, allowed, *, primary_budget, relation_budget, include_related=True):
    """证据清单只用于按键查取；关联证据不计入主命中。"""
    manifest = receipt.evidence_set
    evidence_by_key = {item.key: item for item in manifest.evidence}
    groups_by_key = {group.anchor_key: group for group in manifest.groups}
    gaps = []
    failures = [
        {"channel": channel.channel_key, "reason": channel.reason or channel.status}
        for channel in manifest.channel_status if channel.status != "COMPLETE"
    ]
    if manifest.status == "FAILED":
        gaps.append("RETRIEVAL_FAILED")
    if manifest.status == "PARTIAL":
        gaps.append("ENGINE_PARTIAL")
    if not manifest.scope_exhausted:
        gaps.append("RETRIEVAL_SCOPE_UNEXHAUSTED")

    def identity(item):
        candidate = allowed.get(item.key.document_version_ref)
        if (candidate is None or item.key.owner_binding_ref != "aiops"
                or item.key.parse_run_ref != candidate["parse_run_id"]):
            return None
        return candidate

    async def read(item, candidate, byte_budget):
        descriptor, chunks = await retrieval.stream_content(
            access, retrieval_ref=receipt.retrieval_ref, stable_key=item.key.stable_key,
        )
        body = bytearray()
        try:
            async for block in chunks:
                byte_budget[0] -= len(block)
                if byte_budget[0] < 0 or len(body) + len(block) > 1024 * 1024:
                    return None
                body.extend(block)
        finally:
            await chunks.aclose()
        quote = bytes(body).decode("utf-8")
        if not quote.strip():
            return None
        return {
            "document_version_id": candidate["document_version_id"],
            "parse_run_id": item.key.parse_run_ref,
            "index_generation_id": candidate["index_generation_id"],
            "quote": quote,
            "locator": {
                "evidence_key": encode(item.key), "locators": encode(item.locators),
                "source_keys": encode(item.source_keys), "source_spans": encode(item.source_spans),
                "content_hash": descriptor.content_hash, "owner_binding_ref": "aiops",
            },
        }

    async def text(item, candidate, byte_budget):
        if byte_budget[0] <= 0 or not item.mime_type.startswith("text/") or item.payload_status == "UNAVAILABLE":
            return None
        try:
            return await read(item, candidate, byte_budget)
        except (AlgorithmError, OSError, UnicodeError):
            return None

    primary_rows = []
    for hit in sorted(manifest.primary_hits, key=lambda row: row.rank):
        item = evidence_by_key.get(hit.evidence_key)
        candidate = identity(item) if item else None
        if not candidate:
            gaps.append("PRIMARY_IDENTITY_UNAVAILABLE")
            continue
        citation = await text(item, candidate, primary_budget)
        if not citation:
            gaps.append("CONTENT_BUDGET" if primary_budget[0] <= 0 else "PRIMARY_CONTENT_UNAVAILABLE")
            continue
        primary_rows.append((hit, item, citation))

    hits = []
    for hit, item, citation in primary_rows:
        primary = {**citation, "rank": hit.rank, "evidence_key": item.key.stable_key}
        linked = groups_by_key.get(hit.evidence_key) if include_related else None
        related, relation_gaps = [], []
        if linked is not None:
            if linked.manifest_status != "COMPLETE":
                relation_gaps.append(linked.incomplete_reason or "RELATION_INCOMPLETE")
            if len(linked.linked_keys) > 8:
                relation_gaps.append("RELATION_BUDGET")
            for key in linked.linked_keys[:8]:
                linked_item = evidence_by_key.get(key)
                source = identity(linked_item) if linked_item else None
                if not source:
                    relation_gaps.append("RELATION_UNAVAILABLE")
                    continue
                # 关联关系不是额外授权，不隐式发送图像。
                payload = await text(linked_item, source, relation_budget)
                if payload is None:
                    relation_gaps.append("RELATION_CONTENT_UNAVAILABLE")
                related.append({
                    **(payload or {"document_version_id": source["document_version_id"],
                        "parse_run_id": key.parse_run_ref,
                        "index_generation_id": source["index_generation_id"],
                        "locator": {"evidence_key": encode(key), "locators": encode(linked_item.locators)}}),
                    "evidence_key": key.stable_key, "readable": payload is not None,
                })
        # 关联缺口单列；不可读的图不影响主命中已读正文。
        hits.append((citation, {"primary": primary, "related": related,
            "relation_gaps": list(dict.fromkeys(relation_gaps))}))
    return hits, gaps, failures
