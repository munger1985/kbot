"""发布资格与正文相关性分离的知识检索；所有外部调用均在数据库事务之外。"""

from dataclasses import dataclass
from datetime import UTC, datetime
import math
from uuid import UUID
from typing import Any

from aiops_agent.application.case_knowledge_projection import case_projection_is_current, case_context_for_citation
from aiops_agent.application.case_source_validation import validate_case_sources, validate_case_effect
from aiops_agent.application.case_applicability import (
    ACTION_PURPOSES, load_case_search_context, assess_case_candidates, case_reuse_decision,
)
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.application.knowledge_ranking import knowledge_rank_scorecard, knowledge_rank_key, RANKING_POLICY
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from platform_clients.knowledge_core import KnowledgeCoreClientError


@dataclass(frozen=True)
class KnowledgeSearchBudget:
    """业务范围扫描和 KC 分片预算，不是模型生成长度或用户要求的结果数。"""

    max_candidates: int = 512
    registry_page_size: int = 64
    discovery_scope_size: int = 1000
    discovery_per_collection: int = 100
    evidence_batch_size: int = 128
    semantic_case_budget: int = 20

    def __post_init__(self):
        if not 1 <= self.max_candidates <= 1000:
            raise ValueError("知识候选预算须处于 1 至 1000，保持两平台精确范围查询一致")
        if not 1 <= self.registry_page_size <= 128 or not 1 <= self.discovery_scope_size <= 1000:
            raise ValueError("知识范围分页或分片预算无效")
        if not 1 <= self.discovery_per_collection <= 100 or not 1 <= self.evidence_batch_size <= 128:
            raise ValueError("KC 召回预算超出共享合同")
        if not 1 <= self.semantic_case_budget <= 20:
            raise ValueError("案例适用性单批预算须处于 1 至 20")


async def _eligible_case(service, *, uow, domain_id, agent_id, asset, version):
    if str(asset.asset_kind) != "DIAGNOSIS_CASE":
        return True
    if not await validate_case_sources(
        uow=uow, domain_id=domain_id, version_id=version.asset_version_id,
        agent_id=agent_id, plan_capture_reader=getattr(service, "_plan_capture_reader", None),
    ) or not await validate_case_effect(uow=uow, domain_id=domain_id, profile=version.profile_json):
        return False
    try:
        profile = DiagnosisCaseProfile.model_validate(version.profile_json)
        return profile.knowledge_projection is not None and case_projection_is_current(
            title=profile.knowledge_projection.title, profile=profile,
        )
    except (TypeError, ValueError):
        return False


def _snapshot(asset, version, index):
    profile = dict(version.profile_json or {})
    manifest = profile.get("knowledge_projection") or {}
    return {
        "asset_id": str(asset.asset_id), "asset_version_id": str(version.asset_version_id),
        "asset_kind": str(asset.asset_kind), "title": str(manifest.get("title") or asset.display_name),
        "version": int(version.version_no), "profile": profile,
        "created_at": getattr(version, "created_at", None),
        "collection_id": index.collection_id, "bundle_revision_id": index.bundle_revision_id,
        "bundle_id": getattr(index, "bundle_id", None) if str(asset.asset_kind) == "DIAGNOSIS_CASE" else None,
    }


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else 0.0
    except (ValueError, TypeError):
        return 0.0


def _evidence_rank(item):
    ranks = [_number((entry.get("evidence") or {}).get("local_rank"))
        for entry in item.get("items") or () if isinstance(entry, dict)]
    ranks = [rank for rank in ranks if rank > 0]
    return min(ranks) if ranks else 1.0


def _lineage(candidate):
    if candidate["asset_kind"] != "DIAGNOSIS_CASE":
        return candidate["asset_id"]
    source = dict(candidate["profile"].get("source") or {})
    return str(source.get("source_fingerprint") or source.get("report_id")
        or source.get("effect_result_artifact_id") or candidate["asset_id"])


async def search_published_knowledge(
    service, *, domain_id, agent_id, request, context, target_id, filters, kc_scope,
):
    from aiops_agent.application.operations_knowledge import (
        _profile_dimensions_from_profile, _catalog_topics_from_profile,
    )

    async with service._uow_factory() as uow:
        authorization = await _current_knowledge_authorization(
            uow=uow, domain_id=domain_id, agent_id=agent_id, target_id=target_id, request=request,
        )
    if isinstance(authorization, str):
        return {"status": authorization, "results": [], "warnings": ["当前 Agent 的知识能力、知识库或来源范围不可用"]}
    request, knowledge_base_ids = authorization
    filters = {**filters, "TAG_ID": tuple(str(value).upper() for value in request.tag_ids)}
    filters = {key: values for key, values in filters.items() if values}
    budget = getattr(service, "_knowledge_search_budget", None) or KnowledgeSearchBudget()
    scope = dict(domain_id=domain_id, target_id=target_id,
        source_kinds=request.source_kinds, knowledge_base_ids=knowledge_base_ids,
        max_security_level=request.max_security_level, scope_filters=filters)
    snapshots, scanned, cursor, exhausted = [], 0, None, False
    # 短分页事务；禁止把 ORM 对象带到 KC 调用后再访问。
    while scanned < budget.max_candidates:
        page_size = min(budget.registry_page_size, budget.max_candidates - scanned)
        async with service._uow_factory() as uow:
            rows = await uow.operations_knowledge.list_published_candidates(
                **scope, limit=page_size, after_index_id=cursor,
            )
            for asset, version, index in rows:
                scanned += 1
                if await _eligible_case(service, uow=uow, domain_id=domain_id,
                    agent_id=agent_id, asset=asset, version=version):
                    snapshots.append(_snapshot(asset, version, index))
            if len(rows) < page_size:
                exhausted = True
            elif rows:
                next_cursor = rows[-1][2].index_ref_id
                if cursor is not None and str(next_cursor) == str(cursor):
                    raise AIOpsApplicationError(code="AIOPS_KNOWLEDGE_CURSOR_CONFLICT",
                        message="知识范围分页未前进，请刷新后重试", status_code=409)
                cursor = next_cursor
        if exhausted:
            break
    warnings = []
    if not exhausted:
        async with service._uow_factory() as uow:
            remaining = await uow.operations_knowledge.list_published_candidates(
                **scope, limit=1, after_index_id=cursor,
            )
        if remaining:
            warnings.append("候选范围超过本次检索预算，结果仅覆盖已扫描范围；请缩小知识库或问题范围。")
    diagnostics = {"scanned_candidates": scanned, "candidate_budget": budget.max_candidates,
        "registry_scope_complete": not warnings, "discovery_shards": 0, "evidence_batches": 0}

    def response(status, results, extra_warnings=()):
        return {"status": status, "results": results,
            "warnings": list(dict.fromkeys([*warnings, *extra_warnings])), "coverage": diagnostics}

    if not snapshots:
        return response("NO_APPLICABLE_KNOWLEDGE", [])

    case_search_context = None

    async def revalidate_snapshots():
        """目录和正文输出使用同一当前授权／精确版本围栏，目录也不能泄漏撤权资产。"""
        current = set()
        async with service._uow_factory() as uow:
            checked = await _current_knowledge_authorization(
                uow=uow, domain_id=domain_id, agent_id=agent_id, target_id=target_id, request=request,
            )
            if isinstance(checked, str) or checked != (request, knowledge_base_ids):
                return None
            if case_search_context is not None:
                current_context = await load_case_search_context(
                    uow=uow, domain_id=domain_id, agent_id=agent_id, target_id=target_id,
                )
                if current_context is None or current_context.fingerprint != case_search_context.fingerprint:
                    return None
            rows = await uow.operations_knowledge.list_published_candidates(
                **scope, limit=None,
                asset_version_ids=tuple(UUID(row["asset_version_id"]) for row in snapshots),
                bundle_revision_ids=tuple(row["bundle_revision_id"] for row in snapshots),
            )
            for asset, version, index in rows:
                if await _eligible_case(service, uow=uow, domain_id=domain_id,
                    agent_id=agent_id, asset=asset, version=version):
                    current.add((str(version.asset_version_id), str(index.bundle_revision_id)))
        return [row for row in snapshots if (row["asset_version_id"], str(row["bundle_revision_id"])) in current]

    if request.purpose == "CATALOG":
        catalog = await revalidate_snapshots()
        if catalog is None:
            return response("KNOWLEDGE_SCOPE_CHANGED", [], ["读取期间 Agent 知识授权或范围已变化，请重新检索。"])
        results = [{
            **{key: row[key] for key in ("asset_id", "asset_version_id", "asset_kind", "title", "version")},
            "authority": "APPROVED_MANUAL" if row["asset_kind"] == "MANUAL" else row["profile"].get("case_kind"),
            "applicability": "CATALOG", "topics": list(_catalog_topics_from_profile(row["profile"])),
            "citation_pack": [],
        } for row in catalog[:request.max_results]]
        if len(catalog) > request.max_results:
            warnings.append("目录仅显示本次请求的结果数，不代表全部已发布资产。")
        return response("READY" if results else "NO_APPLICABLE_KNOWLEDGE", results)
    if any(row["asset_kind"] == "DIAGNOSIS_CASE" for row in snapshots):
        async with service._uow_factory() as uow:
            case_search_context = await load_case_search_context(
                uow=uow, domain_id=domain_id, agent_id=agent_id, target_id=target_id,
            )
        if case_search_context is None:
            return response("KNOWLEDGE_SCOPE_CHANGED", [], ["当前 Agent 或目标已变化，请重新检索。"])
    by_revision = {str(row["bundle_revision_id"]): row for row in snapshots}
    discoveries, evidence_items, discovery_scores = {}, [], {}
    try:
        for offset in range(0, len(snapshots), budget.discovery_scope_size):
            shard = snapshots[offset:offset + budget.discovery_scope_size]
            allowed = {str(row["bundle_revision_id"]) for row in shard}
            discovery = await service._kc.discover(
                **kc_scope, query=request.query,
                collection_ids=sorted({row["collection_id"] for row in shard}, key=str),
                bundle_revision_ids=[row["bundle_revision_id"] for row in shard],
                agent_id=agent_id, auth_context=context, max_security_level=request.max_security_level,
                per_collection_limit=budget.discovery_per_collection,
            )
            diagnostics["discovery_shards"] += 1
            warnings.extend(str(value) for value in discovery.get("warnings") or ())
            for rank, candidate in enumerate(discovery.get("candidates") or (), start=1):
                revision = str(candidate.get("bundle_revision_id") or "")
                if revision not in allowed:
                    continue
                expected = by_revision[revision]
                if candidate.get("collection_id") is not None and str(candidate["collection_id"]) != str(expected["collection_id"]):
                    continue
                discoveries[revision] = dict(candidate)
                discovery_scores[revision] = max(discovery_scores.get(revision, 0.0),
                    _number(candidate.get("rrf_score")) or 1.0 / (60 + rank))
        if not discoveries:
            return response("NO_EVIDENCE_MATCH", [], ["当前发布版本没有形成正文命中"])
        candidates = list(discoveries.values())
        for offset in range(0, len(candidates), budget.evidence_batch_size):
            batch = candidates[offset:offset + budget.evidence_batch_size]
            allowed = {str(candidate["bundle_revision_id"]) for candidate in batch}
            evidence = await service._kc.retrieve_evidence(
                **kc_scope, query=request.query, candidates=batch,
                agent_id=agent_id, auth_context=context, max_security_level=request.max_security_level,
                max_evidence=min(128, max(request.max_results * 2, len(batch))),
            )
            diagnostics["evidence_batches"] += 1
            warnings.extend(str(value) for value in evidence.get("warnings") or ())
            evidence_items.extend(item for item in evidence.get("citations") or ()
                if isinstance(item, dict) and str(item.get("bundle_revision_id") or "") in allowed)
        # 核对 CONTENT 文件的精确正文哈希后才扩展上下文，MANIFEST 不能充当诊断正文。
        case_documents = {}
        for revision_id in dict.fromkeys(str(item.get("bundle_revision_id") or "") for item in evidence_items):
            candidate = by_revision.get(revision_id)
            if candidate is None or candidate["asset_kind"] != "DIAGNOSIS_CASE":
                continue
            if candidate["bundle_id"] is None:
                continue
            revision = await service._kc.get_revision_status(
                **kc_scope, bundle_id=candidate["bundle_id"], bundle_revision_id=UUID(revision_id),
                include_members=True, auth_context=context,
            )
            if (
                str(revision.get("bundle_revision_id") or "") != revision_id
                or revision.get("status") != "READY"
                or revision.get("approval_status") != "APPROVED"
                or revision.get("withdrawn_at") is not None
            ):
                continue
            expected_hash = candidate["profile"]["knowledge_projection"]["body_sha256"]
            case_documents[revision_id] = {
                str(member["document_version_id"]) for member in revision.get("members") or ()
                if member.get("document_version_id") and member.get("document_role") == "CONTENT"
                and member.get("content_hash_sha256") == expected_hash
                and member.get("member_status") in {"READY", "COMPLETED"}
            }
    except KnowledgeCoreClientError as exc:
        raise AIOpsApplicationError(code="AIOPS_KNOWLEDGE_KC_UNAVAILABLE",
            message="运维知识正文检索暂时不可用", status_code=503, retryable=True) from exc

    # 正文及语义判断完成后统一重验，期间不持有数据库事务。
    grouped, scores = {}, {}
    for item in evidence_items:
        revision = str(item.get("bundle_revision_id") or "")
        candidate = by_revision.get(revision)
        if candidate is None or (
            item.get("collection_id") is not None and str(item["collection_id"]) != str(candidate["collection_id"])
        ):
            continue
        citation = knowledge_citation_pack(dict(item))
        if citation is None:
            continue
        if candidate["asset_kind"] == "DIAGNOSIS_CASE":
            document_id = str(item.get("document_version_id") or "")
            if document_id not in case_documents.get(revision, set()):
                warnings.append("部分案例的精确正文版本或内容哈希无法核对，已从本次结果排除。")
                continue
            try:
                citation["locator"]["case_context"] = case_context_for_citation(
                    profile=candidate["profile"], citation=item, document_version_id=document_id,
                )
            except ValueError:
                warnings.append("部分案例缺少完整可引用上下文，已排除；请在诊断案例库复核。")
                continue
        key = candidate["asset_version_id"]
        if key not in grouped:
            dimensions = _profile_dimensions_from_profile(candidate["profile"])
            matched = sorted(key for key, wanted in filters.items()
                if dimensions.get(key) and not dimensions[key].isdisjoint(wanted))
            mismatched = sorted(key for key, wanted in filters.items()
                if dimensions.get(key) and dimensions[key].isdisjoint(wanted))
            unknown = sorted(key for key in filters if not dimensions.get(key))
            grouped[key] = {
                **{name: candidate[name] for name in ("asset_id", "asset_version_id", "asset_kind", "title", "version")},
                "authority": "APPROVED_MANUAL" if candidate["asset_kind"] == "MANUAL" else candidate["profile"].get("case_kind"),
                "applicability": "CONFLICT" if mismatched else "PARTIAL" if unknown else "MATCHED",
                "matched_dimensions": matched, "mismatched_dimensions": mismatched,
                "unknown_dimensions": unknown, "citation_pack": [],
            }
            if candidate["asset_kind"] == "DIAGNOSIS_CASE":
                profile = candidate["profile"]
                grouped[key]["reuse_context"] = {
                    "role": "HISTORICAL_KNOWLEDGE_NOT_CURRENT_EVIDENCE",
                    "case_kind": profile["case_kind"],
                    "applicability": profile.get("applicability") or [],
                    "contraindications": profile.get("contraindications") or [],
                    "unresolved_questions": profile.get("unresolved_questions") or [],
                    "verification": {name: profile["verification"].get(name)
                        for name in ("result", "method", "comparability", "attribution_strength", "causal_limitations")},
                    "requires_current_verification": True,
                    "action_use": "WARNING_ONLY" if profile["case_kind"] == "NEGATIVE_EXPERIENCE"
                        else "REFERENCE_REQUIRES_LOCAL_VALIDATION",
                }
        if citation not in grouped[key]["citation_pack"]:
            grouped[key]["citation_pack"].append(citation)
        scores[key] = max(scores.get(key, 0.0), discovery_scores.get(revision, 0.0) / _evidence_rank(item))
    candidates_by_version = {row["asset_version_id"]: row for row in by_revision.values()}
    case_candidates = sorted((row for row in candidates_by_version.values()
        if row["asset_kind"] == "DIAGNOSIS_CASE" and row["asset_version_id"] in grouped),
        key=lambda row: (-scores[row["asset_version_id"]], row["asset_version_id"]))
    if case_candidates:
        assessments, semantic_status, incomplete, semantic_provenance = await assess_case_candidates(
            service, candidates=case_candidates, context=case_search_context, request=request,
            model_scope=kc_scope, max_cases=budget.semantic_case_budget,
        )
        diagnostics.update(semantic_case_budget=budget.semantic_case_budget,
            assessed_cases=len(assessments), semantic_scope_complete=not incomplete)
        if incomplete:
            warnings.append("部分案例的前提或禁忌尚未完成当前环境核验，仅可参考；不能当作处置推荐。")
        for candidate in case_candidates:
            key = candidate["asset_version_id"]
            decision = case_reuse_decision(profile=candidate["profile"], context=case_search_context,
                judgments=assessments.get(key, ()),
                semantic_status=semantic_status if key in assessments or semantic_status != "ASSESSED" else "BUDGET_EXCEEDED",
                semantic_provenance=semantic_provenance if key in assessments else {})
            grouped[key]["applicability_decision"] = decision
            grouped[key]["applicability"] = decision["status"]
            grouped[key]["reuse_context"].update(
                action_use=decision["action_use"], applicability_decision=decision)
            # 调用方即便只保留 Citation，仍携带业务判断；不把判断伪装成 KC 原文。
            for citation in grouped[key]["citation_pack"]:
                citation["locator"]["case_applicability"] = decision
            if request.purpose in ACTION_PURPOSES and not decision["action_recommendation_allowed"] and decision["action_use"] != "WARNING_ONLY":
                del grouped[key]
                warnings.append("部分案例不满足当前处置推荐条件，已排除；可改以诊断参考用途核对。")
    # 所有外部调用之后重新核验确切发布版本、Agent 模型和当前 Target，不能重取 Top-K。
    checked = await revalidate_snapshots()
    if checked is None:
        return response("KNOWLEDGE_SCOPE_CHANGED", [], ["检索期间 Agent 授权、模型或目标环境已变化，请重新检索。"])
    current = {(row["asset_version_id"], str(row["bundle_revision_id"])) for row in checked}
    grouped = {key: row for key, row in grouped.items()
        if (key, str(candidates_by_version[key]["bundle_revision_id"])) in current}
    ranking_time = datetime.now(UTC)
    for key, row in grouped.items():
        row["ranking"] = knowledge_rank_scorecard(candidate=candidates_by_version[key], result=row,
            kc_relevance=scores[key], purpose=request.purpose, now=ranking_time)
    diagnostics["ranking_policy"] = RANKING_POLICY
    ordered = sorted(grouped.values(), key=knowledge_rank_key)
    results, seen_lineage = [], set()
    for row in ordered:
        lineage = _lineage(candidates_by_version[row["asset_version_id"]])
        if lineage in seen_lineage:
            continue
        seen_lineage.add(lineage)
        results.append(row)
        if len(results) >= request.max_results:
            break
    return response("READY" if results else "NO_APPLICABLE_KNOWLEDGE" if case_candidates else "NO_EVIDENCE_MATCH", results,
        () if results else ("当前发布范围没有形成可安全用于本次用途的正文结果",))


async def _current_knowledge_authorization(*, uow, domain_id, agent_id, target_id, request):
    """知识服务自身复核当前 Agent 策略，后台路径也不能借显式请求绕过绑定和开关。"""
    try:
        identity = UUID(str(agent_id))
    except (TypeError, ValueError):
        return "NO_AGENT_BOUND"
    agent = await uow.agents.get(domain_id=domain_id, agent_id=identity)
    if agent is None or agent.status != "ACTIVE" or agent.current_version_id is None:
        return "NO_AGENT_BOUND"
    if target_id is not None and not await uow.agents.version_has_target(
        agent_version_id=agent.current_version_id, target_id=target_id,
    ):
        return "TARGET_NOT_AUTHORIZED"
    policy = await uow.agents.version_knowledge_policy(agent_version_id=agent.current_version_id)
    if not policy.get("enabled", False):
        return "KNOWLEDGE_DISABLED"
    allowed_sources = set(policy.get("source_kinds") or ())
    source_kinds = tuple(kind for kind in request.source_kinds if kind in allowed_sources)
    if not source_kinds:
        return "NO_AUTHORIZED_KNOWLEDGE_SOURCES"
    bound = {UUID(str(value)) for value in policy.get("knowledge_base_ids") or ()}
    bases = tuple(sorted(bound.intersection(request.knowledge_base_ids) if request.knowledge_base_ids else bound, key=str))
    if not bases:
        return "NO_KNOWLEDGE_BASE_BOUND"
    active_bases = await uow.operations_knowledge.active_knowledge_base_ids(
        domain_id=domain_id, knowledge_base_ids=bases,
    )
    bases = tuple(sorted(set(bases).intersection(active_bases), key=str))
    if not bases:
        return "NO_KNOWLEDGE_BASE_BOUND"
    tags = tuple(sorted({str(value) for value in request.tag_ids} | {str(value) for value in policy.get("tag_ids") or ()}))
    return request.model_copy(update={
        "source_kinds": source_kinds, "knowledge_base_ids": bases, "tag_ids": tags,
        "max_security_level": min(request.max_security_level, int(policy.get("max_security_level") or 1)),
    }), bases


def knowledge_citation_pack(item):
    """把 KC CitationGroup 规范化为 AIOps 可直接引用的正文与定位。"""
    direct_text = next(
        (
            str(item.get(key) or "").strip()
            for key in ("quote", "text", "content", "content_text")
            if str(item.get(key) or "").strip()
        ),
        "",
    )
    excerpts: list[str] = [direct_text] if direct_text else []
    evidence_refs: list[dict[str, Any]] = []
    seen_evidence: set[str] = set()
    raw_items = item.get("items")
    if isinstance(raw_items, list):
        for raw_group_item in raw_items:
            if not isinstance(raw_group_item, dict):
                continue
            evidence = raw_group_item.get("evidence")
            if not isinstance(evidence, dict):
                continue
            evidence_id = str(evidence.get("evidence_id") or "")
            dedupe_key = evidence_id or str(evidence.get("content_hash") or "")
            if dedupe_key and dedupe_key in seen_evidence:
                continue
            if dedupe_key:
                seen_evidence.add(dedupe_key)
            content_text = str(evidence.get("content_text") or "").strip()
            if content_text:
                excerpts.append(content_text)
            evidence_refs.append(
                {
                    "evidence_id": evidence_id or None,
                    "role": str(raw_group_item.get("final_role") or ""),
                    "heading_path": list(evidence.get("heading_path") or ()),
                    "locator": dict(evidence.get("locator") or {}),
                }
            )
    quote = "\n\n".join(dict.fromkeys(excerpts)).strip()
    if not quote:
        return None
    locator = dict(item.get("locator") or {})
    for key in (
        "citation_id",
        "collection_id",
        "bundle_id",
        "bundle_revision_id",
        "document_version_id",
        "parse_view_id",
        "primary_evidence_ids",
        "structural_context_ids",
        "neighbor_evidence_ids",
    ):
        if item.get(key) is not None:
            locator[key] = item[key]
    if evidence_refs:
        locator["evidence_refs"] = evidence_refs
    return {**item, "quote": quote, "locator": locator}
