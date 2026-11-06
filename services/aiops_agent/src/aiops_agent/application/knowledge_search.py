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


@dataclass(frozen=True)
class KnowledgeSearchBudget:
    """业务范围扫描和本 App 检索分片预算，不是模型生成长度或用户要求的结果数。"""

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
            raise ValueError("召回预算超出本 App 合同")
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
        "index_ref_id": str(index.index_ref_id),
        **{key: str(getattr(index, key)) for key in
           ("document_id", "document_version_id", "ingestion_job_id", "parse_run_id", "index_generation_id")},
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
    # 短分页事务；禁止把 ORM 对象带到检索调用后再访问。
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
                document_version_ids=tuple(row["document_version_id"] for row in snapshots),
            )
            for asset, version, index in rows:
                if await _eligible_case(service, uow=uow, domain_id=domain_id,
                    agent_id=agent_id, asset=asset, version=version):
                    current.add((str(version.asset_version_id), str(index.document_version_id), str(index.parse_run_id), str(index.index_generation_id)))
        return [row for row in snapshots if (row["asset_version_id"], row["document_version_id"], row["parse_run_id"], row["index_generation_id"]) in current]

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
    by_revision = {str(row["document_version_id"]): row for row in snapshots}
    evidence_items, discovery_scores, case_documents = [], {}, {}
    from datetime import timedelta
    from platform_core.identity import uuid7
    from knowledge_algorithms.contracts.query import QueryInput, QueryMode
    from knowledge_algorithms.contracts.wire import encode
    access = service._documents.access(context)
    retrieval = service._documents.retrieval
    batch_size = min(retrieval.limits.max_versions, budget.discovery_scope_size)
    for offset in range(0, len(snapshots), batch_size):
        shard = snapshots[offset:offset + batch_size]
        async with service._uow_factory() as uow:
            resource_ref = await uow.knowledge_documents.create_query_ticket(access, ticket_id=uuid7(),
                agent_id=agent_id, target_id=target_id, request=request, snapshots=shard, filters=filters,
                expires_at=retrieval.clock() + timedelta(seconds=retrieval.limits.scope_ttl_seconds))
            await uow.commit()
        authorized = await retrieval.create_scope(access, resource_ref=resource_ref)
        receipt = await retrieval.retrieve(access, scope=authorized, profile_ref="default",
            query=QueryInput(QueryMode.RELEVANCE, min(100, max(request.max_results * 3, len(shard))), ("TEXT", "TABLE"), text=request.query))
        diagnostics["discovery_shards"] += 1
        diagnostics["evidence_batches"] += 1
        allowed = {row["document_version_id"]: row for row in shard}
        for rank, evidence in enumerate(receipt.evidence_set.evidence, start=1):
            candidate = allowed.get(evidence.key.document_version_ref)
            if (candidate is None or evidence.key.owner_binding_ref != "aiops"
                or evidence.key.parse_run_ref != candidate["parse_run_id"]):
                continue
            if not evidence.mime_type.startswith("text/"):
                continue
            descriptor, chunks = await retrieval.stream_content(access, retrieval_ref=receipt.retrieval_ref,
                stable_key=evidence.key.stable_key)
            body = b""
            async for block in chunks:
                body += block
            text = body.decode("utf-8")
            identity = candidate["document_version_id"]
            discovery_scores[identity] = max(discovery_scores.get(identity, 0.), 1. / (60 + rank))
            evidence_items.append({"document_version_id": identity, "parse_run_id": evidence.key.parse_run_ref, "quote": text,
                "locator": {"evidence_key": encode(evidence.key), "locators": encode(evidence.locators),
                    "source_keys": encode(evidence.source_keys), "source_spans": encode(evidence.source_spans),
                    "retrieval_ref": receipt.retrieval_ref, "content_hash": descriptor.content_hash,
                    "scope_hash": authorized.scope_hash, "owner_binding_ref": "aiops"}})
            if candidate["asset_kind"] == "DIAGNOSIS_CASE":
                record = await service._documents.inspect(access, UUID(identity))
                if record["source_hash"] == candidate["profile"]["knowledge_projection"]["body_sha256"]:
                    case_documents[identity] = {identity}
    # 正文及语义判断完成后统一重验，期间不持有数据库事务。
    grouped, scores = {}, {}
    for item in evidence_items:
        revision = str(item.get("document_version_id") or "")
        candidate = by_revision.get(revision)
        if candidate is None:
            continue
        citation = dict(item) if str(item.get("quote") or "").strip() else None
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
    current = {(row["asset_version_id"], str(row["document_version_id"])) for row in checked}
    # 多文件版本逐条重验，不能用最后一个文档的授权代替其它文件。
    for key in tuple(grouped):
        row = grouped[key]
        row["citation_pack"] = [citation for citation in row["citation_pack"]
            if (key, str(citation["document_version_id"])) in current]
        if not row["citation_pack"]:
            del grouped[key]
        else:
            cited = {item["document_version_id"] for item in row["citation_pack"]}
            row["document_bindings"] = [{name: value for name, value in item.items() if name != "created_at"}
                for item in checked if item["asset_version_id"] == key and item["document_version_id"] in cited]
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
