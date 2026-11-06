"""固定 App 后端的完整召回编排，不调用回答模型或独立重排模型。"""

import asyncio
from dataclasses import replace

from ..contracts.content import ContentKind
from ..contracts.errors import AlgorithmError, ErrorCode, require
from ..contracts.identity import AuthorizedScope, EvidenceKey, fingerprint
from ..contracts.query import QueryInput, QueryMode
from ..contracts.wire import encode
from ..contracts.retrieval import (ChannelRequest, ChannelStatus, EvidenceSet, PrimaryHit, QueryPlan,
                                   RelationRequest, RetrievalProfile, StageRecord)
from ..execution import Deadline
from ..ports.retrieval import AppBoundBackend
from ..projections.contracts import ModelGateway, ModelInput
from ..references.relations import LinkedManifest, RelationType, direct_figure_manifest
from .fusion import RankedChannel, merge_channel_shards, reciprocal_rank_fusion
from .selection import (CandidateTargetMatch, CoverageTarget, EvidenceGroup, MatchStatus, PayloadUnit,
                         select_evidence_groups)


_TEXT = {ContentKind.PARAGRAPH, ContentKind.SECTION, ContentKind.CAPTION, ContentKind.LIST_ITEM,
         ContentKind.TABLE_ROW, ContentKind.CODE, ContentKind.FORMULA, ContentKind.FOOTNOTE}


class RetrievalEngine:
    def __init__(self, backend: AppBoundBackend, model_gateway: ModelGateway | None):
        self.backend, self.model_gateway = backend, model_gateway
        self.binding = backend.binding

    async def retrieve(self, query: QueryInput, scope: AuthorizedScope, profile: RetrievalProfile,
                       *, targets: tuple[CoverageTarget, ...] = ()) -> EvidenceSet:
        self.binding.check_owner(scope.owner_binding_ref)
        self.backend.capabilities.validate(query)
        require(query.requested_count <= profile.max_primary_count, "用户数量超过服务上限，不得静默截断")
        require(len(query.image_handles) <= profile.candidate_budget, "查询图片数超过显式工作预算")
        if query.combination == "COMPOSED" or query.text and query.image_handles:
            raise AlgorithmError(ErrorCode.MODALITY_UNAVAILABLE, "尚未登记已评测的组合输入能力", stage="plan")
        return await Deadline(profile.timeout_seconds).run(self._retrieve(query, scope, profile, targets), stage="retrieval")

    async def _retrieve(self, query, scope, profile, targets):
        backend = self.backend
        snapshot = await backend.snapshot(scope)
        snapshot.validate_scope(self.binding, scope)
        profile_hash = fingerprint(encode(profile))
        request_id = fingerprint({"binding": self.binding.binding_ref, "query": query.query_hash,
                                  "scope": scope.scope_hash, "snapshot": snapshot.snapshot_ref,
                                  "profile": profile_hash, "targets": encode(targets)})
        channels, statuses, input_matches, trace, representation_hashes = [], [], [], [], []
        planned_channels = tuple(channel for channel in profile.channels if (
            channel.direction.startswith("TEXT_") and query.text
            or channel.direction.startswith("IMAGE_") and query.image_handles))
        continuation, scope_exhausted = None, False
        if query.mode != QueryMode.RELEVANCE:
            page = await backend.search_ordered(query, scope, snapshot)
            require(page.snapshot == snapshot and page.query_hash == query.query_hash, "原生排序快照或查询身份不一致")
            require(len(page.hits) <= query.requested_count and len(set(page.hits)) == len(page.hits), "原生排序主命中数量或身份无效")
            require(page.scope_exhausted == (page.continuation is None), "排序未穷尽却没有续取")
            ranked_keys = page.hits
            scores = {}
            continuation, scope_exhausted = page.continuation, page.scope_exhausted
        else:
            require(bool(planned_channels), "相关性计划没有登记适用于当前输入的召回通道")
            tasks = [asyncio.create_task(self._channel(query, scope, snapshot, channel, profile)) for channel in planned_channels]
            try:
                results = await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            for channel, result in zip(planned_channels, results):
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    if isinstance(result, AlgorithmError) and result.code == ErrorCode.SCOPE_CLOSURE_BROKEN:
                        raise result
                    reason = result.code.value if isinstance(result, AlgorithmError) else ErrorCode.CHANNEL_FAILED.value
                    statuses.append(ChannelStatus(channel.channel_key, channel.required, "FAILED", reason, False, False))
                else:
                    ranked, matched, represented = result
                    channels.append(ranked)
                    input_matches.extend(matched)
                    representation_hashes.extend(represented)
                    statuses.append(ChannelStatus(channel.channel_key, channel.required,
                                                   "PARTIAL" if ranked.status_reason else "COMPLETE", ranked.status_reason,
                                                   ranked.approximate, ranked.exhausted))
            fused = reciprocal_rank_fusion(tuple(channels), weights={c.channel_key: c.weight for c in profile.channels
                if any(r.channel_key == c.channel_key for r in channels)}, constant=profile.rrf_constant)
            initial = tuple(dict.fromkeys(hit.key for channel in channels for hit in channel.hits))
            ranked_keys = tuple(hit.key for hit in fused)
            require(set(initial) == set(ranked_keys), "融合丢失或伪造身份", stage="fusion")
            trace.append(StageRecord("fusion", initial, ranked_keys, ()))
            scores = {hit.key: hit.score for hit in fused}
            scope_exhausted = all(s.exhausted for s in statuses) and all(s.status == "COMPLETE" for s in statuses)
        for key in ranked_keys:
            self.binding.check_owner(key.owner_binding_ref)
        admitted = ranked_keys[:profile.candidate_budget]
        trace.append(StageRecord("candidate_budget", ranked_keys, admitted,
                     tuple((key, "CANDIDATE_BUDGET") for key in ranked_keys[len(admitted):])))
        items = await self._load(admitted, scope, snapshot, trace, "load_primary")
        valid = await self._validate(tuple(items), scope, snapshot, trace, "validate_candidates")
        items = {key: item for key, item in items.items() if key in valid}
        output_types = set(query.output_types)
        legal = tuple(key for key in admitted if key in items and ("MIXED" in output_types
                    or items[key].kind.value in output_types or "TEXT" in output_types and items[key].kind in _TEXT))
        trace.append(StageRecord("output_type", tuple(items), legal,
                     tuple((key, "OUTPUT_TYPE_MISMATCH") for key in items if key not in legal)))
        items = {key: items[key] for key in legal}
        manifests, relation_failed = [], False
        for key in legal:
            item = items[key]
            anchor_kind = "FIGURE" if item.kind in {ContentKind.FIGURE, ContentKind.IMAGE, ContentKind.PHOTO, ContentKind.PAGE} else (
                "REGION" if item.kind == ContentKind.REGION else "CAPTION" if item.kind == ContentKind.CAPTION else "TEXT")
            edges, cursor, exhausted, relation_reason = [], None, False, None
            seen_cursors = set()
            try:
                for _ in range(profile.max_relation_pages):
                    page = await backend.follow_relations(RelationRequest(item.source_keys, profile.relation_page_size, cursor), scope, snapshot)
                    require(len(page.relations) <= profile.relation_page_size and page.exhausted == (page.continuation is None), "关联分页合同无效")
                    for relation in page.relations:
                        self.binding.check_owner(relation.source_key.owner_binding_ref)
                        self.binding.check_owner(relation.target_key.owner_binding_ref)
                    edges.extend(page.relations)
                    exhausted = page.exhausted
                    if exhausted:
                        break
                    require(page.continuation not in seen_cursors, "关联续取游标循环")
                    seen_cursors.add(page.continuation)
                    cursor = page.continuation
            except AlgorithmError as exc:
                if exc.code == ErrorCode.SCOPE_CLOSURE_BROKEN:
                    raise
                relation_failed = True
                relation_reason = exc.code.value
            except Exception:
                relation_failed = True
                relation_reason = ErrorCode.CHANNEL_FAILED.value
            # 多页重复边去重但不吞掉同图的不同引用出现。
            by_edge = {}
            for edge in edges:
                require(edge.relation_key not in by_edge or edge == by_edge[edge.relation_key], "相同关系身份内容冲突")
                by_edge[edge.relation_key] = edge
            linked, selected_edges = [], {}
            for source_key in item.source_keys:
                relevant = tuple(edge for edge in by_edge.values() if not (
anchor_kind == "TEXT" and edge.relation_type in {RelationType.REFERS_TO, RelationType.CITES_DOCUMENT}
                    and edge.source_key == source_key and item.source_spans
                    and not any(span.node_key == edge.source_span.node_key
                        and span.char_start <= edge.source_span.char_start < edge.source_span.char_end <= span.char_end
                        for span in item.source_spans)))
                manifest = direct_figure_manifest(source_key, anchor_kind=anchor_kind, relations=relevant,
                                                  relation_pages_exhausted=exhausted)
                linked.extend(manifest.linked_keys)
                selected_edges.update({edge.relation_key: edge for edge in manifest.relations})
            manifests.append(LinkedManifest(key, tuple(dict.fromkeys(linked)), tuple(selected_edges.values()),
                "COMPLETE" if exhausted else "PARTIAL", continuation=None if exhausted else cursor,
                incomplete_reason=relation_reason or (None if exhausted else "RELATION_PAGE_BUDGET")))
        linked_keys = tuple(dict.fromkeys(k for m in manifests for k in m.linked_keys if k not in items))
        items.update(await self._load(linked_keys, scope, snapshot, trace, "load_linked"))
        final_valid = await self._validate(tuple(items), scope, snapshot, trace, "final_revalidation")
        items = {k: item for k, item in items.items() if k in final_valid}
        legal = tuple(k for k in legal if k in items)
        final_manifests = []
        for manifest in manifests:
            if manifest.anchor_key not in items:
                continue
            linked = tuple(k for k in manifest.linked_keys if k in items)
            final_manifests.append(replace(manifest, linked_keys=linked,
                relations=tuple(edge for edge in manifest.relations if all(k in items or k in items[manifest.anchor_key].source_keys
                    for k in (edge.source_key, edge.target_key))),
                manifest_status="PARTIAL" if len(linked) != len(manifest.linked_keys) else manifest.manifest_status))
        input_targets = tuple(CoverageTarget(f"input:{i}", "INPUT_IMAGE", query.combination == "ALL", f"query-image:{i}",
                              profile_hash) for i in range(len(query.image_handles)))
        all_targets = targets + input_targets
        # 输入图片覆盖由原生子查询记录；业务目标匹配器不能再为同一对身份追加 UNKNOWN。
        mapped = await backend.match_targets(tuple(items), targets, scope, snapshot) if targets else ()
        # 输入方向命中仅是覆盖线索，不是答案支持判断。
        matches = tuple(mapped) + tuple(m for m in input_matches if m.evidence_key in items)
        groups = tuple(EvidenceGroup(m.anchor_key.stable_key, m.anchor_key,
            tuple(PayloadUnit(items[k].key, items[k].estimated_cost) for k in (m.anchor_key,) + m.linked_keys),
            tuple(edge.relation_key for edge in m.relations)) for m in final_manifests)
        selection = select_evidence_groups(groups, targets=all_targets, matches=matches,
            budget=profile.payload_budget, max_cover_states=profile.max_cover_states)
        primary = tuple(PrimaryHit(key, index, key.stable_key, scores.get(key))
                        for index, key in enumerate(legal[:query.requested_count], 1))
        dropped = tuple((key, "USER_RESULT_COUNT") for key in legal[query.requested_count:])
        trace.append(StageRecord("presentation", legal, tuple(hit.evidence_key for hit in primary), dropped))
        trace.append(StageRecord("context_selection", tuple(items), selection.selected_evidence_keys,
                     tuple((key, "CONTEXT_BUDGET_OR_COVERAGE") for key in items if key not in selection.selected_evidence_keys)))
        failed = any(s.status == "FAILED" for s in statuses)
        partial = (failed or relation_failed or any(s.status != "COMPLETE" or s.approximate or not s.exhausted for s in statuses)
                   or any(m.manifest_status != "COMPLETE" for m in final_manifests)
                   or any(record.excluded for record in trace if record.stage not in {"presentation", "context_selection"})
                   or len(legal) > profile.max_primary_count)
        if primary:
            status, reason = ("PARTIAL" if partial else "COMPLETE"), None
            if not selection.selected_evidence_keys:
                reason = ErrorCode.CONTEXT_BUDGET_INSUFFICIENT.value
        elif failed:
            status, reason = "FAILED", ErrorCode.CHANNEL_FAILED.value
        elif ranked_keys:
            status, reason = "PARTIAL", ErrorCode.NO_ACCESSIBLE_PAYLOAD.value
        elif partial:
            status, reason = "PARTIAL", next((s.reason for s in statuses if s.reason), "SEARCH_NOT_EXHAUSTED")
        else:
            status, reason = "EMPTY", ErrorCode.NO_MATCH.value
        plan = QueryPlan(query.query_hash, scope.scope_hash, profile.profile_version, query.mode,
                         tuple(c.channel_key for c in planned_channels) if query.mode == QueryMode.RELEVANCE else (),
                         all_targets, profile_hash, tuple(representation_hashes))
        return EvidenceSet(request_id, self.binding.binding_ref, scope.scope_ref, scope.scope_hash, snapshot,
            query.query_hash, query.mode, status, reason, primary, tuple(items.values()), tuple(final_manifests),
            selection, matches, tuple(statuses), tuple(trace), plan, continuation, scope_exhausted)

    async def _channel(self, query, scope, snapshot, channel, profile):
        async def run():
            spec = channel.representation_spec
            image_query = channel.direction.startswith("IMAGE_")
            require(bool(query.image_handles) if image_query else bool(query.text), "通道与输入方向不匹配")
            indexes = range(len(query.image_handles)) if image_query else (None,)
            ranked_results, matches, represented = [], [], []
            for index in indexes:
                representation = None
                if spec is not None:
                    if self.model_gateway is None:
                        raise AlgorithmError(ErrorCode.MODALITY_UNAVAILABLE, "未注入查询表示模型", stage="query_model")
                    value = ModelInput("IMAGE", image_handle=query.image_handles[index]) if image_query else ModelInput("TEXT", text=query.text)
                    representation = await self.model_gateway.represent(value, spec)
                    representation.validate(spec, profile.max_representation_elements)
                    represented.append((channel.channel_key, index, fingerprint(encode(representation))))
                request = ChannelRequest(query, channel, representation, index)
                operation = {"LEXICAL": self.backend.search_lexical, "DENSE": self.backend.search_dense,
                             "MULTIVECTOR": self.backend.search_multivector,
                             "NEAR_DUPLICATE": self.backend.search_near_duplicate}[channel.kind]
                shards = await operation(request, scope, snapshot)
                if len(shards) > profile.max_shards or any(len(s.hits) > channel.limit for s in shards):
                    raise AlgorithmError(ErrorCode.RESOURCE_LIMIT, "通道分片或候选超过登记上限", stage="channel")
                for shard in shards:
                    for hit in shard.hits:
                        self.binding.check_owner(hit.key.owner_binding_ref)
                    require(shard.snapshot == snapshot and shard.channel_key == channel.channel_key
                            and shard.signal_family == channel.signal_family, "通道身份或快照不匹配")
                    if spec is not None:
                        require(shard.space_id == spec.space_id, "通道索引空间与查询模型不一致")
                ranked = merge_channel_shards(shards)
                ranked = replace(ranked, hits=ranked.hits[:channel.limit],
                                 exhausted=ranked.exhausted and len(ranked.hits) <= channel.limit)
                ranked_results.append(ranked)
                if index is not None:
                    matches.extend(CandidateTargetMatch(f"input:{index}", hit.key, MatchStatus.MATCHED,
                        ("SUBQUERY_HIT", channel.channel_key), "native-query-direction") for hit in ranked.hits)
            best = {}
            for ranked in ranked_results:
                for rank, hit in enumerate(ranked.hits, 1):
                    if hit.key not in best or rank < best[hit.key][0]:
                        best[hit.key] = (rank, hit)
            ordered = tuple(value[1] for _, value in sorted(best.items(), key=lambda pair: (pair[1][0], pair[0].stable_key)))
            return RankedChannel(channel.channel_key, channel.signal_family, snapshot, ordered[:channel.limit],
                any(r.approximate for r in ranked_results), all(r.exhausted for r in ranked_results)
                and len(ordered) <= channel.limit,
                next((r.status_reason for r in ranked_results if r.status_reason), None)), tuple(matches), tuple(represented)
        return await Deadline(channel.timeout_seconds).run(run(), stage="channel")

    async def _load(self, keys, scope, snapshot, trace, stage):
        if not keys:
            return {}
        batch = await self.backend.load_evidence(keys, scope, snapshot)
        actual = [item.key for item in batch.items]
        missing = [key for key, _ in batch.missing]
        for key in actual + missing:
            self.binding.check_owner(key.owner_binding_ref)
        require(len(set(actual + missing)) == len(actual + missing) and set(actual + missing) == set(keys),
                "证据读取不能漏报、重复或替换身份", stage="assembly")
        usable = {item.key: item for item in batch.items if item.payload_status != "UNAVAILABLE"}
        excluded = batch.missing + tuple((item.key, "NO_ACCESSIBLE_PAYLOAD") for item in batch.items
                                       if item.payload_status == "UNAVAILABLE")
        trace.append(StageRecord(stage, keys, tuple(usable), excluded))
        return usable

    async def _validate(self, keys, scope, snapshot, trace, stage):
        if not keys:
            return set()
        result = await self.backend.revalidate(keys, scope, snapshot)
        all_keys = result.valid_keys + tuple(key for key, _ in result.excluded)
        for key in all_keys:
            self.binding.check_owner(key.owner_binding_ref)
        require(len(set(all_keys)) == len(all_keys) and set(all_keys) == set(keys), "复核结果身份不守恒")
        trace.append(StageRecord(stage, keys, result.valid_keys, result.excluded))
        return set(result.valid_keys)
