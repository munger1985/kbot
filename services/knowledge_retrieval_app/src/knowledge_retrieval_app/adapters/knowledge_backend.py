"""固定 App 的有界精确召回适配器；每次调用重新核权，不持有业务事务。"""

from dataclasses import replace
import json

from knowledge_algorithms.contracts.cursor import CursorBinding
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.contracts.identity import canonical_json, fingerprint
from knowledge_algorithms.contracts.query import BackendCapabilities, QueryMode
from knowledge_algorithms.contracts.retrieval import EvidenceBatch, RelationPage, ValidationResult
from knowledge_algorithms.ports.ordered import OrderedPage
from knowledge_algorithms.retrieval.fusion import ChannelShard
from knowledge_algorithms.retrieval.selection import CandidateTargetMatch, MatchStatus
from ..domain.knowledge_ingestion import BINDING


class PublishedKnowledgeBackend:
    binding = BINDING
    capabilities = BackendCapabilities(
        frozenset({"title", "business_asset_ref", "created_at", "document_version_ref", "kind"}),
        frozenset({"EQ", "NE", "LT", "LE", "GT", "GE", "PREFIX"}),
        frozenset({"title", "created_at", "document_version_ref", "kind", "stable_key"}),
        "stable_key", frozenset({QueryMode.FILTERED_ORDER, QueryMode.EXHAUSTIVE}))

    def __init__(self, *, index, items, relations, scope, snapshot, subject_ref,
                 expires_at, profile_hash, cursor_codec, validate, clock,
                 primary_cursor=None, image_fingerprints=None):
        self.index, self.items, self.relations = index, items, tuple(relations)
        self.scope, self.fixed_snapshot, self.subject_ref = scope, snapshot, subject_ref
        self.expires_at, self.profile_hash = expires_at, profile_hash
        self.codec, self.validate, self.clock = cursor_codec, validate, clock
        self.primary_cursor = primary_cursor
        self.image_fingerprints = image_fingerprints or {}
        self.text_scan = None
        self.scan_policy = None

    async def check(self, scope, snapshot=None):
        self.binding.check_owner(scope.owner_binding_ref)
        require(scope == self.scope and (snapshot is None or snapshot == self.fixed_snapshot),
                "后端范围或索引快照不一致")
        await self.validate()

    async def snapshot(self, scope):
        await self.check(scope)
        return self.fixed_snapshot

    def cursor_binding(self, *, role, query_hash, mode, order, position=None):
        return CursorBinding(self.binding.binding_ref, self.subject_ref, self.scope.scope_hash,
            self.fixed_snapshot.snapshot_ref, query_hash, self.profile_hash, role, mode,
            canonical_json(order), canonical_json(position or {}), self.expires_at)

    async def search_ordered(self, query, scope, snapshot):
        await self.check(scope, snapshot)
        self.capabilities.validate(query)
        keys = self.index.ordered(query)
        binding = self.cursor_binding(role="PRIMARY", query_hash=query.query_hash, mode=query.mode.value,
                                      order=[(o.field, o.descending) for o in query.order_by])
        start = 0
        if self.primary_cursor:
            decoded = self.codec.decode(self.primary_cursor, expected=binding, now=int(self.clock().timestamp()))
            previous = json.loads(decoded.position_json).get("last_key")
            positions = [i for i, key in enumerate(keys) if key.stable_key == previous]
            require(len(positions) == 1, "排序游标的精确位置已经失效")
            start = positions[0] + 1
        hits = keys[start:start + query.requested_count]
        exhausted = start + len(hits) == len(keys)
        continuation = None if exhausted else self.codec.encode(replace(
            binding, position_json=canonical_json({"last_key": hits[-1].stable_key})))
        return OrderedPage(snapshot, query.query_hash, hits, exhausted, continuation)

    async def search_channel(self, request, scope, snapshot):
        await self.check(scope, snapshot)
        image_hash, visual_hash = None, None
        if request.channel.kind == "NEAR_DUPLICATE":
            handle = request.query.image_handles[request.input_index]
            require(handle in self.image_fingerprints, "查询图像未获当前主体授权")
            image_hash, visual_hash = self.image_fingerprints[handle]
        hits = self.index.search(request, image_sha256=image_hash, visual_hash=visual_hash)
        scan_complete, scan_reason = True, None
        if request.channel.kind == "LEXICAL" and self.text_scan is not None and self.scan_policy is not None:
            from knowledge_algorithms.ports.text_scan import lexical_family_order
            from knowledge_algorithms.retrieval.fusion import NativeHit
            try:
                scan_request = self.scan_policy(request.query)
            except Exception:
                scan_request = None
                scan_complete, scan_reason = False, "TEXT_SCAN_PLAN_FAILED"
            if scan_request is not None:
                try:
                    scanned = await self.text_scan.scan(scan_request, scope, snapshot)
                    require(all(match.key in self.items for match in scanned.matches), "扫描返回未授权对象")
                    eligible = {indexed.key for indexed in self.index.units if self.index.eligible(indexed, request.query)}
                    matches = tuple(match for match in scanned.matches if match.key in eligible)
                    keys = lexical_family_order(tuple(hit.key for hit in hits), matches,
                        pattern_count=len(scan_request.patterns), combination=scan_request.combination)
                    hits = tuple(NativeHit(key, float(len(keys) - position)) for position, key in enumerate(keys))
                    scan_complete = scanned.exhaustive
                    scan_reason = None if scanned.status == "COMPLETE" else "TEXT_SCAN_" + scanned.status
                except Exception:
                    # 可选扫描故障不能清空 BM25 命中；故障状态必须向调用方可见。
                    scan_complete, scan_reason = False, "TEXT_SCAN_FAILED"
        spec = request.channel.representation_spec
        return (ChannelShard(request.channel.channel_key, request.channel.signal_family,
            spec.space_id if spec else request.channel.kind.lower() + "/v1",
            "authorized-exact/" + request.channel.kind.lower() + "/v1",
            self.index.statistics_ref, snapshot, hits[:request.channel.limit],
            exhausted=scan_complete and len(hits) <= request.channel.limit, status_reason=scan_reason),)

    search_lexical = search_channel
    search_dense = search_channel
    search_multivector = search_channel
    search_near_duplicate = search_channel

    async def load_evidence(self, keys, scope, snapshot):
        await self.check(scope, snapshot)
        for key in keys:
            self.binding.check_owner(key.owner_binding_ref)
        return EvidenceBatch(tuple(self.items[key] for key in keys if key in self.items),
                             tuple((key, "EVIDENCE_NOT_IN_ALLOWED_VIEW") for key in keys if key not in self.items))

    async def revalidate(self, keys, scope, snapshot):
        await self.check(scope, snapshot)
        for key in keys:
            self.binding.check_owner(key.owner_binding_ref)
        return ValidationResult(tuple(key for key in keys if key in self.items),
                                tuple((key, "EVIDENCE_NOT_IN_ALLOWED_VIEW") for key in keys if key not in self.items))

    async def follow_relations(self, request, scope, snapshot):
        await self.check(scope, snapshot)
        require(0 < request.limit <= 1000 and all(key in self.items for key in request.source_keys),
                "关联请求超出允许对象或分页预算")
        sources = set(request.source_keys)
        figures = sources | {edge.target_key for edge in self.relations
                             if edge.source_key in sources and (edge.relation_type.value in {"REFERS_TO", "CAPTION_OF"}
                                or edge.relation_type.value == "PART_OF" and edge.basis == "EXPLICIT_TRANSCRIPTION_REGION")}
        parents = {edge.target_key for edge in self.relations
                   if edge.source_key in figures and edge.relation_type.value == "PART_OF"}
        edges = tuple(sorted((edge for edge in self.relations if (
            edge.source_key in sources or edge.target_key in figures
            or edge.source_key in figures | parents and edge.relation_type.value in {"HAS_ASSET", "PART_OF"})),
            key=lambda edge: edge.relation_key))
        binding = self.cursor_binding(role="RELATION",
            query_hash=fingerprint(sorted(key.stable_key for key in sources)), mode="RELATION", order=["relation_key"])
        start = 0
        if request.cursor:
            decoded = self.codec.decode(request.cursor, expected=binding, now=int(self.clock().timestamp()))
            position = json.loads(decoded.position_json).get("last_key")
            positions = [i for i, edge in enumerate(edges) if edge.relation_key == position]
            require(len(positions) == 1, "关联游标位置无效")
            start = positions[0] + 1
        selected = edges[start:start + request.limit]
        exhausted = start + len(selected) == len(edges)
        continuation = None if exhausted else self.codec.encode(replace(binding,
            position_json=canonical_json({"last_key": selected[-1].relation_key})))
        return RelationPage(selected, exhausted, continuation)

    async def match_targets(self, keys, targets, scope, snapshot):
        await self.check(scope, snapshot)
        # 没有业务证明时保留 UNKNOWN，不调用模型逐条删候选。
        return tuple(CandidateTargetMatch(target.target_key, key, MatchStatus.UNKNOWN,
                     (), "app-target-policy/not-asserted") for target in targets for key in keys)
