"""纯内存 fixture，不访问数据库、网络或真实资料。"""

from dataclasses import replace
from hashlib import sha256

from knowledge_algorithms.contracts.content import ContentKind, ContentNode, Provenance, ProvenanceKind
from knowledge_algorithms.contracts.identity import AppBackendBinding, AuthorizedScope, EvidenceKey, Locator, RetrievalSnapshot
from knowledge_algorithms.contracts.query import BackendCapabilities, QueryMode
from knowledge_algorithms.contracts.retrieval import (ChannelSpec, EvidenceBatch, EvidenceItem, RelationPage,
                                                      RetrievalProfile, ValidationResult)
from knowledge_algorithms.parsing.contracts import LayoutProfile, ParseProfile, SourceDescriptor
from knowledge_algorithms.parsing.events import ParseRunIdentity, local_key
from knowledge_algorithms.ports.ordered import OrderedPage
from knowledge_algorithms.projections.contracts import (AllowedContent, ModelInput, ProjectionProfile,
                                                       Representation, RepresentationSpec, UnitSource)
from knowledge_algorithms.retrieval.fusion import ChannelShard, NativeHit
from knowledge_algorithms.retrieval.maxsim import TokenMatrix
from knowledge_algorithms.retrieval.selection import PayloadCost


def parse_profile(**overrides):
    return replace(ParseProfile("parse-fixture/v1", "ir/v1", ("zh", "en"), 100000, 100, 1000,
        100000, 100000, 10, 2.0, 2, ("OCR", "MODEL_TRANSCRIPTION"), 0.1, 0.8, True,
        LayoutProfile(0.62, 0.18)), **overrides)


def source(text="第一段。\n\n第二段。", owner="app-a"):
    class MemorySource:
        descriptor = SourceDescriptor(owner, "doc-v1", sha256(text.encode()).hexdigest(), "text/plain",
                                      len(text.encode()), "source-1")

        async def chunks(self):
            encoded = text.encode()
            for i in range(0, len(encoded), 3):
                yield encoded[i:i + 3]
    return MemorySource()


def run_for(reader, profile):
    return ParseRunIdentity(reader.descriptor.owner_binding_ref, "job-1", "parse-1", 1, 1,
                            reader.descriptor.sha256, profile.profile_hash)


def original(name, owner="app-a"):
    return EvidenceKey(owner, "doc-v1", "parse-1", name)


def candidate(name, owner="app-a"):
    return replace(original(name, owner), projection_ref="projection-1")


def provenance(kind=ProvenanceKind.NATIVE):
    return Provenance(kind, "fixture", "fixture/v1", "a" * 64)


def node(name="p", text="如图 1-1 所示。", kind=ContentKind.PARAGRAPH, **overrides):
    native_ref = "native:" + name
    key = original(local_key(algorithm_version="ir/v1", native_ref=native_ref, kind=kind.value))
    return replace(ContentNode(key, kind, text, (Locator(logical_ref=native_ref),), provenance(), native_ref, 0), **overrides)


class CharacterTokenizer:
    revision = "fixture-character-tokenizer/v1"

    def count(self, text):
        return len(text)


def projection_profile(**overrides):
    return replace(ProjectionProfile("projection-fixture/v1", 100, 100000, 1000, 2, 1000, 2.0, ()), **overrides)


def spec(kind="DENSE", input_kind="TEXT", direction="TEXT_TO_TEXT"):
    return RepresentationSpec("fixture-model", "fixture/v1", "processor/v1", "raw", 2, kind,
                               input_kind, "DOT", "NONE", frozenset({direction}))


class FakeModel:
    def __init__(self):
        self.calls = []

    async def represent(self, value, specification):
        self.calls.append(value)
        if specification.kind == "DENSE":
            return Representation(specification.space_id, vector=(1.0, 0.0))
        return Representation(specification.space_id, matrix=TokenMatrix(specification.space_id, ((1.0, 0.0),), (True,)))


class AllowedReader:
    def __init__(self, contents, owner="app-a", view_hash="view-hash"):
        self.binding = AppBackendBinding(owner, owner)
        self.document_version_ref, self.parse_run_ref = "doc-v1", "parse-1"
        self.allowed_view_ref, self.allowed_view_hash = "allowed-v1", view_hash
        self.values, self.closed = contents, False

    async def contents(self):
        try:
            for content in self.values:
                yield content
        finally:
            self.closed = True


def allowed(content_node):
    sources = (UnitSource(0, len(content_node.text), content_node.key, 0, len(content_node.text)),) if content_node.text else ()
    return AllowedContent(content_node.key, content_node.kind, content_node.text, sources,
                          content_node.locators, content_node.provenance)


def retrieval_profile(channels=None, **overrides):
    if channels is None:
        channels = (ChannelSpec("lexical", "lexical", "LEXICAL", "TEXT_TO_TEXT", 1.0, 100, 1.0, True),)
    return replace(RetrievalProfile("retrieval-fixture/v1", channels, 60, 100, 10,
        PayloadCost(100, 10, 10000), 100, 100, 3, 1000, 2.0), **overrides)


class FakeBackend:
    def __init__(self, owner="app-a"):
        self.binding = AppBackendBinding(owner, owner)
        self.capabilities = BackendCapabilities(frozenset({"category"}), frozenset({"EQ"}),
            frozenset({"date", "id"}), "id", frozenset({QueryMode.FILTERED_ORDER, QueryMode.EXHAUSTIVE}))
        self.scope = AuthorizedScope(owner, "scope-ref", "scope-hash", "auth-v1")
        self.fixed_snapshot = RetrievalSnapshot(owner, "scope-hash", "snapshot-1", "index-1", "withdrawal-1")
        self.items, self.rows, self.relations, self.calls = {}, {}, (), []
        self.revoked, self.final_revoked = set(), set()
        self.matches, self.relation_error = (), None

    def add(self, name, kind=ContentKind.PARAGRAPH, cost=1, projected=True):
        raw = original(name, self.binding.binding_ref)
        key = candidate(name, self.binding.binding_ref) if projected else raw
        self.items[key] = EvidenceItem(key, (raw,), kind, (Locator(logical_ref=name),), provenance(),
            "view-1", "content-" + name, sha256(name.encode()).hexdigest(),
            "image/png" if kind in {ContentKind.FIGURE, ContentKind.IMAGE} else "text/plain", cost,
            PayloadCost(cost, int(kind in {ContentKind.FIGURE, ContentKind.IMAGE}), cost), 2000000000)
        return key

    def check(self, scope, snapshot=None):
        self.binding.check_owner(scope.owner_binding_ref)
        if snapshot is not None:
            assert snapshot == self.fixed_snapshot

    async def snapshot(self, scope):
        self.check(scope)
        self.calls.append("snapshot")
        return self.fixed_snapshot

    async def _search(self, request, scope, snapshot):
        self.check(scope, snapshot)
        self.calls.append(request.channel.kind)
        rows = self.rows.get((request.channel.channel_key, request.input_index), self.rows.get(request.channel.channel_key, ()))
        if isinstance(rows, Exception):
            raise rows
        space = request.channel.representation_spec.space_id if request.channel.representation_spec else "lexical-space"
        return (ChannelShard(request.channel.channel_key, request.channel.signal_family, space, "native/v1",
                             "global-stats", snapshot, tuple(NativeHit(key, score) for key, score in rows)),)

    search_lexical = _search
    search_dense = _search
    search_multivector = _search
    search_near_duplicate = _search

    async def search_ordered(self, query, scope, snapshot):
        self.check(scope, snapshot)
        self.calls.append("ordered")
        keys = tuple(row[0] for row in self.rows.get("ordered", ()))[:query.requested_count]
        return OrderedPage(snapshot, query.query_hash, keys, True, None)

    async def load_evidence(self, keys, scope, snapshot):
        self.check(scope, snapshot)
        self.calls.append("load")
        return EvidenceBatch(tuple(self.items[k] for k in keys if k in self.items),
                             tuple((k, "NO_ACCESSIBLE_PAYLOAD") for k in keys if k not in self.items))

    async def revalidate(self, keys, scope, snapshot):
        self.check(scope, snapshot)
        self.calls.append("validate")
        invalid = self.final_revoked if self.calls.count("validate") > 1 else self.revoked
        return ValidationResult(tuple(k for k in keys if k not in invalid),
                                 tuple((k, "SCOPE_CHANGED") for k in keys if k in invalid))

    async def follow_relations(self, request, scope, snapshot):
        self.check(scope, snapshot)
        self.calls.append("relations")
        if self.relation_error:
            raise self.relation_error
        return RelationPage(self.relations, True, None)

    async def match_targets(self, keys, targets, scope, snapshot):
        self.check(scope, snapshot)
        return self.matches
