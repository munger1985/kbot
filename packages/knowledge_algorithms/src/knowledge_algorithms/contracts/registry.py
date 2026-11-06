"""K0 固定线合同注册表；这里只选择类型，不选择 App 或数据库。"""

from dataclasses import MISSING, fields
from enum import Enum
from typing import get_args, get_origin, get_type_hints

from .cursor import CursorBinding
from .errors import require
from .identity import AppBackendBinding, AuthorizedScope, EvidenceKey, RetrievalSnapshot
from .query import QueryInput
from .retrieval import EvidenceSet, QueryPlan, RetrievalProfile
from .runtime import (CitationLabelMap, DelegationClaims, ErrorEnvelope, GroundedAnswerArtifact,
ModelEvidenceDelivery, RetrievalEvidenceArtifact, RetrievalRequest,
                      AppRetrievalArtifact, KnowledgeCompositionArtifact)
from .wire import decode, encode
from ..parsing.contracts import ParseProfile, SourceDescriptor
from ..parsing.engine import ParsedDocument
from ..parsing.events import ParseArtifact, ParseEventType, ParseRunIdentity
from ..projections.contracts import ProjectionBatch, ProjectionProfile, RepresentationSpec


CONTRACTS = {
    "knowledge.backend_binding.v1": AppBackendBinding,
    "knowledge.authorized_scope.v1": AuthorizedScope,
    "knowledge.retrieval_snapshot.v1": RetrievalSnapshot,
    "knowledge.evidence_key.v1": EvidenceKey,
    "knowledge.query_input.v1": QueryInput,
    "knowledge.query_plan.v1": QueryPlan,
    "knowledge.retrieval_profile.v1": RetrievalProfile,
    "knowledge.evidence_set.v1": EvidenceSet,
    "knowledge.source_descriptor.v1": SourceDescriptor,
    "knowledge.parse_profile.v1": ParseProfile,
    "knowledge.parsed_document.v1": ParsedDocument,
    "knowledge.projection_profile.v1": ProjectionProfile,
    "knowledge.projection_batch.v1": ProjectionBatch,
    "knowledge.representation_spec.v1": RepresentationSpec,
    "knowledge.delegation_claims.v1": DelegationClaims,
    "knowledge.retrieval_request.v1": RetrievalRequest,
    "knowledge.citation_label_map.v1": CitationLabelMap,
    "knowledge.retrieval_evidence.v1": RetrievalEvidenceArtifact,
    "knowledge.model_evidence_delivery.v1": ModelEvidenceDelivery,
    "knowledge.grounded_answer.v1": GroundedAnswerArtifact,
    "knowledge.error.v1": ErrorEnvelope,
    "knowledge.cursor.v1": CursorBinding,
    "knowledge.app_retrieval.v1": AppRetrievalArtifact,
    "knowledge.composition.v1": KnowledgeCompositionArtifact,
}


def to_envelope(schema_version: str, value: object) -> dict:
    require(type(schema_version) is str and schema_version in CONTRACTS
            and type(value) is CONTRACTS[schema_version], "未登记的合同或类型")
    return {"schema_version": schema_version, **encode(value)}


def from_envelope(value: dict) -> object:
    require(type(value) is dict and type(value.get("schema_version")) is str
            and value["schema_version"] in CONTRACTS, "未登记的线协议版本")
    contract = CONTRACTS[value["schema_version"]]
    payload = dict(value)
    if "schema_version" not in {field.name for field in fields(contract)}:
        payload.pop("schema_version")
    return decode(contract, payload)


def schema_manifest() -> dict:
    """冻结所有可达嵌套字段、必需性及枚举；不依赖模型/数据库生成。"""
    found = {}

    def describe(t):
        origin, args = get_origin(t), get_args(t)
        if origin is not None:
            return {"container": origin.__name__, "args": [describe(a) if a is not Ellipsis else "..." for a in args]}
        if isinstance(t, type) and issubclass(t, Enum):
            return {"enum": t.__name__, "values": [item.value for item in t]}
        if hasattr(t, "__dataclass_fields__"):
            if t.__name__ not in found:
                found[t.__name__] = []
                hints = get_type_hints(t)
                found[t.__name__] = [{"name": f.name, "type": describe(hints[f.name]),
                    "required": f.default is MISSING and f.default_factory is MISSING} for f in fields(t)]
            return {"contract": t.__name__}
        return t.__name__

    for contract in CONTRACTS.values():
        describe(contract)
    describe(ParseRunIdentity)
    describe(ParseArtifact)
    return {"schemas": {name: c.__name__ for name, c in CONTRACTS.items()}, "contracts": found,
            "parse_event": {"schema_version": "knowledge.parse_event.v1", "shape": "FLAT_RUN_IDENTITY",
                "run_fields": [f.name for f in fields(ParseRunIdentity)],
                "fields": {"schema_version": "str", "sequence": "int", "batch_key": "str",
                    "event_type": describe(ParseEventType), "payload_hash": "str",
                    "dependency_keys": "list[str]", "payload": "bounded-json-object",
                    "artifact_handles": "list[ParseArtifact]"},
                "payloads": {
                    "RUN_STARTED": {"profile_hash": "sha256", "parser_revision": "str"},
                    "PAGE_BATCH": {"objects": "list[Page]"},
                    "NODE_BATCH": {"objects": "list[ContentNode+local_key]"},
                    "ASSET_BATCH": {"objects": "list[VisualAsset+local_key]"},
                    "REFERENCE_BATCH": {"objects": "list[FigureResolution+local_key]"},
                    "STRUCTURE_FINALIZED": {"relations": "list[ResolvedRelation]",
                        "alternatives": "list[RepairAlternative]", "warnings": "list[str]"},
                    "RUN_COMPLETED": {"counts": "dict[object_event_type,int]", "event_count": "int",
                        "ordered_digest": "sha256", "quality_report": {"status": ["COMPLETE", "PARTIAL", "FAILED"],
                            "failed_regions": "list[str]"}},
                    "RUN_FAILED": {"error_code": "ErrorCode", "stage": "str"}}}}
