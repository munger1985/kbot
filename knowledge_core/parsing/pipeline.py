"""End-to-end pure KC parsing pipeline after file conversion."""

from dataclasses import asdict, dataclass
from typing import Any

from docling_core.types.doc import DoclingDocument

from .contracts import build_output_fingerprint, canonical_json_hash
from .docling_adapter import DoclingAtomNormalizer
from .evidence_planner import EvidencePlanner, EvidencePolicy, PlannedEvidence
from .quality import StructureQualityEvaluator, StructureQualityReport
from .reading_order import ReadingOrderResolver
from .structure_builder import OutlineResolver
from .spreadsheet_artifact import build_spreadsheet_artifact


@dataclass(frozen=True)
class ParserOutput:
    artifacts: dict[str, Any]
    artifact_schemas: dict[str, str]
    evidences: tuple[PlannedEvidence, ...]
    quality_report: StructureQualityReport
    output_fingerprint: str


class KcParsingPipeline:
    def __init__(self, *, parser_version: str, evidence_policy: EvidencePolicy | None = None):
        self._parser_version = parser_version
        self._normalizer = DoclingAtomNormalizer(generator_version=parser_version)
        self._reading_order = ReadingOrderResolver()
        self._outline = OutlineResolver()
        self._quality = StructureQualityEvaluator()
        self._planner = EvidencePlanner(evidence_policy)

    @property
    def parser_version(self) -> str:
        return self._parser_version

    def parse(
        self, *, document_version_id: int, parse_view_id: int, document: DoclingDocument,
    ) -> ParserOutput:
        raw_docling = document.export_to_dict(mode="json", by_alias=True, exclude_none=True)
        atom_ir = self._normalizer.normalize(
            document_version_id=document_version_id, document=document,
        )
        reading_order = self._reading_order.resolve(atom_ir)
        structure_ir = self._outline.build(atom_ir, reading_order)
        quality = self._quality.evaluate(atom_ir, reading_order, structure_ir)
        evidences = self._planner.plan(
            parse_view_id=parse_view_id, atom_ir=atom_ir, structure_ir=structure_ir,
        ) if quality.passed else ()
        quality = self._quality.evaluate_evidence(
            quality, evidences, max_tokens=self._planner.policy.max_tokens,
        )
        evidence_manifest = {
            "schema": "kc-evidence-manifest/v1",
            "document_version_id": document_version_id,
            "parse_view_id": parse_view_id,
            "evidences": [{
                "evidence_key": evidence.evidence_key,
                "evidence_type": evidence.evidence_type,
                "source_spans_sha256": canonical_json_hash(evidence.source_spans),
                "content_sha256": canonical_json_hash(evidence.content_text),
                "locator_sha256": canonical_json_hash(evidence.locator),
            } for evidence in evidences],
            "quality_report": quality.as_dict(),
        }
        artifacts = {
            "raw_docling": raw_docling,
            "atom_ir": asdict(atom_ir),
            "structure_ir": asdict(structure_ir),
            "evidence_manifest": evidence_manifest,
        }
        spreadsheet_artifact = build_spreadsheet_artifact(document)
        if spreadsheet_artifact is not None:
            artifacts["spreadsheet_artifact"] = spreadsheet_artifact
        artifact_schemas = {
            "raw_docling": "docling-document/v1",
            "atom_ir": atom_ir.ir_version,
            "structure_ir": structure_ir.ir_version,
            "evidence_manifest": "kc-evidence-manifest/v1",
        }
        if spreadsheet_artifact is not None:
            artifact_schemas["spreadsheet_artifact"] = "kc-spreadsheet/v1"
        output_fingerprint = build_output_fingerprint(
            artifact_hashes={name: canonical_json_hash(value) for name, value in artifacts.items()},
            evidence_keys=[evidence.evidence_key for evidence in evidences],
        )
        return ParserOutput(
            artifacts=artifacts, artifact_schemas=artifact_schemas,
            evidences=evidences, quality_report=quality,
            output_fingerprint=output_fingerprint,
        )
