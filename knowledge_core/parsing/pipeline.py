"""End-to-end pure KC parsing pipeline after file conversion."""

from uuid import UUID
from dataclasses import asdict, dataclass
from typing import Any

from docling_core.types.doc import DoclingDocument

from .contracts import build_output_fingerprint, canonical_json_hash
from .deepseek_ocr_adapter import DeepSeekOcrAtomAdapter
from .docling_adapter import DoclingAtomNormalizer
from .evidence_planner import EvidencePlanner, EvidencePolicy, PlannedEvidence
from .quality import StructureQualityEvaluator, StructureQualityReport
from .reading_order import ReadingOrderResolver
from .structure_builder import OutlineResolver
from .spreadsheet_artifact import build_spreadsheet_artifact
from .visual_page_adapter import VisualPageAtomAdapter


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
        self._deepseek_ocr_adapter = DeepSeekOcrAtomAdapter()
        self._reading_order = ReadingOrderResolver()
        self._outline = OutlineResolver()
        self._quality = StructureQualityEvaluator()
        self._planner = EvidencePlanner(evidence_policy)
        self._visual_adapter = VisualPageAtomAdapter()

    @property
    def parser_version(self) -> str:
        return self._parser_version

    def parse(
        self,
        *,
        document_version_id: UUID,
        parse_view_id: UUID,
        document: DoclingDocument,
        ocr_enrichment=None,
        visual_enrichment=None,
        visual_embedding_enabled: bool = False,
    ) -> ParserOutput:
        raw_docling = document.export_to_dict(mode="json", by_alias=True, exclude_none=True)
        atom_ir = self._normalizer.normalize(
            document_version_id=document_version_id, document=document,
        )
        atom_ir = self._deepseek_ocr_adapter.apply(
            atom_ir,
            ocr_enrichment,
        )
        atom_ir = self._visual_adapter.apply(atom_ir, visual_enrichment)
        reading_order = self._reading_order.resolve(atom_ir)
        structure_ir = self._outline.build(atom_ir, reading_order)
        quality = self._quality.evaluate(atom_ir, reading_order, structure_ir)
        evidences = self._planner.plan(
            parse_view_id=parse_view_id, atom_ir=atom_ir, structure_ir=structure_ir,
        ) if quality.passed else ()
        quality = self._quality.evaluate_evidence(
            quality, evidences, max_tokens=self._planner.policy.max_tokens,
        )
        if visual_enrichment is not None:
            metrics = dict(quality.metrics)
            metrics["visual_enrichment"] = {
                "strategy": visual_enrichment.strategy,
                "enabled": visual_enrichment.enabled,
                "skip_reason": visual_enrichment.skip_reason,
                "picture_description_count": (
                    visual_enrichment.picture_description_count
                ),
                "analyzed_page_count": len(visual_enrichment.page_results),
                "replaced_page_count": sum(
                    page.replace_docling
                    for page in visual_enrichment.page_results
                ),
                "failed_page_numbers": list(
                    visual_enrichment.failed_page_numbers
                ),
            }
            quality = StructureQualityReport(
                passed=quality.passed,
                hard_failures=quality.hard_failures,
                warnings=quality.warnings,
                metrics=metrics,
                evaluator_version=quality.evaluator_version,
            )
        metrics = dict(quality.metrics)
        metrics["image_processing"] = {
            "vlm": {
                "enabled": bool(
                    visual_enrichment is not None
                    and visual_enrichment.enabled
                ),
                "skip_reason": (
                    visual_enrichment.skip_reason
                    if visual_enrichment is not None
                    else "COMPONENT_NOT_AVAILABLE"
                ),
            },
            "visual_embedding": {
                "enabled": visual_embedding_enabled,
                "status": (
                    "DEFERRED_TO_INDEX"
                    if visual_embedding_enabled
                    else "MODEL_NOT_CONFIGURED"
                ),
            },
        }
        quality = StructureQualityReport(
            passed=quality.passed,
            hard_failures=quality.hard_failures,
            warnings=quality.warnings,
            metrics=metrics,
            evaluator_version=quality.evaluator_version,
        )
        if ocr_enrichment is not None:
            metrics = dict(quality.metrics)
            metrics["deepseek_ocr"] = {
                "model": ocr_enrichment.served_model_name,
                "recognized_page_count": len(
                    ocr_enrichment.page_results
                ),
                "failed_page_numbers": list(
                    ocr_enrichment.failed_page_numbers
                ),
                "picture_description_count": (
                    ocr_enrichment.picture_description_count
                ),
            }
            quality = StructureQualityReport(
                passed=quality.passed,
                hard_failures=quality.hard_failures,
                warnings=quality.warnings,
                metrics=metrics,
                evaluator_version=quality.evaluator_version,
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
        if visual_enrichment is not None and (
            visual_enrichment.page_results
            or visual_enrichment.picture_description_count
            or visual_enrichment.failed_page_numbers
        ):
            artifacts["visual_analysis"] = visual_enrichment.as_dict()
        if ocr_enrichment is not None:
            artifacts["deepseek_ocr_analysis"] = (
                ocr_enrichment.as_dict()
            )
        artifact_schemas = {
            "raw_docling": "docling-document/v1",
            "atom_ir": atom_ir.ir_version,
            "structure_ir": structure_ir.ir_version,
            "evidence_manifest": "kc-evidence-manifest/v1",
        }
        if spreadsheet_artifact is not None:
            artifact_schemas["spreadsheet_artifact"] = "kc-spreadsheet/v1"
        if "visual_analysis" in artifacts:
            artifact_schemas["visual_analysis"] = "kc-visual-analysis/v1"
        if "deepseek_ocr_analysis" in artifacts:
            artifact_schemas["deepseek_ocr_analysis"] = (
                "kc-deepseek-ocr/v1"
            )
        output_fingerprint = build_output_fingerprint(
            artifact_hashes={name: canonical_json_hash(value) for name, value in artifacts.items()},
            evidence_keys=[evidence.evidence_key for evidence in evidences],
        )
        return ParserOutput(
            artifacts=artifacts, artifact_schemas=artifact_schemas,
            evidences=evidences, quality_report=quality,
            output_fingerprint=output_fingerprint,
        )
