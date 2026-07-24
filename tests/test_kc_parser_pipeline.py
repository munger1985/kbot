import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from knowledge_core.adapters.local_parser_artifact_store import LocalParserArtifactStore
from knowledge_core.parsing import canonical_json_hash
from knowledge_core.parsing.evidence_planner import EvidencePolicy
from knowledge_core.parsing.pipeline import KcParsingPipeline
from knowledge_core.workers.parser.visual_enricher import (
    PageQualityAssessment,
    VisualEnrichmentResult,
    VisualPageResult,
)
from tests.test_kc_docling_atom_normalizer import FakeDocument, item


class LongTable:
    self_ref = "#/tables/0"
    label = "table"
    formatting = None
    annotations = []

    def __init__(self, prov):
        self.prov = prov

    def export_to_markdown(self, document):
        rows = ["| Name | Value |", "| --- | --- |"]
        rows.extend(f"| row-{index} | value-{index} |" for index in range(12))
        return "\n".join(rows)


class SpreadsheetTable(LongTable):
    parent = SimpleNamespace(cref="#/groups/0")

    def __init__(self, prov):
        super().__init__(prov)
        self.data = SimpleNamespace(
            num_rows=2, num_cols=2,
            table_cells=[
                SimpleNamespace(
                    start_row_offset_idx=0, end_row_offset_idx=1,
                    start_col_offset_idx=0, end_col_offset_idx=1,
                    text="Name", column_header=True, row_header=False,
                ),
                SimpleNamespace(
                    start_row_offset_idx=1, end_row_offset_idx=2,
                    start_col_offset_idx=0, end_col_offset_idx=1,
                    text="Asset A", column_header=False, row_header=True,
                ),
            ],
        )

    def export_to_markdown(self, document):
        return "| Name | Value |\n| --- | --- |\n| Asset A | 1 |"


class ParserPipelineTest(unittest.TestCase):
    def test_pipeline_produces_all_artifacts_and_traceable_evidence(self):
        document = FakeDocument([
            item(source_ref="#/texts/0", label="section_header", text="1 Overview"),
            item(source_ref="#/texts/1", label="text", text="Install the service before startup."),
        ])
        pipeline = KcParsingPipeline(
            parser_version="3.5-test",
            evidence_policy=EvidencePolicy(min_tokens=1, target_tokens=20, max_tokens=50),
        )

        output = pipeline.parse(document_version_id=10, parse_view_id=20, document=document)

        self.assertTrue(output.quality_report.passed)
        self.assertEqual(set(output.artifacts), {
            "raw_docling", "atom_ir", "structure_ir", "evidence_manifest",
        })
        self.assertTrue(output.evidences)
        self.assertTrue(all(value.evidence_key.startswith("ev1:20:") for value in output.evidences))
        self.assertFalse(
            output.quality_report.metrics["image_processing"][
                "visual_embedding"
            ]["enabled"]
        )
        self.assertEqual(
            output.quality_report.metrics["image_processing"]["vlm"][
                "skip_reason"
            ],
            "COMPONENT_NOT_AVAILABLE",
        )

    def test_local_artifact_store_is_content_addressed_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalParserArtifactStore(Path(directory))
            payload = {"ir_version": "kc-atom/v1", "atoms": []}
            digest = canonical_json_hash(payload)
            descriptor = asyncio.run(store.put_json(
                job_id=5, artifact_name="atom_ir", payload=payload,
                expected_sha256=digest, schema="kc-atom/v1", generator="test/1",
            ))
            replay = asyncio.run(store.put_json(
                job_id=5, artifact_name="atom_ir", payload=payload,
                expected_sha256=digest, schema="kc-atom/v1", generator="test/1",
            ))

            self.assertEqual(descriptor, replay)
            self.assertTrue(Path(descriptor["uri"]).is_file())
            asyncio.run(store.delete_manifest({"atom_ir": descriptor}))
            self.assertFalse(Path(descriptor["uri"]).exists())

    def test_long_text_and_table_create_precise_fragments(self):
        title = item(source_ref="#/texts/0", label="section_header", text="1 Data")
        long_text = item(
            source_ref="#/texts/1", label="text",
            text="one two three four five six seven eight nine ten eleven twelve",
        )
        table = LongTable(title.prov)
        pipeline = KcParsingPipeline(
            parser_version="3.5-test",
            evidence_policy=EvidencePolicy(min_tokens=1, target_tokens=5, max_tokens=6),
        )

        output = pipeline.parse(
            document_version_id=10, parse_view_id=20,
            document=FakeDocument([title, long_text, table]),
        )

        paragraphs = [value for value in output.evidences if value.evidence_type == "PARAGRAPH"]
        table_rows = [value for value in output.evidences if value.evidence_type == "TABLE_ROW"]
        self.assertGreaterEqual(len(paragraphs), 2)
        self.assertIn("char_start", paragraphs[0].source_spans[0])
        self.assertTrue(table_rows)
        self.assertTrue(all(value.parent_evidence_key for value in table_rows))

    def test_spreadsheet_creates_sheet_and_cell_range_evidence(self):
        base = item(source_ref="#/texts/0", label="text", text="Workbook")
        table = SpreadsheetTable(base.prov)
        document = FakeDocument([base, table])
        document.groups = [SimpleNamespace(
            self_ref="#/groups/0", label=SimpleNamespace(value="sheet"), name="Assets",
        )]

        output = KcParsingPipeline(parser_version="3.5-test").parse(
            document_version_id=10, parse_view_id=20, document=document,
        )

        sheet = next(value for value in output.evidences if value.evidence_type == "SHEET")
        cell_range = next(value for value in output.evidences if value.evidence_type == "CELL_RANGE")
        self.assertIn("spreadsheet_artifact", output.artifacts)
        self.assertEqual("kc-spreadsheet/v1", output.artifact_schemas["spreadsheet_artifact"])
        self.assertEqual("A1", output.artifacts["spreadsheet_artifact"]["sheets"][0]["cells"][0]["address"])
        self.assertEqual(cell_range.locator["cell_range"], "A1:B2")
        self.assertEqual(cell_range.parent_evidence_key, sheet.evidence_key)

    def test_visual_description_is_citable_image_evidence(self):
        picture = item(source_ref="#/pictures/0", label="picture", text="")
        picture.annotations = [SimpleNamespace(
            text="A diagram showing three service nodes", provenance="vlm_inference",
        )]

        output = KcParsingPipeline(parser_version="3.5-test").parse(
            document_version_id=10, parse_view_id=20,
            document=FakeDocument([picture]),
        )

        image = next(value for value in output.evidences if value.evidence_type == "IMAGE")
        self.assertIn("three service nodes", image.content_text)
        self.assertIn("VLM", image.provenance["extractors"])

    def test_low_quality_page_is_replaced_by_visual_markdown(self):
        document = FakeDocument([
            item(
                source_ref="#/texts/0",
                label="text",
                text="乱码",
            ),
            item(
                source_ref="#/texts/1",
                label="text",
                text="设备序列号 SN-001",
            ),
        ])
        enrichment = VisualEnrichmentResult(
            strategy="AUTO",
            picture_description_count=0,
            page_assessments=(
                PageQualityAssessment(
                    page_no=1,
                    text_characters=2,
                    mean_confidence=0.4,
                    gibberish_ratio=0,
                    image_available=True,
                    requires_visual=True,
                    reasons=("LOW_TEXT_COVERAGE",),
                ),
            ),
            page_results=(
                VisualPageResult(
                    page_no=1,
                    markdown=(
                        "# 设备巡检\n\n"
                        "数据库实例运行正常。\n\n"
                        "| 指标 | 数值 |\n| --- | --- |\n| CPU | 20% |"
                    ),
                    served_model_name="vlm-a",
                    confidence=0.82,
                    replace_docling=True,
                    selection_reasons=("LOW_TEXT_COVERAGE",),
                ),
            ),
            failed_page_numbers=(),
        )

        output = KcParsingPipeline(parser_version="4.0-test").parse(
            document_version_id=10,
            parse_view_id=20,
            document=document,
            visual_enrichment=enrichment,
            visual_embedding_enabled=True,
        )

        contents = [evidence.content_text for evidence in output.evidences]
        self.assertNotIn("乱码", contents)
        self.assertTrue(
            any("数据库实例运行正常" in content for content in contents)
        )
        self.assertTrue(
            any("SN-001" in content for content in contents)
        )
        self.assertTrue(
            any(
                evidence.evidence_type == "TABLE"
                for evidence in output.evidences
            )
        )
        self.assertIn("visual_analysis", output.artifacts)
        self.assertEqual(
            1,
            output.quality_report.metrics[
                "visual_enrichment"
            ]["replaced_page_count"],
        )
        self.assertTrue(
            output.quality_report.metrics["image_processing"]["vlm"][
                "enabled"
            ]
        )
        self.assertEqual(
            output.quality_report.metrics["image_processing"][
                "visual_embedding"
            ]["status"],
            "DEFERRED_TO_INDEX",
        )

    def test_hybrid_visual_result_corrects_matching_heading(self):
        document = FakeDocument([
            item(
                source_ref="#/texts/0",
                label="text",
                text="2 部署步骤",
            ),
            item(
                source_ref="#/texts/1",
                label="text",
                text="先安装数据库，再启动服务。",
            ),
        ])
        enrichment = VisualEnrichmentResult(
            strategy="HYBRID",
            picture_description_count=0,
            page_assessments=(),
            page_results=(
                VisualPageResult(
                    page_no=1,
                    markdown="# 2 部署步骤\n\n> 服务依赖关系图",
                    served_model_name="vlm-a",
                    confidence=0.82,
                    replace_docling=False,
                    selection_reasons=(),
                ),
            ),
            failed_page_numbers=(),
        )

        output = KcParsingPipeline(parser_version="4.0-test").parse(
            document_version_id=10,
            parse_view_id=20,
            document=document,
            visual_enrichment=enrichment,
        )

        headings = [
            evidence
            for evidence in output.evidences
            if evidence.evidence_type == "SECTION"
        ]
        images = [
            evidence
            for evidence in output.evidences
            if evidence.evidence_type == "IMAGE"
        ]
        self.assertEqual(headings[0].heading_level, 1)
        self.assertIn("服务依赖关系图", images[0].content_text)


if __name__ == "__main__":
    unittest.main()
