import unittest

from pydantic import ValidationError

from knowledge_core.api.parse_task_router import EvidenceItemRequest

from knowledge_core.parsing import (
    build_evidence_key,
    build_output_fingerprint,
    evidence_fingerprint,
    validate_artifact_manifest,
    validate_quality_report,
    validate_locator,
    validate_source_spans,
)


class ParserContractTest(unittest.TestCase):
    def test_parser_evidence_rejects_precomputed_embedding(self):
        payload = {
            "evidence_key": "ev1:1:a:0:PARAGRAPH",
            "evidence_type": "PARAGRAPH",
            "ordinal": 0,
            "fragment_index": 0,
            "content_text": "content",
            "source_spans": [{"atom_id": "atom:1"}],
            "locator_schema_version": "document-logical/v1",
            "locator": {"source_refs": ["#/texts/0"], "coordinate_space": "logical_document"},
            "provenance": {"method": "docling"},
            "embedding": [0.1, 0.2],
        }

        with self.assertRaises(ValidationError):
            EvidenceItemRequest.model_validate(payload)

    def test_artifact_manifest_requires_all_replay_artifacts(self):
        digest = "a" * 64
        manifest = {
            name: {
                "uri": f"kc://parse/1/{name}.json",
                "sha256": digest,
                "schema": f"kc-{name}/v1",
                "generator": "parser/3.5",
            }
            for name in ("raw_docling", "atom_ir", "structure_ir", "evidence_manifest")
        }

        validate_artifact_manifest(manifest)
        del manifest["atom_ir"]

        with self.assertRaisesRegex(ValueError, "atom_ir"):
            validate_artifact_manifest(manifest)

    def test_quality_report_rejects_hard_failures(self):
        validate_quality_report({"passed": True, "hard_failures": [], "metrics": {}})

        with self.assertRaisesRegex(ValueError, "hard_failures"):
            validate_quality_report({
                "passed": True,
                "hard_failures": ["ATOM_COVERAGE"],
                "metrics": {},
            })

    def test_source_spans_require_atom_and_valid_character_bounds(self):
        validate_source_spans([{"atom_id": "atom:1", "char_start": 0, "char_end": 8}])

        with self.assertRaises(ValueError):
            validate_source_spans([{"atom_id": "atom:1", "char_start": 8, "char_end": 2}])

    def test_evidence_fingerprint_covers_source_and_locator(self):
        first = evidence_fingerprint(
            content_text="same",
            source_spans=[{"atom_id": "atom:1"}],
            locator={"pages": [{"page_no": 1}]},
        )
        second = evidence_fingerprint(
            content_text="same",
            source_spans=[{"atom_id": "atom:2"}],
            locator={"pages": [{"page_no": 1}]},
        )

        self.assertNotEqual(first, second)

    def test_evidence_key_is_derived_from_source_spans(self):
        key = build_evidence_key(
            parse_view_id=9,
            source_spans=[{"atom_id": "atom:1"}],
            fragment_index=0,
            evidence_type="PARAGRAPH",
        )

        self.assertTrue(key.startswith("ev1:9:"))
        self.assertTrue(key.endswith(":0:PARAGRAPH"))

    def test_output_fingerprint_is_shared_by_worker_and_kc(self):
        fingerprint = build_output_fingerprint(
            artifact_hashes={
                name: str(index) * 64 for index, name in enumerate(
                    ("raw_docling", "atom_ir", "structure_ir", "evidence_manifest"), start=1,
                )
            },
            evidence_keys=["ev1:1:a:0:SECTION"],
        )

        self.assertEqual(len(fingerprint), 64)

    def test_locator_schema_is_strict(self):
        validate_locator("document-logical/v1", {
            "source_refs": ["#/texts/0"], "coordinate_space": "logical_document",
        })
        validate_locator("spreadsheet/v1", {
            "sheet_name": "Assets", "sheet_ref": "#/groups/0", "cell_range": "A1:F20",
        })

        with self.assertRaisesRegex(ValueError, "normalized"):
            validate_locator("document/v1", {"pages": [{
                "page_no": 1, "bbox": [0, -1, 1, 1],
                "coordinate_space": "page_normalized_top_left",
            }]})


if __name__ == "__main__":
    unittest.main()
