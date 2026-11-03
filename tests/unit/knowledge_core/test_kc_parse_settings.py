import unittest
from types import SimpleNamespace
from uuid import UUID
from platform_core.identity import uuid7

from knowledge_core.domain.parse_settings import (
    normalize_collection_parse_policy,
)
from knowledge_core.application.parse_policy import build_parse_plan
from knowledge_core.workers.parser.visual_evidence import (
    build_visual_page_evidence,
)


class ParseSettingsTest(unittest.TestCase):
    def test_native_text_auto_uses_generic_text_profile_without_ocr(self):
        for mime in ("text/plain", "text/markdown", "text/x-markdown; charset=utf-8"):
            plan = build_parse_plan(
                collection=SimpleNamespace(models_json={}, parse_policy_json={"parse_strategy": "AUTO"}),
                version=SimpleNamespace(detected_mime_type=mime), overrides={"parse_strategy": "VISUAL"},
            )
            self.assertEqual("TEXT", plan.policy["parse_strategy"])
            self.assertEqual("AUTO", plan.policy["configured_parse_strategy"])
            self.assertEqual("NATIVE_STRUCTURED_TEXT", plan.policy["content_profile"])
            self.assertFalse(plan.policy["do_ocr"])
            self.assertFalse(plan.policy["extract_picture_images"])
            self.assertEqual("kc-docling-pipeline", plan.parser_name)

    def test_explicit_admin_strategy_and_binary_manual_auto_remain_authoritative(self):
        model_id = str(uuid7())
        for strategy, mime in (("VISUAL", "text/markdown"), ("AUTO", "application/pdf"), ("TEXT", "text/markdown")):
            plan = build_parse_plan(
                collection=SimpleNamespace(models_json={"parser_ocr": model_id}, parse_policy_json={"parse_strategy": strategy}),
                version=SimpleNamespace(detected_mime_type=mime), overrides={"parse_strategy": "TEXT"},
            )
            self.assertEqual(strategy, plan.policy["parse_strategy"])
            self.assertEqual("DOCUMENT", plan.policy["content_profile"])
        with self.assertRaisesRegex(ValueError, "OCR"):
            build_parse_plan(
                collection=SimpleNamespace(models_json={}, parse_policy_json={"parse_strategy": "AUTO"}),
                version=SimpleNamespace(detected_mime_type="application/pdf"), overrides={},
            )

    def test_content_profile_and_model_snapshot_are_part_of_plan_fingerprint(self):
        collection = SimpleNamespace(models_json={"parser_ocr": str(uuid7())}, parse_policy_json={})
        text = build_parse_plan(collection=collection, version=SimpleNamespace(detected_mime_type="text/markdown"), overrides={})
        pdf = build_parse_plan(collection=collection, version=SimpleNamespace(detected_mime_type="application/pdf"), overrides={})
        self.assertNotEqual(text.fingerprint, pdf.fingerprint)
        self.assertEqual(collection.models_json["parser_ocr"], text.policy["models"]["parser_ocr"])

    def test_text_disables_ocr_and_all_image_generation(self):
        policy = normalize_collection_parse_policy({
            "parse_strategy": "TEXT",
        })

        self.assertFalse(policy["do_ocr"])
        self.assertFalse(policy["force_full_page_ocr"])
        self.assertFalse(policy["extract_page_images"])
        self.assertFalse(policy["extract_picture_images"])

    def test_auto_uses_only_object_level_images(self):
        policy = normalize_collection_parse_policy({
            "parse_strategy": "AUTO",
        })

        self.assertTrue(policy["do_ocr"])
        self.assertFalse(policy["force_full_page_ocr"])
        self.assertFalse(policy["extract_page_images"])
        self.assertTrue(policy["extract_picture_images"])

    def test_visual_baseline_does_not_generate_page_images(self):
        policy = normalize_collection_parse_policy({
            "parse_strategy": "VISUAL",
        })

        self.assertTrue(policy["do_ocr"])
        self.assertFalse(policy["extract_page_images"])

    def test_hybrid_is_rejected_and_image_switch_cannot_override_mode(self):
        with self.assertRaises(ValueError):
            normalize_collection_parse_policy({"parse_strategy": "HYBRID"})
        policy = normalize_collection_parse_policy({
            "parse_strategy": "AUTO",
            "extract_page_images": True,
        })
        self.assertFalse(policy["extract_page_images"])

    def test_visual_page_evidence_is_split_at_token_limit(self):
        items = build_visual_page_evidence(
            parse_view_id=UUID(int=1),
            enrichment=SimpleNamespace(page_results=[SimpleNamespace(
                page_no=1,
                markdown="字" * 1801,
                served_model_name="vlm-a",
                confidence=0.82,
            )]),
        )

        self.assertEqual(3, len(items))
        self.assertLessEqual(max(item["token_count"] for item in items), 900)


if __name__ == "__main__":
    unittest.main()
