import unittest
from types import SimpleNamespace
from pathlib import Path
import tempfile

from PIL import Image

from knowledge_core.workers.parser.visual_enricher import KcVisualEnricher
from knowledge_core.workers.parser.visual_cache import LocalVisualResultCache
from tests.unit.knowledge_core.test_kc_docling_atom_normalizer import item


class FakeModelClient:
    calls = 0

    async def get_vlm_answer(
        self,
        served_model_name,
        image,
        prompt,
        **kwargs,
    ):
        type(self).calls += 1
        if "Markdown" in prompt:
            return "# 系统架构\n\n> 三个服务节点通过消息队列连接"
        return "三个服务节点通过消息队列连接"


def document_with_page(*, text="短文本", confidence=0.9, image=None):
    text_item = item(
        source_ref="#/texts/0",
        label="text",
        text=text,
    )
    if confidence is not None:
        text_item.prov[0].confidence = confidence
    elif hasattr(text_item.prov[0], "confidence"):
        del text_item.prov[0].confidence
    picture_image = image or object()
    picture = SimpleNamespace(
        image=SimpleNamespace(pil_image=picture_image),
        annotations=[],
    )
    return SimpleNamespace(
        pictures=[picture],
        pages={
            1: SimpleNamespace(
                page_no=1,
                image=SimpleNamespace(pil_image=picture_image),
            )
        },
        iterate_items=lambda: iter(((text_item, 0),)),
    ), picture


class VisualEnricherTest(unittest.IsolatedAsyncioTestCase):
    async def test_adds_policy_model_description_with_provenance(self):
        document, picture = document_with_page()
        enricher = KcVisualEnricher(client_factory=FakeModelClient)

        result = await enricher.enrich(
            document,
            served_model_name="vlm-a",
            prompt="Describe visible facts",
        )

        self.assertEqual(result.picture_description_count, 1)
        self.assertEqual(
            picture.annotations[0].text,
            "三个服务节点通过消息队列连接",
        )
        self.assertIn("vlm_kc_v2", picture.annotations[0].provenance)
        self.assertFalse(result.page_results)

    async def test_does_nothing_without_policy_model(self):
        document, _ = document_with_page()
        result = await KcVisualEnricher(
            client_factory=FakeModelClient
        ).enrich(
            document, served_model_name=None, prompt="Describe",
        )
        self.assertEqual(result.picture_description_count, 0)
        self.assertFalse(result.page_results)

    async def test_hybrid_strategy_is_rejected(self):
        document, _ = document_with_page(text="稳定正文" * 50)
        with self.assertRaisesRegex(ValueError, "不支持"):
            await KcVisualEnricher(
                client_factory=FakeModelClient
            ).enrich(
                document,
                served_model_name="vlm-a",
                prompt="描述图片",
                strategy="HYBRID",
            )

    async def test_visual_analyzes_every_page_but_not_picture_objects(self):
        document, picture = document_with_page(text="稳定正文" * 50)
        result = await KcVisualEnricher(
            client_factory=FakeModelClient
        ).enrich(
            document,
            served_model_name="vlm-a",
            prompt="描述图片",
            strategy="VISUAL",
        )
        self.assertEqual([item.page_no for item in result.page_results], [1])
        self.assertFalse(result.page_results[0].replace_docling)
        self.assertFalse(picture.annotations)

    async def test_missing_docling_confidence_is_not_treated_as_zero(self):
        document, _ = document_with_page(
            text="稳定正文" * 50, confidence=None
        )
        result = await KcVisualEnricher(
            client_factory=FakeModelClient
        ).enrich(
            document,
            served_model_name="vlm-a",
            prompt="描述图片",
        )
        self.assertIsNone(result.page_assessments[0].mean_confidence)
        self.assertNotIn(
            "LOW_EXTRACTION_CONFIDENCE",
            result.page_assessments[0].reasons,
        )

    async def test_successful_picture_result_is_reused_from_cache(self):
        FakeModelClient.calls = 0
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (256, 256), color="white")
            cache = LocalVisualResultCache(Path(directory))
            for _ in range(2):
                document, _ = document_with_page(image=image)
                await KcVisualEnricher(
                    client_factory=FakeModelClient,
                    cache=cache,
                ).enrich(
                    document,
                    served_model_name="vlm-a",
                    prompt="描述图片",
                )
        self.assertEqual(FakeModelClient.calls, 1)

    async def test_visual_strategy_requires_model(self):
        document, _ = document_with_page()
        with self.assertRaisesRegex(ValueError, "必须配置 VLM"):
            await KcVisualEnricher(
                client_factory=FakeModelClient
            ).enrich(
                document,
                served_model_name=None,
                prompt="描述图片",
                strategy="VISUAL",
            )


if __name__ == "__main__":
    unittest.main()
