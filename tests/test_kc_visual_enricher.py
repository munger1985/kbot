import unittest
from types import SimpleNamespace

from knowledge_core.workers.parser.visual_enricher import KcVisualEnricher
from tests.test_kc_docling_atom_normalizer import item


class FakeModelClient:
    async def get_vlm_answer(
        self,
        served_model_name,
        image,
        prompt,
        **kwargs,
    ):
        if "Markdown" in prompt:
            return "# 系统架构\n\n> 三个服务节点通过消息队列连接"
        return "三个服务节点通过消息队列连接"


def document_with_page(*, text="短文本", confidence=0.9):
    text_item = item(
        source_ref="#/texts/0",
        label="text",
        text=text,
    )
    text_item.prov[0].confidence = confidence
    picture = SimpleNamespace(
        image=SimpleNamespace(pil_image=object()),
        annotations=[],
    )
    return SimpleNamespace(
        pictures=[picture],
        pages={
            1: SimpleNamespace(
                page_no=1,
                image=SimpleNamespace(pil_image=object()),
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
        self.assertEqual([page.page_no for page in result.page_results], [1])
        self.assertTrue(result.page_results[0].replace_docling)
        self.assertIn(
            "LOW_TEXT_COVERAGE",
            result.page_results[0].selection_reasons,
        )

    async def test_does_nothing_without_policy_model(self):
        document, _ = document_with_page()
        result = await KcVisualEnricher(
            client_factory=FakeModelClient
        ).enrich(
            document, served_model_name=None, prompt="Describe",
        )
        self.assertEqual(result.picture_description_count, 0)
        self.assertFalse(result.page_results)

    async def test_hybrid_analyzes_healthy_page_without_replacing_docling(self):
        document, _ = document_with_page(text="稳定正文" * 50)

        result = await KcVisualEnricher(
            client_factory=FakeModelClient
        ).enrich(
            document,
            served_model_name="vlm-a",
            prompt="描述图片",
            strategy="HYBRID",
        )

        self.assertEqual(len(result.page_results), 1)
        self.assertFalse(result.page_results[0].replace_docling)

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
