"""4.0 多图片检索的专项契约测试。"""

import asyncio
import base64
from pathlib import Path
import tempfile
import unittest
from uuid import UUID

from PIL import Image

from agent_runtime.application.attachments import ConversationAttachmentStore
from knowledge_core.workers.parser.worker import KcParserWorker
from platform_core.contracts import (
    AgentExecutionSpec,
    ConversationQueryImage,
    CreateConversationTurnRequest,
)


class _ImageHolder:
    def __init__(self, image):
        self.pil_image = image


class _Page:
    page_no = 1

    def __init__(self, image):
        self.image = _ImageHolder(image)


class _Document:
    pictures = ()

    def __init__(self, image):
        self.pages = {1: _Page(image)}


class VisualImageSearchTest(unittest.IsolatedAsyncioTestCase):
    async def test_attachment_store_is_content_addressed(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (8, 8), "red")
            from io import BytesIO

            stream = BytesIO()
            image.save(stream, format="PNG")
            encoded = base64.b64encode(stream.getvalue()).decode("ascii")
            query = ConversationQueryImage(
                file_name="query.png",
                mime_type="image/png",
                content_base64=encoded,
            )
            store = ConversationAttachmentStore(Path(directory))
            first = await store.put_images(
                conversation_id=UUID(
                    "01900000-0000-7000-8000-000000000001"
                ),
                images=(query,),
            )
            second = await store.put_images(
                conversation_id=UUID(
                    "01900000-0000-7000-8000-000000000001"
                ),
                images=(query,),
            )
            self.assertEqual(first, second)
            self.assertTrue(Path(first[0]["storage_uri"]).is_file())

    async def test_parser_exports_page_for_visual_index(self):
        assets = KcParserWorker._visual_assets(
            _Document(Image.new("RGB", (10, 10), "blue")),
            None,
        )
        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["asset_type"], "PAGE")
        self.assertEqual(assets[0]["mime_type"], "image/png")
        self.assertEqual(len(assets[0]["content_sha256"]), 64)

    async def test_turn_accepts_multiple_images(self):
        encoded = base64.b64encode(b"small-image").decode("ascii")
        request = CreateConversationTurnRequest(
            input="查找相似图片",
            expected_conversation_version=1,
            execution_spec=AgentExecutionSpec(
                schema_version="1.0",
                owner_app_id="knowledge_retrieval",
                domain_id=20,
                consumer_agent_id=UUID(
                    "01900000-0000-7000-8000-000000000001"
                ),
                consumer_agent_version_id=UUID(
                    "01900000-0000-7000-8000-000000000002"
                ),
                agent_kind="KNOWLEDGE_RETRIEVAL",
                display_name="文档助手",
                enabled_capabilities=("document",),
                models={
                    "composer_llm": UUID(
                        "01900000-0000-7000-8000-000000000003"
                    )
                },
            ),
            images=(
                ConversationQueryImage(
                    file_name="a.png",
                    mime_type="image/png",
                    content_base64=encoded,
                ),
                ConversationQueryImage(
                    file_name="b.webp",
                    mime_type="image/webp",
                    content_base64=encoded,
                ),
            ),
        )
        self.assertEqual(len(request.images), 2)


if __name__ == "__main__":
    unittest.main()
