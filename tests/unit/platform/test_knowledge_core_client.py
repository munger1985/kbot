"""Knowledge Core 内部客户端路由与载荷测试。"""

import unittest
from uuid import UUID

from platform_clients import KnowledgeCoreClient


class _Response:
    status = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        return None

    async def json(self):
        return {
            "bundle_revision_id": (
                "019f8eae-2c25-7d48-b044-350ec3f5a022"
            ),
            "generation": "019f8eae-2c25-7d48-b044-350ec3f5a024",
            "scheduled_file_count": 1,
        }


class _Session:
    def __init__(self):
        self.method = None
        self.url = None
        self.kwargs = None

    def request(self, method, url, **kwargs):
        self.method = method
        self.url = url
        self.kwargs = kwargs
        return _Response()


class KnowledgeCoreClientTest(unittest.IsolatedAsyncioTestCase):
    async def test_get_collection_model_policy_uses_scoped_route(self):
        session = _Session()
        client = KnowledgeCoreClient(
            base_url="http://knowledge-core.internal",
            caller_service="main-api",
            audience="knowledge-core",
            session=session,
        )
        client._headers = lambda context: {}  # type: ignore[method-assign]
        collection_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")

        await client.get_collection_model_policy(
            domain_id=100,
            collection_id=collection_id,
            auth_context=object(),  # type: ignore[arg-type]
        )

        self.assertEqual("GET", session.method)
        self.assertEqual(
            (
                "http://knowledge-core.internal/internal/v1/knowledge/"
                f"domains/100/collections/{collection_id}/model-policy"
            ),
            session.url,
        )

    async def test_list_processing_uses_scoped_catalog_route(self):
        session = _Session()
        client = KnowledgeCoreClient(
            base_url="http://knowledge-core.internal",
            caller_service="km-asset-app",
            audience="knowledge-core",
            session=session,
        )
        client._headers = lambda context: {}  # type: ignore[method-assign]
        collection_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")

        await client.list_processing(
            domain_id=100,
            collection_id=collection_id,
            page=2,
            page_size=50,
            auth_context=object(),  # type: ignore[arg-type]
        )

        self.assertEqual("GET", session.method)
        self.assertEqual(
            (
                "http://knowledge-core.internal/internal/v1/knowledge/"
                "domains/100/catalog/processing?"
                f"collection_id={collection_id}&page=2&page_size=50"
            ),
            session.url,
        )

    async def test_reindex_discovery_uses_internal_route(self):
        session = _Session()
        client = KnowledgeCoreClient(
            base_url="http://knowledge-core.internal",
            caller_service="km-asset-app",
            audience="knowledge-core",
            session=session,
        )
        client._headers = lambda context: {}  # type: ignore[method-assign]
        collection_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
        bundle_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a021")
        revision_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a022")

        await client.reindex_discovery(
            domain_id=100,
            collection_id=collection_id,
            bundle_id=bundle_id,
            bundle_revision_id=revision_id,
            auth_context=object(),  # type: ignore[arg-type]
        )

        self.assertEqual("POST", session.method)
        self.assertEqual(
            (
                "http://knowledge-core.internal/internal/v1/knowledge/"
                f"domains/100/bundles/{bundle_id}/revisions/{revision_id}"
                "/reindex-discovery"
            ),
            session.url,
        )
        self.assertEqual(
            {"collection_id": str(collection_id)},
            session.kwargs["json"],
        )

    async def test_get_reindex_discovery_status_uses_generation_route(self):
        session = _Session()
        client = KnowledgeCoreClient(
            base_url="http://knowledge-core.internal",
            caller_service="km-asset-app",
            audience="knowledge-core",
            session=session,
        )
        client._headers = lambda context: {}  # type: ignore[method-assign]
        bundle_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a021")
        revision_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a022")
        generation = UUID("019f8eae-2c25-7d48-b044-350ec3f5a024")

        await client.get_reindex_discovery_status(
            domain_id=100,
            bundle_id=bundle_id,
            bundle_revision_id=revision_id,
            generation=generation,
            auth_context=object(),  # type: ignore[arg-type]
        )

        self.assertEqual("GET", session.method)
        self.assertEqual(
            (
                "http://knowledge-core.internal/internal/v1/knowledge/"
                f"domains/100/bundles/{bundle_id}/revisions/{revision_id}"
                f"/reindex-discovery/{generation}"
            ),
            session.url,
        )

    async def test_reprocess_revision_uses_internal_route_and_uuid_json(self):
        session = _Session()
        client = KnowledgeCoreClient(
            base_url="http://knowledge-core.internal",
            caller_service="main-api",
            audience="knowledge-core",
            session=session,
        )
        client._headers = lambda context: {}  # type: ignore[method-assign]

        collection_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
        bundle_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a021")
        revision_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a022")
        version_id = UUID("019f8eae-2c25-7d48-b044-350ec3f5a023")

        result = await client.reprocess_revision(
            domain_id=100,
            collection_id=collection_id,
            bundle_id=bundle_id,
            bundle_revision_id=revision_id,
            document_version_id=version_id,
            auth_context=object(),  # type: ignore[arg-type]
        )

        self.assertEqual(1, result["scheduled_file_count"])
        self.assertEqual("POST", session.method)
        self.assertEqual(
            (
                "http://knowledge-core.internal/internal/v1/knowledge/"
                f"domains/100/bundles/{bundle_id}/revisions/{revision_id}"
                "/reprocess"
            ),
            session.url,
        )
        self.assertEqual(
            {
                "collection_id": str(collection_id),
                "document_version_id": str(version_id),
            },
            session.kwargs["json"],
        )


if __name__ == "__main__":
    unittest.main()
