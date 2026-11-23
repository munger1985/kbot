"""源文件的内部路由在核权后为 BFF 提供安全 MIME，响应本身仍为附件。"""

from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from knowledge_retrieval_app.api.knowledge_library import create_library_router, library_access


@pytest.mark.parametrize(("mime_type", "expected"), [
    ("application/pdf", "application/pdf"),
    ("text/plain", "text/plain"),
    ("text/markdown", "text/markdown"),
    ("image/jpeg", "image/jpeg"),
    ("text/html", "application/octet-stream"),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/octet-stream"),
])
def test_internal_source_media_type(mime_type, expected):
    async def stream():
        yield b"source"

    app = FastAPI()
    app.include_router(create_library_router())
    access = object()
    app.dependency_overrides[library_access] = lambda: access
    app.state.knowledge_library_service = AsyncMock()
    source = app.state.knowledge_library_service.source
    source.return_value = ({"mime_type": mime_type, "byte_count": 6}, stream())
    library_id = UUID(int=1)
    version_id = UUID(int=2)
    path = f"/internal/v1/knowledge-retrieval/knowledge/libraries/{library_id}/versions/{version_id}/source"

    with TestClient(app) as client:
        response = client.get(path, headers={"Accept": "text/html"})
        assert response.status_code == 200
        assert response.headers["Content-Type"].split(";")[0] == expected
        assert response.headers["Content-Disposition"] == "attachment"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.content == b"source"
        source.assert_awaited_once_with(access, library_id, version_id)

        source.side_effect = HTTPException(403)
        denied = client.get(path)
        assert denied.status_code == 403
        assert "Content-Disposition" not in denied.headers
        assert denied.content != b"source"
