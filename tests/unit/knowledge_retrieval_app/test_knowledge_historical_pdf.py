"""历史引用原文按冻结版本与现行授权读取。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_retrieval_app.application.knowledge_library import KnowledgeLibraryService


@pytest.mark.asyncio
async def test_historical_pdf_uses_pinned_version_and_stops_after_revocation():
    library_id, version_id = uuid4(), uuid4()
    access = object()
    hit = SimpleNamespace(locators=[SimpleNamespace(page_no=2)],
                          key=SimpleNamespace(document_version_ref=str(version_id)))
    reauthorize = AsyncMock(side_effect=[f"library:{library_id}", f"library:{library_id}", PermissionError()])
    closed = []

    async def source_chunks():
        try:
            yield b"%PDF-"
            yield b"tail"
        finally:
            closed.append(True)

    app = SimpleNamespace(
        retrieval=SimpleNamespace(historical_evidence=AsyncMock(return_value=(hit, reauthorize))),
        source=AsyncMock(return_value=({"mime_type": "application/pdf", "byte_count": 9}, source_chunks())),
    )
    _, chunks = await KnowledgeLibraryService.historical_pdf(app, access, "receipt", "key")
    app.retrieval.historical_evidence.assert_awaited_once_with(access, retrieval_ref="receipt", stable_key="key")
    app.source.assert_awaited_once_with(access, library_id, version_id, published_only=True)
    assert await anext(chunks) == b"%PDF-"
    with pytest.raises(PermissionError):
        await anext(chunks)
    assert closed == [True]


@pytest.mark.asyncio
async def test_historical_pdf_refuses_a_reference_without_page_location():
    hit = SimpleNamespace(locators=[])
    reauthorize = AsyncMock(return_value=f"library:{uuid4()}")
    app = SimpleNamespace(
        retrieval=SimpleNamespace(historical_evidence=AsyncMock(return_value=(hit, reauthorize))),
        source=AsyncMock(),
    )
    with pytest.raises(AlgorithmError):
        await KnowledgeLibraryService.historical_pdf(app, object(), "receipt", "key")
    app.source.assert_not_awaited()
