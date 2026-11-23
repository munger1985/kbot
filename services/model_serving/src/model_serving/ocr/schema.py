"""OCR 推理 HTTP 合同。"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: UUID
    expected_revision: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    image_base64: str = Field(min_length=1, max_length=14_000_000)
    mime_type: str = Field(pattern=r"^image/")


class OCRResponse(BaseModel):
    model_id: UUID
    provider: str
    text: str
    blocks: list[dict[str, Any]]
    model_revision: str
