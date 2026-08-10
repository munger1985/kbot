"""Model Serving 独立 OCR 推理能力。"""

from .schema import OCRRequest, OCRResponse
from .service import OCRService

__all__ = ["OCRRequest", "OCRResponse", "OCRService"]
