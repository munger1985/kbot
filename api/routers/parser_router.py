from fastapi import APIRouter, status
from api.controllers.parser_controller import parser_controller
from api.schemas.base_response import SuccessResponse
from core.auth.shortcuts import AnyAuth

router = APIRouter(prefix="/parser", tags=["Parser Management"])


@router.get(
    "/ocr-engines",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="Get available OCR engine list"
)
async def handle_get_ocr_engines(auth: AnyAuth):
    """Get the list of OCR engines available on the current system.

    Includes:
    - AI OCR models: read from config (e.g. whether DeepSeek OCR is deployed)
    - Builtin OCR engines: auto-detect local installation status (EasyOCR / Tesseract / RapidOCR)

    Access: User / Service

    Returns:
        SuccessResponse: data contains ai_ocr and builtin fields.
    """
    return await parser_controller.get_ocr_engines()
