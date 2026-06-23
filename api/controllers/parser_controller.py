from api.schemas.base_response import SuccessResponse
from core.config.settings import get_dsocr_config, detect_builtin_ocr_engines, OCR_ENGINE_LABELS
from core.dictionary import OCRProvider


class ParserController:
    """Parser configuration controller."""

    async def get_ocr_engines(self) -> SuccessResponse:
        """Get available OCR engine list (AI models + builtin engine detection).

        Returns:
            SuccessResponse: data contains ai_ocr and builtin fields.
        """
        dsocr_config = get_dsocr_config()
        builtin_status = detect_builtin_ocr_engines()

        # AI OCR models (from config, not database)
        ai_ocr = {
            "available": dsocr_config.enabled,
            "models": [
                {"value": "deepseek-ocr-2", "label": "DeepSeek OCR v2"}
            ] if dsocr_config.enabled else []
        }

        # Builtin OCR engine (actual installation detection)
        builtin = []
        for engine in OCRProvider:
            engine_value = engine.value
            if engine_value in builtin_status:
                builtin.append({
                    "value": engine_value,
                    "label": OCR_ENGINE_LABELS.get(engine_value, engine_value),
                    "available": builtin_status[engine_value]
                })

        return SuccessResponse(
            message="OCR engine list retrieved successfully",
            data={"ai_ocr": ai_ocr, "builtin": builtin}
        )


parser_controller = ParserController()
