"""OpenAI 兼容接口共享的错误响应。"""

from fastapi.responses import JSONResponse


def openai_error_response(
    *,
    status_code: int,
    message: str,
    code: str,
    error_type: str = "invalid_request_error",
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": code,
            }
        },
    )
