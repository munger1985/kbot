"""Responses 适配器的稳定错误。"""

from platform_core.contracts import GENERATIVE_ERROR_CODES


class GenerativeAdapterError(RuntimeError):
    """将上游失败映射为供应商中立的生成错误码。"""

    def __init__(self, code: str, message: str, *, status_code: int = 502):
        if code not in GENERATIVE_ERROR_CODES:
            raise ValueError(f"未知生成错误码：{code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
