"""知识检索应用应用错误。"""


class KnowledgeRetrievalApplicationError(ValueError):
    """将绑定、Run 与媒体校验转换为稳定的内部 API 错误。"""

    def __init__(self, code: str, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
