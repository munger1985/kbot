"""AIOps Task Handler 的类型化执行错误。"""


class RetryableTaskError(RuntimeError):
    """通知运行内核按任务重试策略重新调度当前 Handler。"""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code
