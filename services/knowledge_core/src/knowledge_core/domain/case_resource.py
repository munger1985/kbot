"""系统案例资源冲突，不携带数据库或凭证详情。"""


class CaseResourceConflict(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)
