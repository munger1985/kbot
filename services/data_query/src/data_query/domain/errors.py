"""Data Query 稳定领域错误。"""


class DataQueryPersistenceError(RuntimeError):
    """Data Query 数据库约束或事务提交失败。"""


class DataSourceConnectionError(ValueError):
    """连接参数无法通过真实数据库校验；不泄露驱动或服务器细节。"""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code
        self.public_message = {
            "DATA_SOURCE_AUTHENTICATION_FAILED": "数据库身份验证失败，请检查只读账号和密码。",
            "DATA_SOURCE_DATABASE_NOT_FOUND": "数据库或 Oracle Service Name 不存在，请检查名称和监听器配置。",
            "DATA_SOURCE_HOST_NOT_FOUND": "无法解析数据库主机地址，请检查主机名或 DNS。",
            "DATA_SOURCE_CONNECTION_REFUSED": "数据库端口拒绝连接，请确认数据库已启动、端口正确且允许当前主机访问。",
            "DATA_SOURCE_CONNECTION_TIMEOUT": "数据库连接超时，请检查网络、防火墙和访问控制。",
            "DATA_SOURCE_TLS_FAILED": "数据库监听未接受 TLS/TCPS 连接。本地普通 TCP Listener 请关闭“使用加密连接”；生产环境请正确配置数据库证书后重试。",
            "DATA_SOURCE_METADATA_PERMISSION_DENIED": "连接已建立，但只读账号缺少读取数据库版本或 Schema 元数据的权限。",
            "DATA_SOURCE_CONNECTION_FAILED": "数据库连接失败，请检查主机、端口、数据库名称及连接加密方式。",
        }.get(code, "数据库连接失败，请检查连接配置。")
