# nacos_config.py
from pydantic import BaseModel
from nacos import NacosClient

class NacosSettings(BaseModel):
    """Nacos配置模型"""
    server_addr: str = "localhost:8848"
    namespace: str = "public"
    group: str = "DEFAULT_GROUP"
    username: str | None = None
    password: str | None = None

    class Config:
        env_prefix = "NACOS_"
        env_file = ".env"

class NacosConfigManager:
    """Nacos配置管理器"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        """初始化配置"""
        self.settings = NacosSettings()
        self.client = NacosClient(
            server_addresses=self.settings.server_addr,
            namespace=self.settings.namespace,
            username=self.settings.username,
            password=self.settings.password
        )

    def get_config(self, data_id: str, group: str | None = None) -> str:
        """获取指定配置"""
        return self.client.get_config( # type: ignore
            data_id=data_id,
            group=group or self.settings.group
        )

    def add_watcher(self, data_id: str, cb: callable, group: str | None = None): # type: ignore
        """添加配置监听"""
        self.client.add_config_watcher(
            data_id=data_id,
            group=group or self.settings.group,
            cb=cb
        )

# 单例实例
nacos_manager = NacosConfigManager()