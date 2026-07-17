from abc import ABC, abstractmethod
import pandas as pd
from typing import Any, cast
from ..schemas.db_config import BaseDBConfig

class BaseDriver(ABC):
    """所有数据库驱动的基类"""
    
    def __init__(self, config: Any):
        self.config = config
        self.connection = None

    @abstractmethod
    async def connect(self):
        """建立异步连接（或池化连接）"""
        pass

    @abstractmethod
    async def execute_query(self, sql: str, params: tuple | dict | None = None) -> pd.DataFrame:
        """执行 SQL 并返回 DataFrame。params 为可选的参数化查询绑定值（防 SQL 注入）。"""
        pass

    async def execute_non_query(self, sql: str, params: tuple | dict | None = None) -> str: # type: ignore
        """执行无结果集的管理命令。子类可按需覆写以支持参数化。"""
        pass

    @abstractmethod
    async def close(self):
        """关闭连接"""
        pass

    def format_results(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        """统一将结果转换为 JSON 友好的格式"""
        return cast(list[dict[str, Any]], df.to_dict(orient="records"))