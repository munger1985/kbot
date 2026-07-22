from pydantic import BaseModel, Field
from typing import Any

# --- 请求模型 ---
class ExecuteRequest(BaseModel):
    kb_id: str = Field(..., description="业务知识库ID")
    db_type: str = Field(..., description="数据库引擎类型, e.g., oracle, mysql")
    connection_config: dict[str, Any] = Field(..., description="数据库连接配置")
    sql: str = Field(..., description="待执行的SQL语句")
    limit: int | None = Field(100, description="行数限制")


class OpsExecuteRequest(BaseModel):
    instance_id: str = Field(..., description="运维物理/云端实例唯一ID")
    db_type: str = Field(..., description="数据库引擎类型, e.g., oracle, mysql")
    sql: str = Field(..., description="待执行的专家运维 SQL 或命令（可包含命名占位符如 :sid）")
    connection_config: dict[str, Any] = Field(..., description="动态连接配置")
    environment: str = Field("prod", description="环境标签: prod, stg, dev")
    run_mode: str = Field("read_only", description="运行模式: read_only (指标探测) 或 mutation (高危变更)")
    limit: int | None = Field(None, description="行数限制")
    params: dict[str, Any] | None = Field(None, description="SQL 占位符参数绑定字典（防注入参数化查询）")