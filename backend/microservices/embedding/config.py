import os
from typing import Dict, Any, Optional
from pydantic import BaseSettings, Field

class EmbeddingServiceConfig(BaseSettings):
    """嵌入服务配置"""
    
    # 服务配置
    host: str = Field(default="0.0.0.0", description="服务监听主机")
    port: int = Field(default=8000, description="服务监听端口")
    workers: int = Field(default=1, description="工作进程数")
    log_level: str = Field(default="info", description="日志级别")
    
    # 模型池配置
    max_idle_time: int = Field(default=3600, description="模型最大空闲时间（秒）")
    
    # 批处理配置
    max_batch_size: int = Field(default=64, description="最大批处理大小")
    max_wait_time: float = Field(default=0.1, description="最大等待时间（秒）")
    
    # 健康检查配置
    health_check_interval: int = Field(default=300, description="健康检查间隔（秒）")
    
    # 默认模型配置
    default_models: Dict[str, Dict[str, Any]] = Field(
        default={},
        description="默认模型配置，格式为 {model_id: config}"
    )
    
    class Config:
        env_prefix = "EMBEDDING_"
        env_file = ".env"

def load_config() -> EmbeddingServiceConfig:
    """加载配置"""
    return EmbeddingServiceConfig()

def get_model_config_from_env(model_id: str) -> Optional[Dict[str, Any]]:
    """从环境变量加载特定模型的配置"""
    env_prefix = f"EMBEDDING_MODEL_{model_id.upper()}_"
    config = {}
    
    # 检查是否有此模型的环境变量
    has_config = False
    for key in os.environ:
        if key.startswith(env_prefix):
            has_config = True
            break
    
    if not has_config:
        return None
    
    # 提取配置
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            config_key = key[len(env_prefix):].lower()
            config[config_key] = value
    
    return config