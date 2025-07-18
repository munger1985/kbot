"""
LLM基础配置和接口定义
包含：
1. 所有配置数据类LLMConfig
2. 基础接口类BaseLLM
"""

from pydantic import BaseModel, field_validator, ConfigDict
from typing import Optional, Dict, List
from tenacity import retry, stop_after_attempt
from prometheus_client import Counter, Histogram
from core.config import settings

class LLMConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # 禁止额外字段
    
    model_name: str
    api_key: str
    provider: str
    endpoint: Optional[str] = None
    temperature: float = settings['llm']['temperature']
    top_p: float = settings['llm']['top_p']
    top_k: int = settings['llm']['top_k']
    max_tokens: int = settings['llm']['max_tokens']
    max_retries: int = settings['llm']['max_retries']
    timeout: int = settings['llm']['timeout']
    
    @field_validator('temperature')
    def validate_temp(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    

class BaseLLM:
    ERROR_COUNTER = Counter('llm_errors', 'Errors by provider', ['provider'])
    LATENCY_HIST = Histogram('llm_latency', 'Generation latency', ['model_type'])
    
    async def startup(self):
        """异步初始化资源"""
        pass
    
    async def shutdown(self):
        """异步释放资源"""
        pass
    
    @retry(stop=stop_after_attempt(3))
    async def generate(self, prompt: str, **kwargs) -> str:
        """异步生成文本"""
        raise NotImplementedError
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """异步对话"""
        raise NotImplementedError