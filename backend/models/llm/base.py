"""
LLM基础配置和接口定义
包含：
1. 所有配置数据类（BaseLLMConfig/LocalLLMConfig/CloudLLMConfig）
2. 基础接口类BaseLLM
"""

from pydantic import BaseModel, field_validator, ConfigDict
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
import numpy as np
from tenacity import retry, stop_after_attempt
from functools import lru_cache
import torch
from prometheus_client import Counter, Histogram

class BaseLLMConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')  # 禁止额外字段
    
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 30
    
    @field_validator('temperature')
    def validate_temp(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v

class LocalLLMConfig(BaseLLMConfig):
    model_path: str
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    trust_remote_code: bool = False
    compile_model: bool = True
    
    @field_validator('device')
    def validate_device(cls, v):
        assert v in ['cpu', 'cuda', 'mps'], "Invalid device"
        return v

class CloudLLMConfig(BaseLLMConfig):
    api_key: str
    provider: str
    endpoint: Optional[str] = None
    region: Optional[str] = None

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