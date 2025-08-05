"""
VLM (Vision-Language Model) base configuration and interface definition.
Contains:
1. Base configuration class VLMConfig
2. Base interface class BaseVLM
"""
from PIL import Image
from abc import ABC, abstractmethod
from pydantic import BaseModel
from prometheus_client import Counter, Histogram
from typing import Any


class VLMConfig(BaseModel):
    """Base configuration for VLM models."""   
    model_name: str
    provider: str
    max_tokens: int = 512

class LocalVLMConfig(VLMConfig):
    model_path: str | None = None             # 模型本地路径（优先级高于model_name）
    device: str | None = None                 # 如 "cuda:0"/"cpu"/"mps"
    device_map: str | None = None             # 多卡部署策略（如 "auto"/"balanced"/"sequential"）
    max_memory: dict[int, str] | None = None  # 每卡显存分配（如 {0: "20GB", 1: "20GB"}）
    trust_remote_code: bool = False           # 是否信任远程代码（如自定义模型）
    use_fp16: bool = False                    # 是否使用半精度（FP16）
    local_files_only: bool = False            # 是否强制离线加载
    compile_model: bool = True                # 是否启用PyTorch 2.0+编译优化
    quantization: str | None = None           # 量化精度设置（4bit/8bit）

class RemoteVLMConfig(VLMConfig):
    """Cloud API configuration"""
    api_key: str                   # Required for cloud
    api_endpoint: str | None = None
    api_version: str = "2023-08-01"
    request_timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7       # Override base temperature

class BaseVLM(ABC):
    """Base class for VLM implementations."""
    
    LATENCY_HIST = Histogram(
        'vlm_latency_seconds', 
        'vlm latency in seconds',
        ['model_type']
    )
    ERROR_COUNTER = Counter(
        'vlm_errors_total', 
        'Total number of vlm errors', 
        ['provider']
    )
    
    @abstractmethod
    async def startup(self) -> None:
        """Initialize resources asynchronously."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Release resources asynchronously."""
        pass

    @abstractmethod
    async def inference(self, text: str, image: str | Image.Image, **kwargs) -> str:
        """
        1. input: text and image
        2. output: the generated text chunk
        """
        pass

    @abstractmethod
    async def health_check(self) -> dict[str, Any]:
        """Health check for a remote or local model"""
        pass
