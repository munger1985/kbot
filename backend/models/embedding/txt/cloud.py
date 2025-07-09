import numpy as np
import openai
import aiohttp
from typing import List, Dict, Any, Optional, Callable, Protocol, Union
from tenacity import retry, stop_after_attempt, wait_exponential
from .base import CloudEmbeddingConfig, BaseEmbedding

class EmbeddingClient(Protocol):
    """Protocol for embedding clients"""
    async def get_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings for texts"""
        ...

class OpenAIClient:
    """Client for OpenAI embeddings"""
    def __init__(self, api_key: str, endpoint: Optional[str] = None, timeout: int = 30):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=endpoint if endpoint else None,
            timeout=timeout
        )
    
    async def get_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings from OpenAI API"""
        response = await self.client.embeddings.create(
            model=model_name,
            input=texts
        )
        if response and hasattr(response, 'data'):
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        else:
            raise RuntimeError("Invalid response from OpenAI API")

class AzureOpenAIClient:
    """Client for Azure OpenAI embeddings"""
    def __init__(self, api_key: str, endpoint: str, timeout: int = 30):
        if not endpoint:
            raise ValueError("Azure OpenAI requires an endpoint")
        self.client = openai.AsyncAzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version="2023-05-15",  # 可以从配置中获取
            timeout=timeout
        )
    
    async def get_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings from Azure OpenAI API"""
        response = await self.client.embeddings.create(
            model=model_name,
            input=texts
        )
        if response and hasattr(response, 'data'):
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings)
        else:
            raise RuntimeError("Invalid response from Azure OpenAI API")

class CohereClient:
    """Client for Cohere embeddings"""
    def __init__(self, api_key: str, endpoint: Optional[str] = None, timeout: int = 30):
        self.api_key = api_key
        self.endpoint = endpoint or "https://api.cohere.ai/v1/embed"
        self.timeout = timeout
    
    async def get_embeddings(self, texts: List[str], model_name: str) -> np.ndarray:
        """Get embeddings from Cohere API"""
        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "texts": texts,
                "model": model_name,
                "truncate": "END"
            }
            
            async with session.post(
                self.endpoint, 
                json=payload, 
                headers=headers, 
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"Cohere API error: {response.status} - {error_text}")
                
                data = await response.json()
                if "embeddings" in data:
                    return np.array(data["embeddings"])
                else:
                    raise RuntimeError("Invalid response from Cohere API")

class CloudEmbedding(BaseEmbedding):
    def __init__(self, config: CloudEmbeddingConfig):
        self.config = config
        self.client = None
        
    async def startup(self) -> None:
        """Initialize the appropriate client based on provider"""
        try:
            provider = self.config.provider.lower()
            
            if provider == "openai":
                self.client = OpenAIClient(
                    api_key=self.config.api_key,
                    endpoint=self.config.endpoint,
                    timeout=self.config.timeout
                )
            elif provider == "azure_openai":
                self.client = AzureOpenAIClient(
                    api_key=self.config.api_key,
                    endpoint=self.config.endpoint,
                    timeout=self.config.timeout
                )
            elif provider == "cohere":
                self.client = CohereClient(
                    api_key=self.config.api_key,
                    endpoint=self.config.endpoint,
                    timeout=self.config.timeout
                )
            else:
                raise ValueError(f"Unsupported cloud provider: {provider}")
        except Exception as e:
            self.ERROR_COUNTER.labels(provider=self.config.provider).inc()
            raise
    
    async def shutdown(self) -> None:
        """Release resources"""
        self.client = None
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts using cloud API"""
        with self.LATENCY_HIST.labels(model_type='cloud').time():
            try:
                if self.client is None:
                    raise RuntimeError("Client not initialized. Call startup() first.")
                
                return await self.client.get_embeddings(texts, self.config.model_name)
            except Exception as e:
                self.ERROR_COUNTER.labels(provider=self.config.provider).inc()
                raise