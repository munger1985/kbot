"""
大模型服务核心模块
包含LLM和Embedding两类模型的实现
"""
from .llm import LLMProvider
from .embedding.txt import EmbeddingProvider

__all__ = ['LLMProvider', 'EmbeddingProvider']
__version__ = '0.1.0'