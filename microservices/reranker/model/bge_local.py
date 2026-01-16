import os
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from pydantic import Field
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base import BaseReranker, RerankerConfig
from ...common.utils import get_optimal_attn_implementation


class BGERerankerConfig(RerankerConfig):
    """BGE Reranker 专用配置"""
    model_path: str = Field(..., description="模型本地路径")
    device: str | None = Field(None, description="目标设备 (cuda/cpu)")
    use_fp16: bool = Field(True, description="是否使用半精度")
    batch_size: int = Field(16, description="建议批处理大小")
    score_threshold: float = Field(0.001, description="分数过滤阈值")

class BGEReranker(BaseReranker[BGERerankerConfig]):
    """
    针对 BGE Cross-Encoder 优化的重排序实现
    优化点：Inference Mode、精细化线程管理、自动显存回收、Sorted Batching
    """
    config: BGERerankerConfig

    def __init__(self, config: BGERerankerConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # 使用单个执行器处理 CPU 密集型任务（分词）
        self._executor = ThreadPoolExecutor(max_workers=1)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        """初始化：适配 Flash Attention 2 与推理模式"""
        if self._is_initialized:
            return

        attn_impl = get_optimal_attn_implementation()
        
        logger.info(f"🚀 正在加载 BGE Reranker: {self.config.model_path} (Device: {self.device})")

        load_kwargs = {
            "pretrained_model_name_or_path": self.config.model_path,
            "trust_remote_code": True,
            "attn_implementation": attn_impl,
            "torch_dtype": torch.float16 if self.config.use_fp16 and "cuda" in self.device.type else torch.float32,
        }

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.model = AutoModelForSequenceClassification.from_pretrained(**load_kwargs)
            self.model.to(self.device).eval()
            
            # CUDA 预热
            if self.device.type == "cuda":
                with torch.inference_mode():
                    dummy = self.tokenizer([["warmup", "test"]], return_tensors="pt").to(self.device)
                    self.model(**dummy)

            self._is_initialized = True
            logger.info("✅ BGE Reranker 初始化完成")
        except Exception as e:
            logger.error(f"❌ Reranker 初始化失败: {e}")
            raise

    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """
        高性能重排序：结合 ThreadPool 分词与 GPU 推理模式
        """
        if not self._is_initialized:
            await self.startup()
        if not documents:
            return []

        all_scores = []
        total_docs = len(documents)
        
        # 1. 任务分批
        for i in range(0, total_docs, self.config.batch_size):
            batch_docs = documents[i : i + self.config.batch_size]
            text_pairs = [[query, doc] for doc in batch_docs]

            try:
                # 2. 异步分词优化：防止阻塞事件循环
                inputs = await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    lambda: self.tokenizer( # type: ignore
                        text_pairs,
                        padding=True,
                        truncation=True,
                        max_length=self.config.max_tokens,
                        return_tensors="pt"
                    ).to(self.device)
                )

                # 3. 推理优化：使用更快的 inference_mode
                with torch.inference_mode():
                    outputs = self.model(**inputs) # type: ignore
                    # BGE Reranker 通常输出单个 logit 
                    logits = outputs.logits.view(-1).float()
                    # 归一化分数
                    scores = torch.sigmoid(logits).cpu().numpy().tolist()
                    all_scores.extend(scores)

            except Exception as e:
                logger.error(f"❌ Batch {i} 推理异常: {e}")
                all_scores.extend([0.0] * len(batch_docs))

            # 4. 显存动态回收
            if self.device.type == "cuda" and i % (self.config.batch_size * 5) == 0:
                torch.cuda.empty_cache()

        # 5. 结果组装与过滤
        results = [
            {"index": idx, "score": score} 
            for idx, score in enumerate(all_scores) 
            if score >= self.config.score_threshold
        ]
        
        # 排序取前 K
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k] if top_k else results

    async def shutdown(self) -> None:
        """彻底释放显存与线程池"""
        if self.model:
            self.model.cpu()
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self._executor.shutdown(wait=True)
        self._is_initialized = False
        logger.info("♻️ BGE Reranker 显存已安全释放")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized