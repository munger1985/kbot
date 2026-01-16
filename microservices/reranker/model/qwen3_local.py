import os
import torch
from typing import Any
from pydantic import Field
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseReranker, RerankerConfig

class Qwen3RerankerConfig(RerankerConfig):
    """Qwen3 Reranker 官方适配配置"""
    model_path: str = Field(..., description="本地路径")
    device: str | None = Field(None, description="设备")
    use_fp16: bool = Field(True, description="RTX 5080/4090 建议设为 True 使用 BF16")
    batch_size: int = Field(8, description="批处理大小")
    score_threshold: float = Field(-10.0, description="分数阈值")
    max_tokens: int = Field(1024, description="最大序列长度")
    instruction: str | None = Field(
        "Given a query and a relevant document, retrieve the relevance score of the document to the query.", 
        description="官方默认指令"
    )

class Qwen3Reranker(BaseReranker[Qwen3RerankerConfig]):
    """
    针对生成式 Qwen 架构优化的 Reranker 实现
    优化点：Inference Mode、BF16 精度适配、更健壮的 Token 定位
    """

    def __init__(self, config: Qwen3RerankerConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self._is_initialized = False
        # 统一处理设备对象
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.yes_id = None
        self.no_id = None
        
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    async def startup(self) -> None:
        """初始化：加载模型并启用针对 RTX 50 系列的硬件优化"""
        if self._is_initialized:
            return

        from ...common.utils import get_optimal_attn_implementation
        attn_impl = get_optimal_attn_implementation() 
        
        try:
            # 1. 初始化 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, trust_remote_code=True)
            self.tokenizer.padding_side = "left" # 生成式 Reranker 必须左填充
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 2. 预存 Token ID (避免在推理循环中重复调用 encode)
            self.yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[-1]
            self.no_id = self.tokenizer.encode("no", add_special_tokens=False)[-1]

            # 3. 加载模型：优先使用 bfloat16 (RTX 5080/4090/A100 等显卡)
            compute_dtype = torch.bfloat16 if (self.config.use_fp16 and "cuda" in self.device.type) else torch.float32
            
            logger.info(f"🚀 加载 Qwen Reranker: {self.config.model_path} (Dtype: {compute_dtype}, Impl: {attn_impl})")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                torch_dtype=compute_dtype,
                device_map={"": self.device} # 明确映射到单个设备
            )
            self.model.eval()
            
            # CUDA 预热
            if self.device.type == "cuda":
                with torch.inference_mode():
                    warmup_text = "warmup"
                    self.model(**self.tokenizer([warmup_text], return_tensors="pt").to(self.device))

            self._is_initialized = True
            logger.info(f"✅ Qwen3 Reranker 就绪 (Yes ID: {self.yes_id})")
        except Exception as e:
            logger.error(f"❌ 加载失败: {e}")
            raise

    async def rerank(
        self, 
        query: str, 
        documents: list[str], 
        top_k: int | None = None
    ) -> list[dict[str, Any]]:
        if not self._is_initialized:
            await self.startup()
        if not documents:
            return []

        all_results = []
        
        # 1. 批量推理
        for i in range(0, len(documents), self.config.batch_size):
            batch_docs = documents[i : i + self.config.batch_size]
            
            # 采用官方建议的 Prompt 结构
            formatted_texts = [
                f"<Instruct>: {self.config.instruction}\n<Query>: {query}\n<Document>: {d}\nRelevant (yes/no):"
                for d in batch_docs
            ]

            try:
                # 2. 编码 (左填充确保最后一个 token 是预测位)
                inputs = self.tokenizer( # type: ignore
                    formatted_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_tokens,
                    return_tensors="pt"
                ).to(self.device)

                # 3. 高性能推理模式
                with torch.inference_mode():
                    outputs = self.model(**inputs) # type: ignore
                    # 获取 batch 中每个序列最后一个 token 的 logits
                    last_token_logits = outputs.logits[:, -1, :] 
                    
                    # 4. 计算得分：logit(yes) - logit(no)
                    # 使用 float() 确保在 BF16 模式下减法精度正确
                    yes_logits = last_token_logits[:, self.yes_id].float()
                    no_logits = last_token_logits[:, self.no_id].float()
                    scores = (yes_logits - no_logits).cpu().numpy().tolist()

                for j, score in enumerate(scores):
                    all_results.append({
                        "index": i + j,
                        "score": score,
                        "document": batch_docs[j]
                    })
                    
            except Exception as e:
                logger.error(f"❌ Batch 推理失败 [{i}]: {e}")
                continue

        # 5. 全局排序与过滤
        all_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = [r for r in all_results if r["score"] >= self.config.score_threshold]
        
        return final_results[:top_k] if top_k else final_results

    async def shutdown(self) -> None:
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        self._is_initialized = False
        logger.info("♻️ Qwen3 Reranker 资源已完全回收")

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized