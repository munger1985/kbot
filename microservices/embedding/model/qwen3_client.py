import torch
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from typing import Any

from .local_client import LocalEmbedding, LocalEmbeddingConfig, EmbeddingResponse, EmbeddingDataItem


# class Qwen3EmbeddingConfig(LocalEmbeddingConfig):
#     """Qwen3 Embedding 模型的专用配置"""
#     use_flash_attention: bool = True  # 是否使用 flash attention 2
#     task_description: str = "Generate the embedding vector."  # 任务描述
#     max_length: int = 8192  # Qwen3 Embedding 支持的最大长度


class Qwen3Embedding(LocalEmbedding):
    """
    Qwen3 Embedding 模型的专用实现，支持指令格式和 last token pooling。
    继承自 LocalEmbedding，重写关键方法以适应 Qwen3 的特殊需求。
    """

    def __init__(self, config: LocalEmbeddingConfig):
        """
        初始化 Qwen3 Embedding 模型
        
        参数:
            config: LocalEmbeddingConfig 配置对象
        """
        
        super().__init__(config)
        self.task_description: str = "Generate the embedding vector."  # 任务描述
        self.max_length: int = 8192  # Qwen3 Embedding 支持的最大长度
        self.use_flash_attention: bool = True  # 是否使用 flash attention 2

    def _load_model(self) -> torch.nn.Module:
        """使用 flash attention 2 加载 Qwen3 模型"""
        load_kwargs = {
            "pretrained_model_name_or_path": self.name_or_path,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
            "local_files_only": self.local_files_only,
            "cache_dir": self.cache_dir
        }

        # 添加 flash attention 2 支持
        if self.use_flash_attention and torch.cuda.is_available():
            try:
                load_kwargs.update({
                    "attn_implementation": "flash_attention_2",
                    "torch_dtype": torch.float16
                })
                logger.info("启用 flash_attention_2 加速")
            except Exception as e:
                logger.warning(f"启用 flash_attention_2 失败: {e}")

        # 确定设备配置
        if self.device_map is not None:
            load_kwargs.update({
                "device_map": self.device_map,
                "max_memory": self.max_memory,
            })
            self._using_device_map = True
            target_device = None
        else:
            self._using_device_map = False
            target_device = self.device
            if "torch_dtype" not in load_kwargs:
                load_kwargs["torch_dtype"] = torch.float16 if self.use_fp16 else torch.float32

        try:
            model = AutoModel.from_pretrained(**load_kwargs)
            
            if not self._using_device_map and target_device is not None:
                model = model.to(target_device)
                logger.debug(f"模型移动到设备: {target_device}")
            
            logger.debug("Qwen3 embedding 模型加载成功")
            return model
            
        except Exception as e:
            logger.error(f"加载 Qwen3 embedding 模型失败: {str(e)}")
            raise

    def _load_tokenizer(self) -> Any:
        """加载 Qwen3 分词器，设置 padding_side='left'"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=self.name_or_path,
                trust_remote_code=self.trust_remote_code,
                use_fast=True,
                model_max_length=self.max_length,
                padding_side='left',  # Qwen3 需要 left padding
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir
            )
            logger.debug("Qwen3 分词器加载成功")
            return tokenizer
        except Exception as e:
            logger.error(f"加载 Qwen3 分词器失败: {str(e)}")
            raise

    def _format_query_text(self, text: str) -> str:
        """
        格式化查询文本，添加指令前缀
        
        参数:
            text: 原始查询文本
            
        返回:
            格式化后的查询文本
        """
        return f'Instruct: {self.task_description}\nQuery: {text}'

    # def _is_query_text(self, text: str) -> bool:
    #     """
    #     判断文本是否为查询文本（需要添加指令前缀）
        
    #     参数:
    #         text: 待判断的文本
            
    #     返回:
    #         bool: 是否为查询文本
    #     """
    #     # 简单的启发式判断：如果文本看起来像问题或查询
    #     query_indicators = ['?', 'what', 'how', 'why', 'when', 'where', 'explain', 'describe']
    #     text_lower = text.lower()
    #     return any(indicator in text_lower for indicator in query_indicators)

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        normalize: bool = True,
        raise_on_error: bool = True,
        is_query: bool = True,
        **kwargs
    ) -> EmbeddingResponse:
        """
        生成 Qwen3 嵌入向量，支持查询和文档两种模式
        
        参数:
            texts: 文本列表
            batch_size: 批次大小，0 表示自动选择
            normalize: 是否归一化嵌入向量
            raise_on_error: 是否在错误时抛出异常
            is_query: 是否为查询文本（None 表示自动判断）
            
        返回:
            EmbeddingResponse: 嵌入向量响应
        """
        # 预处理文本：为查询文本添加指令前缀
        processed_texts = []
        for text in texts:
            if is_query is True:
                processed_texts.append(self._format_query_text(text))
            else:
                processed_texts.append(text)  # 文档文本保持原样

        # 调用父类的 embed 方法
        return await super().embed(
            texts=processed_texts,
            batch_size=batch_size,
            normalize=normalize,
            raise_on_error=raise_on_error
        )

    async def embed_queries(self, queries: list[str], **kwargs) -> EmbeddingResponse:
        """
        专门用于嵌入查询文本的便捷方法
        
        参数:
            queries: 查询文本列表
            **kwargs: 其他参数传递给 embed 方法
            
        返回:
            EmbeddingResponse: 嵌入向量响应
        """
        return await self.embed(queries, is_query=True, **kwargs)

    async def embed_documents(self, documents: list[str], **kwargs) -> EmbeddingResponse:
        """
        专门用于嵌入文档文本的便捷方法
        
        参数:
            documents: 文档文本列表
            **kwargs: 其他参数传递给 embed 方法
            
        返回:
            EmbeddingResponse: 嵌入向量响应
        """
        return await self.embed(documents, is_query=False, **kwargs)

    def _last_token_pooling(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Qwen3 专用的 last token pooling 方法
        
        参数:
            last_hidden_states: 最后隐藏状态，形状为 (batch_size, seq_len, hidden_size)
            attention_mask: 注意力掩码，形状为 (batch_size, seq_len)
            
        返回:
            torch.Tensor: 池化后的嵌入向量，形状为 (batch_size, hidden_size)
        """
        # 检查是否为左填充
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        
        if left_padding:
            # 左填充：取最后一个token
            return last_hidden_states[:, -1]
        else:
            # 右填充：取每个序列的最后一个有效token
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths
            ]

    async def _process_single_batch(
        self,
        batch: list[str],
        normalize: bool
    ) -> tuple[torch.Tensor, int]:
        """处理单个文本批次，使用 Qwen3 的特殊池化方法"""
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("模型和分词器必须已初始化")
        
        # 分词
        encoded_input = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 跳过空输入
        if encoded_input['input_ids'].numel() == 0:
            return torch.empty((0, self.embedding_dim)), 0
        
        # 移动到正确的设备
        if not self._using_device_map and hasattr(self.model, 'device'):
            device = self.model.device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        
        # 使用 Qwen3 专用的 last token pooling
        embeddings = self._last_token_pooling(
            outputs.last_hidden_state,
            encoded_input['attention_mask']
        )
        
        # 归一化
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 计算此批次中的令牌数
        tokens = encoded_input['input_ids'].numel()
        
        return embeddings.cpu(), tokens

    # @property
    # def embedding_dim(self) -> int:
    #     """获取 Qwen3 嵌入向量的输出维度"""
    #     if self.model is None:
    #         raise RuntimeError("模型未初始化")
    #     return self.model.config.hidden_size