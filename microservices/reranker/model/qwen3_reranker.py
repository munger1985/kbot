from typing import Any
from pydantic import Field
from loguru import logger

# 优雅降级导入
try:
    import torch
    TORCH_AVAILABLE = True
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("警告: PyTorch 不可用，将使用备用方案")

from .base import BaseReranker, RerankerConfig


class Qwen3RerankerConfig(RerankerConfig):
    """Qwen3 Reranker 专用配置"""
    model_name: str = Field("Qwen/Qwen3-Reranker-0.6B", description="Qwen3 模型名称")
    model_path: str | None = Field(None, description="模型文件的本地路径")
    device: str | None = Field(None, description="目标设备，None则自动选择")
    max_tokens: int = Field(8192, description="Qwen3 支持 8192 tokens")
    use_fp16: bool = Field(True, description="使用半精度推理")
    use_flash_attention: bool = Field(True, description="使用 Flash Attention 加速")
    instruction: str | None = Field(None, description="自定义指令")
    batch_size: int = Field(1, description="批次大小，Qwen3 建议为1")


class Qwen3Reranker(BaseReranker):
    """适配 Qwen3 Reranker 的类，支持优雅降级"""
    
    def __init__(self, config: Qwen3RerankerConfig):
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.model_name = config.model_name
        self.model_path = config.model_path
        self.max_tokens = config.max_tokens
        self.use_fp16 = config.use_fp16 and TORCH_AVAILABLE  # 只有 PyTorch 可用时才启用 FP16
        self.use_flash_attention = config.use_flash_attention and TORCH_AVAILABLE
        self.instruction = config.instruction
        self.batch_size = 1  # 强制批次大小为1
        
        # 设备检测和设置
        self.device = self._setup_device(config.device)
        
        # Qwen3 特殊属性
        self.token_false_id: int | None = None
        self.token_true_id: int | None = None
        self.prefix_tokens: list[int] = []
        self.suffix_tokens: list[int] = []
        
        self._is_initialized = False
        self._fallback_mode = not TORCH_AVAILABLE  # 降级模式标志
    
    def _setup_device(self, device_config: str | None) -> str:
        """设置设备，确保 CUDA 可用性"""
        if device_config:
            return device_config
        
        # 自动检测设备
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    
    def _check_flash_attention_available(self) -> bool:
        """检查 flash attention 2 是否可用"""
        if not TORCH_AVAILABLE:
            return False
            
        try:
            # 尝试导入 flash attention 2
            import flash_attn
            # 检查 CUDA 是否可用
            if not torch.cuda.is_available():
                logger.warning("CUDA 不可用，禁用 Flash Attention 2")
                return False
            # 检查 transformers 版本是否支持
            from transformers.utils import is_flash_attn_2_available
            return is_flash_attn_2_available()
        except ImportError:
            logger.warning("flash-attn 不可用，使用标准注意力机制")
            return False
        except Exception as e:
            logger.warning(f"检查 Flash Attention 2 可用性时出错: {e}")
            return False

    async def startup(self) -> None:
        """初始化 Qwen3 reranker 模型"""
        if self._is_initialized:
            return

        # 如果 PyTorch 不可用，进入降级模式
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch 不可用，Qwen3 Reranker 进入降级模式")
            self._is_initialized = True
            self._fallback_mode = True
            return

        logger.info(f"正在初始化 Qwen3 Reranker: {self.model_name}，设备: {self.device}")
        
        # 确定模型路径
        model_path = self.model_path or self.model_name
        
        try:
            # 加载分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                padding_side='left',  # 官方要求
                trust_remote_code=True
            )
            
            # 设置 pad_token
            if self.tokenizer.pad_token is None: # type: ignore
                if self.tokenizer.eos_token is not None: # type: ignore
                    self.tokenizer.pad_token = self.tokenizer.eos_token # type: ignore
                    logger.info(f"设置 pad_token 为 eos_token: {self.tokenizer.pad_token}") # type: ignore
                else:
                    self.tokenizer.pad_token = self.tokenizer.unk_token # type: ignore
                    logger.info(f"设置 pad_token 为 unk_token: {self.tokenizer.pad_token}") # type: ignore
            
            # 准备模型加载参数
            load_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "trust_remote_code": True,
            }
            
            # 设备相关的配置
            if self.device.startswith('cuda'):
                # CUDA 设备：使用 FP16 和可能的 Flash Attention
                if self.use_fp16:
                    load_kwargs["torch_dtype"] = torch.float16
                
                # 只有在 CUDA 上且可用时才启用 Flash Attention
                if self.use_flash_attention:
                    self.use_flash_attention = self._check_flash_attention_available()
                    if self.use_flash_attention:
                        try:
                            load_kwargs["attn_implementation"] = "flash_attention_2"
                            logger.info("启用 Flash Attention 2")
                        except Exception as e:
                            logger.warning(f"启用 Flash Attention 2 失败: {e}")
                            self.use_flash_attention = False
                    else:
                        logger.info("Flash Attention 2 不可用，使用标准注意力机制")
            else:
                # CPU 设备：使用 FP32，禁用 Flash Attention
                load_kwargs["torch_dtype"] = torch.float32
                self.use_flash_attention = False
                logger.info("CPU 设备，禁用 Flash Attention")
            
            # 关键：使用 AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            
            # 移动模型到设备
            self.model = self.model.to(self.device) # type: ignore
            self.model.eval()
            
            # 初始化 Qwen3 特殊 tokens
            self._init_qwen3_special_tokens()
            
            self._is_initialized = True
            self._fallback_mode = False
            logger.info(f"Qwen3 Reranker 初始化成功")
            
        except Exception as e:
            logger.error(f"Qwen3 Reranker 初始化失败: {e}")
            # 初始化失败时进入降级模式
            self._is_initialized = True
            self._fallback_mode = True
            logger.warning("Qwen3 Reranker 进入降级模式")

    def _init_qwen3_special_tokens(self):
        """初始化 Qwen3 特殊 tokens"""
        if not TORCH_AVAILABLE or self.tokenizer is None:
            return
            
        try:
            # 官方示例中的 token id 获取方式
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no") # type: ignore
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes") # type: ignore
            
            # 官方示例中的 prefix 和 suffix
            prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            
            self.prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False) # type: ignore
            self.suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False) # type: ignore
            
            logger.debug(f"特殊 tokens 初始化完成: token_true_id={self.token_true_id}, token_false_id={self.token_false_id}")
            
        except Exception as e:
            logger.error(f"初始化特殊 tokens 失败: {e}")
            # 设置默认值
            self.token_false_id = 0
            self.token_true_id = 1
    
    def _format_instruction(self, query: str, doc: str) -> str:
        """格式化输入指令"""
        instruction = self.instruction or 'Given a web search query, retrieve relevant passages that answer the query'
        return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"
    
    def _compute_single_score_fallback(self, query: str, document: str) -> float:
        """降级模式下的分数计算"""
        # 基于文本长度的简单启发式分数
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        # 计算 Jaccard 相似度
        if len(query_words) == 0 or len(doc_words) == 0:
            return 0.0
            
        intersection = len(query_words.intersection(doc_words))
        union = len(query_words.union(doc_words))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # 添加基于长度的权重
        length_penalty = min(len(document) / 1000, 1.0)  # 文档长度惩罚
        
        return jaccard_similarity * 0.7 + length_penalty * 0.3

    def _compute_single_score(self, query: str, document: str) -> float:
        """计算单个文档的分数"""
        if not self._is_initialized:
            raise RuntimeError("模型未初始化")
        
        # 如果处于降级模式，使用备用方案
        if self._fallback_mode or not TORCH_AVAILABLE:
            return self._compute_single_score_fallback(query, document)
        
        try:
            # 格式化输入
            formatted_text = self._format_instruction(query, document)
            
            # 第一步：基础分词
            inputs = self.tokenizer( # type: ignore
                formatted_text,
                padding=False,
                truncation='longest_first',
                return_attention_mask=False,
                max_length=self.max_tokens - len(self.prefix_tokens) - len(self.suffix_tokens)
            )
            
            # 第二步：添加特殊 tokens
            input_ids = inputs['input_ids']
            input_ids = self.prefix_tokens + input_ids + self.suffix_tokens
            
            # 第三步：填充
            inputs = {'input_ids': [input_ids]}
            inputs = self.tokenizer.pad( # type: ignore
                inputs,
                padding=True,
                return_tensors="pt",
                max_length=self.max_tokens
            )
            
            # 移动输入到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 官方示例的推理逻辑
            with torch.no_grad():
                batch_scores = self.model(**inputs).logits[:, -1, :] # type: ignore
                true_vector = batch_scores[:, self.token_true_id]
                false_vector = batch_scores[:, self.token_false_id]
                batch_scores = torch.stack([false_vector, true_vector], dim=1)

                # 归一化分数到 [0, 1] 范围
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                score = batch_scores[:, 1].exp().item()
            
            return score
            
        except Exception as e:
            logger.error(f"计算分数失败: {e}")
            # 返回降级分数而不是抛出异常，避免影响其他文档
            return self._compute_single_score_fallback(query, document)
    
    async def rerank(self, query: str, documents: list[str], top_k: int | None = None) -> list[dict[str, Any]]:
        """
        对文档进行重排序
        """
        if not self._is_initialized:
            raise RuntimeError("模型未初始化，请先调用 startup() 方法")
        
        if not documents:
            return []
        
        top_k = min(top_k, len(documents)) if top_k else len(documents)
        
        try:
            scores = []
            total_docs = len(documents)
            
            if self._fallback_mode:
                logger.warning(f"使用降级模式对 {total_docs} 个文档进行重排序")
            else:
                logger.info(f"开始处理 {total_docs} 个文档，设备: {self.device}")
            
            # 逐文档处理，完全避免批次padding问题
            for i, doc in enumerate(documents):
                try:
                    score = self._compute_single_score(query, doc)
                    scores.append(score)
                    
                    if (i + 1) % 10 == 0 or (i + 1) == total_docs:
                        logger.debug(f"已处理 {i + 1}/{total_docs} 个文档")
                        
                except Exception as e:
                    logger.error(f"处理文档 {i} 时出错: {e}")
                    scores.append(0.0)  # 出错时返回默认分数
            
            # 创建结果并排序
            scored_results = [(i, score) for i, score in enumerate(scores)]
            scored_results.sort(key=lambda x: x[1], reverse=True)
            scored_results = scored_results[:top_k]
            
            mode_info = "降级模式" if self._fallback_mode else "正常模式"
            logger.info(f"重排序完成({mode_info})，返回前 {top_k} 个结果")
            return [{"index": idx, "score": float(score)} for idx, score in scored_results]
        
        except Exception as e:
            logger.exception(f"重排序过程中发生错误: {e}")
            # 在错误时返回基于索引的默认排序
            return [{"index": i, "score": 1.0 - (i * 0.01)} for i in range(min(top_k, len(documents)))]

    async def shutdown(self) -> None:
        """清理资源"""
        if TORCH_AVAILABLE and self.model is not None:
            # 移动模型到 CPU 释放 GPU 内存
            if str(self.device).startswith('cuda'):
                self.model = self.model.to('cpu')
            
            # 清理 GPU 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        self._is_initialized = False
        logger.info("Qwen3 Reranker 资源已释放")

    @property
    def is_fallback_mode(self) -> bool:
        """检查是否处于降级模式"""
        return self._fallback_mode or not TORCH_AVAILABLE

    @property
    def supports_flash_attention(self) -> bool:
        """检查是否支持 Flash Attention"""
        return self.use_flash_attention and TORCH_AVAILABLE and not self._fallback_mode