import oci
import math
import json
from loguru import logger
from .base import BaseEmbedding, EmbeddingConfig, EmbeddingResponse, EmbeddingDataItem

class OCIEmbeddingConfig(EmbeddingConfig):
    """OCI 嵌入客户端的配置。"""
    compartment_id: str
    config_file: dict | str
    api_endpoint: str


class OCIEmbedding(BaseEmbedding):
    """OCI 嵌入客户端实现。"""

    def __init__(self, config: OCIEmbeddingConfig):
        """初始化 OCI 嵌入客户端。
        
        参数:
            config: OCI 嵌入配置
        """
        self.config = config
        self.client = None
        self._is_running = False
        self.batch_size = config.batch_size or 1
    
    async def startup(self) -> None:
        """初始化 OCI 客户端。"""
        try:
            # 处理配置文件（支持字符串或字典格式）
            if isinstance(self.config.config_file, str):
                oci_config = json.loads(self.config.config_file)
            else:
                oci_config = self.config.config_file

            # 创建 OCI 客户端
            self.client = oci.generative_ai_inference.GenerativeAiInferenceClient(
                config=oci_config,
                service_endpoint=self.config.api_endpoint,
                retry_strategy=oci.retry.NoneRetryStrategy(),
                timeout=(10, 240)  # 连接超时10秒，读取超时240秒
            )
            
            self._is_running = True
            logger.info("OCI 客户端初始化成功")
            
        except json.JSONDecodeError as e:
            logger.error(f"OCI 配置文件 JSON 解析错误: {str(e)}")
            raise RuntimeError(f"OCI 配置文件格式无效: {str(e)}")
        except oci.exceptions.ConfigFileNotFound as e:
            logger.error(f"OCI 配置文件未找到: {str(e)}")
            raise RuntimeError(f"OCI 配置文件未找到: {str(e)}")
        except oci.exceptions.InvalidConfig as e:
            logger.error(f"OCI 配置无效: {str(e)}")
            raise RuntimeError(f"OCI 配置无效: {str(e)}")
        except Exception as e:
            logger.error(f"初始化 OCI 客户端时发生错误: {str(e)}")
            raise RuntimeError(f"初始化 OCI 客户端失败: {str(e)}")
        
    async def shutdown(self) -> None:
        """关闭 OCI 客户端。"""
        try:
            if self.client:
                self.client = None
            self._is_running = False
            logger.info("OCI 客户端已关闭")
        except Exception as e:
            logger.warning(f"关闭 OCI 客户端时发生错误: {str(e)}")

    async def embed(
        self,
        texts: list[str],
        batch_size: int = 0,
        normalize: bool = True,  # 保持参数一致性，虽然OCI可能不支持
        raise_on_error: bool = True,
        **kwargs
    ) -> EmbeddingResponse:
        """批量处理文本列表并生成嵌入向量。
        
        参数:
            texts: 要嵌入的文本列表
            batch_size: 批次大小，0表示使用默认值
            normalize: 是否归一化嵌入向量（OCI可能不支持）
            raise_on_error: 是否在出错时抛出异常
            
        返回:
            EmbeddingResponse: 包含嵌入向量的响应对象
        """
        # 检查客户端状态
        if not self._is_running or self.client is None:
            await self.startup()
        
        # 确定批次大小
        effective_batch_size = batch_size if batch_size > 0 else self.batch_size
        
        # 验证输入文本
        valid_texts = []
        original_indices = []
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                original_indices.append(i)
            else:
                logger.warning(f"跳过索引 {i} 处的无效或空文本")
        
        if not valid_texts:
            logger.warning("没有提供有效的文本用于嵌入")
            return self._empty_response()

        all_embeddings = []
        processed_count = 0

        try:
            # 分批处理文本
            for i in range(0, len(valid_texts), effective_batch_size):
                batch = valid_texts[i:i + effective_batch_size]
                
                current_batch = i // effective_batch_size + 1
                total_batches = math.ceil(len(valid_texts) / effective_batch_size)
                
                logger.info(f"处理批次 {current_batch}/{total_batches}, 大小: {len(batch)}")

                # 准备嵌入请求
                embed_text_detail = oci.generative_ai_inference.models.EmbedTextDetails()
                embed_text_detail.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                    model_id=self.config.model_name
                )
                embed_text_detail.inputs = batch
                embed_text_detail.truncate = "NONE"  # 不截断文本
                embed_text_detail.compartment_id = self.config.compartment_id
                
                # 发送请求并获取响应
                embed_text_response = self.client.embed_text(embed_text_detail) # type: ignore
                
                # 处理响应
                if hasattr(embed_text_response.data, 'embeddings'): # type: ignore
                    all_embeddings.extend(embed_text_response.data.embeddings) # type: ignore
                    processed_count += len(batch)
                    logger.debug(f"批次 {current_batch} 处理成功，获得 {len(batch)} 个嵌入向量")
                else:
                    logger.warning(f"批次 {current_batch} 响应中未找到嵌入向量数据")

            # 构建响应数据
            return self._build_response(all_embeddings, original_indices, len(texts))

        except oci.exceptions.ServiceError as e:
            error_msg = f"OCI 服务错误: {e.status} - {e.message}"
            logger.error(error_msg)
            if raise_on_error:
                raise RuntimeError(error_msg) from e
            return self._empty_response()
            
        except oci.exceptions.RequestException as e:
            error_msg = f"OCI 请求错误: {str(e)}"
            logger.error(error_msg)
            if raise_on_error:
                raise RuntimeError(error_msg) from e
            return self._empty_response()
            
        except Exception as e:
            error_msg = f"OCI 嵌入处理过程中发生未知错误: {str(e)}"
            logger.exception(error_msg)
            if raise_on_error:
                raise RuntimeError(error_msg) from e
            return self._empty_response()

    def _empty_response(self) -> EmbeddingResponse:
        """返回空的响应对象。"""
        return EmbeddingResponse(
            data=[],
            model=self.config.model_name,
            object="list",
            usage={}
        )

    def _build_response(
        self, 
        embeddings: list, 
        original_indices: list[int], 
        total_texts: int
    ) -> EmbeddingResponse:
        """构建嵌入响应对象。
        
        参数:
            embeddings: 嵌入向量列表
            original_indices: 原始文本索引列表
            total_texts: 总文本数量
            
        返回:
            EmbeddingResponse: 格式化后的响应对象
        """
        # 确保嵌入向量数量与处理的文本数量匹配
        if len(embeddings) != len(original_indices):
            logger.warning(
                f"嵌入向量数量 ({len(embeddings)}) 与文本数量 ({len(original_indices)}) 不匹配"
            )
        
        # 创建嵌入数据项
        embeddings_data = []
        for idx, (original_idx, embedding) in enumerate(zip(original_indices, embeddings)):
            embeddings_data.append(
                EmbeddingDataItem(
                    embedding=embedding,
                    index=original_idx,  # 使用原始索引
                    object="embedding"
                )
            )
        
        # 记录使用情况统计
        usage_stats = {
            "prompt_tokens": len(embeddings) * 100,  # 估算值，实际需要根据API响应调整
            "total_tokens": len(embeddings) * 100,
            "processed_texts": len(embeddings),
            "total_texts": total_texts
        }

        logger.info(f"成功处理 {len(embeddings)}/{total_texts} 个文本的嵌入")

        return EmbeddingResponse(
            data=embeddings_data,
            model=self.config.model_name,
            object="list",
            usage=usage_stats
        )

    @property
    def is_running(self) -> bool:
        """检查客户端是否正在运行。"""
        return self._is_running and self.client is not None
