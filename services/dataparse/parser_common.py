"""
文件解析通用方法类
"""
import os
import uuid
from loguru import logger
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import FileStatus
from utils.model_client import CallModel


class ParserCommonMethods:

    def __init__(self):
        self.file_repo = KbotMdKbFilesRepository()
        self.prompt_repo = KbotMdPromptRepository()

    async def save_chunks(self, kb_id: int, file_id: str, chunks: list[KbotBizTxtEmbedding]):
        """
        保存嵌入向量到数据库（包含错误处理）

        Args:
            kb_id: 知识库ID
            file_id: 文件ID
            chunks: 文本片段列表
            dims: 嵌入向量维度

        Returns:
            bool: 保存成功返回True，失败返回False
        """
        try:
            # 创建嵌入向量仓库对象
            repo = await EmbeddingRepositoryFactory.create_repository(kb_id)
            if not repo:
                raise

            await repo.create(kb_id=kb_id, embeddings=chunks)

            await self.update_file_status(file_id, FileStatus.PARSED, f"成功保存 {len(chunks)} 个嵌入向量")

        except Exception as e:
            msg = f"保存嵌入向量时发生异常: {str(e)}"
            logger.error(msg)
            await self.update_file_status(file_id, FileStatus.PARSE_FAILED, msg)
        
    async def update_file_status(self, file_id: str, status: FileStatus, message: str) -> None:
            """
            更新文件状态辅助方法

            Args:
                file_id: 文件ID
                status: 文件状态
                message: 状态信息
            """
            await self.file_repo.update_file_status(
                file_id=file_id,
                status=status,
                log_msg=message
            )

    async def get_embeddings(self, texts: list[str], file_params: FileParams) -> list[KbotBizTxtEmbedding] | None:
        """
        获取文本的嵌入向量
        
        Args:
            texts: 文本列表
            file_params: 文件参数对象
            
        Returns:
            list[KbotBizTxtEmbedding]: 每个文本的嵌入向量列表
        """
        model = file_params.txt_embed_model
        if not model:
            logger.error(f"知识库 {file_params.kb_id} 未配置文本嵌入模型")
            return None
        
        try:
            batch_size = len(texts) if len(texts) <= 8 else 8
            response = await CallModel().call_embedding_model(model, texts, batch_size)
            chunks = []
            if response:
                chunk_num = 0
                for embedding, text in zip(response, texts):
                    chunk = KbotBizTxtEmbedding(
                        embed_id=str(uuid.uuid4()),
                        chunk_metadata={"chunk_num": chunk_num},
                        biz_metadata=file_params.biz_metadata,
                        security_level=file_params.security_level,
                        kb_id=file_params.kb_id,
                        file_id=file_params.file_id,
                        chunk_doc=text,
                        embedding=embedding.embedding
                    )
                    chunk_num += 1
                    chunks.append(chunk)
            
            return chunks
        
        except Exception as e:
            logger.error(f"获取嵌入向量时发生异常: {str(e)}")
            return None

    

    async def check_file(self, file_params: FileParams) -> bool:
        """
        检查文件嵌入模型和文件存在性
        
        Args:
            file_params: 文件参数对象
        """
        try:
            # 检查文本嵌入模型是否指定
            if file_params.txt_embed_model is None:
                msg = f"知识库 {file_params.kb_id} 未指定文本嵌入模型"
                logger.error(msg)
                # 更新文件状态为处理失败
                await self.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return False

            # 检查文件是否存在
            if not os.path.exists(file_params.file_path):
                logger.error(f"文件路径不存在: {file_params.file_path}")
                await self.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件路径不存在")
                return False
            
            return True
                
        except Exception as e:
            msg = f"处理文本文件 {file_params.file_id} 时发生错误: {str(e)}"
            logger.error(msg)
            await self.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        
    async def get_prompt_content(self, prompt_unique_name: str) -> str | None:
        """
        根据提示词唯一名称获取提示词内容
        
        Args:
            prompt_unique_name: 提示词唯一名称
            
        Returns:
            str: 提示词内容
        """
        return await self.prompt_repo.get_prompt_by_unique_name(prompt_unique_name)