import json
import os
import uuid
from loguru import logger
from datetime import datetime

from .file_params import FileParams
from .txt_to_md import TxtToMarkdownParser
from .summary_parser import SummaryParser
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.repositories.kbot_md_prompt_repo import KbotMdPromptRepository
from core.dictionary import FileStatus, ProcessPriority
from utils.encoder import DecimalEncoder
from utils.parser_client import CallParser
from utils.model_client import CallModel


class FileProcessor:
    """文件处理类，负责文件解析和处理的业务逻辑"""
    def __init__(self):
        self.file_repo = KbotMdKbFilesRepository()
        self.kb_repo = KbotMdKbRepository()
        self.prompt_repo = KbotMdPromptRepository()
        self.summary_parser = SummaryParser()   
    
    async def get_pending_files(self) -> list[tuple[int, float, FileParams]]:
        """
        从数据库获取待处理的文件
        
        返回:
            包含(优先级, 时间戳, 文件参数)元组的列表
        """
        result = []

        files = await self.file_repo.get_by_status(FileStatus.APPROVED)
        
        if files is None or len(files) == 0:
            return result
            
        for file in files:
            file_params = FileParams(
                file_id=file.file_id,
                app_id=file.app_id,
                kb_id=file.kb_id,
                batch_id=file.batch_id,
                file_path=file.file_path, # type: ignore
                file_ext=file.file_ext,
                enable_summary = True if file.enable_summary == 1 else False,
                img2txt = file.is_img2txt,
                tab_head = file.is_table_head_fill,
                priority = file.process_priority or ProcessPriority.MEDIUM.value,
                security_level = file.security_level,
                parser={},
                biz_metadata={},
                img2txt_model=None,
                img_embed_model=None,
                txt_embed_model=None,
                summary_model=None,
                )
            # 检查 chunk_parser 是否已经是字典类型
            if isinstance(file.chunk_parser, dict):
                file_params.parser = file.chunk_parser
            elif file.chunk_parser is None:
                # 如果是 None，则使用空字典
                file_params.parser = {}
                logger.warning(f"文件ID {file.file_id} 的 chunk_parser 为 None，使用空字典")
            else:
                # 如果是字符串，则解析为 JSON
                file_params.parser = json.loads(file.chunk_parser, cls=DecimalEncoder) # type: ignore

            # 解析 biz_metadata 字段
            if isinstance(file.biz_metadata, dict):
                file_params.biz_metadata = file.biz_metadata
            elif file.biz_metadata is None:
                file_params.biz_metadata = {}
                logger.warning(f"文件ID {file.file_id} 的 biz_metadata 为 None，使用空字典")
            else:
                file_params.biz_metadata = json.loads(file.biz_metadata, cls=DecimalEncoder) # type: ignore
            
            logger.debug(f"文件参数: {file_params.__dict__}")

            models = await self.kb_repo.get_model_by_kbid(file.kb_id)

            logger.debug(f"模型配置: {models}")

            # 确保 models 不为空且至少有一个元素，并且第一个元素不为None
            if models:
                file_params.kb_category = models[0]
                file_params.img2txt_model = models[1]
                file_params.img_embed_model = models[2]
                file_params.txt_embed_model = models[3]
                if file_params.enable_summary:
                    file_params.summary_model = models[4]
            else:
                logger.warning(f"未找到知识库ID {file.kb_id} 对应的模型配置")
                return result

            timestamp = datetime.now().timestamp()  # 获取当前时间戳
            # 将文件状态更新为处理中
            await self.file_repo.update_file_status(file_params.file_id, FileStatus.PARSING)
            # 添加到结果列表
            result.append((file_params.priority, timestamp, file_params))
            logger.info(f"已添加文件到处理队列: {file_params.file_path} (优先级: {ProcessPriority(file_params.priority)})")
            
        return result


    async def process_file(self, file_params: FileParams):
        """
        处理文件的入口方法
        
        参数:
            file_params: 文件参数对象

        """

        try:
            logger.info(f"开始处理文件: {file_params.file_path}...")

            chunks = []

            # 生成解析参数
            kwargs = {
                "file_path": file_params.file_path,
                "in_memory": False,
                "file_content": None,
                "output_format": "chunks", # 固定为 "chunks", 切片后用于RAG
                "do_ocr": file_params.parser.get("do_ocr", False),
                "generate_picture_images": file_params.parser.get("generate_picture_images", False),
                "images_scale": file_params.parser.get("images_scale", 2.0),
                "use_vlm": False,
                "vlm_model": file_params.img2txt_model,
                "vlm_prompt": None,
                "chunk_size": file_params.parser.get("chunk_size", 512),
                "overlap": file_params.parser.get("overlap", 50),
                "min_chunk_len": file_params.parser.get("min_chunk_len", 10)
            }
            
            if file_params.file_ext == ".txt":
                # 因为docling不支持直接解析txt文件，所以先转换为md
                md_content = TxtToMarkdownParser().process(file_params.file_path)
                # 注入转换后的md内容
                kwargs["file_content"] = md_content
                kwargs["in_memory"] = True
            else:
                # 获取 VLM 提示词
                prompt_unique_name = file_params.parser.get("image_parse_prompt_unique_name", None)
                if prompt_unique_name:
                    vlm_prompt = await self._get_prompt_content(prompt_unique_name)
                    if vlm_prompt:
                        kwargs["vlm_prompt"] = vlm_prompt
                        kwargs["use_vlm"] = True
                    else:
                        logger.warning(f"提示词唯一名称 {prompt_unique_name} 不存在")
                        kwargs["use_vlm"] = False
                else:
                    kwargs["use_vlm"] = False

            # 调用 Docling 处理文件
            chunks = await CallParser().call_doc_parser_service(**kwargs)
        
            if not chunks:
                logger.error(f"文件 {file_params.file_path} 解析结果为空")
                # 更新文件状态为处理失败
                await self.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件解析结果为空")
            
            txt_chunks = None
            if isinstance(chunks, list):
                txt_chunks= await self._get_embeddings(chunks, file_params)
            else:
                logger.error(f"文件 {file_params.file_path} 解析结果不是列表")
                # 更新文件状态为处理失败
                await self.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件解析结果不是列表")

            # 保存嵌入向量
            if txt_chunks:
                await self.save_chunks(file_params.kb_id, file_params.file_id, txt_chunks)

            # 生成摘要
            if file_params.enable_summary and txt_chunks:
                logger.info(f"开始生成摘要...")
                await self.summary_parser.process_summary(file_params=file_params, embed_entities=txt_chunks)
           
                
        except Exception as e:
            msg = f"处理文件 {file_params.file_path} 时发生错误: {str(e)}"
            logger.error(msg)
            # 更新文件状态为处理失败
            await self.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        
    
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

    async def _get_embeddings(self, texts: list[str], file_params: FileParams) -> list[KbotBizTxtEmbedding] | None:
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

    async def _check_file(self, file_params: FileParams) -> bool:
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
        
    async def _get_prompt_content(self, prompt_unique_name: str) -> str | None:
        """
        根据提示词唯一名称获取提示词内容
        
        Args:
            prompt_unique_name: 提示词唯一名称
            
        Returns:
            str: 提示词内容
        """
        return await self.prompt_repo.get_prompt_by_unique_name(prompt_unique_name)

