import json
from loguru import logger
from datetime import datetime

from .file_params import FileParams
from .txt_to_md import TxtToMarkdownParser
from .summary_parser import SummaryParser
from .parser_common import ParserCommonMethods
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository

from core.dictionary import FileStatus, ProcessPriority
from utils.encoder import DecimalEncoder
from utils.parser_client import CallParser
from api.schemas.parser_schema import ParserParams



class FileProcessor:
    """文件处理类，负责文件解析和处理的业务逻辑"""
    def __init__(self):
        self.file_repo = KbotMdKbFilesRepository()
        self.kb_repo = KbotMdKbRepository()
        self.common = ParserCommonMethods()
    
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
                parser=None,
                biz_metadata=None,
                img2txt_model=None,
                img_embed_model=None,
                txt_embed_model=None,
                summary_model=None,
                )
            # 解析 chunk_parser 字段
            if file.chunk_parser:
                if isinstance(file.chunk_parser, str):
                    parser_dict = json.loads(file.chunk_parser, cls=DecimalEncoder) # type: ignore
                    parser_params = ParserParams(**parser_dict)
                    file_params.parser = parser_params
                
                elif isinstance(file.chunk_parser, dict):
                    parser_dict = file.chunk_parser
                    parser_params = ParserParams(**parser_dict)
                    file_params.parser = parser_params
                else:
                    logger.warning(f"文件ID {file.file_id} 的 chunk_parser 格式未知，使用默认值") 
            else:
                msg = f"文件ID {file.file_id} 的 chunk_parser 为空，跳过解析"
                logger.warning(msg)
                await self.common.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
                return []

            # 解析 biz_metadata 字段
            if file.biz_metadata:
                if isinstance(file.biz_metadata, str):
                    file_params.biz_metadata = json.loads(file.biz_metadata, cls=DecimalEncoder) # type: ignore
                elif isinstance(file.biz_metadata, dict):
                    file_params.biz_metadata = file.biz_metadata
                else:
                    logger.warning(f"文件ID {file.file_id} 的 biz_metadata 格式未知，使用空字典")
            else:
                logger.warning(f"文件ID {file.file_id} 的 biz_metadata 为 None，使用空字典")
            

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
            await self.common.update_file_status(file_params.file_id, FileStatus.PARSING, "文件解析中")
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
        if not await self.common.check_file(file_params):
            return

        try:
            logger.info(f"开始处理文件: {file_params.file_path}...")
            chunks = []

            # 调用 Docling 处理文件
            result = await CallParser().call_doc_parser_service(
                file_path=file_params.file_path,
                parser_params=file_params.parser,
                output_format="chunks") # 期望返回 chunks 格式
        
            if isinstance(result, list):
                embeddings = await self.common.get_embeddings(result, file_params)
                if embeddings is None:
                    logger.error(f"获取文件 {file_params.file_path} 的 embedding 向量失败")
                    # 更新文件状态为处理失败
                    await self.common.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "获取 embedding 向量失败")
                    return
                else:
                    # 保存嵌入向量
                    await self.common.save_chunks(file_params.kb_id, file_params.file_id, embeddings)
            else:
                logger.error(f"文件 {file_params.file_path} 解析结果不是期望的列表格式")
                # 更新文件状态为处理失败
                await self.common.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件解析结果不是期望的列表格式")
                return

            # 生成摘要
            if file_params.enable_summary and embeddings:
                logger.info(f"开始生成摘要...")
                summary_parser = SummaryParser()
                await summary_parser.process_summary(file_params=file_params, embed_entities=embeddings)
                
        except Exception as e:
            msg = f"处理文件 {file_params.file_path} 时发生错误: {str(e)}"
            logger.error(msg)
            # 更新文件状态为处理失败
            await self.common.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        
    
    

    

