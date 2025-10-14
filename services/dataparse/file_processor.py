import json
from loguru import logger
from datetime import datetime

from .csv_parser_dataframe import process_csv
from .excel_parser_openpyxl import process_excel
from .markdown_parser_bs4 import process_markdown
from .txt_parser import process_txt
from .img_parser import process_img
from .pdf_parser_pdfplumber import process_pdf
from .office_parser import process_word_ppt_by_converter
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from core.dictionary import FileStatus, ProcessPriority
from utils.decimal_encoder import DecimalEncoder


class FileProcessor:
    """文件处理类，负责文件解析和处理的业务逻辑"""
    
    @staticmethod
    async def get_pending_files() -> list[tuple[int, float, FileParams]]:
        """
        从数据库获取待处理的文件
        
        返回:
            包含(优先级, 时间戳, 文件参数)元组的列表
        """
        result = []
        kb_repo = KbotMdKbRepository()       
        file_repo = KbotMdKbFilesRepository()
        files = await file_repo.get_by_status(FileStatus.APPROVED)
        
        if files is None or len(files) == 0:
            return result
            
        for file in files:
            file_params = FileParams()
            file_params.file_id = file.file_id
            file_params.app_id = file.app_id
            file_params.kb_id = file.kb_id
            file_params.batch_id = file.batch_id
            file_params.file_path = file.file_path # type: ignore
            file_params.file_ext = file.file_ext
            file_params.enable_summary = True if file.enable_summary == 1 else False
            file_params.img2txt = file.is_img2txt
            file_params.tab_head = file.is_table_head_fill
            file_params.priority = file.process_priority or ProcessPriority.MEDIUM.value
            file_params.security_level = file.security_level
            
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

            models = await kb_repo.get_model_by_kbid(file.kb_id)

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
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSING)
            # 添加到结果列表
            result.append((file_params.priority, timestamp, file_params))
            logger.info(f"已添加文件到处理队列: {file_params.file_path} (优先级: {ProcessPriority(file_params.priority)})")
            
        return result

    @staticmethod
    async def process_file(file_params: FileParams) -> bool:
        """
        处理文件的入口方法
        
        参数:
            file_params: 文件参数对象
            
        返回:
            处理是否成功
        """
        file_repo = KbotMdKbFilesRepository()

        try:
            logger.info(f"开始处理文件: {file_params.file_path}...")

            # 处理文本文件
            if file_params.file_ext == ".txt":
                logger.info(f"处理文本文件: {file_params.file_path}...")
                return await process_txt(file_params)
            elif file_params.file_ext == ".md":
                logger.info(f"处理Markdown文件: {file_params.file_path}...")
                return await process_markdown(file_params)
            elif file_params.file_ext == ".csv":
                logger.info(f"处理  CSV 文件: {file_params.file_path}...")
                return await process_csv(file_params)
            elif file_params.file_ext == ".pdf":
                logger.info(f"处理PDF文件: {file_params.file_path}...")
                return await process_pdf(file_params)
            elif file_params.file_ext == ".xlsx":
                logger.info(f"处理Excel文件: {file_params.file_path}...")
                return await process_excel(file_params)
            elif file_params.file_ext in [".doc", ".docx", ".pptx", ".ppt"]:
                logger.info(f"处理Word/PPT文件: {file_params.file_path}...")
                return await process_word_ppt_by_converter(file_params)
            elif file_params.file_ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"]:
                logger.info(f"处理图片文件: {file_params.file_path}...")
                return await process_img(file_params)
            else:
                msg = f"不支持的文件类型 {file_params.file_ext}，跳过处理..."
                logger.info(msg)
                # 更新文件状态为已处理
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED, msg)
                return True
                
        except Exception as e:
            msg = f"处理文件 {file_params.file_path} 时发生错误: {str(e)}"
            logger.error(msg)
            # 更新文件状态为处理失败
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False