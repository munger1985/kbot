import json
from loguru import logger
from datetime import datetime
from .txt_parser import process_txt
from .pdf_parser import process_pdf
from .file_params import FileParams
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from dao.data_dict import FileStatus, ProcessPriority
from core.config import settings
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
            file_params.summary = file.enable_summary
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
                logger.warning(f"chunk_parser is None for file_id: {file.file_id}, using empty dict")
            else:
                # 如果是字符串，则解析为 JSON
                file_params.parser = json.loads(file.chunk_parser, cls=DecimalEncoder) # type: ignore
            
            logger.debug(f"File params: {file_params.__dict__}")

            models = await kb_repo.get_model_by_kbid(file.kb_id)

            logger.debug(f"Models: {models}")

            # 确保 models 不为空且至少有一个元素，并且第一个元素不为None
            if models:
                file_params.kb_category = models[0]
                file_params.img2txt_model = models[1]
                file_params.img_embed_model = models[2]
                file_params.txt_embed_model = models[3]
            else:
                logger.warning(f"No models found for kb_id: {file.kb_id}")
                return result

            timestamp = datetime.now().timestamp()  # 获取当前时间戳
            # 将文件状态更新为处理中
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSING)
            # 添加到结果列表
            result.append((file_params.priority, timestamp, file_params))
            logger.info(f"Added file to process list: {file_params.file_path} (priority: {ProcessPriority(file_params.priority)})")
            
        return result

    @staticmethod
    async def process_file(file_params: FileParams) -> bool:
        """
        处理文件的入口方法
        
        参数:
            file_params: 文件参数
            
        返回:
            处理是否成功
        """
        try:
            logger.info(f"Processing {file_params.file_path}...")

            # 处理文本文件
            if file_params.file_ext == ".txt":
                logger.info(f"Processing text file {file_params.file_path}...")
                return await process_txt(file_params)
            if file_params.file_ext == ".pdf":
                return await process_pdf(file_params)
            else:
                logger.info(f"File {file_params.file_path} is not a text file, skipping...")
                # 更新文件状态为已处理
                file_repo = KbotMdKbFilesRepository()
                await file_repo.update_file_status(file_params.file_id, FileStatus.PARSED)
                return True
                
        except Exception as e:
            logger.error(f"Error processing {file_params.file_path}: {str(e)}")
            # 更新文件状态为处理失败
            file_repo = KbotMdKbFilesRepository()
            await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED)
            return False

    