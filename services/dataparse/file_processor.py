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
                "ocr_engine": file_params.parser.get("ocr_engine", "easyocr"),
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
                    vlm_prompt = await self.common.get_prompt_content(prompt_unique_name)
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
                await self.common.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件解析结果为空")
            
            txt_chunks = None
            if isinstance(chunks, list):
                txt_chunks= await self.common.get_embeddings(chunks, file_params)
            else:
                logger.error(f"文件 {file_params.file_path} 解析结果不是列表")
                # 更新文件状态为处理失败
                await self.common.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, "文件解析结果不是列表")

            # 保存嵌入向量
            if txt_chunks:
                await self.common.save_chunks(file_params.kb_id, file_params.file_id, txt_chunks)

            # 生成摘要
            if file_params.enable_summary and txt_chunks:
                logger.info(f"开始生成摘要...")
                summary_parser = SummaryParser()
                await summary_parser.process_summary(file_params=file_params, embed_entities=txt_chunks)
           
                
        except Exception as e:
            msg = f"处理文件 {file_params.file_path} 时发生错误: {str(e)}"
            logger.error(msg)
            # 更新文件状态为处理失败
            await self.common.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        
    
    

    

