import os
from loguru import logger
import json
from services.dataparse.parser_common import ParserCommonMethods
from services.dataparse.file_params import FileParams
from services.dataparse.txt_to_md import TxtToMarkdownParser
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from core.config.settings import get_prompt_config


from core.exceptions import ResourceNotFoundException, DatabaseException, ValidationException, InternalServerError
from utils.parser_client import CallParser



class FileTransformService:
    """文件转换服务类，负责文件解析和转换的业务逻辑"""
    def __init__(self):
        self.file_repo = KbotMdKbFilesRepository()
        self.kb_repo = KbotMdKbRepository()
        self.common = ParserCommonMethods()
    

    async def _get_file_params(self, file_id: str) -> tuple[str, dict] | None:
        """
        从数据库获取待处理的文件参数
        
        返回:
            文件路径和解析参数 tuple[str, dict]
        """
        try:
            file = await self.file_repo.get_by_id(file_id=file_id)
        except ResourceNotFoundException as e:
            logger.warning(e.detail)
            return None
        except DatabaseException as e:
            logger.error(f"数据库查询错误: {str(e)}")
            return None
        except Exception as e:
            logger.exception(f"获取待处理文件失败: {str(e)}")
            return None
        
        if not file:
            logger.warning(f"未找到文件 {file_id}")
            return None
        
        if not file.chunk_parser:
            msg = f"文件 {file.file_name} 的解析器参数为空，跳过处理"
            logger.warning(msg)
            return None
        
        if not file.file_path:
            msg = f"文件 {file.file_name} 的文件路径为空，跳过处理"
            logger.warning(msg)
            return None

        parser_params = None
        try:
            parser_params = json.loads(file.chunk_parser)
        except json.JSONDecodeError as e:
            logger.error(f"文件 {file.file_name} 的解析器参数 JSON 解析错误: {str(e)}")
            return None

        if not parser_params:
            msg = f"文件 {file.file_name} 的解析器参数为空，跳过处理"
            logger.warning(msg)
            return None

        return file.file_path, parser_params


    async def transform_file(self, file_id: str, override_existing: bool = False, output_format: str = "markdown"):
        """
        转换文件为指定格式
        
        参数:
            file_id: 文件ID
            override_existing: 是否覆盖已存在文件，默认 False
            output_format: 输出格式，默认 markdown
        """

        result = await self._get_file_params(file_id=file_id)
        if not result:
            raise ResourceNotFoundException(f"未找到文件 {file_id} 的文件路径或解析器参数", resource_type="KB_FILE", resource_id=file_id)
        file_path, parser_params = result

        # 获取目录路径
        directory = os.path.dirname(file_path)

        # 构建新的文件路径：目录 + file_id + .输出格式后缀
        new_file = os.path.join(directory, f"{file_id}.{output_format}")

        # 检查新文件是否已存在
        if os.path.exists(new_file):
            if override_existing:
                logger.info(f"文件 {new_file} 已存在，进行覆盖")
            else:
                logger.info(f"文件 {new_file} 已存在，跳过转换直接返回")
                return new_file
        
        # 如果新文件不存在，继续转换
        try:
            logger.info(f"开始转换文件: {file_path}...")

            # 生成解析参数
            kwargs = {
                "file_path": file_path,
                "in_memory": False,
                "file_content": None,
                "output_format": output_format,
                "do_ocr": parser_params.get("do_ocr", False),
                "ocr_engine": parser_params.get("ocr_engine", None),
                "generate_picture_images": parser_params.get("generate_picture_images", False),
                "images_scale": parser_params.get("images_scale", 2.0),
                "use_vlm": parser_params.get("use_vlm", False),
                "vlm_model": parser_params.get("vlm_model", None),
                "vlm_prompt": None,
                "chunk_size": parser_params.get("chunk_size", 512),
                "overlap": parser_params.get("overlap", 50),
                "min_chunk_len": parser_params.get("min_chunk_len", 10)
            }

            if file_path.endswith(".txt"):
                # 因为docling不支持直接解析txt文件，所以先转换为md
                md_content = TxtToMarkdownParser().process(file_path)
                # 注入转换后的md内容
                kwargs["file_content"] = md_content
                kwargs["in_memory"] = True
            

            chunks = []

            # 获取 VLM 提示词
            prompt_unique_name = parser_params.get("image_parse_prompt_unique_name", None)
            if not prompt_unique_name:
                logger.warning(f"文件 {file_id} 的解析器参数中未指定 image_parse_prompt_unique_name，跳过转换")
                prompt_unique_name = get_prompt_config().image2text
            vlm_prompt = await self.common.get_prompt_content(prompt_unique_name)
            if vlm_prompt:
                kwargs["vlm_prompt"] = vlm_prompt
            else:
                logger.warning(f"文件 {file_id} 的解析器参数中指定的 image_parse_prompt_unique_name {prompt_unique_name} 未找到，跳过转换")
                kwargs["vlm_prompt"] = "请对图片中的内容进行详细描述"
            
            # 调用 Docling 处理文件
            md = await CallParser().call_doc_parser_service(**kwargs)
        
            if not md:
                logger.error(f"文件 {file_path} 转换markdown格式结果为空")
                # 更新文件状态为处理失败
                return None
            
            if not isinstance(md, str):
                logger.error(f"文件 {file_path} 转换markdown格式结果不是字符串")
                return None
            
            # 写入新文件
            with open(new_file, "w", encoding="utf-8") as f:
                f.write(md)

            logger.info(f"文件 {file_path} 转换{output_format}格式成功，保存为 {new_file}")
            return new_file
                        
        except Exception as e:
            msg = f"处理文件 {file_path} 时发生错误: {str(e)}"
            logger.error(msg)
            return None
        