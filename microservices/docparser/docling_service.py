"""文档解析服务层。

本模块负责业务逻辑组装，包括文件校验、参数预处理以及调用底层的 DoclingDocProcessor。
"""

import os
from loguru import logger

from core.config.settings import get_parser_config
from docling.datamodel.base_models import InputFormat
from .parsers.docling_parser import DoclingDocProcessor, OutputFormat
from .parser_schema import ParserParams


class ParserService:
    """文档解析服务类。
    
    协调文件格式检查与解析引擎调用。
    """

    # 1. 输入格式校验字典
    SUPPORTED_EXTENSIONS_MAP = {
        ".pdf": InputFormat.PDF,
        ".docx": InputFormat.DOCX,
        ".pptx": InputFormat.PPTX,
        ".xlsx": InputFormat.XLSX,
        ".html": InputFormat.HTML,
        ".xhtml": InputFormat.HTML,
        ".md": InputFormat.MD,
        ".markdown": InputFormat.MD,
        ".asciidoc": InputFormat.ASCIIDOC,
        ".csv": InputFormat.CSV,
        ".png": InputFormat.IMAGE,
        ".jpg": InputFormat.IMAGE,
        ".jpeg": InputFormat.IMAGE,
        ".vtt": InputFormat.VTT,
    }

    # 2. 输出配置映射字典 (Key: 用户传入字符串, Value: 对应枚举)
    OUTPUT_CONFIG_MAP = {
        "markdown": OutputFormat.MARKDOWN,
        "html": OutputFormat.HTML,
        "json": OutputFormat.JSON,
        "doctags": OutputFormat.DOCTAGS,
        "chunks": OutputFormat.CHUNKS,  # 内部 Processor 已支持 CHUNKS 枚举
    }

    def __init__(self):
        """初始化服务。

        Args:
            en_model_path: 英文 Tokenizer 路径。
            zh_model_path: 中文 Tokenizer 路径。
        """
        # Processor 现在是无状态的 VLM 配置，只需初始化分词器
        config = get_parser_config()
        self.processor = DoclingDocProcessor(
            en_tokenizer_path=config.tokenizer.en,
            zh_tokenizer_path=config.tokenizer.zh,
            local_artifacts_path=config.local_artifacts_path,
            max_workers=config.max_workers,
        )

    async def parse_file(self, parser_params: ParserParams) -> str | dict | list[dict]:
        """执行文件解析任务。

        Args:
            parser_params: 包含文件路径、VLM 配置及输出格式的参数对象。

        Returns:
            解析后的文本内容或切片列表。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 不支持的输入或输出格式。
        """
        file_path = parser_params.file_path
        req_format = parser_params.output_format.lower()

        # 1. 物理存在校验
        if not os.path.exists(file_path):
            error_msg = f"未找到文件: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 2. 输入后缀校验
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS_MAP:
            error_msg = f"Docling 不支持输入格式: {ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 3. 输出格式配置检索
        if req_format not in self.OUTPUT_CONFIG_MAP:
            valid_fmts = list(self.OUTPUT_CONFIG_MAP.keys())
            raise ValueError(f"不支持的输出格式 '{req_format}'。可选范围: {valid_fmts}")

        target_fmt = self.OUTPUT_CONFIG_MAP[req_format]

        # 4. 调用 Processor 接口，透传 VLM 相关参数
        try:
            logger.info(
                f"开始解析任务 | 文件: {os.path.basename(file_path)} | "
                f"模式: {req_format} | VLM: {parser_params.vlm_model or '禁用'}"
            )
            
            # 直接调用更新后的接口，传入模型名称和提示词
            return await self.processor.convert_document(parser_params)
            
        except Exception as e:
            logger.exception(f"文件 [{os.path.basename(file_path)}] 解析过程中发生异常")
            raise e
  