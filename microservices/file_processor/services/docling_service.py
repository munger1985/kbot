"""文档解析服务层。

该模块负责业务逻辑组装，包括文件验证、参数预处理，以及调用底层的 DoclingDocProcessor 解析器。
"""

import os
import asyncio
from loguru import logger
from concurrent.futures import ProcessPoolExecutor
from core.config.settings import get_parser_config
from docling.datamodel.base_models import InputFormat
from ..parsers.engine import DoclingEngine, OutputFormat
from ..parser_schema import DocParserParams, ChunkResult


class ParserService:
    """文档解析服务类。
    
    协调文件格式检查和解析引擎调用的核心服务类，对外提供统一的文档解析接口。
    """

    # 1. 输入格式验证字典：映射文件扩展名到 Docling 支持的输入格式
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

    # 2. 输出配置映射字典（键：用户输入字符串，值：对应的枚举值）
    OUTPUT_CONFIG_MAP = {
        "markdown": OutputFormat.MARKDOWN,
        "html": OutputFormat.HTML,
        "json": OutputFormat.JSON,
        "doctags": OutputFormat.DOCTAGS,
        "chunks": OutputFormat.CHUNKS,  # 内部处理器支持的 CHUNKS 枚举值
    }

    def __init__(self):
        """初始化解析服务。
        
        加载解析配置，初始化 Docling 文档处理器实例，配置工件存储路径和并发数。
        """
        # 处理器现在是无状态的（包含VLM配置），仅需初始化分词器相关配置
        config = get_parser_config()
        # max_workers 建议设为核心数的一半，避免干扰主线程 IO
        self.executor = ProcessPoolExecutor(max_workers=config.queue_workers)
        # 将进程池注入 Engine
        self.processor = DoclingEngine(
            artifacts_path=config.local_artifacts_path,
            pool_executor=self.executor
        )

    async def parse_file(
        self, 
        file_id: str,
        file_path: str, 
        parser_params: DocParserParams,  
        output_format: str = "markdown"
    ) -> str | dict | list[ChunkResult]:
        """执行文件解析任务。

        对外提供的核心解析接口，包含文件校验、格式转换、解析调用全流程。

        Args:
            file_id: 待解析文件的 ID
            file_path: 待解析文件的完整路径
            parser_params: 解析参数对象，包含文件路径、VLM配置、输出格式等信息
            output_format: 期望的输出格式，默认为 "markdown"

        Returns:
            解析后的文本内容（字符串）或分块列表（字典列表），具体取决于输出格式

        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValueError: 传入不支持的输入/输出格式时抛出
            Exception: 解析过程中发生的其他异常会被捕获并重新抛出
        """
        req_format = output_format.lower()  # 统一转为小写，避免格式大小写问题

        # 1. 物理文件存在性验证
        if not os.path.exists(file_path):
            error_msg = f"文件不存在：{file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 2. 输入文件扩展名验证
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS_MAP:
            error_msg = f"Docling 不支持该输入格式：{ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 3. 输出格式配置查找
        if req_format not in self.OUTPUT_CONFIG_MAP:
            valid_fmts = list(self.OUTPUT_CONFIG_MAP.keys())
            raise ValueError(f"不支持的输出格式 '{req_format}'，有效选项：{valid_fmts}")

        target_fmt = self.OUTPUT_CONFIG_MAP[req_format]

        # 4. 调用处理器接口，透传 VLM 相关参数
        try:
            logger.info(
                f"开始解析任务 | 文件：{os.path.basename(file_path)} | "
                f"格式：{req_format} | VLM模型：{parser_params.vlm_model or '未启用'}"
            )
            
            # 直接调用更新后的接口，传入模型名称和提示词
            return await self.processor.convert_document(
                file_id,
                file_path, 
                parser_params, 
                output_format=target_fmt
            )
            
        except Exception as e:
            logger.exception(f"文件 [{os.path.basename(file_path)}] 解析过程中发生异常")
            raise e
        
    def shutdown(self):
        """服务关闭时清理进程池，防止僵尸进程"""
        if self.executor:
            logger.info("正在关闭解析服务子进程池...")
            self.executor.shutdown(wait=True)