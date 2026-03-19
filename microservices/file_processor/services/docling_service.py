"""Document parsing service layer.

This module is responsible for business logic assembly, including file validation, 
parameter preprocessing, and invoking the underlying DoclingDocProcessor.
"""

import os
from loguru import logger

from core.config.settings import get_parser_config
from docling.datamodel.base_models import InputFormat
from ..parsers.docling_parser import DoclingDocProcessor, OutputFormat
from ..parser_schema import DocParserParams


class ParserService:
    """Document parsing service class.
    
    Coordinates file format checking and parsing engine invocation.
    """

    # 1. Input format validation dictionary
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

    # 2. Output configuration mapping dictionary (Key: user input string, Value: corresponding enum)
    OUTPUT_CONFIG_MAP = {
        "markdown": OutputFormat.MARKDOWN,
        "html": OutputFormat.HTML,
        "json": OutputFormat.JSON,
        "doctags": OutputFormat.DOCTAGS,
        "chunks": OutputFormat.CHUNKS,  # CHUNKS enum supported by internal Processor
    }

    def __init__(self):
        """Initialize parsing service."""
        # Processor is now stateless with VLM configuration, only tokenizer initialization needed
        config = get_parser_config()
        self.processor = DoclingDocProcessor(
            local_artifacts_path=config.local_artifacts_path,
            max_workers=config.queue_workers,
        )

    async def parse_file(
        self, 
        file_path: str, 
        parser_params: DocParserParams,  
        output_format: str = "markdown"
    ) -> str | dict | list[dict]:
        """Execute file parsing task.

        Args:
            file_path: Path to the file to be parsed.
            parser_params: Parameter object containing file path, VLM configuration, and output format.
            output_format: Desired output format, default is "markdown".

        Returns:
            Parsed text content or chunk list.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If unsupported input/output format is provided.
        """
        req_format = output_format.lower()

        # 1. Physical existence validation
        if not os.path.exists(file_path):
            error_msg = f"File not found: {file_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        # 2. Input file extension validation
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS_MAP:
            error_msg = f"Docling does not support input format: {ext}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # 3. Output format configuration lookup
        if req_format not in self.OUTPUT_CONFIG_MAP:
            valid_fmts = list(self.OUTPUT_CONFIG_MAP.keys())
            raise ValueError(f"Unsupported output format '{req_format}'. Valid options: {valid_fmts}")

        target_fmt = self.OUTPUT_CONFIG_MAP[req_format]

        # 4. Call Processor interface, pass through VLM-related parameters
        try:
            logger.info(
                f"Starting parsing task | File: {os.path.basename(file_path)} | "
                f"Mode: {req_format} | VLM: {parser_params.vlm_model or 'Disabled'}"
            )
            
            # Directly call updated interface with model name and prompt
            return await self.processor.convert_document(
                file_path, 
                parser_params, 
                output_format=target_fmt
            )
            
        except Exception as e:
            logger.exception(f"Exception occurred during parsing of file [{os.path.basename(file_path)}]")
            raise e