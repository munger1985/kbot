from loguru import logger
from .file_params import FileParams
from .pdf_converter import OfficeToPDFConverter
from .pdf_parser_pdfplumber import process_pdf
from utils.common_methods import check_text_file


async def process_word_ppt_by_converter(file_params: FileParams) -> bool:
    """
    处理word/PPT文件，将其转换为pdf文件后，调用pdf_parser.py处理
    
    参数:
        file_params: 文件参数类
        
    返回:
        是否成功处理文件
    """
    
    if not await check_text_file(file_params):
        return False
    
    try:
        logger.debug(f"Processing word/ppt file: {file_params.file_path}")

        converter = OfficeToPDFConverter()
        input_path = file_params.file_path
        output_path = await converter.convert_to_pdf(input_path)
        file_params.file_path = output_path

        return await process_pdf(file_params)
        
    except Exception as e:
        logger.exception(f"Error in processing word/ppt file: {file_params.file_path}, {str(e)}")
        return False

        
        
        