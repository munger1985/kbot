from loguru import logger
from .file_params import FileParams
from .pdf_parser_pdfplumber import process_pdf
from .common import check_text_file
from utils.file_converter import OfficeToPDF


async def process_word_ppt_by_converter(file_params: FileParams) -> bool:
    """
    处理Word/PPT文件，将其转换为PDF文件后，调用PDF解析器处理
    
    参数:
        file_params: 文件参数对象
        
    返回:
        bool: 文件处理是否成功
    """
    
    if not await check_text_file(file_params):
        return False
    
    try:
        logger.debug(f"正在处理Word/PPT文件: {file_params.file_path}")

        converter = OfficeToPDF()
        input_path = file_params.file_path
        output_path = await converter.convert_to_pdf(input_path)
        if output_path is None:
            logger.error(f"Word/PPT文件转换失败: {file_params.file_path}")
            return False
        
        file_params.file_path = output_path

        return await process_pdf(file_params)
        
    except Exception as e:
        logger.exception(f"处理Word/PPT文件时发生错误: {file_params.file_path}, 错误信息: {str(e)}")
        return False