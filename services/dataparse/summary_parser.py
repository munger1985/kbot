from loguru import logger
from .file_params import FileParams
from core.dictionary import FileStatus
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository

async def process_summary(chunk: str, file_params: FileParams) -> bool:


    file_repo = KbotMdKbFilesRepository()

    if file_params.summary_model is None:
        msg = f"摘要总结模型不存在，无法进行摘要总结，file_id: {file_params.file_id}"
        logger.error(msg)
        await file_repo.update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False
    
    # TODO: 调用模型进行摘要总结
    # TODO: 将摘要结果写入数据库
    # TODO: 更新文件状态为已处理
    # TODO: 更新文件状态为处理失败

    return True