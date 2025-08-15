import shutil
from loguru import logger
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_repo import KbotBizTxtEmbeddingRepository
from dao.data_dict import FileStatus

class KBProcedure():
    """
    知识库流程管理类
    """
    def __init__(self):
        pass

    async def reparse_files(self, kb_id: int, files: list[str]) -> bool:
        """
        将KB中的文件标记为未解析，触发重新解析

        Args:
            files (list[str]): 文件ID列表
            
        Returns:
            bool: 操作是否成功
        """
        file_repo = KbotMdKbFilesRepository()
        chunk_repo = KbotBizTxtEmbeddingRepository()
        # 1. 删除文件对应的chunk数据
        try:
            await chunk_repo.delete_by_file_ids(kb_id=kb_id, file_ids=files)
            logger.debug(f"Deleted chunks for files: {files} of KB {kb_id}.")
            # 2. 删除文件对应的解析图片和表格数据（部分PDF，PPT等文件有）
            # todo: ...
            #logger.debug(f"Deleted images and tables for files: {files} of KB {kb_id}.")
            # 3. 重置文件状态为未解析
            await file_repo.batch_update_file_status(file_ids=files, status=FileStatus.APPROVED, log_msg="reparse")
            logger.info(f"Files {files} are marked for reprocessing.")
            return True
        except Exception as e:
            logger.exception(e)
            return False
