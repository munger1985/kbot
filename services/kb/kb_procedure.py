from loguru import logger
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding_factory import EmbeddingRepositoryFactory
from core.dictionary import FileStatus

class KBProcedure():
    """
    知识库流程管理类
    """
    def __init__(self):
        pass

    async def reparse_files(self, kb_id: int, file_ids: list[str]) -> bool:
        """
        重新解析知识库中的指定文件
        
        将指定文件标记为未解析状态，触发重新解析流程
        
        Args:
            kb_id (int): 知识库ID
            file_ids (list[str]): 需要重新解析的文件ID列表
            
        Returns:
            bool: 操作是否成功。True表示成功，False表示失败
        """
        file_repo = KbotMdKbFilesRepository()
        chunk_repo = await EmbeddingRepositoryFactory.create_repository(kb_id=kb_id)

        try:
            # 1. 删除文件对应的文本片段数据
            await chunk_repo.delete_by_file_ids(kb_id=kb_id, file_ids=file_ids)
            logger.debug(f"已删除知识库 {kb_id} 中文件 {file_ids} 对应的文本片段数据")
            
            # 2. 删除文件对应的解析图片和表格数据（部分PDF，PPT等文件有）
            # todo: 待实现图片和表格数据的删除逻辑
            # logger.debug(f"已删除知识库 {kb_id} 中文件 {file_ids} 对应的图片和表格数据")
            
            # 3. 重置文件状态为未解析
            await file_repo.batch_update_file_status(
                file_ids=file_ids, 
                status=FileStatus.APPROVED, 
                log_msg="重新解析文件"
            )
            logger.info(f"文件 {file_ids} 已标记为待重新解析状态")
            return True
            
        except Exception as e:
            logger.exception(f"重新解析文件失败: {e}")
            return False