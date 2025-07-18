import os
import shutil
from pathlib import Path
from typing import List, Optional
from loguru import logger
from core.config import settings
from dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from dao.repositories.kbot_biz_txt_embedding import KbotBizTxtEmbeddingRepository
  

async def delete_files(domain_id: int, 
                       kb_id: Optional[int], 
                       batch_name: Optional[str],
                       file_paths: Optional[List[str]]) -> tuple[int, int]:
    '''
    Delete files from disk by file IDs or batch ID or knowledge base ID.
    根据文件ID或批次ID或知识库ID从磁盘删除文件
    
    Args/参数:
        domain_id: ID of the domain where the files reside
        文件所在的业务域ID
        kb_id: ID of the knowledge base where the files reside (optional)
        文件所在的知识库ID(可选)
        batch_name: Name of the batch where the files reside (optional)
        文件所在的批次名称(可选)
        file_paths: List of file paths to delete (optional)
        要删除的文件路径列表(可选)
    
    Returns/返回:
        result: The number of successfully deleted files and the number of failed files
        包含成功删除的文件数和失败文件数的结果对象
    '''
    success_cnt = 0
    failed_cnt = 0
    # 模式1: 通过文件ID删除
    if file_paths is not None:
        for file in file_paths:
            logger.info("Deleting file: {}", file)
            if os.path.exists(Path(file)):
                try:
                    os.remove(Path(file))
                    logger.info("Successfully deleted file: {}", file)
                    success_cnt += 1
                except Exception as e:
                    logger.error(f"Failed to delete file {file}: {str(e)}")
                    failed_cnt += 1
            else:
                logger.error(f"File {file} does not exist")
                failed_cnt += 1
        return success_cnt, failed_cnt
    
    # 模式2: 通过批次ID删除
    elif batch_name is not None and kb_id is not None:
        # 使用知识库ID和批次名称构建完整目标路径
        toml_config_path = settings["kbot"]["file_root_path"]
        if not toml_config_path:
            # 动态计算项目根目录的同级路径
            project_root = Path(__file__).parent.parent.parent
            toml_config_path = project_root.parent / "KBOT_FILES"

        root_path = Path(toml_config_path).resolve()  # 转换为绝对路径
        target_path = root_path / str(domain_id) / str(kb_id) / "source" / batch_name

        file_count = 0
        for files in os.walk(target_path):
            file_count += len(files)
        
        try:
            logger.info(f"Deleting batch files: {str(target_path)}")
            shutil.rmtree(target_path)
            success_cnt = file_count            
            return success_cnt, failed_cnt
        except Exception as e:
            logger.error(f"Failed to delete batch files: {str(target_path)}: {str(e)}")
            failed_cnt = file_count    
            return success_cnt, failed_cnt
    
    # 模式3: 通过知识库ID删除
    elif kb_id is not None:
        # 使用知识库ID构建完整目标路径
        toml_config_path = settings["kbot"]["file_root_path"]
        if not toml_config_path:
            # 动态计算项目根目录的同级路径
            project_root = Path(__file__).parent.parent.parent
            toml_config_path = project_root.parent / "KBOT_FILES"

        root_path = Path(toml_config_path).resolve()  # 转换为绝对路径
        target_path = root_path / str(domain_id) / str(kb_id)

        file_count = 0
        for files in os.walk(target_path):
            file_count += len(files)
        
        try:
            logger.info(f"Deleting files in knowlodge base {str(kb_id)}: {str(target_path)}")
            shutil.rmtree(target_path)
            success_cnt = file_count            
            return success_cnt, failed_cnt
        except Exception as e:
            logger.error(f"Failed to delete files in knowlodge base {str(kb_id)}: {str(target_path)}, error: {str(e)}")
            failed_cnt = file_count    
            return success_cnt, failed_cnt
    
    else:
        logger.error("Invalid parameters")
        return success_cnt, failed_cnt

async def delete_metadata(kb_id: Optional[int], 
                          batch_id: Optional[int], 
                          file_ids: Optional[List[int]]) -> bool:
    """
    Delete file metadata either by individual file IDs or by batch ID or by kb ID.
    根据文件ID或批次ID或知识库ID删除文件元数据
    
    Args/参数:
        kb_id: ID of the knowledge base to delete all contained files (optional)
        知识库ID(用于整个知识库删除)(可选)
        batch_id: ID of the batch to delete all contained files (optional)
        要删除的批次ID(将删除该批次所有文件)(可选)
        file_ids: List of specific file IDs to delete (optional)
        要删除的特定文件ID列表(可选)
        
    Returns/返回:
        result: Object containing counts of successfully and failed deletions
        包含成功和失败删除计数的结果对象
    
    Note/注意:
        - Either file_ids or batch_id or kb_id must be provided (but only one parameter at a time)
          必须提供file_ids或batch_id或kb_id之一(但一次只能提供一个参数)
        - This is an async function and needs to be awaited
          这是一个异步函数，需要await调用
    
    Example/示例:
        >>> # Delete by kb_id
        >>> result = await delete_file_metadata(kb_id=123, batch_id=None, file_ids=None)
        >>> # Delete by batch
        >>> result = await delete_file_metadata(kb_id=None, batch_id=456, file_ids=None)
        >>> # Delete specific files
        >>> result = await delete_file_metadata(kb_id=None, batch_id=None, file_ids=[1,2,3])
    """
    file_repo = KbotMdKbFilesRepository()

    # Delete all files in the knowledge base
    # 删除整个知识库中的所有文件
    if kb_id is not None:
        try:
            rowcnt = await file_repo.delete(kb_id, None, None)
            logger.info(f"Successfully deleted {rowcnt} files in kb {kb_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete files in kb {kb_id}: {str(e)}")
            return False
        
    # Batch deletion logic
    # 批次删除逻辑
    elif batch_id is not None:
        try:
            rowcnt = await file_repo.delete(None, batch_id, None)
            logger.info(f"Successfully deleted {rowcnt} files in batch {batch_id}")           
            return True           
        except Exception as e:
            logger.error(f"Failed to delete files in batch {batch_id}: {str(e)}")
            return False
    
    # Individual file deletion logic
    # 单个文件删除逻辑
    elif file_ids is not None:
        try:
            rowcnt = await file_repo.delete(None, None, file_ids)
            logger.info(f"Successfully deleted {rowcnt} files")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file(s): {str(file_ids)}: {str(e)}")
            return False
    else:
        logger.error("Invalid deletion parameters: must provide either kb_id, batch_id, or file_ids")
        return False

async def delete_vec_data(kb_id: int, 
                          batch_id: Optional[int], 
                          file_ids: Optional[List[int]]) -> int:
    """
    Delete vector data from the database by file IDs and delete file metadata finally. 
    根据文件ID从数据库中删除向量数据，最后彻底删除文件元数据。
    
    Args/参数:
        kb_id: Knowledge base ID //知识库ID
        batch_id: Batch ID //批次ID
        file_ids: List of file IDs to delete //要删除的文件ID列表
    
    Returns/返回:
        result: Row count of deleted records //删除的记录行数
    
    Note/注意:
        - This is an async function and needs to be awaited //这是一个异步函数，需要await调用
    """

    embed_repo = KbotBizTxtEmbeddingRepository()
    file_repo = KbotMdKbFilesRepository()
    vec_cnt = 0
    # Mode 1: Delete by file IDs //模式1：通过文件ID删除
    if file_ids is not None:
        try:
            logger.debug(f"Deleting vector data for {len(file_ids)} files in vector base")
            vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
            logger.debug(f"Successfully deleted {vec_cnt} records in vector base")
            return vec_cnt
        except Exception as e:
            logger.error(f"Failed to delete vector data: {str(e)}")
            return 0
    # Mode 2: Delete by batch ID //模式2：通过批次ID删除
    elif batch_id is not None:
        try:
            file_repo = KbotMdKbFilesRepository()
            files = await file_repo.get_by_batch_id(batch_id)
            file_ids = []
            if files is None:
                logger.error(f"Files in batch {batch_id} not found")
                return 0
            for file in files:
                file_ids.append(file.file_id)
            logger.debug(f"Deleting vector data for {len(file_ids)} files in vector base")
            vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
            logger.debug(f"Successfully deleted {vec_cnt} records in vector base")
            return vec_cnt
        except Exception as e:
            logger.error(f"Failed to delete vector data: {str(e)}")
            return 0
    # Mode 3: Delete by knowledge base ID //模式3：通过知识库ID删除
    else:
        try:
            file_repo = KbotMdKbFilesRepository()
            files = await file_repo.get_by_kb_id(kb_id)
            file_ids = []
            if files is None:
                logger.error(f"Files in knowledge base {kb_id} not found")
                return 0
            for file in files:
                file_ids.append(file.file_id)
            logger.debug(f"Deleting vector data for {len(file_ids)} files in vector base")
            vec_cnt = await embed_repo.delete_by_file_ids(kb_id, file_ids)
            logger.debug(f"Successfully deleted {vec_cnt} records in vector base")
            return vec_cnt
        except Exception as e:
            logger.error(f"Failed to delete vector data: {str(e)}")
            return 0


async def delete_file_service(
    app_id: int,
    domain_id: int,
    kb_id: int, 
    batch_id: Optional[int], 
    batch_name: Optional[str],
    file_ids: Optional[List[int]],
    file_paths: Optional[List[str]]
) -> dict:
    """
    Unified file deletion service that handles multiple deletion scenarios.
    统一文件删除服务，处理多种删除场景
    
    Args/参数:
        kb_id: Knowledge base ID (for full KB deletion) 
        知识库ID(用于整个知识库删除)
        batch_id: Batch ID (for batch deletion)
        批次ID(用于批次删除)
        batch_name: Batch name (for file path construction)
        批次名称(用于文件路径构建)
        file_ids: List of file IDs (for specific file deletion)
        文件ID列表(用于特定文件删除)
        file_paths: List of file paths (for physical file deletion)
        文件路径列表(用于物理文件删除)
    
    Returns/返回:
        SuccessWithErrorResponse: Standard response object with detailed deletion results
        标准响应对象，包含详细的删除结果
    
    Note/注意:
        - Supports three deletion modes: single files, batch, or entire knowledge base
          支持三种删除模式：单个文件、批次或整个知识库
        - Returns HTTP 207 (Multi-Status) if partial failures occur
          如果部分失败会返回HTTP 207(多状态)
    
    Example/示例:
        >>> # Delete specific files
        >>> await delete_file_service(None, "kb1", None, None, [1,2], ["/path1","/path2"])
        >>> # Delete entire batch
        >>> await delete_file_service(None, "kb1", 123, "batch1", None, None)
        >>> # Delete entire knowledge base
        >>> await delete_file_service(None, "kb1", 123, None, None, None)
    """
    # Initialize result objects
    # 初始化结果对象
    result = {"success_file_cnt": 0, "failed_file_cnt": 0, "meta_cnt": 0, "vec_cnt": 0}

    # Mode 1: Delete specific files by IDs and paths
    # 模式1：通过ID和路径删除特定文件
    if file_paths is not None and file_ids is not None:
        logger.info(f"Starting to delete files, total {len(file_paths)} files...")
        # 1. Delete vector data //删除向量数据
        result["vec_cnt"] = await delete_vec_data(kb_id, None, file_ids)
        # 2. Delete file metadata //删除文件元数据
        result["meta_cnt"] = await delete_metadata(None, None, file_ids)
        # 3. Delete files physically //物理删除文件
        result["success_file_cnt"], result["failed_file_cnt"] = await delete_files(domain_id, None, None, file_paths)
        return result
    # Mode 2: Delete entire batch
    # 模式2：删除整个批次
    elif batch_name is not None and batch_id is not None:
        logger.info(f"Starting to delete files in batch: {batch_name}")
        # 1. Delete vector data //删除向量数据
        result["vec_cnt"] = await delete_vec_data(kb_id, batch_id, None)
        # 2. Delete file metadata //删除文件元数据
        result["meta_cnt"] = await delete_metadata(None, batch_id, None)
        # 3. Delete files physically //物理删除文件
        result["success_file_cnt"], result["failed_file_cnt"] = await delete_files(domain_id, kb_id, batch_name, None)
        return result
    # Mode 3: Delete entire knowledge base
    # 模式3：删除整个知识库
    elif kb_id is not None and batch_id is None and file_ids is None:
        logger.info(f"Starting to delete knowledge base: {kb_id}")
        # 1. Delete vector data //删除向量数据
        result["vec_cnt"] = await delete_vec_data(kb_id, None, None)
        # 2. Delete file metadata //删除文件元数据
        result["meta_cnt"] = await delete_metadata(kb_id, None, None)
        # 3. Delete files physically //物理删除文件
        result["success_file_cnt"], result["failed_file_cnt"] = await delete_files(domain_id, kb_id, None, None)
        return result
    else:
        logger.error("Invalid deletion parameters: must provide either kb_id, batch_id, or file_ids")
        return result

