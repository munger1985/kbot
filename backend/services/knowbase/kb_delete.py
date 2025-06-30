import os
import shutil
from pathlib import Path
from typing import List, Optional

from backend.api.schemas.kb_response import SuccessWithErrorResponse
from backend.core.log.logger import logger
from backend.core.config import settings
from backend.dao.entities.kbot_md_kb_files import FileStatus
from backend.dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from backend.dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from backend.dao.repositories.kbot_biz_txt_embedding_repo import (
    KbotBizTxtEmbeddingOracleRepo,
    KbotBizTxtEmbeddingPGRepo,
    KbotBizTxtEmbeddingMySQLRepo
)
from backend.dao.repositories.kbot_biz_img_embedding_repo import (
    KbotBizImgEmbeddingOracleRepo,
    KbotBizImgEmbeddingPGRepo,
    KbotBizImgEmbeddingMySQLRepo
)
from backend.dao.repositories.kbot_md_kb_batch_repo import KbotMdKbBatchRepository

# define the result object // 定义结果对象
class DELRESULT():
        success_file_cnt: int = 0
        failed_file_cnt: int = 0
        success_meta_cnt: int = 0
        failed_meta_cnt: int = 0
def delete_files(file_paths: List[str]) -> DELRESULT:
    '''
    Delete files from disk by file paths.
    根据文件路径从磁盘删除文件
    
    Args/参数:
        file_paths: List of file paths to delete
        要删除的文件路径列表
    
    Returns/返回:
        result: The number of successfully deleted files and the number of failed files
        包含成功删除的文件数和失败文件数的结果对象
    
    Example/示例:
        >>> result = delete_files(['/path/to/file1.txt', '/path/to/file2.txt'])
        >>> print(f"Success: {result.success_file_cnt}, Failed: {result.failed_file_cnt}")
    '''
    # Initialize result object to track deletion counts
    # 初始化结果对象用于记录删除计数
    result = DELRESULT()
    
    # Iterate through each file path in the input list
    # 遍历输入列表中的每个文件路径
    for file in file_paths:
        # Log the current file being processed
        # 记录当前正在处理的文件
        logger.info("Deleting file: {}", file)
        
        # Check if the file exists before attempting deletion
        # 在尝试删除前检查文件是否存在
        if os.path.exists(Path(file)):
            try:
                # Attempt to delete the file
                # 尝试删除文件
                os.remove(Path(file))
                # Increment success counter if deletion succeeds
                # 如果删除成功则增加成功计数器
                result.success_file_cnt += 1
            except Exception as e:
                # Log error if deletion fails
                # 如果删除失败则记录错误
                logger.error(f"Failed to delete file {file}: {str(e)}")
                # Increment failure counter
                # 增加失败计数器
                result.failed_file_cnt += 1
        else:
            # Log error if file doesn't exist
            # 如果文件不存在则记录错误
            logger.error(f"File {file} does not exist")
            # Increment failure counter
            # 增加失败计数器
            result.failed_file_cnt += 1
    
    # Return the result object with success/failure counts
    # 返回包含成功/失败计数的结果对象
    return result

def delete_batch_files(domain_id: int, kb_id: int, batch_name: str) -> DELRESULT:
    '''
    Delete all files in a batch from the knowledge base.
    从知识库中删除指定批次的所有文件
    
    Args/参数:
        domain_id: ID of the domain where the batch resides
        批次所在的业务域ID
        kb_id: ID of the knowledge base where the batch resides
        批次所在的知识库ID
        batch_name: Name of the batch to be deleted
        要删除的批次名称
    
    Returns/返回:
        result: Object containing count of successfully deleted files and failed files
        包含成功删除文件数和失败文件数的结果对象
    
    Example/示例:
        >>> result = delete_batch_files("my_kb", "batch_2023")
        >>> print(f"Deleted {result.success_file_cnt} files successfully")
    
    Note/注意:
        This function deletes the entire batch directory recursively
        本函数会递归删除整个批次目录
    '''
    # Initialize result object to track deletion status
    # 初始化结果对象用于记录删除状态
    result = DELRESULT()
    
    # Construct the complete target path using knowledge id and batch names
    # 使用知识库ID和批次名称构建完整目标路径
    toml_config_path = settings.get("kbot", {}).get("file_root_path")
    if not toml_config_path:
        # 动态计算项目根目录的同级路径
        project_root = Path(__file__).parent.parent.parent
        toml_config_path = project_root.parent / "KBOT_FILES"

    root_path = Path(toml_config_path).resolve()  # 转换为绝对路径
    target_path = root_path / str(domain_id) / str(kb_id) / "source" / batch_name

    
    # Count total files in the batch directory for reporting
    # 统计批次目录中的文件总数用于报告
    file_count = 0
    for files in os.walk(target_path):
        file_count += len(files)
    
    try:
        # Log the deletion operation
        # 记录删除操作日志
        logger.info(f"Deleting batch files: {str(target_path)}")
        
        # Recursively delete the entire directory tree
        # 递归删除整个目录树
        shutil.rmtree(target_path)
        
        # Set success count to total files (since we deleted the whole batch)
        # 设置成功计数为文件总数(因为我们删除了整个批次)
        result.success_file_cnt = file_count
        
        return result
    except Exception as e:
        # Log error if deletion fails
        # 如果删除失败则记录错误
        logger.error(f"Failed to delete batch files: {str(target_path)}: {str(e)}")
        
        # Set failure count to total files (since the whole batch failed)
        # 设置失败计数为文件总数(因为整个批次删除失败)
        result.failed_meta_cnt = file_count
        
        return result

async def delete_file_metadata(file_ids: Optional[List[int]], batch_id: Optional[int]) -> DELRESULT:
    """
    Delete file metadata either by individual file IDs or by batch ID.
    根据文件ID或批次ID删除文件元数据
    
    Args/参数:
        file_ids: List of specific file IDs to delete (optional)
        要删除的特定文件ID列表(可选)
        batch_id: ID of the batch to delete all contained files (optional)
        要删除的批次ID(将删除该批次所有文件)(可选)
    
    Returns/返回:
        result: Object containing counts of successfully and failed deletions
        包含成功和失败删除计数的结果对象
    
    Note/注意:
        - Either file_ids or batch_id must be provided (but not both)
          必须提供file_ids或batch_id之一(但不能同时提供)
        - This is an async function and needs to be awaited
          这是一个异步函数，需要await调用
    
    Example/示例:
        >>> # Delete by batch
        >>> result = await delete_file_metadata(None, batch_id=123)
        >>> # Delete specific files
        >>> result = await delete_file_metadata([1,2,3], None)
    """
    # Initialize repository for metadata operations
    # 初始化元数据操作仓库
    file_repo = KbotMdKbFilesRepository()
    
    # Initialize result object to track deletion counts
    # 初始化结果对象用于记录删除计数
    result = DELRESULT()
    
    # Batch deletion logic
    # 批次删除逻辑
    if batch_id is not None:
        try:
            # Perform batch delete operation
            # 执行批次删除操作
            success_cnt, failed_cnt = await file_repo.batch_delete(batch_id)
            
            # Log operation results
            # 记录操作结果
            logger.info(
                f"Successfully deleted {success_cnt} files in batch {batch_id}, "
                f"failed to delete {failed_cnt} files"
            )
            
            # Update result counts
            # 更新结果计数
            result.success_meta_cnt = success_cnt
            result.failed_meta_cnt = failed_cnt
            return result
            
        except Exception as e:
            # Log batch deletion failure
            # 记录批次删除失败
            logger.error(f"Failed to delete files in batch {batch_id}: {str(e)}")
            return result
    
    # Individual file deletion logic
    # 单个文件删除逻辑
    else:
        # Validate file_ids parameter
        # 验证file_ids参数
        if file_ids is None:
            logger.error("No file IDs provided")
            return result
            
        failed_cnt = 0
        success_cnt = 0
        
        try:
            # Process each file ID individually
            # 逐个处理每个文件ID
            for file_id in file_ids:
                # Attempt to delete metadata
                # 尝试删除元数据
                delmeta = await file_repo.delete(file_id)
                
                if delmeta:
                    # Log successful deletion
                    # 记录成功删除
                    logger.info(f"Successfully deleted file and chunks: {str(file_id)}")
                    success_cnt += 1
                else:
                    # Log failed deletion
                    # 记录删除失败
                    logger.error(f"Failed to delete file: {str(file_id)}")
                    failed_cnt += 1
                    
        except Exception as e:
            # Log bulk operation failure
            # 记录批量操作失败
            logger.error(f"Failed to delete file(s): {str(file_ids)}: {str(e)}")
            return result
            
        # Update final counts
        # 更新最终计数
        result.success_meta_cnt = success_cnt
        result.failed_meta_cnt = failed_cnt
        return result
        
async def delete_vecdata_service(kb_id: int, file_id: int) -> bool:
    """Delete vector data associated with a specific file in a knowledge base.
    删除知识库中特定文件关联的向量数据
    
    Args/参数:
        kb_id: ID of the knowledge base containing the target file
        包含目标文件的知识库ID
        file_id: ID of the file whose vector data should be deleted
        需要删除向量数据的文件ID
        
    Returns/返回:
        bool: True if deletion succeeded, False if failed
        成功返回True，失败返回False
    
    Note/注意:
        - This is an async function and needs to be awaited
          这是一个异步函数，需要await调用
        - Currently only supports Oracle database (can be extended)
          当前仅支持Oracle数据库(可扩展)
    
    Example/示例:
        >>> success = await delete_vecdata_service(123, 456)
        >>> print("Deletion succeeded" if success else "Deletion failed")
    """
    # Initialize knowledge base repository
    # 初始化知识库仓库
    kb_repo = KbotMdKbRepository()
    
    # Get database configuration for the knowledge base
    # 获取知识库的数据库配置
    mddbconf = await kb_repo.get_dbconf_by_kbid(kb_id)
    if mddbconf is None:
        logger.error("Failed to get database configuration for kb_id: {}", kb_id)
        return False
    
    # Process database configuration with type safety
    # 类型安全地处理数据库配置
    dbtype = mddbconf.db_type
    dbstr = mddbconf.db_conn_str
    dbconf = {"db_type": dbtype, "db_conn_str": dbstr}    
    
    logger.info(f"Starting to delete vector data of file: {file_id}")
     
    try:
        # Initialize appropriate repository based on database type
        # 根据数据库类型初始化相应的仓库
        vec_txt_repo = None
        vec_img_repo = None
        if dbtype and dbtype.lower() == "oracle":  # Case-insensitive comparison
            vec_txt_repo = KbotBizTxtEmbeddingOracleRepo(dbconf)
            vec_img_repo = KbotBizImgEmbeddingOracleRepo(dbconf)
        elif dbtype and dbtype.lower() == "mysql":  # Case-insensitive comparison
            vec_txt_repo = KbotBizTxtEmbeddingMySQLRepo(dbconf)
            vec_img_repo = KbotBizImgEmbeddingMySQLRepo(dbconf)
        elif dbtype and dbtype.lower() == "postgresql":  # Case-insensitive comparison
            vec_txt_repo = KbotBizTxtEmbeddingPGRepo(dbconf)
            vec_img_repo = KbotBizImgEmbeddingPGRepo(dbconf)
        # Other database types can be added here
        else:
            # Log unsupported database type
            # 记录不支持的数据库类型
            logger.error("Unsupported database type or failed to get repository")
            return False
        r1 = None
        r2 = None
        # Perform the actual deletion
        # 执行实际删除操作
        if vec_txt_repo is not None:            
            r1 = await vec_txt_repo.delete_by_file_id(file_id)
        if vec_img_repo is not None:
            r2 = await vec_img_repo.delete_by_file_id(file_id)
        
        if r1 and r2:
            return True
        elif r1 and not r2:
            logger.error("Failed to delete img vecdata for file {}", file_id)
            return True
        elif not r1 and r2:
            logger.error("Failed to delete txt vecdata for file {}", file_id)
            return True
        else:
            logger.error("Failed to delete vecdata for file {}", file_id)
            return False
        
    except Exception as e:
        # Log any exceptions during deletion
        # 记录删除过程中的任何异常
        logger.error("Failed to delete metadata for kb_id {}: {}", kb_id, str(e))
        return False
    
async def delete_file_service(
    domain_id: int,
    kb_id: int, 
    kb_name: str, 
    batch_id: Optional[int], 
    batch_name: Optional[str],
    file_ids: Optional[List[int]],
    file_paths: Optional[List[str]]
) -> SuccessWithErrorResponse:
    """
    Unified file deletion service that handles multiple deletion scenarios.
    统一文件删除服务，处理多种删除场景
    
    Args/参数:
        kb_id: Knowledge base ID (for full KB deletion) 
        知识库ID(用于整个知识库删除)
        kb_name: Knowledge base name (for file path construction)
        知识库名称(用于文件路径构建)
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
    """
    # Initialize result trackers
    # 初始化结果跟踪器
    file_result = DELRESULT()  # Physical file deletion results 物理文件删除结果
    meta_result = DELRESULT()  # Metadata deletion results 元数据删除结果
    total_result = DELRESULT() # Aggregated results 聚合结果

    # Mode 1: Delete specific files by IDs and paths
    # 模式1：通过ID和路径删除特定文件
    if file_paths is not None and file_ids is not None:
        logger.info(f"Starting to delete files, total {len(file_paths)} files...")
        # Delete file metadata (async)
        # 删除文件元数据(异步)
        meta_result = await delete_file_metadata(file_ids, None)
        # Delete physical files (sync)
        # 删除物理文件(同步)
        file_result = delete_files(file_paths)
        
    # Mode 2: Delete entire batch
    # 模式2：删除整个批次
    elif batch_name is not None and kb_id is not None:
        logger.info(f"Starting to delete files in batch: {batch_name}")
        # Delete batch metadata (async)
        # 删除批次元数据(异步)
        meta_result = await delete_file_metadata(None, batch_id)
        # Delete batch files (sync)
        # 删除批次文件(同步)
        file_result = delete_batch_files(domain_id, kb_id, batch_name)

    # Mode 3: Delete entire knowledge base
    # 模式3：删除整个知识库
    elif kb_id is not None:
        batch_repo = KbotMdKbBatchRepository()
        # Get all batches in the knowledge base
        # 获取知识库中所有批次
        batches = await batch_repo.get_by_kb_id(kb_id)
        temp_result = DELRESULT()
        
        for batch in batches:
            logger.info(f"Starting to delete files in batch: {batch.batch_name}")
            # Delete batch metadata (async)
            # 删除批次元数据(异步)
            temp_result = await delete_file_metadata(None, batch.batch_id)
            meta_result.success_meta_cnt += temp_result.success_meta_cnt
            meta_result.failed_meta_cnt += temp_result.failed_meta_cnt
            
            # Delete batch files (sync)
            # 删除批次文件(同步)
            temp_result = delete_batch_files(domain_id, kb_id, batch.batch_name)
            file_result.success_file_cnt += temp_result.success_file_cnt
            file_result.failed_file_cnt += temp_result.failed_file_cnt
    
    # Aggregate results from all operations
    # 聚合所有操作结果
    total_result.success_file_cnt = file_result.success_file_cnt
    total_result.failed_file_cnt = file_result.failed_file_cnt
    total_result.success_meta_cnt = meta_result.success_meta_cnt
    total_result.failed_meta_cnt = meta_result.failed_meta_cnt

    # Determine HTTP status code
    # 确定HTTP状态码
    code = 200  # Default to success 默认成功
    if total_result.failed_file_cnt != 0 or total_result.failed_meta_cnt != 0:
        code = 207  # Multi-Status if partial failures 部分失败时使用多状态码

    return SuccessWithErrorResponse(
        success=True,
        code=code,
        message="Successfully deleted files",
        details={
            "success_file_cnt": total_result.success_file_cnt,
            "failed_file_cnt": total_result.failed_file_cnt,
            "success_meta_cnt": total_result.success_meta_cnt,
            "failed_meta_cnt": total_result.failed_meta_cnt
        }
    )
