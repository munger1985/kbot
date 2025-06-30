import os

from typing import List, Optional
from pathlib import Path
from fastapi import UploadFile
from datetime import datetime


from backend.api.schemas.kb_response import KBErrorResponse, KBUploadResponse, KBItem
from backend.core.log.logger import logger
from backend.core.config import settings
from backend.dao.entities.kbot_md_kb_batch import KbotMdKbBatch
from backend.dao.entities.kbot_md_kb_files import (
    KbotMdKbFiles,
    FileStatus
    )
from backend.dao.repositories.kbot_md_kb_repo import KbotMdKbRepository
from backend.dao.repositories.kbot_md_kb_files_repo import KbotMdKbFilesRepository
from backend.utils.common_methods import run_in_thread_pool


def save_file(file: UploadFile, knowledge_base_name: str, overwrite: bool, batch_name:str) -> dict:
        '''
        Save single file to disk and return the file path // 保存单个文件到磁盘并返回文件路径
        '''
        filename = file.filename
        if filename is None:
                raise ValueError("Filename cannot be None")
        try:
            logger.debug(f"Start saving file: {filename} to knowledge base: {knowledge_base_name}")
            file_content = file.file.read()

            # Construct the target path. // 构建目标路径
            root_path = Path(str(settings.file_root_path))
            target_path = root_path / knowledge_base_name / "source" / batch_name
            target_path.mkdir(parents=True, exist_ok=True)
            file_path = target_path / filename

            # Get file parameters. // 获取文件相关参数
            name, ext = os.path.splitext(filename)

            fileparams = {"file_path": str(file_path), 
                          "file_name": filename, 
                          "file_ext": ext, 
                          "is_overwrite": "Y" if overwrite else "N",
                          "file_version": 1, 
                          "file_size": len(file_content)}          
            
            # Handle filename conflicts. // 处理文件名冲突
            if file_path.exists() and not overwrite:
                
                # Append a numeric suffix to the filename until the conflict is resolved. // 添加数字后缀直到文件名不冲突
                counter = 1
                new_filename = ""
                while file_path.exists():
                    new_filename = f"{name}-{counter}{ext}"
                    file_path = target_path / new_filename
                    counter += 1
                fileparams["file_name"] = new_filename
                fileparams["file_version"] = counter
            
            # Save the file. // 保存文件
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"File saved successfully: {filename} -> {file_path}")
            return fileparams

        except Exception as e:
            logger.error(f"Failed to save file {filename if 'filename' in locals() else 'unknown'}: {str(e)}")
            return {}
        
def save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          overwrite: bool,
                          batch_name :str) -> List[dict]:
    '''
    Save uploaded files to corresponding knowledge base directory using multi-threading. // 通过多线程将上传的文件保存到对应知识库目录内
    Generator returns save results: {"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}} // 生成器返回保存结果
    '''
    params = [{"file": file, "knowledge_base_name": knowledge_base_name,
               "overwrite": overwrite, "batch_name": batch_name} for file in files]
    results = list(run_in_thread_pool(save_file, params=params))
    return results

async def upload_files(files: List[UploadFile], 
                 app_id: int,
                 kb_name: str, 
                 overwrite: bool,
                 batch_name:str,
                 batch_id: Optional[int] = None,
                 status: Optional[int] = FileStatus.UPLOADED,
                 security_level: Optional[str] = None,
                 biz_metadata: Optional[dict] = None,
                 created_by: Optional[str] = None,
                 ) -> KBUploadResponse | KBErrorResponse:
    '''
    Save upload records // 保存上传记录
    '''
    
    kb_repo = KbotMdKbRepository()
    kb_id = kb_repo.get_by_name(kb_name)
    if kb_id is None:
        msg = f"Knowledge base {kb_name} does not exist."
        logger.error(msg)
        return KBErrorResponse(
            success=False,
            code=404, 
            message=msg,
            error_type="knowledge_base_not_found",
            details=None
        )
    
    # Save the file. // 保存文件
    logger.info(f"Start uploading {len(files)} files to knowledge base: {kb_name}")
    fileparams = save_files_in_thread(files, kb_name, overwrite, batch_name)
    logger.debug(f"Files saved to disk: {[fp['file_name'] for fp in fileparams]}")

    # Construct the batch entities for saving to the database. //构造 batch 的实体用于保存到数据库
    batch_entity = KbotMdKbBatch(
        batch_id=batch_id,
        app_id=app_id,
        batch_name=batch_name,
        kb_id=kb_id,
        created_by=created_by
    )

    # Construct the file entities for batch saving to the database. //构造 file 的实体列表用于批量保存到数据库
    file_entitities = []
    for fileparam in fileparams:

        file_entitity = KbotMdKbFiles(
            app_id = app_id,
            kb_id = kb_id,
            batch_id = batch_id,
            file_path = fileparam["file_path"],
            file_name = fileparam["file_name"],
            file_ext = fileparam["file_ext"],
            status=status,
            file_version = fileparam["file_version"],
            is_overwrite = fileparam["is_overwrite"],
            security_level = security_level,
            file_size = fileparam["file_size"],
            biz_metadata = biz_metadata,
            created_by=created_by
        )
        file_entitities = file_entitities + [file_entitity]
    
    # Save upload records to database. // 保存上传记录到数据库
    kb_files_repo = KbotMdKbFilesRepository()
    try:
        logger.debug(f"Start saving {len(file_entitities)} files to database for knowledge base: {kb_name}")
        r = await kb_files_repo.create(batch_entity, file_entitities)
        logger.info(f"Successfully saved {len(file_entitities)} files to database")
        if r:
            data = KBItem(
                id=str(kb_id),
                name=kb_name,
                description=f"Knowledge base for {kb_name}",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat()
            )
            return KBUploadResponse(
                success=True,
                message="Files uploaded successfully", 
                data=data,
                code=200
            )
        else:
            return KBErrorResponse(
                success=False,
                code=500,
                message="Upload records save failed.",
                error_type="database_error",
                details=None
            )
    except Exception as e:
        msg = f"Upload records save failed. Error: {e}"
        logger.error(msg)
        return KBErrorResponse(
            success=False,
            code=500,
            message=msg,
            error_type="database_error",
            details={"file_params": fileparams}  # use the actual saved file parameters as details. //使用实际保存的文件参数作为details
        )
    