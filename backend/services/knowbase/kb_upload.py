import os
import json
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


def save_file(file: UploadFile, domain_id: int, kb_id: int, batch_name:str, overwrite: bool) -> dict:
        '''
        Save single file to disk and return the file path // 保存单个文件到磁盘并返回文件路径
        Args:
            file: File to upload // 要上传的文件
            domain_id: Target domain id // 业务域id
            kb_id: Target knowledge base id // 目标知识库id
            batch_name: Batch name for this upload // 本次上传的批次名称
            overwrite: Whether to overwrite existing files // 是否覆盖已存在的文件
               
        Returns:
            dict: File saving result containing: // 文件保存结果，包含:
                {
                    "file_path": str,  // 文件保存路径
                    "file_name": str,  // 文件名
                    "file_ext": str,   // 文件扩展名
                    "is_overwrite": str,  // 是否覆盖(Y/N)
                    "file_version": int,  // 文件版本号
                    "file_size": int     // 文件大小
                }
            or empty dict on error // 或出错时返回空字典
        '''
        filename = file.filename
        if filename is None:
                raise ValueError("Filename cannot be None")
        try:
            logger.debug(f"Start saving file: {filename} to knowledge base: {kb_id}")
            file_content = file.file.read()

            # Construct the target path. // 构建目标路径
            toml_config_path = settings.get("kbot", {}).get("file_root_path")
            if not toml_config_path:
                # 动态计算项目根目录的同级路径
                project_root = Path(__file__).parent.parent.parent
                toml_config_path = project_root.parent / "KBOT_FILES"

            root_path = Path(toml_config_path).resolve()  # 转换为绝对路径
            target_path = root_path / str(domain_id) / str(kb_id) / "source" / batch_name
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
            # Retrieve the maximum version number of the file before overwriting. //覆盖前获取该文件的最大版本号
            elif file_path.exists() and overwrite:
                counter = 1
                new_filename = ""
                 
                # After retrieving the maximum version number, the original file must still be overwritten, 
                # and subsequent saves will continue using the same file_path.
                # 在获取最大版本号之后仍然需要覆盖最初的文件，后续保存文件仍然使用 file_path
                new_path = file_path

                while new_path.exists():
                    new_filename = f"{name}-{counter}{ext}"
                    new_path = target_path / new_filename
                    counter += 1
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
                          domain_id: int,
                          kb_id: int,
                          batch_name: str,
                          overwrite: bool) -> List[dict]:
    '''
    Save uploaded files to corresponding knowledge base directory using multi-threading. // 通过多线程将上传的文件保存到对应知识库目录内
    Args:
        files: List of files to upload // 要上传的文件列表
        domain_id: Target domain id // 业务域id
        kb_id: Target knowledge base id // 目标知识库id
        batch_name: Batch name for this upload // 本次上传的批次名称
        overwrite: Whether to overwrite existing files // 是否覆盖已存在的文件
    
    Returns:
        List[dict]: List of file saving results, each contains: // 文件保存结果列表，每个结果包含:
            {
                "file_path": str,  // 文件保存路径
                "file_name": str,  // 文件名
                "file_ext": str,   // 文件扩展名
                "is_overwrite": str,  // 是否覆盖(Y/N)
                "file_version": int,  // 文件版本号
                "file_size": int     // 文件大小
            }
    '''
    params = [{"file": file, "domain_id": domain_id, "kb_id": kb_id,
               "batch_name": batch_name, "overwrite": overwrite} for file in files]
    results = list(run_in_thread_pool(save_file, params=params))
    return results

async def upload_files(files: List[UploadFile], 
                 app_id: int,
                 domain_id: int,
                 kb_id: int,
                 batch_name:str,
                 overwrite: bool,
                 batch_id: Optional[int] = None,
                 security_level: Optional[str] = None,
                 biz_metadata: Optional[dict] = None,
                 created_by: Optional[str] = None,
                 ) -> KBUploadResponse | KBErrorResponse:
    '''
    Upload files to knowledge base and save records to database. // 上传文件到知识库并保存记录到数据库
    Args:
        files: List of files to upload // 要上传的文件列表
        app_id: Application ID // 应用ID
        domain_id: Target domain id // 业务域id
        kb_id: Target knowledge base id // 目标知识库id
        batch_name: Batch name for this upload // 本次上传的批次名称
        overwrite: Whether to overwrite existing files // 是否覆盖已存在的文件
        batch_id: Optional batch ID // 可选的批次ID
        security_level: File security level // 文件安全级别
        biz_metadata: Business metadata in JSON format // 业务元数据(JSON格式)
        created_by: Creator identifier // 创建者标识
    
    Returns:
        KBUploadResponse: On successful upload // 上传成功时返回
        KBErrorResponse: On error // 出错时返回
    '''
    
    # kb_repo = KbotMdKbRepository()
    # kb_entity = await kb_repo.get_by_id(kb_id)
    # if kb_entity:
    #    kb_id = kb_entity.kb_id
    # else:
    #    msg = f"Knowledge base {kb_id} does not exist."
    #    logger.error(msg)
    #    return KBErrorResponse(
    #         success=False,
    #         code=404, 
    #         message=msg,
    #         error_type="knowledge_base_not_found",
    #         details=None
    #     )
    
    # Save the file. // 保存文件
    logger.info(f"Start uploading {len(files)} files to knowledge base: {kb_id}")
    fileparams = save_files_in_thread(files=files, domain_id=domain_id, kb_id=kb_id, batch_name=batch_name, overwrite=overwrite)
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
            status=FileStatus.UPLOADED.value,
            file_version = fileparam["file_version"],
            is_overwrite = fileparam["is_overwrite"],
            security_level = security_level,
            file_size = fileparam["file_size"],
            biz_metadata = json.dumps(biz_metadata) if biz_metadata is not None else None,
            created_by=created_by
        )
        file_entitities = file_entitities + [file_entitity]
    
    # Save upload records to database. // 保存上传记录到数据库
    kb_files_repo = KbotMdKbFilesRepository()
    try:
        logger.debug(f"Start saving {len(file_entitities)} files to database for knowledge base: {kb_id}")
        r = await kb_files_repo.create(batch_entity, file_entitities)
        logger.info(f"Successfully saved {len(file_entitities)} files to database")
        if r:
            data = KBItem(
                id=str(kb_id),
                name="Knowledge Base",
                description=f"Successfully saved {len(file_entitities)} files to database",
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
    