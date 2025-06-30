from typing import List
from fastapi import UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from backend.services.knowbase.kb_upload import upload_files
from backend.api.schemas.kb_response import KBErrorResponse
from backend.api.schemas.kb_upload_schema import KBUploadForm


async def upload_knowledge_base_files(
    form: KBUploadForm
) -> JSONResponse:
    """
    Upload files to the knowledge base.
    上传文件到知识库
    """
    result = await upload_files(
        files=form.files,
        app_id=form.metadata.app_id,
        domain_id=form.metadata.domain_id,
        kb_id=form.metadata.kb_id,
        overwrite=form.metadata.overwrite,
        batch_name=form.metadata.batch_name,
        batch_id=form.metadata.batch_id,
        security_level=form.metadata.security_level,
        biz_metadata=form.metadata.biz_metadata,
        created_by=form.metadata.created_by
    )

    if isinstance(result, KBErrorResponse):
        raise HTTPException(
            status_code=result.code,
            detail={
                "message": result.message,
                "error_type": result.error_type,
                "details": result.details
            }
        )

    # 直接返回upload_files的结果，它已经是正确的响应类型
    return JSONResponse(
        status_code=result.code,
        content=result.dict()
    )