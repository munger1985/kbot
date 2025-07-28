from services.kb.kb_upload import upload_file_service
from services.kb.kb_delete import delete_file_service
from api.schemas.kb_schema import KBUploadForm, KBDeleteForm


async def upload_knowledge_base_files(
    form: KBUploadForm
) -> bool:
    """
    Upload files to the knowledge base.
    上传文件到知识库
    """
    try:
        result = await upload_file_service(
            files=form.files,
            app_id=form.metadata.app_id,
            domain_id=form.metadata.domain_id,
            kb_id=form.metadata.kb_id,
            overwrite=form.metadata.overwrite,
            batch_name=form.metadata.batch_name,
            batch_id=form.metadata.batch_id,
            biz_metadata=form.metadata.biz_metadata,
            created_by=form.metadata.created_by
        )
        return result
    except Exception as e:
        raise e

async def delete_knowledge_base_files(
    form: KBDeleteForm
) -> dict:
    """
    Delete files from the knowledge base.
    从知识库中删除文件
    """
    try:
        result = await delete_file_service(
            app_id=form.metadata.app_id,
            domain_id=form.metadata.domain_id,
            kb_id=form.metadata.kb_id,
            batch_id=form.metadata.batch_id,
            batch_name=form.metadata.batch_name,
            file_ids=form.metadata.file_ids,
            file_paths=form.metadata.file_paths,
        )
        return result
    except Exception as e:
        raise e
    
