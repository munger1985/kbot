from services.knowbase.kb_upload import upload_files
from api.schemas.kb_upload_schema import KBUploadForm


async def upload_knowledge_base_files(
    form: KBUploadForm
) -> bool:
    """
    Upload files to the knowledge base.
    上传文件到知识库
    """
    try:
        result = await upload_files(
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