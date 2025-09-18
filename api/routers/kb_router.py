
import json
from fastapi import APIRouter, UploadFile, File, Form, status, Depends
from fastapi.responses import HTMLResponse
from api.controllers.security_controller import AuthController
from fastapi.responses import FileResponse
from api.controllers.kb_controller import *
from api.schemas.kb_schema import *
from api.schemas.base_response import SuccessResponse, ErrorResponse

router = APIRouter(
    prefix="/kb",
    tags=["Knowledge Base"]
)

@router.post(
    "/upload",
    description="上传一个或多个文件到指定知识库的接口",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_upload_files(
    files: list[UploadFile] = File(...),
    metadata: str = Form(...)
):
    try:
        # Parse and validate as form model
        metadata_dict = json.loads(metadata)
        form = KBUploadForm(files=files, **metadata_dict)
        
        # Call the controller
        result = await upload_kb_files(form)
        
        if result:
            return SuccessResponse(
                code=200,
                success=True,
                message="文件上传成功"
            )
        else:
            return ErrorResponse(
                code=400,
                success=False,
                message="文件上传失败"
            )
        
    except json.JSONDecodeError as e:
        return ErrorResponse(
            code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            success=False,
            message=f"请求参数格式错误: {str(e)}"
        )
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
    
@router.post(
    "/delete",
    description="从指定的知识库中删除文件或所有文件以及其知识库或批次的接口",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_delete_files(
    metadata: str = Form(...)
):
    try:
        # Parse and validate as form model
        metadata_dict = json.loads(metadata)
        form = KBDeleteForm(**metadata_dict)
        
        # Call the controller
        result = await delete_kb_files(form)


        if result["failed_file_cnt"] == 0 and result["meta_cnt"] > 0:
            return SuccessResponse(
                code=200,
                success=True,
                message="删除文件成功"
            )
        else:
            return ErrorResponse(
                code=400,
                success=False,
                message=f"删除文件失败: {result['failed_file_cnt']}个文件删除失败，详情请查看日志"
                )
        
    except json.JSONDecodeError as e:
        return ErrorResponse(
            code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            success=False,
            message=f"请求参数格式错误: {str(e)}"
        )
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
    
@router.get(
    "/download",
    description="从知识库中下载文件的接口",
    response_model=None,
    # dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_download_file(
    file_id: str
):
    try:
        result = await get_kb_files(file_id, download=True)
        
        if result:
            return FileResponse(
                path=result["file_path"],
                filename=result["file_name"],
                media_type="multipart/form-data",
                headers={
                    "Content-Disposition": "attachment"
                },
                content_disposition_type=None # type: ignore
                )
        else:
            return ErrorResponse(
                code=status.HTTP_404_NOT_FOUND,
                success=False,
                message="文件不存在"
            )
        
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )

@router.get(
    "/preview",
    description="从知识库中预览文件的接口",
    response_model=None,
    # dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_preview_file(
    file_id: str,
    page_num: int = 0
):
    try:
        result = await get_kb_files(file_id, download=False, page_num=page_num)
        
        if result:
            if result["file_ext"] == ".txt":
                with open(result["file_path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <body>
                    <div class="content">{content}</div>
                </body>
                </html>
                """
                return HTMLResponse(content=html_content)
            
            else:
                return FileResponse(
                        path=result["file_path"],
                        filename=result["file_name"],
                        media_type="image/png",
                        headers={"Content-Disposition": "inline"}
                    )
        else:
            return ErrorResponse(
                code=status.HTTP_404_NOT_FOUND,
                success=False,
                message="文件不存在"
            )
        
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
    
@router.post(
    "/file/reparse",
    description="重新解析文件的接口",
    response_model=SuccessResponse,
    dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_reparse_files(
    metadata: str = Form(...)
):
    try:
        metadata_dict = json.loads(metadata)
        form = KBReparseForm(**metadata_dict)

        result = await reparse_kb_files(form=form)
        if result:
            return SuccessResponse(
                code=200,
                success=True,
                message="重解析文件成功"
                )
        else:
            return ErrorResponse(
                code=400,
                success=False,
                message="重解析文件失败"
            )
            
    except json.JSONDecodeError as e:
        return ErrorResponse(
            code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            success=False,
            message=f"请求参数格式错误: {str(e)}"
        )
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
    

@router.post(
    "/file/preview/v2",
    description="从知识库中预览文件的接口",
    response_model=None,
    # dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_preview_kb_file(
    form: KBFilePreviewForm
):
    try:
        kwargs = {
            "file_id": form.file_id,
            "max_text_length": form.max_text_length,
            "max_pages": form.max_pages,
            "max_sheets": form.max_sheets,
            "max_slides": form.max_slides,
            "pdf_pages": form.pdf_pages,
            "word_page": form.word_page,
            "sheet_index": form.sheet_index,
            "start_index": form.start_index,
            "slide": form.slide
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        return await preview_kb_file(**kwargs)
         
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )