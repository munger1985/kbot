
import json
import urllib.parse
from fastapi import APIRouter, UploadFile, File, Form, status, Depends, Body
from fastapi.responses import HTMLResponse
from api.controllers.security_controller import AuthController
from fastapi.responses import FileResponse
from api.controllers.kb_controller import kb_controller as controller
from api.schemas.kb_schema import *
from api.schemas.base_response import *

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])

@router.post(
    "/upload",
    summary="上传一个或多个文件到指定知识库",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_upload_files(
    files: list[UploadFile] = File(...),
    metadata: str = Form(...)
) -> SuccessResponse | ErrorResponse:
    """
    上传一个或多个文件到指定知识库
    
    Args:
    - **files**: 上传文件列表
    - **metadata**: 上传文件元数据，json格式，包含以下字段
    ```
        app_id: int
        domain_id: int
        kb_id: int
        overwrite: bool
        batch_name: str
        batch_id: int | None = None
        biz_metadata: dict | None = None
        created_by: str | None = None
    ```
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        # 解析并验证为表单模型
        metadata_dict = json.loads(metadata)
        form = KBUploadForm(files=files, **metadata_dict)
        
        result, error_msg = await controller.upload_kb_files(form)
        
        if result:
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="文件上传成功"
            )
        else:
            # 如果有具体的错误信息，使用它；否则使用默认消息
            message = error_msg if error_msg else "文件上传失败"
            
            # 根据错误类型设置适当的状态码
            if error_msg and "知识库" in error_msg and "不存在" in error_msg:
                code = status.HTTP_404_NOT_FOUND
            else:
                code = status.HTTP_400_BAD_REQUEST
                
            return ErrorResponse(
                code=code,
                success=False,
                message=message
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
    summary="从指定的知识库中删除文件或所有文件以及其知识库或批次",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_delete_files(
    form: KBDeleteForm = Body(...)
) -> SuccessResponse | ErrorResponse:
    """
    从指定的知识库中删除文件或所有文件以及其知识库或批次
    
    Args:
    - **form**: 删除文件元数据，json格式，包含以下字段
    ```
        app_id: int
        domain_id: int
        kb_id: int
        batch_id: int | None = None
        file_id: str | None = None
        delete_batch: bool = False
        delete_kb: bool = False
    ```
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        
        result = await controller.delete_kb_files(form)

        if result["failed_file_cnt"] == 0 and result["meta_cnt"] > 0:
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="删除文件成功"
            )
        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
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
    summary="从知识库中下载文件",
    response_model=None,
    # dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_download_file(
    file_id: str
) -> FileResponse | ErrorResponse:
    """
    从知识库中下载文件
    
    Args:
    - **file_id**: 文件ID
    
    Returns:
    - **FileResponse**: 文件响应
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        result = await controller.get_kb_files(file_id, download=True)
        
        if result:
            return FileResponse(
                path=result["file_path"],
                filename=urllib.parse.quote(result["file_name"], encoding='utf-8'),
                media_type="multipart/form-data",
                headers={
                    "Content-Disposition": "attachment; filename*=UTF-8''{}".format(urllib.parse.quote(result["file_name"], encoding='utf-8'))
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
    summary="从知识库中预览文件",
    response_model=None,
    # dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_preview_file(
    file_id: str,
    page_num: int = 0
) -> HTMLResponse | FileResponse | ErrorResponse:
    """
    从知识库中预览文件
    
    Args:
    - **file_id**: 文件ID
    - **page_num**: 页码
    
    Returns:
    - **HTMLResponse**: HTML响应
    - **FileResponse**: 图片响应
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        result = await controller.get_kb_files(file_id, download=False, page_num=page_num)
        
        if result:
            if result["file_ext"] == ".txt":
                encoding = result.get("encoding", "utf-8")
                with open(result["file_path"], 'r', encoding=encoding) as f:
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
                        filename=urllib.parse.quote(result["file_name"], encoding='utf-8'),
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
    summary="重新解析文件",
    response_model=SuccessResponse,
    dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_reparse_files(
    form: KBReparseForm = Body(...)
) -> SuccessResponse | ErrorResponse:
    """
    重新解析文件
    
    Args:
    - **form**: 重解析文件元数据
    ```
        kb_id: int
        file_ids: list[str]
    ```
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        # metadata_dict = json.loads(metadata)
        # form = KBReparseForm(**metadata_dict)

        result = await controller.reparse_kb_files(form)
        if result:
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="重解析文件成功"
                )
        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
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
    "/file/preview/v1",
    summary="从知识库中预览文件 v1",
    response_model=None,
    # dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_preview_kb_file_v1(
    form: KBFilePreviewForm
):
    """
    从知识库中预览文件
    
    Args:
    - **form**: 文件预览元数据
    ```
        file_id: str = Field(..., description="文件ID")
        max_length: int | None = Field(None, description="最大长度")
        pages: int | list[int] | None = Field(None, description="页数")
        sheet_index: int | None = Field(None, description="Sheet索引")
        preview_rows: int | None = Field(None, description="预览行数")
        slide: int | None = Field(None, description="幻灯片页码")
    ```
    
    Returns:
    - **dict**: dict响应，包含文件预览信息
    ```
        {
            "file_id": "文件ID",
            "file_name": "文件名称",
            "mime_type": "mime类型",
            "file_size": "文件大小",
            "success": True,
            "preview_type": "预览文件的类型",
            "content": "base64编码的文件内容",
            "total_pages": "总页数",
            "extracted_pages": "提取的页数",
            "page_count": "提取的页码",
            "message": "预览信息"
        }
    ```

    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        kwargs = {
            "file_id": form.file_id,
            "max_length": form.max_length,
            "pages": form.pages,
            "sheet_index": form.sheet_index,
            "preview_rows": form.preview_rows,
            "slide": form.slide
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        
        return await controller.preview_kb_file(**kwargs)
         
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
    
@router.get(
    "/file/preview/v2",
    summary="在浏览器中直接预览文件 v2",
    response_model=None,
    # dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_preview_kb_file_v2(
    file_id: str
) -> FileResponse | ErrorResponse:
    """
    在浏览器中直接预览文件
    
    Args:
    - **file_id**: str = Field(..., description="文件ID")
    
    Returns:
    - **FileResponse**: 文件响应
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        result = await controller.get_kb_files(file_id, download=True)
        
        if result:
            # 获取文件扩展名以确定内容类型
            file_extension = result["file_ext"].lower()
            
            # 常见文件类型的媒体类型映射
            content_types = {
                '.pdf': 'application/pdf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.bmp': 'image/bmp',
                '.txt': 'text/plain',
                '.html': 'text/html',
                '.htm': 'text/html',
                '.css': 'text/css',
                '.js': 'application/javascript',
                '.json': 'application/json',
                '.xml': 'application/xml',
                '.csv': 'text/csv',
                '.mp4': 'video/mp4',
                '.mp3': 'audio/mpeg',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
                '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                '.ppt': 'application/vnd.ms-powerpoint'
            }
            
            # 设置默认内容类型为二进制流
            media_type = content_types.get(file_extension, 'application/octet-stream')
            
            file_path = result["file_path"]
            filename = result["file_name"]
            encoding = result.get("encoding", "utf-8")
            
            # 对于文本文件，检测实际编码
            headers = {}
            encoded_filename = urllib.parse.quote(filename, encoding=encoding)
            
            # 如果是文本文件，根据编码设置字符集
            if file_extension in ['.txt', '.csv', '.html', '.htm', '.css', '.js', '.json', '.xml']:
                media_type = f"{media_type}; charset={encoding}"
                    
            headers["Content-Disposition"] = f"inline; filename*=UTF-8''{encoded_filename}"

            return FileResponse(
                path=file_path,
                filename=filename,
                media_type=media_type,
                headers=headers
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
    "/file/chunk",
    summary="更改或删除知识库文件的分片内容",
    response_model=SuccessResponse | ErrorResponse,
    dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_edit_file_chunk(
    form: KBFileChunkEditForm
) -> SuccessResponse | ErrorResponse:
    """
    更改或删除知识库文件的分片内容
    
    Args:
    - **form**: 文件分片编辑元数据
    ```
        kb_id: int = Field(..., description="知识库ID")
        file_id: str = Field(..., description="文件ID")
        embed_id: str = Field(..., description="分片ID")
        new_chunk: str | None = Field(None, description="新分片内容")
        action: str = Field(..., description="操作类型")
    ```
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        if form.action == "update":
            if form.new_chunk is None or form.new_chunk.strip() == "":
                return ErrorResponse(
                    code=status.HTTP_400_BAD_REQUEST,
                    success=False,
                    message="更新操作需要提供新的分片内容"
                )
            result = await controller.edit_kb_file_chunk(
                kb_id=form.kb_id,
                file_id=form.file_id,
                embed_id=form.embed_id,
                new_chunk=form.new_chunk
            )
        elif form.action == "delete":
            result = await controller.delete_kb_file_chunk(
                kb_id=form.kb_id,
                file_id=form.file_id,
                embed_id=form.embed_id
            )
            
        elif form.action == "enable":
            result = await controller.toogle_kb_file_chunk_status(
                kb_id=form.kb_id,
                chunk_id=form.embed_id,
                status=1
            )
        elif form.action == "disable":
            result = await controller.toogle_kb_file_chunk_status(
                kb_id=form.kb_id,
                chunk_id=form.embed_id,
                status=0
            )

        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                success=False,
                message="无效的操作类型，仅支持 'update', 'delete', 'enable' 和 'disable' 四种操作"
            )
        if result:
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="切换文件分片状态成功"
                )
        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                success=False,
                message="切换文件分片状态失败"
            )
            
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
    
@router.get(
    "/file/get_chunks",
    summary="根据文件ID获取文件的分片内容",
    response_model=SuccessQueryResponse | ErrorResponse,
    dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_get_file_chunks(
    kb_id: int,
    file_id: str
) -> SuccessQueryResponse | ErrorResponse:
    """
    根据文件ID获取文件的分片内容
    
    Args:
    - **kb_id**: int = Field(..., description="知识库ID")
    - **file_id**: str = Field(..., description="文件ID")
    
    Returns:
    - **SuccessQueryResponse**: 成功查询模型参数
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
        data: dict | list[dict] = Field(..., description="响应返回的数据")
    ```
    - **data**: 模型参数
    ```
        {
            embed_id: str = Field(..., description="分片ID")
            kb_id: int = Field(..., description="知识库ID")
            file_id: str = Field(..., description="文件ID")
            chunk_doc: str = Field(..., description="分片内容")
            chunk_metadata: str = Field(..., description="分片元数据")
            biz_metadata: str = Field(..., description="业务元数据")
            embedding = [], # embedding 不返回，防止接口数据过大
            security_level: int = Field(..., description="安全级别")
            status: int = Field(..., description="状态")
        }
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        result = await controller.get_kb_file_chunk_by_id(
            kb_id=kb_id,
            file_id=file_id
        )
        if result:
            return SuccessQueryResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="获取文件分片成功",
                data=result
            )
        else:
            return ErrorResponse(
                code=status.HTTP_404_NOT_FOUND,
                success=False,
                message="未找到文件分片"
            )
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
    
@router.post(
    "/file/chunk/update_description",
    summary="更新知识库文件的分片描述",
    response_model=SuccessResponse | ErrorResponse,
    dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_update_chunk_description(
    form: KBFileChunkUpdateDescriptionForm
) -> SuccessResponse | ErrorResponse:
    """
    更新知识库文件的分片描述
    
    Args:
    - **kb_id**: int = Field(..., description="知识库ID")
    - **embed_id**: str = Field(..., description="分片ID")
    - **description**: str = Field(..., description="分片描述")
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        result = await controller.update_kb_file_chunk_description(
            kb_id=form.kb_id,
            embed_id=form.embed_id,
            description=form.description
        )
        if result:
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="更新文件分片描述成功"
            )
        else:
            return ErrorResponse(
                code=status.HTTP_404_NOT_FOUND,
                success=False,
                message="未找到文件分片"
            )
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )

@router.post(
    "/file/chunk/update_tags",
    summary="更新知识库文件的分片标签",
    response_model=SuccessResponse | ErrorResponse,
    dependencies=[Depends(AuthController.get_current_accessor)],
    status_code=status.HTTP_200_OK
)
async def handle_update_chunk_tags(
    form: KBFileChunkUpdateTagsForm
) -> SuccessResponse | ErrorResponse:
    """
    更新知识库文件的分片标签
    
    Args:
    - **kb_id**: int = Field(..., description="知识库ID")
    - **file_id**: str = Field(..., description="文件ID")
    - **tags**: list[str] = Field(..., description="文件分片标签")
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    try:
        result = await controller.update_kb_file_chunk_tags(
            kb_id=form.kb_id,
            file_id=form.file_id,
            tags=form.tags
        )
        if result:
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="更新文件分片标签成功"
            )
        else:
            return ErrorResponse(
                code=status.HTTP_404_NOT_FOUND,
                success=False,
                message="未找到文件分片"
            )
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message=f"服务器内部错误: {str(e)}"
        )
