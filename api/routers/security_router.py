from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from core.security.auth import *
from api.controllers.security_controller import AuthController
from api.schemas.accessor_schema import *
from api.schemas.base_response import SuccessResponse, ErrorResponse, SuccessQueryResponse

router = APIRouter(prefix="/security", tags=["API Security"])

@router.post(
        "/get_token",
        summary="获取JWT令牌"
)
async def handle_login_for_access_token(form: OAuth2PasswordRequestForm = Depends()) -> SuccessQueryResponse | ErrorResponse:
    """获取JWT令牌
    Args:
    - **form**: 登录请求表单
    ```
        username: str = Field(..., description="用户名")
        password: str = Field(..., description="密码")
    ```
    Returns:
    - **SuccessQueryResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
        data: dict = Field(..., description="响应返回的数据")
    ```
    - **data**: 登录成功响应数据
    ```
        {
            "access_token": "string",
            "token_type": "bearer"
        }
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    token = await AuthController.login_for_access_token(
        username=form.username,
        password=form.password
    )
    if token:
        return SuccessQueryResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="登录成功",
            data={"access_token": token, "token_type": "bearer"}
        )
    else:
        return ErrorResponse(
            code=status.HTTP_401_UNAUTHORIZED,
            success=False,
            message="登录失败"
        )

@router.post(
        "/create_accessor",
        summary="创建访问者",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_create_accessor(form: AccessorForm) -> SuccessResponse | ErrorResponse:
    """创建访问者
    Args:
    - **form**: 访问者创建表单
    ```
        app_id: int = Field(..., description="应用ID")
        accessor: str = Field(..., description="访问者")
        accessor_type: int = Field(..., description="访问者类型")
        plain_password: str = Field(..., description="明文密码")
        status: int = Field(0, description="状态")
        descs: str | None = Field(None, description="描述")
        by: str | None = Field(None, description="创建人")
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
    if await AuthController.create_accessor(form):
        return SuccessResponse(
        code=status.HTTP_200_OK,
        success=True,
        message="创建访问者成功"
    )
    else:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message="创建访问者失败"
        )
    

@router.post(
        "/change_password",
        summary="修改密码",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_change_password(form: ChangePasswordForm) -> SuccessResponse | ErrorResponse:
    """访问者修改密码
    Args:
    - **form**: 访问者修改密码表单
    ```
        accessor: str = Field(..., description="访问者")
        plain_password: str = Field(..., description="明文密码")
        new_plain_password: str = Field(..., description="新明文密码")
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
    if await AuthController.change_password(form):      
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="修改密码成功"
        )
    else:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message="修改密码失败"
        )
