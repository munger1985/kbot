from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from core.security.auth import *
from api.controllers.security_controller import AuthController
from api.schemas.accessor_schema import *
from api.schemas.base_response import SuccessResponse, ErrorResponse, SuccessQueryResponse

router = APIRouter(
    prefix="/security",
    tags=["API Security"]
)

@router.post(
        "/get_token",
        description="获取JWT令牌的端点"
)
async def handle_login_for_access_token(form: OAuth2PasswordRequestForm = Depends()):
    """获取JWT令牌的端点"""
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
        description="创建访问者",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_create_accessor(form: AccessorForm):
    """创建访问者"""
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
        description="修改密码",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_change_password(form: ChangePasswordForm):
    """访问者修改密码"""
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
