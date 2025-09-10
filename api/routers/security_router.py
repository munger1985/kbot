from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from core.security.auth import *
from api.controllers.security_controller import AuthController
from api.schemas.accessor_schema import *
from api.schemas.kb_response import SuccessResponse

router = APIRouter(
    prefix="/security",
    tags=["API Security"]
)

@router.post(
        "/get_token",
        description="获取JWT令牌的端点",
        response_model=None,
        status_code=status.HTTP_200_OK
)
async def handle_login_for_access_token(form: OAuth2PasswordRequestForm = Depends()):
    """获取JWT令牌的端点"""
    token = await AuthController.login_for_access_token(
        username=form.username,
        password=form.password
    )
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": token, "token_type": "bearer"}

@router.post(
        "/create_accessor",
        description="创建访问者",
        response_model=SuccessResponse,
        dependencies=[Depends(AuthController.get_current_accessor)],
        status_code=status.HTTP_200_OK
)
async def handle_create_accessor(form: AccessorForm):
    """创建访问者"""
    result = await AuthController.create_accessor(form)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Create accessor failed"
        )
    return SuccessResponse(
        code=200,
        success=True,
        message="Create accessor successfully."
    )

# @router.post(
#         "/change_password",
#         description="修改密码",
#         response_model=SuccessResponse,
#         # dependencies=[Depends(AuthController.get_current_accessor)],
#         status_code=status.HTTP_200_OK
# )
# async def handle_change_password(form: ChangePasswordForm):
#     """创建访问者"""
#     result = await 
#     if not result:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Create accessor failed"
#         )
#     return SuccessResponse(
#         code=200,
#         success=True,
#         message="Create accessor successfully."
#     )
