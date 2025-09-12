from fastapi import APIRouter, Depends, status
from api.controllers.model_controller import ModelController
from api.controllers.security_controller import AuthController
from api.schemas.model_schema import *
from api.schemas.base_response import SuccessResponse, ErrorResponse
from core.dictionary import ModelCategory

router = APIRouter(
    prefix="/model",
    tags=["API Security"]
)

   
@router.post(
        "/toggle",
        description="启用/禁用指定模型",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_enable_model(form: ToggleModelForm):
    """
    启用/禁用指定模型
    
    Args:
        form (ToogleModelForm): 启用或禁用模型请求表单
        
    Returns:
        dict: 结果消息
        
    Raises:
        HTTPException: 失败时抛出500错误
    """
    controller = ModelController()
    enable = True if form.switch == 1 else False
    result = await controller.toggle(form.model_id, enable=enable)
    if result:
        return SuccessResponse(
            code=200,
            success=True,
            message="模型 {form.model_id} 操作成功"
        )
    else:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message="模型 {form.model_id} 操作失败"
        )
    
    
    
@router.post(
        "/params",
        description="从数据库中获取指定模型的参数",
        # dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_get_model_params(form: ModelForm):
    """
    获取指定模型的参数
    
    Returns:
        dict: 指定模型的参数
    """
    controller = ModelController()
    model = await controller.get_model_by_id(form.model_id)
    if model:
        return model
    else:
        return ErrorResponse(
            code=status.HTTP_404_NOT_FOUND,
            success=False,
            message="模型 {form.model_id} 未找到"
        )

@router.get(
        "/available",
        description="从数据库中获取指定类别的可用模型",
        # dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_get_all_model_params(model_category: int):
    """
    获取数据库中所有可用模型
    
    Returns:
        list[dict]: 可用模型列表
    """
    controller = ModelController()
    models = await controller.get_all_available_models(model_category)
    if models:
        return models
    else:
        return ErrorResponse(
            code=status.HTTP_404_NOT_FOUND,
            success=False,
            message=f"未找到可用的 {ModelCategory(model_category)} 模型"
        )

@router.post(
        "/test",
        description="测试指定模型是否可用",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_test_model(form: TestModelForm):
    """测试指定模型是否可用"""

    controller = ModelController()
    if await controller.verify_model(form.model_id, form.model_category):
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="模型 {model_id} 可用"
        )
    else:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message="模型 {model_id} 不可用"
        )