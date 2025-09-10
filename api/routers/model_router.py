from fastapi import APIRouter, Depends, status
from api.controllers.model_controller import ModelController
from api.controllers.security_controller import AuthController
from api.schemas.model_schema import *
from api.schemas.base_response import SuccessResponse, ErrorResponse

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
    result = await controller.toggle(form.model_unique_name, enable=form.enable)
    if result:
        return SuccessResponse(
            code=200,
            success=True,
            message="模型 {form.model_unique_name} 操作成功"
        )
    else:
        return ErrorResponse(
            code=500,
            success=False,
            message="模型 {form.model_unique_name} 操作失败"
        )
    
    
    
@router.post(
        "/params",
        description="从数据库中获取指定模型的参数",
        # dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_get_model_params(form: ModelForm):
    """
    获取Redis中所有可用模型
    
    Returns:
        list[dict]: 可用模型列表
    """
    controller = ModelController()
    model = await controller.get_model_params_by_uname(form.model_unique_name)
    if model:
        return model
    else:
        return ErrorResponse(
            code=404,
            success=False,
            message="模型 {form.model_unique_name} 未找到"
        )

@router.post(
        "/available",
        description="从数据库中获取指定类别的可用模型",
        # dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_get_all_model_params(form: AvailableModelForm):
    """
    获取Redis中所有可用模型
    
    Returns:
        list[dict]: 可用模型列表
    """
    controller = ModelController()
    models = await controller.get_all_available_models(form.model_category)
    if models:
        return models
    else:
        return ErrorResponse(
            code=404,
            success=False,
            message="未找到 {form.model_category} 类型的可用模型"
        )
    