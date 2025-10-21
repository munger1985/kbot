from fastapi import APIRouter, Depends, status
from api.controllers.model_controller import ModelController
from api.controllers.security_controller import AuthController
from api.schemas.model_schema import *
from api.schemas.base_response import *
from core.dictionary import ModelCategory

router = APIRouter(prefix="/model", tags=["Model Management"])
   
@router.post(
        "/toggle",
        summary="启用/禁用指定模型",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_enable_model(form: ToggleModelForm) -> SuccessResponse | ErrorResponse:
    """
    启用/禁用指定模型
    
    Args:
    - **form**: 启用或禁用模型请求表单
    ```
        model_id: int = Field(..., description="模型ID")
        switch: int = Field(..., description="开关状态, 1: 启用, 0: 禁用")
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
    controller = ModelController()
    enable = True if form.switch == 1 else False
    result = await controller.toggle(form.model_id, enable=enable)
    if result:
        return SuccessResponse(
            code=200,
            success=True,
            message="操作成功"
        )
    else:
        return ErrorResponse(
            code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            success=False,
            message="操作失败"
        )
    
    
    
@router.post(
        "/params",
        summary="从数据库中获取指定模型的参数",
        # dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_get_model_params(form: ModelForm) -> SuccessQueryResponse | ErrorResponse:
    """
    获取指定模型的参数
    
    Args:
    - **form**: 模型参数请求表单
    ```
        model_id: int = Field(..., description="模型ID")
    ```
    
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
            "model_id": int,
            "model_name": "string",
            "display_name": "string",
            "category": int,
            "provider": "string",
            "model_params": "dict",
            "api_endpoint": "string",
            "api_key": "string"   
        }
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    controller = ModelController()
    model = await controller.get_model_by_id(form.model_id)
    if model:
        return SuccessQueryResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="模型参数获取成功",
            data=model
        )
    else:
        return ErrorResponse(
            code=status.HTTP_404_NOT_FOUND,
            success=False,
            message="模型 {form.model_id} 未找到"
        )

@router.get(
        "/available",
        summary="从数据库中获取指定类别的可用模型",
        # dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_get_all_model_params(model_category: int) -> SuccessQueryResponse | ErrorResponse:
    """
    获取数据库中所有可用模型
    
    Args:
    - **model_category**: int = Field(..., description="模型类别")
    
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
        [
            {
                "model_id": int,
                "model_name": "string",
                "display_name": "string",
                "category": int,
                "provider": "string",
                "model_params": "dict",
                "api_endpoint": "string",
                "api_key": "string"   
            }
        ]
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
    controller = ModelController()
    models = await controller.get_all_available_models(model_category)
    if models:
        return SuccessQueryResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="模型参数获取成功",
            data=list(models)
        )
    else:
        return ErrorResponse(
            code=status.HTTP_404_NOT_FOUND,
            success=False,
            message=f"未找到可用的 {ModelCategory(model_category)} 模型"
        )

@router.post(
        "/test",
        summary="测试指定模型是否可用",
        dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_test_model(form: TestModelForm) -> SuccessResponse | ErrorResponse:
    """测试指定模型是否可用
    Args:
    - **form**: 测试模型请求表单
    ```
        model_id: int = Field(..., description="模型ID")
        model_category: int = Field(..., description="模型类别")
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

    controller = ModelController()
    if await controller.verify_model(form.model_id, form.model_category):
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="模型可用"
        )
    else:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message="模型不可用"
        )