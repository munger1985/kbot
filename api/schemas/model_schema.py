from pydantic import BaseModel, Field

class ModelForm(BaseModel):
    """获取模型参数请求表单"""
    model_id: int = Field(..., description="模型ID")

class ToggleModelForm(ModelForm):
    """启用或禁用模型请求表单"""
    switch: int = Field(..., description="开关状态, 1: 启用, 0: 禁用")

class TestModelForm(ModelForm):
    """测试模型可用性请求表单"""
    model_category: int = Field(..., description="模型类别")