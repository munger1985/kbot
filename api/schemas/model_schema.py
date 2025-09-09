from pydantic import BaseModel

class ModelForm(BaseModel):
    """获取模型参数请求表单"""
    model_unique_name: str

class ToggleModelForm(BaseModel):
    """启用或禁用模型请求表单"""
    model_unique_name: str
    enable: bool

class AvailableModelForm(BaseModel):
    """按类别获取可用模型请求表单"""
    model_category: int