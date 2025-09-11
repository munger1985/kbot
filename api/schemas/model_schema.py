from pydantic import BaseModel

class ModelForm(BaseModel):
    """获取模型参数请求表单"""
    model_unique_name: str

class ToggleModelForm(ModelForm):
    """启用或禁用模型请求表单"""
    enable: bool

class TestModelForm(ModelForm):
    """测试模型可用性请求表单"""
    model_category: int