from pydantic import BaseModel, Field

class ModelForm(BaseModel):
    """获取模型参数请求表单"""
    model_id: int = Field(..., description="模型ID")
    model_category: int = Field(..., description="模型类别")