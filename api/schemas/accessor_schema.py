from pydantic import BaseModel, Field

class AccessorForm(BaseModel):
    """API安全访问者表单模型"""
    app_id: int = Field(..., description="应用ID")
    accessor: str = Field(..., description="访问者")
    accessor_type: int = Field(..., description="访问者类型")
    plain_password: str = Field(..., description="明文密码")
    status: int = Field(0, description="状态")
    descs: str|None = Field(None, description="描述")
    by: str|None = Field(None, description="创建人")

class ChangePasswordForm(BaseModel):
    """修改密码表单模型"""
    username: str = Field(..., description="用户名")
    old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., description="新密码")

    