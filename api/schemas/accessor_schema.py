from pydantic import BaseModel
from datetime import datetime

class AccessorForm(BaseModel):
    """API安全访问者表单模型"""
    app_id: int
    accessor: str
    accessor_type: int
    plain_password: str
    status: int = 0
    descs: str | None = None
    by: str | None = None

class ChangePasswordForm(BaseModel):
    """修改密码表单模型"""
    username: str
    plain_password: str

    