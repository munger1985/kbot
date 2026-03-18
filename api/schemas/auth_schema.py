from pydantic import BaseModel, Field

class UserRegisterRequest(BaseModel):
    username: str = Field(..., description="用户名")
    email: str = Field(..., description="邮箱")
    password: str = Field(..., description="密码")


class LoginResponse(BaseModel):
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(..., description="令牌类型")
    user_id: int = Field(..., description="用户ID")
    username: str = Field(..., description="用户名")
    expires_in: int = Field(..., description="过期时间（秒）")

class ChangePasswordRequest(BaseModel):
    username: str = Field(..., description="用户名")
    # old_password: str = Field(..., description="旧密码")
    new_password: str = Field(..., description="新密码")


class CreateAPIKeyRequest(BaseModel):
    service_id: int = Field(..., description="服务ID")
    name: str = Field(..., description="API密钥名称")
    scopes: list[str] | None = Field(None, description="权限范围")
    expires_days: int | None = Field(None, description="过期时间（天）")
    allowed_ips: list[str] | None = Field(None, description="允许IP列表")
    rate_limit: int = Field(0, description="速率限制（次/秒）")
    created_by: str = Field(..., description="创建者用户名")

class CreateServiceRequest(BaseModel):
    service_code: str = Field(..., description="服务代码")
    name: str = Field(..., description="服务名称")
    service_type: str = Field("internal", description="服务类型")
    description: str | None = Field(None, description="服务描述")
    owner: str | None = Field(None, description="服务所有者")
    contact_email: str | None = Field(None, description="联系邮箱")
