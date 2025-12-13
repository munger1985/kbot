from typing import Annotated
from fastapi import Depends
from .dependency import get_current_user, require_api_key, require_user_token

# === 用户相关（明确语义） ===
UserAuth = Annotated[dict, Depends(require_user_token())]

# === 服务相关（明确语义） ===
ServiceAuth = Annotated[dict, Depends(require_api_key())]

# === 混合认证（明确语义） ===
AnyAuth = Annotated[dict, Depends(get_current_user())]

# 带特定权限的快捷方式
UserWithAdminScope = Annotated[dict, Depends(get_current_user(required_scopes=["admin"]))]