from fastapi import status, Request
from fastapi.security import OAuth2PasswordRequestForm
from api.schemas.auth_schema import *
from core.auth import *

class AuthController:

    @staticmethod
    async def register(request: UserRegisterRequest) -> dict[str, str|int|bool]:
        """用户注册"""
        # 检查用户名是否已存在
        existing_user = await UserRepository.get_by_username(request.username)
        if existing_user:
            return {"success": False, "message": "Username already registered", "code": status.HTTP_400_BAD_REQUEST}
        
        # 检查邮箱是否已存在
        existing_email = await UserRepository.get_by_email(request.email)
        if existing_email:
            return {"success": False, "message": "Email already registered", "code": status.HTTP_400_BAD_REQUEST}
        
        # 创建用户
        user = await UserRepository.create(
            username=request.username,
            email=request.email,
            hashed_password=PasswordService.get_password_hash(request.password)
        )
        
        return {
            "success": True,
            "code": status.HTTP_201_CREATED,
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "message": "User registered successfully"
        }

    @staticmethod
    async def login(request: Request, form_data: OAuth2PasswordRequestForm) -> LoginResponse | None:
        """用户登录"""
        # 用户认证
        user_info = await user_auth_service.authenticate_user(
            username=form_data.username,
            password=form_data.password
        )
        
        if not user_info:
            return None
        
        # 创建令牌对
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")
        
        access_token, refresh_token, _ = await user_auth_service.create_user_token_pair(
            user_id=user_info["id"],
            username=user_info["username"],
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            user_id=user_info["id"],
            username=user_info["username"],
            expires_in=user_auth_service.access_token_expire_minutes * 60
        )


    @staticmethod
    async def refresh_token(refresh_token: str) -> dict[str, str|int|bool]:
        """刷新令牌"""
        result = await user_auth_service.refresh_token(refresh_token)
        if not result:
            return {"success": False, "message": "Invalid refresh token", "code": status.HTTP_401_UNAUTHORIZED}
        
        access_token, new_refresh_token = result
        
        return {
            "success": True,
            "code": status.HTTP_200_OK,
            "message": "Token refreshed successfully",
            "access_token": access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }

    @staticmethod
    async def logout(request: Request, auth_info: dict) -> dict[str, str|int|bool]:
        """用户登出"""
        if auth_info["type"] != "user":
            return {"success": False, "message": "Only user tokens can be logged out", "code": status.HTTP_403_FORBIDDEN}
        
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        success = await user_auth_service.logout(token, auth_info["user_id"])
        
        if not success:
            return {"success": False, "message": "Logout failed", "code": status.HTTP_400_BAD_REQUEST}
        
        return {"success": True, "message": "Successfully logged out", "code": status.HTTP_200_OK}

    @staticmethod
    async def create_service_api_key(request_data: CreateAPIKeyRequest, auth_info: dict) -> dict[str, str|int|bool]:
        """创建服务API Key"""
        if auth_info["type"] != "user":
            return {"success": False, "message": "Only users can create API keys", "code": status.HTTP_403_FORBIDDEN}
        
        try:
            api_key_info, full_key = await service_auth_service.create_service_api_key(
                service_id=request_data.service_id,
                name=request_data.name,
                scopes=request_data.scopes,
                expires_days=request_data.expires_days,
                allowed_ips=request_data.allowed_ips,
                rate_limit=request_data.rate_limit,
                created_by=request_data.created_by
            )
            
            return {
                "success": True,
                "code": status.HTTP_201_CREATED,
                **api_key_info,
                "full_key": full_key,  # 仅此次返回
                "message": "Save this key now! It will not be shown again."
            }
            
        except ValueError as e:
            return {"success": False, "message": str(e), "code": status.HTTP_400_BAD_REQUEST}

    @staticmethod
    async def list_service_api_keys(
            service_id: int,
            auth_info: dict,
            active_only: bool = True
        ) -> dict[str, str|int|bool|list[dict]]:

        """获取服务的API Keys列表"""
        # 检查权限
        if auth_info["type"] == "api_key" and auth_info["service_id"] != service_id:
            return {"success": False, "message": "Cannot access other service's API keys", "code": status.HTTP_403_FORBIDDEN}
        keys = await service_auth_service.get_service_api_keys(service_id, active_only)
        return {"success": True, "message": "API Keys retrieved successfully", "code": status.HTTP_200_OK, "keys": keys}

    @staticmethod
    async def revoke_service_api_key(
        key_id: str,
        auth_info: dict,
        reason: str | None = None,
        ) -> dict[str, str|int|bool]:
        """撤销服务API Key"""
        # 用户撤销
        if auth_info["type"] == "user":
            # 这里可以添加权限检查，比如检查用户是否是服务的管理员
            success = await service_auth_service.revoke_api_key(
                key_id=key_id,
                reason=reason
            )
        
        # 服务撤销自己的Key
        elif auth_info["type"] == "api_key":
            success = await service_auth_service.revoke_api_key(
                key_id=key_id,
                service_id=auth_info["service_id"],
                reason=reason
            )
        
        else:
            # 其他类型用户无权限撤销API Key
            return {"success": False, "message": "Permission denied", "code": status.HTTP_403_FORBIDDEN}
        
        if not success:
            return {"success": False, "message": "API Key not found or already revoked", "code": status.HTTP_404_NOT_FOUND}
            
        return {"success": True, "message": "API Key revoked successfully", "code": status.HTTP_200_OK}

    @staticmethod
    async def validate_api_key(api_key: str) -> dict[str, str|int|bool]:
        """验证API Key"""
        is_valid, key_info = await service_auth_service.validate_api_key(api_key)
        
        if not is_valid:
            return {"success": False, "message": "Invalid API Key", "code": status.HTTP_401_UNAUTHORIZED}
            
        
        if not key_info:
            return {"success": False, "message": "API Key not found", "code": status.HTTP_404_NOT_FOUND}
        
        return {
            "success": True,
            "code": status.HTTP_200_OK,
            "message": "API Key is valid",
            "key_id": key_info["key_id"],
            "service_id": key_info["service_id"],
            "service_code": key_info["service_code"],
            "scopes": key_info["scopes"]
        }


    # 服务管理端点
    @staticmethod
    async def create_service(
        service_code: str,
        name: str,
        auth_info: dict,
        service_type: str = "internal",
        description: str | None = None,
        owner: str | None = None,
        contact_email: str | None = None
        ) -> dict[str, str|int|bool]:
        """创建服务（需要管理员权限）"""
        # 检查用户是否为超级用户
        if not auth_info.get("is_superuser", False):
            return {"success": False, "message": "Admin privileges required", "code": status.HTTP_403_FORBIDDEN}
        
        # 检查服务代码是否已存在
        existing_service = await ServiceRepository.get_by_code(service_code)
        if existing_service:
            return {"success": False, "message": "Service code already exists", "code": status.HTTP_400_BAD_REQUEST}
        
        service = await ServiceRepository.create(
            service_code=service_code,
            name=name,
            service_type=service_type,
            description=description,
            owner=owner,
            contact_email=contact_email
        )
        
        return {
            "success": True,
            "code": status.HTTP_201_CREATED,
            "id": service.id,
            "service_code": service.service_code,
            "name": service.name,
            "service_type": service.service_type.value,
            "created_at": service.created_at.isoformat()
        }

    @staticmethod
    async def list_services(auth_info: dict) -> dict[str, str|int|bool|list[dict]]:
        """获取服务列表"""
        services = await ServiceRepository.list_active()
        
        return {
                "success": True,
                "code": status.HTTP_200_OK,
                "services": [
                {
                    "id": service.id,
                    "service_code": service.service_code,
                    "name": service.name,
                    "service_type": service.service_type.value,
                    "description": service.description,
                    "owner": service.owner,
                    "contact_email": service.contact_email,
                    "created_at": service.created_at.isoformat()
                }
                for service in services
            ]
        }

