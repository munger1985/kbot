import os
from typing import Callable
from fastapi import Depends, HTTPException, status, Request
from loguru import logger

from .auth_service import (
    JWTService, 
    UserAuthService, 
    ServiceAuthService
)

from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("KBOT_AUTH_ENCRYPTION_KEY")
if not SECRET_KEY:
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail="KBOT_AUTH_ENCRYPTION_KEY must be set in environment"
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("KBOT_AUTH_EXPIRE_MINUTES", "30"))

# 初始化服务
jwt_service = JWTService(
    secret_key=SECRET_KEY,
    algorithm=ALGORITHM
)

user_auth_service = UserAuthService(
    jwt_service=jwt_service,
    access_token_expire_minutes=ACCESS_TOKEN_EXPIRE_MINUTES
)

service_auth_service = ServiceAuthService(jwt_service=jwt_service)


def is_jwt_token(token: str) -> bool:
    """判断是否为JWT令牌"""
    parts = token.split('.')
    return len(parts) == 3


def get_current_user(
    required_scopes: list[str] | None = None,
    allow_api_key: bool = True
) -> Callable:
    """获取当前用户的依赖工厂函数"""
    
    async def dependency(request: Request) -> dict:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        
        try:
            # Bearer Token 认证
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                
                # 判断是 JWT 还是 API Key
                if is_jwt_token(token):
                    return await _authenticate_user_token(token, request, required_scopes)
                else:
                    if not allow_api_key:
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="API Key not allowed for this endpoint"
                        )
                    return await _authenticate_api_key(token, request, required_scopes)
            
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme"
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    return dependency


async def _authenticate_user_token(
    token: str, 
    request: Request,
    required_scopes: list[str] | None
) -> dict:
    """验证用户令牌"""
    user_info = await user_auth_service.validate_token(token)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )
    
    # 检查权限范围
    user_scopes = user_info.get("scopes", [])
    if required_scopes and not all(scope in user_scopes for scope in required_scopes):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions"
        )
    
    auth_info = {
        "authenticated": True,
        "type": "user",
        "user_id": user_info["user_id"],
        "username": user_info["username"],
        "is_superuser": bool(user_info.get("is_superuser", False)),
        "scopes": user_scopes,
        "payload": user_info.get("payload", {})
    }
    
    # 存储到请求状态
    request.state.auth = auth_info
    request.state.user_id = user_info["user_id"]
    
    return auth_info


async def _authenticate_api_key(
    api_key: str, 
    request: Request,
    required_scopes: list[str] | None
) -> dict:
    """验证API Key"""
    client_ip = request.client.host if request.client else None
    
    is_valid, key_info = await service_auth_service.validate_api_key(
        api_key=api_key,
        client_ip=client_ip,
        required_scopes=required_scopes
    )
    
    if not is_valid or not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )
    
    auth_info = {
        "authenticated": True,
        "type": "api_key",
        "key_id": key_info["key_id"],
        "service_id": key_info["service_id"],
        "service_code": key_info["service_code"],
        "service_name": key_info["service_name"],
        "scopes": key_info["scopes"],
        "key_name": key_info["name"]
    }
    
    # 存储到请求状态
    request.state.auth = auth_info
    request.state.service_id = key_info["service_id"]
    request.state.service_code = key_info["service_code"]
    
    return auth_info


def require_api_key(required_scopes: list[str] | None = None) -> Callable:
    """要求API Key认证的依赖工厂函数"""
    
    async def dependency(request: Request) -> dict:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key required"
            )
        
        try:
            # 只支持 Bearer 头
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                # 验证是否为API Key格式
                if is_jwt_token(token):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="API Key required, not JWT"
                    )
                return await _authenticate_api_key(token, request, required_scopes)
            
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication scheme"
                )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API Key authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key"
            )
    
    return dependency


def require_user_token(required_scopes: list[str] | None = None) -> Callable:
    """要求用户Token认证的依赖工厂函数"""
    
    async def dependency(request: Request) -> dict:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required"
            )
        
        token = auth_header[7:]
        # 验证是否为JWT格式
        if not is_jwt_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User token required, not API Key"
            )
        return await _authenticate_user_token(token, request, required_scopes)
    
    return dependency


def require_superuser(required_scopes: list[str] | None = None) -> Callable:
    """要求超级管理员权限的依赖工厂函数"""
    
    async def dependency(request: Request) -> dict:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Bearer token required"
            )
        
        token = auth_header[7:]
        # 验证是否为JWT格式
        if not is_jwt_token(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User token required, not API Key"
            )
        auth_info = await _authenticate_user_token(token, request, required_scopes)
        
        # 检查是否为超级管理员
        if not auth_info.get("is_superuser", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Superuser privileges required"
            )
        
        return auth_info
    
    return dependency