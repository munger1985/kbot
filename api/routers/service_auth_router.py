# api/routers/service_auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
import hashlib

from core.security.auth.service_registry import ServiceRegistry, AuthMethod, ServiceStatus
from core.security.auth.jwt_utils import JWTUtils
from core.security.auth.config import SERVICE_TOKEN_EXPIRE_SECONDS

router = APIRouter(prefix="/service-auth", tags=["service-auth"])
security = HTTPBearer()

# 全局服务注册表实例
service_registry = ServiceRegistry()

# 请求模型
class ServiceAuthRequest(BaseModel):
    service_name: str
    auth_method: AuthMethod = AuthMethod.PRESHARED
    credentials: Dict = Field(..., description="认证凭证，根据auth_method不同而不同")
    requested_permissions: Optional[List[str]] = []
    metadata: Optional[Dict] = {}
    expires_in: Optional[int] = Field(
        None,
        ge=60,
        le=SERVICE_TOKEN_EXPIRE_SECONDS,
        description="Token过期时间（秒）"
    )

class ServiceRegisterRequest(BaseModel):
    service_name: str
    auth_method: AuthMethod = AuthMethod.PRESHARED
    permissions: Optional[List[str]] = []
    metadata: Optional[Dict] = {}
    public_key: Optional[str] = None

class ServiceRegisterResponse(BaseModel):
    service_name: str
    secret: str
    auth_method: AuthMethod
    created_at: datetime

class ServiceTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    service_name: str
    permissions: List[str]
    expires_in: int
    metadata: Dict

class ServiceInfoResponse(BaseModel):
    name: str
    auth_method: AuthMethod
    permissions: List[str]
    metadata: Dict
    status: ServiceStatus
    created_at: datetime
    last_used_at: Optional[datetime]

# 依赖：验证管理密钥
def verify_admin_key(admin_key: Optional[str] = Header(None, alias="X-Admin-Key")):
    """验证管理密钥"""
    admin_secret = os.getenv("KBOT_ADMIN_SECRET")
    if not admin_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin secret not configured"
        )
    
    if not admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin key required"
        )
    
    # 简单的哈希比较
    expected_hash = hashlib.sha256(admin_secret.encode()).hexdigest()
    provided_hash = hashlib.sha256(admin_key.encode()).hexdigest()
    
    if expected_hash != provided_hash:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin key"
        )

# 注册新服务（需要管理员权限）
@router.post("/register", 
             response_model=ServiceRegisterResponse,
             dependencies=[Depends(verify_admin_key)])
async def register_service(request: ServiceRegisterRequest):
    """注册新服务"""
    try:
        secret = service_registry.register_service(
            name=request.service_name,
            auth_method=request.auth_method,
            permissions=request.permissions or [],
            metadata=request.metadata or {},
            public_key=request.public_key
        )
        
        service_info = service_registry.get_service(request.service_name)
        
        return ServiceRegisterResponse(
            service_name=request.service_name,
            secret=secret,
            auth_method=request.auth_method,
            created_at=service_info.created_at
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

# 获取服务Token
@router.post("/token", response_model=ServiceTokenResponse)
async def get_service_token(request: ServiceAuthRequest):
    """获取服务访问令牌"""
    try:
        # 验证服务凭证
        if not service_registry.authenticate_service(
            request.service_name,
            request.credentials
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid service credentials"
            )
        
        # 获取服务信息
        service_info = service_registry.get_service(request.service_name)
        if not service_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service not found"
            )
        
        # 更新最后使用时间
        service_info.update_last_used()
        service_registry.save_to_storage()
        
        # 确定授予的权限
        allowed_permissions = set(service_info.permissions)
        requested_permissions = set(request.requested_permissions or [])
        
        # 只授予服务已有且请求的权限
        granted_permissions = list(allowed_permissions.intersection(requested_permissions))
        if not requested_permissions:
            granted_permissions = service_info.permissions
        
        # 合并元数据
        metadata = {**service_info.metadata, **(request.metadata or {})}
        metadata["auth_method"] = request.auth_method.value
        
        # 创建服务Token
        token = JWTUtils.create_service_token(
            service_name=request.service_name,
            permissions=granted_permissions,
            metadata=metadata,
            expires_in=request.expires_in
        )
        
        return ServiceTokenResponse(
            access_token=token,
            service_name=request.service_name,
            permissions=granted_permissions,
            expires_in=request.expires_in or SERVICE_TOKEN_EXPIRE_SECONDS,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to issue token: {str(e)}"
        )

# 验证服务Token
@router.post("/verify")
async def verify_service_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    required_permissions: Optional[List[str]] = None
):
    """验证服务Token"""
    token = credentials.credentials
    
    try:
        payload = JWTUtils.decode_token(token, "service", audience="internal_services")
        
        # 验证签发者
        if payload.get("iss") != "kbot_auth_service":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token issuer"
            )
        
        service_name = payload.get("service_name")
        
        # 检查服务状态
        service_info = service_registry.get_service(service_name)
        if not service_info or service_info.status != ServiceStatus.ACTIVE:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Service is not active"
            )
        
        # 验证权限
        if required_permissions:
            token_permissions = payload.get("permissions", [])
            missing_perms = [
                perm for perm in required_permissions 
                if perm not in token_permissions
            ]
            if missing_perms:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Missing permissions: {missing_perms}"
                )
        
        return {
            "valid": True,
            "service_name": service_name,
            "permissions": payload.get("permissions", []),
            "metadata": payload.get("metadata", {})
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )

# 列出所有服务（管理员接口）
@router.get("/services", 
            response_model=List[ServiceInfoResponse],
            dependencies=[Depends(verify_admin_key)])
async def list_services():
    """列出所有注册的服务"""
    return [
        ServiceInfoResponse(
            name=info.name,
            auth_method=info.auth_method,
            permissions=info.permissions,
            metadata=info.metadata,
            status=info.status,
            created_at=info.created_at,
            last_used_at=info.last_used_at
        )
        for info in service_registry.services.values()
    ]

# 更新服务信息（管理员接口）
@router.put("/services/{service_name}",
            dependencies=[Depends(verify_admin_key)])
async def update_service(
    service_name: str,
    permissions: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
    status: Optional[ServiceStatus] = None
):
    """更新服务信息"""
    success = service_registry.update_service(
        name=service_name,
        permissions=permissions,
        metadata=metadata,
        status=status
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Service not found"
        )
    
    return {"message": "Service updated successfully"}

# Token验证依赖（供其他路由使用）
def verify_service_token_dependency(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    required_permissions: List[str] = None
):
    """验证服务Token的依赖注入"""
    return verify_service_token(credentials, required_permissions)