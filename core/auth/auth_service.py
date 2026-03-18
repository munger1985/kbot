import secrets
import json
import uuid
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt, ExpiredSignatureError
from passlib.context import CryptContext
from core.dictionary import APIKeyStatus, UserTokenStatus
from dao.repositories import UserRepository, UserTokenRepository, ServiceRepository, APIKeyRepository

# 配置
API_KEY_PREFIX = "sk_"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
api_key_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class PasswordService:
    """密码服务"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)


class JWTService:
    """JWT服务"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def decode_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            if not payload.get("sub"):
                raise ValueError("Missing required field: sub")
            return payload
        except ExpiredSignatureError:
            raise ValueError("Token expired")
        except JWTError as e:
            raise ValueError(f"Invalid token: {str(e)}")
    
    def create_access_token(
        self,
        data: dict,
        expires_delta: timedelta | None = None,
        expire_minutes: int = 30
    ) -> str:
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "jti": str(uuid.uuid4())
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)


class APIKeyService:
    """API Key服务"""
    
    @staticmethod
    def generate_api_key() -> tuple[str, str]:
        """生成API Key，返回(key_id, full_key)"""
        key_id = str(uuid.uuid4()).replace("-", "")[:20]
        secret_part = secrets.token_urlsafe(32)
        full_key = f"{API_KEY_PREFIX}{key_id}.{secret_part}"
        return key_id, full_key
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        return api_key_context.hash(api_key)
    
    @staticmethod
    def verify_api_key(plain_key: str, hashed_key: str) -> bool:
        return api_key_context.verify(plain_key, hashed_key)
    
    @staticmethod
    def extract_key_id(api_key: str) -> str | None:
        if not api_key.startswith(API_KEY_PREFIX):
            return None
        
        parts = api_key[len(API_KEY_PREFIX):].split(".")
        if len(parts) != 2:
            return None
        
        return parts[0]


class UserAuthService:
    """用户认证服务"""
    
    def __init__(self, jwt_service: JWTService, access_token_expire_minutes: int = 30):
        self.jwt_service = jwt_service
        self.access_token_expire_minutes = access_token_expire_minutes
    
    async def authenticate_user(self, username: str, password: str) -> dict | None:
        """用户认证"""
        # 查找用户
        user = await UserRepository.get_by_username(username)
        if not user:
            # 尝试邮箱登录
            user = await UserRepository.get_by_email(username)
        
        if not user or not user.is_active:
            return None
        
        # 验证密码
        if not PasswordService.verify_password(password, user.hashed_password):
            return None
        
        # 更新最后登录时间
        await UserRepository.update_last_login(user.id)
        
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_superuser": user.is_superuser
        }
    
    async def create_user_token_pair(
        self,
        user_id: int,
        username: str,
        device_info: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None
    ) -> tuple[str, str, dict]:
        """创建用户访问令牌和刷新令牌"""
        # 访问令牌
        access_payload = {
            "sub": str(user_id),
            "type": "access",
            "username": username
        }
        
        access_token = self.jwt_service.create_access_token(
            access_payload,
            expire_minutes=self.access_token_expire_minutes
        )
        
        # 解码获取jti
        access_payload_decoded = self.jwt_service.decode_token(access_token)
        access_jti = access_payload_decoded["jti"]
        
        # 刷新令牌
        refresh_payload = {
            "sub": str(user_id),
            "type": "refresh"
        }
        
        refresh_token = self.jwt_service.create_access_token(
            refresh_payload,
            expires_delta=timedelta(days=7)
        )
        
        refresh_payload_decoded = self.jwt_service.decode_token(refresh_token)
        refresh_jti = refresh_payload_decoded["jti"]
        
        # 存储令牌记录
        access_expire = datetime.fromtimestamp(
            access_payload_decoded["exp"], 
            timezone.utc
        )
        refresh_expire = datetime.fromtimestamp(
            refresh_payload_decoded["exp"], 
            timezone.utc
        )
        
        await UserTokenRepository.create(
            jti=access_jti,
            user_id=user_id,
            expires_at=access_expire,
            device_info=device_info,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        await UserTokenRepository.create(
            jti=refresh_jti,
            user_id=user_id,
            expires_at=refresh_expire,
            device_info=device_info,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return access_token, refresh_token, access_payload_decoded
    
    async def refresh_token(self, refresh_token: str) -> tuple[str, str] | None:
        """刷新令牌"""
        try:
            payload = self.jwt_service.decode_token(refresh_token)
            if payload.get("type") != "refresh":
                return None
            
            user_id = int(payload["sub"])
            jti = payload["jti"]
            
            # 验证刷新令牌有效性
            if not await UserTokenRepository.is_valid(jti, user_id):
                return None
            
            # 获取用户信息
            user = await UserRepository.get_by_id(user_id)
            if not user or not user.is_active:
                return None
            
            # 撤销旧的刷新令牌
            await UserTokenRepository.revoke(jti=jti, user_id=user_id, reason="refreshed")
            
            # 创建新的令牌对
            access_token, new_refresh_token, _ = await self.create_user_token_pair(
                user_id=user.id,
                username=user.username
            )
            
            return access_token, new_refresh_token
            
        except (ValueError, JWTError):
            return None
    
    async def logout(self, token: str, user_id: int) -> bool:
        """用户登出"""
        try:
            payload = self.jwt_service.decode_token(token)
            if payload.get("type") != "access":
                return False
            
            jti = payload["jti"]
            return await UserTokenRepository.revoke(jti=jti, user_id=user_id, reason="user_logout")
            
        except (ValueError, JWTError):
            return False
    
    async def validate_token(self, token: str) -> dict | None:
        """验证用户令牌"""
        try:
            payload = self.jwt_service.decode_token(token)
            if payload.get("type") != "access":
                return None
            
            jti = payload.get("jti")
            user_id = int(payload["sub"])
            
            # 检查令牌是否有效
            if not await UserTokenRepository.is_valid(jti, user_id): # type: ignore
                return None
            
            # 获取用户信息
            user = await UserRepository.get_by_id(user_id)
            if not user or not user.is_active:
                return None
            
            return {
                "user_id": user_id,
                "username": payload.get("username"),
                "is_superuser": user.is_superuser,
                "scopes": payload.get("scopes", []),
                "payload": payload
            }
            
        except (ValueError, JWTError):
            return None


class ServiceAuthService:
    """服务认证服务"""
    
    def __init__(self, jwt_service: JWTService):
        self.jwt_service = jwt_service
    
    async def create_service_api_key(
        self,
        service_id: int,
        name: str,
        scopes: list[str] | None = None,
        expires_days: int | None = None,
        allowed_ips: list[str] | None = None,
        rate_limit: int = 0,
        created_by: str | None = None
    ) -> tuple[dict, str]:
        """创建服务API Key"""
        # 检查服务是否存在
        service = await ServiceRepository.get_by_id(service_id)
        if not service or not service.is_active:
            raise ValueError("Service not found or inactive")
        
        # 生成API Key
        key_id, full_key = APIKeyService.generate_api_key()
        
        # 计算过期时间
        expires_at = None
        if expires_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=expires_days)
        
        # 创建API Key记录
        api_key_record = await APIKeyRepository.create(
            key_id=key_id,
            hashed_key=APIKeyService.hash_api_key(full_key),
            key_prefix=full_key[:8],
            name=name,
            service_id=service_id,
            scopes=scopes,
            expires_at=expires_at,
            allowed_ips=allowed_ips,
            rate_limit=rate_limit,
            created_by=created_by
        )
        
        api_key_info = {
            "id": api_key_record.id,
            "key_id": api_key_record.key_id,
            "key_prefix": api_key_record.key_prefix,
            "name": api_key_record.name,
            "service_id": service_id,
            "service_code": service.service_code,
            "expires_at": api_key_record.expires_at.isoformat() if api_key_record.expires_at else None,
            "created_at": api_key_record.created_at.isoformat(),
            "scopes": json.loads(api_key_record.scopes),
            "rate_limit": rate_limit
        }
        
        return api_key_info, full_key
    
    async def validate_api_key(
        self,
        api_key: str,
        client_ip: str | None = None,
        required_scopes: list[str] | None = None
    ) -> tuple[bool, dict | None]:
        """验证API Key"""
        # 提取key_id
        key_id = APIKeyService.extract_key_id(api_key)
        if not key_id:
            return False, None
        
        # 获取API Key记录
        api_key_record = await APIKeyRepository.get_by_key_id(key_id)
        if not api_key_record:
            return False, None
        
        # 检查状态
        if api_key_record.status != APIKeyStatus.ACTIVE:
            return False, None
        
        # 检查过期
        if api_key_record.expires_at:
            # 将date类型转换为datetime类型进行比较
            expires_datetime = datetime.combine(api_key_record.expires_at, datetime.min.time(), tzinfo=timezone.utc)
            if expires_datetime < datetime.now(timezone.utc):
                await APIKeyRepository.mark_expired(key_id)
                return False, None
        
        # 验证密钥
        if not APIKeyService.verify_api_key(api_key, api_key_record.hashed_key):
            return False, None
        
        # 检查IP白名单
        if client_ip:
            allowed_ips = json.loads(api_key_record.allowed_ips or "[]")
            if allowed_ips and client_ip not in allowed_ips:
                return False, None
        
        # 检查权限范围
        if required_scopes:
            key_scopes = json.loads(api_key_record.scopes or "[]")
            if not all(scope in key_scopes for scope in required_scopes):
                return False, None
        
        # 更新使用统计
        await APIKeyRepository.update_usage(key_id, client_ip)
        
        # 获取服务信息
        service = api_key_record.service
        
        return True, {
            "key_id": key_id,
            "service_id": service.id,
            "service_code": service.service_code,
            "service_name": service.name,
            "scopes": json.loads(api_key_record.scopes or "[]"),
            "name": api_key_record.name
        }
    
    async def revoke_api_key(
        self, 
        key_id: str, 
        service_id: int | None = None, 
        reason: str | None = None
    ) -> bool:
        """撤销API Key"""
        return await APIKeyRepository.revoke(key_id, reason, service_id)
    
    async def get_service_api_keys(self, service_id: int, active_only: bool = True) -> list[dict]:
        """获取服务的API Keys"""
        api_keys = await APIKeyRepository.list_by_service(service_id, active_only)
        
        result = []
        for api_key in api_keys:
            service = api_key.service
            
            result.append({
                "id": api_key.id,
                "key_id": api_key.key_id,
                "key_prefix": api_key.key_prefix,
                "name": api_key.name,
                "service_id": service_id,
                "service_code": service.service_code,
                "status": api_key.status.value,
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "last_used_at": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                "usage_count": api_key.usage_count,
                "scopes": json.loads(api_key.scopes),
                "rate_limit": api_key.rate_limit,
                "created_at": api_key.created_at.isoformat(),
                "revoked_at": api_key.revoked_at.isoformat() if api_key.revoked_at else None,
                "revoked_reason": api_key.revoked_reason
            })
        
        return result