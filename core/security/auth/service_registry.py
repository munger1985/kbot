# core/auth/service_registry.py
import os
import secrets
import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum

class ServiceStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class AuthMethod(str, Enum):
    PRESHARED = "preshared"
    MTLS = "mtls"
    ASYMMETRIC = "asymmetric"

@dataclass
class ServiceInfo:
    name: str
    secret_hash: str  # 存储哈希值，不存明文
    auth_method: AuthMethod
    permissions: list[str]
    metadata: dict
    status: ServiceStatus
    created_at: datetime
    last_used_at: datetime | None = None
    public_key: str | None = None  # 用于非对称加密
    certificate_thumbprint: str | None = None  # 用于mTLS
    
    @classmethod
    def create(cls, 
               name: str,
               secret: str,
               auth_method: AuthMethod = AuthMethod.PRESHARED,
               permissions: list[str] = [],
               metadata: dict = {},
               public_key: str | None = None) -> 'ServiceInfo':
        """创建服务信息"""
        return cls(
            name=name,
            secret_hash=cls._hash_secret(secret),
            auth_method=auth_method,
            permissions=permissions or [],
            metadata=metadata or {},
            status=ServiceStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            public_key=public_key
        )
    
    @staticmethod
    def _hash_secret(secret: str) -> str:
        """安全地哈希密钥"""
        # 使用salt增加安全性
        salt = os.getenv("KBOT_AUTH_SALT", "default-salt-change-me")
        return hashlib.sha256((secret + salt).encode()).hexdigest()
    
    def verify_secret(self, secret: str) -> bool:
        """验证密钥"""
        return self.secret_hash == self._hash_secret(secret)
    
    def update_last_used(self):
        """更新最后使用时间"""
        self.last_used_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> dict:
        """转换为字典（排除敏感信息）"""
        data = asdict(self)
        # 移除敏感信息
        data.pop("secret_hash", None)
        if "public_key" in data:
            data["public_key"] = "***" if data["public_key"] else None
        return data

class ServiceRegistry:
    def __init__(self, storage_file: str = "services_registry.json"):
        self.services: dict[str, ServiceInfo] = {}
        self.storage_file = storage_file
        self.load_from_storage()
    
    def load_from_storage(self):
        """从存储加载服务信息"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                
                for service_name, service_data in data.items():
                    # 转换字符串时间回datetime对象
                    service_data['created_at'] = datetime.fromisoformat(service_data['created_at'])
                    if service_data.get('last_used_at'):
                        service_data['last_used_at'] = datetime.fromisoformat(service_data['last_used_at'])
                    
                    service_data['status'] = ServiceStatus(service_data['status'])
                    service_data['auth_method'] = AuthMethod(service_data['auth_method'])
                    
                    self.services[service_name] = ServiceInfo(**service_data)
        except Exception as e:
            print(f"Warning: Failed to load service registry: {e}")
    
    def save_to_storage(self):
        """保存服务信息到存储"""
        try:
            data = {}
            for name, info in self.services.items():
                service_dict = info.to_dict()
                # 转换datetime为字符串
                service_dict['created_at'] = info.created_at.isoformat()
                if info.last_used_at:
                    service_dict['last_used_at'] = info.last_used_at.isoformat()
                data[name] = service_dict
            
            with open(self.storage_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error: Failed to save service registry: {e}")
    
    def register_service(self,
                        name: str,
                        auth_method: AuthMethod = AuthMethod.PRESHARED,
                        permissions: list[str] = [],
                        metadata: dict = {},
                        public_key: str | None = None) -> str:
        """注册新服务并返回生成的密钥"""
        if name in self.services:
            raise ValueError(f"Service '{name}' already registered")
        
        # 生成密钥（对于预共享密钥方式）
        if auth_method == AuthMethod.PRESHARED:
            secret = secrets.token_urlsafe(32)
        else:
            secret = secrets.token_urlsafe(16)  # 较短的secret用于其他认证方式
        
        service_info = ServiceInfo.create(
            name=name,
            secret=secret,
            auth_method=auth_method,
            permissions=permissions or [],
            metadata=metadata or {},
            public_key=public_key
        )
        
        self.services[name] = service_info
        self.save_to_storage()
        
        return secret
    
    def authenticate_service(self, 
                           name: str, 
                           credentials: dict) -> bool:
        """验证服务凭证"""
        if name not in self.services:
            return False
        
        service = self.services[name]
        
        if service.status != ServiceStatus.ACTIVE:
            return False
        
        # 根据认证方式验证
        if service.auth_method == AuthMethod.PRESHARED:
            secret = credentials.get("secret")
            if not secret:
                return False
            return service.verify_secret(secret)
        
        elif service.auth_method == AuthMethod.ASYMMETRIC:
            # 这里简化处理，实际需要验证签名
            signature = credentials.get("signature")
            timestamp = credentials.get("timestamp")
            if not signature or not timestamp:
                return False
            # 实际实现中需要验证签名
            return True
        
        return False
    
    def get_service(self, name: str) -> ServiceInfo | None:
        """获取服务信息"""
        return self.services.get(name)
    
    def list_services(self) -> list[dict]:
        """列出所有服务（不含敏感信息）"""
        return [service.to_dict() for service in self.services.values()]
    
    def update_service(self, 
                      name: str, 
                      permissions: list[str] = [],
                      metadata: dict = {},
                      status: ServiceStatus | None = None) -> bool:
        """更新服务信息"""
        if name not in self.services:
            return False
        
        service = self.services[name]
        
        if permissions:
            service.permissions = permissions
        
        if metadata:
            service.metadata.update(metadata)
        
        if status:
            service.status = status
        
        self.save_to_storage()
        return True
    
    def revoke_service(self, name: str) -> bool:
        """吊销服务"""
        if name not in self.services:
            return False
        
        self.services[name].status = ServiceStatus.SUSPENDED
        self.save_to_storage()
        return True