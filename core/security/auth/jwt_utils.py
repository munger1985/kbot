# core/auth/jwt_utils.py
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt, ExpiredSignatureError
from typing import Any
from core.security.auth.config import (
    SECRET_KEY, SERVICE_SECRET_KEY, ALGORITHM,
    ACCESS_TOKEN_EXPIRE_SECONDS, SERVICE_TOKEN_EXPIRE_SECONDS,
    TokenType
)

class JWTUtils:
    @staticmethod
    def create_token(
        subject: str,
        token_type: TokenType = "access",
        additional_claims: dict[str, Any] | None = None,
        expires_in: int | None = None
    ) -> str:
        """创建JWT令牌
        
        Args:
            subject: 主体标识（用户ID或服务名）
            token_type: Token类型
            additional_claims: 附加声明
            expires_in: 过期时间（秒）
        """
        # 选择密钥
        if token_type == "service":
            secret_key = SERVICE_SECRET_KEY
            default_expire = SERVICE_TOKEN_EXPIRE_SECONDS
        else:
            secret_key = SECRET_KEY
            default_expire = ACCESS_TOKEN_EXPIRE_SECONDS
        
        if secret_key is None:
            raise ValueError(f"Secret key for {token_type} token not found")
        
        # 设置过期时间
        expire_delta = expires_in or default_expire
        
        # 基础声明
        to_encode = {
            "sub": subject,
            "token_type": token_type,
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "exp": int((datetime.now(timezone.utc) + timedelta(seconds=expire_delta)).timestamp())
        }
        
        # 添加附加声明
        if additional_claims:
            to_encode.update(additional_claims)
        
        return jwt.encode(to_encode, secret_key, algorithm=ALGORITHM)
    
    @staticmethod
    def decode_token(
        token: str,
        token_type: TokenType = "access",
        audience: str | None = None
    ) -> dict[str, Any]:
        """解码并验证JWT令牌"""
        try:
            # 根据token类型选择密钥
            if token_type == "service":
                secret_key = SERVICE_SECRET_KEY
            else:
                secret_key = SECRET_KEY
            
            # 解码参数
            decode_options = {
                "require_sub": True,
                "require_exp": True,
                "require_iat": True,
                "audience": ""
            }
            
            if audience:
                decode_options["audience"] = audience
            
            if secret_key is None:
                raise ValueError(f"Secret key for {token_type} token not found")
        
            # 解码Token
            payload = jwt.decode(
                token,
                secret_key,
                algorithms=[ALGORITHM],
                options=decode_options
            )
            
            # 验证Token类型
            if payload.get("token_type") != token_type:
                raise ValueError(f"Invalid token type: expected {token_type}")
            
            return payload
            
        except ExpiredSignatureError:
            raise ValueError("Token expired")
        except JWTError as e:
            raise ValueError(f"Invalid token: {str(e)}")
    
    @staticmethod
    def create_service_token(
        service_name: str,
        permissions: list = [],
        metadata: dict = {},
        expires_in: int | None = None
    ) -> str:
        """创建服务间通信的Token"""
        additional_claims = {
            "service_name": service_name,
            "scope": "service_internal",
            "aud": ["internal_services"],
            "iss": "kbot_auth_service",
        }
        
        if permissions:
            additional_claims["permissions"] = permissions
        
        if metadata:
            additional_claims["metadata"] = metadata
        
        return JWTUtils.create_token(
            subject=service_name,
            token_type="service",
            additional_claims=additional_claims,
            expires_in=expires_in
        )