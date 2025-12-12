# api/middleware/service_auth.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
from typing import List, Optional
import re

from core.security.auth.jwt_utils import JWTUtils

class ServiceAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, public_paths: Optional[List[str]] = None):
        super().__init__(app)
        self.public_paths = public_paths or []
        self.public_patterns = [re.compile(path) for path in self.public_paths]
    
    async def dispatch(self, request: Request, call_next):
        # 检查是否为公开路径
        if self._is_public_path(request.url.path):
            return await call_next(request)
        
        # 检查Authorization头
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=401,
                detail="Authorization header required"
            )
        
        # 验证Bearer token
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                raise HTTPException(
                    status_code=401,
                    detail="Invalid authentication scheme"
                )
            
            # 验证Token
            payload = JWTUtils.decode_token(token, "service", audience="internal_services")
            
            # 将服务信息添加到request state
            request.state.service_name = payload.get("service_name")
            request.state.service_permissions = payload.get("permissions", [])
            request.state.service_metadata = payload.get("metadata", {})
            
            return await call_next(request)
            
        except ValueError as e:
            raise HTTPException(
                status_code=401,
                detail=str(e)
            )
        except Exception as e:
            raise HTTPException(
                status_code=401,
                detail="Invalid token"
            )
    
    def _is_public_path(self, path: str) -> bool:
        """检查路径是否为公开路径"""
        for pattern in self.public_patterns:
            if pattern.match(path):
                return True
        return False