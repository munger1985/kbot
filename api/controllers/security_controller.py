from fastapi.security import OAuth2PasswordBearer
from fastapi import HTTPException, status, Depends
from core.security.auth import *
from api.schemas.accessor_schema import *
from dao.entities.kbot_md_api_security import KbotMdApiSecurity
from dao.repositories.kbot_md_api_security_repo import KbotMdApiSecurityRepository
from loguru import logger

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/security/get_token")

class AuthController:
    """认证控制器"""
    @staticmethod
    async def login_for_access_token(username: str, password: str):
        # 用户登录
        result = await KbotMdApiSecurityRepository().get_hashed_secret(username)
        if not result:
            return None
        if verify_password(password, result.get("hashed_secret")): # type: ignore
            access_token = create_access_token(username, result.get("accessor_type")) # type: ignore
            return access_token
        else:
            return None
        
    @staticmethod
    async def create_accessor(form: AccessorForm) -> bool:
        """创建访问者"""
        security = KbotMdApiSecurity()
        security.app_id = form.app_id
        security.accessor = form.accessor
        security.accessor_type = form.accessor_type
        security.hashed_secret = get_password_hash(form.plain_password)
        security.status = form.status
        security.descs = form.descs
        security.created_by = form.by
        security.updated_by = form.by
        try:
            await KbotMdApiSecurityRepository().create(security)
        except Exception as e:
            logger.exception(f"访问者创建失败: {e}")
            return False
        return True

    @staticmethod
    async def get_current_accessor(authorization: str = Depends(oauth2_scheme)):
        """从JWT令牌中获取当前访问者信息"""
        try:
            payload = decode_token(authorization)
            return payload
        
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e),
                headers={"WWW-Authenticate": "Bearer"}
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
    @staticmethod
    async def change_password(form: ChangePasswordForm) -> bool:
        """访问者修改密码"""
        try:
            # 获取访问者的密码哈希值
            result = await KbotMdApiSecurityRepository().get_hashed_secret(form.username)
            if not result:
                return False
            # 验证旧密码是否正确
            if verify_password(form.old_password, result.get("hashed_secret")): # type: ignore
                # 如果旧密码正确，则更新密码
                hashed_new_secret = get_password_hash(form.plain_password)
                return await KbotMdApiSecurityRepository().change_password(form.username, hashed_new_secret)
            else:
                return False
        except Exception as e:
            logger.exception(f"密码修改失败: {e}")
            return False