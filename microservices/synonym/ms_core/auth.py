import os
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from jose import JWTError, jwt, ExpiredSignatureError
from passlib.context import CryptContext
from .dictionary import AccessorType



load_dotenv()
SECRET_KEY = os.getenv("KBOT_API_AUTH")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("KBOT_AUTH_EXPIRE_MINUTES")) or 30 # type: ignore

# 密码哈希上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码是否正确"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str):
    return pwd_context.hash(password)

def decode_token(token: str) -> dict:
    """解码JWT令牌"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]) # type: ignore
        if not payload.get("sub"):
            raise ValueError("Missing required field: sub")
        return payload
    except ExpiredSignatureError:
        raise ValueError("Token expired")
    except JWTError as e:
        raise ValueError(f"Invalid token: {str(e)}")


def create_access_token(accessor: str, accessor_type: int) -> str:
    """创建JWT令牌"""
    to_encode = {
        "sub": accessor,
        "type": "service" if accessor_type == AccessorType.SERVICE.value else "user"
    }
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode["exp"] = int(expire.timestamp()) # type: ignore
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM) # type: ignore

