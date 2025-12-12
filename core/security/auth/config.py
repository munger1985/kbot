# core/security/auth/config.py
import os
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# 环境变量配置
SECRET_KEY = os.getenv("KBOT_AUTH_ENCRYPTION_KEY")
# 确保SECRET_KEY存在
if not SECRET_KEY:
    raise ValueError("KBOT_AUTH_ENCRYPTION_KEY environment variable is required")

SERVICE_SECRET_KEY = os.getenv("KBOT_SERVICE_SECRET_KEY", SECRET_KEY + "_service")
ALGORITHM = "HS256"

# Token过期时间（单位：秒）
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("KBOT_AUTH_EXPIRE_MINUTES") or 30)
SERVICE_TOKEN_EXPIRE_DAYS = int(os.getenv("KBOT_SERVICE_EXPIRE_DAYS") or 365)

# 转换时间单位为秒
ACCESS_TOKEN_EXPIRE_SECONDS = ACCESS_TOKEN_EXPIRE_MINUTES * 60
SERVICE_TOKEN_EXPIRE_SECONDS = SERVICE_TOKEN_EXPIRE_DAYS * 24 * 3600

# Token类型
TokenType = Literal["access", "refresh", "service"]