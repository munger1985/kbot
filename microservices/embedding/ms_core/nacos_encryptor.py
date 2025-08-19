import os
import base64
from cryptography.fernet import Fernet

# 从环境变量获取加密密钥（必须32字符）
ENCRYPTION_KEY = os.getenv('NACOS_ENCRYPTION_KEY')
# if not ENCRYPTION_KEY or len(ENCRYPTION_KEY) < 32:
#     raise ValueError("NACOS_ENCRYPTION_KEY 环境变量未设置或长度不足32字符")
if not ENCRYPTION_KEY:
    raise ValueError("NACOS_ENCRYPTION_KEY 环境变量未设置")

class ConfigEncryptor:
    """配置加密工具类"""
    _fernet = None

    @classmethod
    def init_cipher(cls):
        """初始化加密器（必须调用）"""
        key = base64.urlsafe_b64encode(ENCRYPTION_KEY.encode()[:32].ljust(32, b'\0')) # type: ignore
        cls._fernet = Fernet(key)

    @classmethod
    def encrypt(cls, plaintext: str) -> str:
        """加密字符串"""
        if not cls._fernet:
            cls.init_cipher()
        return cls._fernet.encrypt(plaintext.encode()).decode() # type: ignore

    @classmethod
    def decrypt(cls, ciphertext: str) -> str:
        """解密字符串"""
        if not cls._fernet:
            cls.init_cipher()
        return cls._fernet.decrypt(ciphertext.encode()).decode() # type: ignore