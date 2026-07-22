# platform_core/security/crypto.py

import os
import base64
from loguru import logger
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from platform_core.exceptions import InternalServerError

class CryptoToolkit:
    """
    企业级凭证加解密安全套件 (AES-256-GCM 工业标准)
    """

    def __init__(self):
        # 从系统环境变量中强行锁死 32 字节(256位)的全局主密钥密钥
        # 生产环境建议通过 OCI Vault 或 K8s Secret 注入
        self.secret_key_str = os.getenv("KBOT_AUTH_ENCRYPTION_KEY")
        if not self.secret_key_str:
            raise ValueError("KBOT_AUTH_ENCRYPTION_KEY 环境变量未设置")
        
        # 2. 确保密钥长度为 32 字节(256位)
        logger.info(f"[Crypto] 正在初始化凭证加解密安全套件，密钥长度: {len(self.secret_key_str)}")
        # 补齐或截断至标准的 32 字节
        self.key_bytes = self.secret_key_str.encode("utf-8").ljust(32, b"\0")[:32]
        self.aesgcm = AESGCM(self.key_bytes)

    def encrypt(self, plain_text: str) -> str:
        """
        将明文密码加密为安全的 Base64 复合密文
        """
        if not plain_text:
            return ""
        try:
            # 1. 每次加密必须生成全新的、不可预测的 12 字节初始化向量 (Nonce/IV)
            nonce = os.urandom(12)
            
            # 2. 执行 AES-GCM 加密，底层会自动追加 16 字节的认证标签 (Tag)
            encrypted_bytes = self.aesgcm.encrypt(nonce, plain_text.encode("utf-8"), None)
            
            # 3. 复合打包: 将 IV 和 密文(含Tag) 拼接，整体转换为 Base64 字符串用于持久化
            combined_payload = nonce + encrypted_bytes
            return base64.b64encode(combined_payload).decode("utf-8")
            
        except Exception as e:
            logger.error(f"[Crypto] 凭证物理加密失败: {str(e)}")
            raise InternalServerError("安全资产防线加密失败")

    def decrypt(self, cipher_text_b64: str) -> str:
        """
        将 Base64 复合密文解密还原为明文字符串
        """
        if not cipher_text_b64:
            return ""
        try:
            # 1. Base64 反序列化为物理字节流
            combined_payload = base64.b64decode(cipher_text_b64.encode("utf-8"))
            
            # 2. 边界防呆检查: 12字节IV + 至少16字节Tag
            if len(combined_payload) < 28:
                raise ValueError("密文物理长度不合法，可能已损坏")
                
            # 3. 物理切片：剥离前 12 字节的 IV，剩下的是加密体
            nonce = combined_payload[:12]
            encrypted_bytes = combined_payload[12:]
            
            # 4. 执行解密与数据完整性校验 (只要密文被篡改过哪怕1个bit，此处会直接抛出 InvalidTag 异常)
            decrypted_bytes = self.aesgcm.decrypt(nonce, encrypted_bytes, None)
            return decrypted_bytes.decode("utf-8")
            
        except Exception as e:
            logger.critical(f"[Crypto] 凭证完整性校验失败或密钥不匹配！密文可能遭到非法篡改！")
            raise InternalServerError("资产安全防线拒绝解密: 凭证完整性受损")
