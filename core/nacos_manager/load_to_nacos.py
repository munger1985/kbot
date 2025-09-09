import os
import json
import requests

from typing import Any
from pydantic import BaseModel, SecretStr
from nacos_encryptor import ConfigEncryptor
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Nacos 配置中心地址
NACOS_SERVER = os.getenv('NACOS_SERVER', 'http://localhost:8848')
NAMESPACE = os.getenv('NACOS_NAMESPACE', 'public')
GROUP = os.getenv('NACOS_GROUP', 'dev')


class SecureDBConfig(BaseModel):
    """数据库安全配置模型"""
    host: str
    port: int
    username: str
    password: SecretStr  # 敏感字段自动加密
    service_name: str

    @property
    def encrypted_config(self) -> dict[str, Any]:
        """返回加密后的配置字典（用于Nacos存储）"""
        return {
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "encrypted_password": ConfigEncryptor.encrypt(self.password.get_secret_value()),
            "service_name": self.service_name
        }
    
def process_sensitive_fields(config_data: dict[str, Any]) -> dict[str, Any]:
    """
    处理配置中的敏感字段
    返回适用于Nacos的安全配置（密码字段被加密）
    """
    # 初始化加密器
    ConfigEncryptor.init_cipher()
    
    # 需要加密的字段路径（支持嵌套）
    sensitive_paths = [
        'database.oracle.password',
        'database.redis.password',
        'api_keys.*'  # 所有api_keys下的字段
    ]

    def encrypt_fields(data: dict[str, Any], path: str = '') -> dict[str, Any]:
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # 如果是字典则递归处理
            if isinstance(value, dict):
                data[key] = encrypt_fields(value, current_path)
            # 检查是否需要加密
            elif any(p in current_path for p in sensitive_paths):
                data[key] = ConfigEncryptor.encrypt(str(value))
        return data

    return encrypt_fields(config_data.copy())

def validate_json(content: str) -> dict[str, Any]:
    """验证JSON格式并返回解析后的字典"""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}")

def load_config_to_nacos():
    """加载项目根目录下configuration目录下的所有JSON配置文件到Nacos（自动加密敏感字段）"""
    try:
        # 添加项目根目录到 Python 路径，确保可以导入配置文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(os.path.dirname(current_dir))
        config_dir = os.path.join(project_dir, 'configuration')
        json_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]

        if not json_files:
            print("未找到任何 .json 文件")
            return

        for file in json_files:
            file_path = os.path.join(config_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            try:
                config_data = validate_json(content)
                # 处理敏感字段加密
                safe_config = process_sensitive_fields(config_data)
                safe_content = json.dumps(safe_config, indent=2)
            except ValueError as e:
                print(f"❌ 配置文件 {file} 格式错误: {str(e)}")
                continue

            data_id = file.replace('.json', '')

            # 调用 Nacos API 上传配置
            url = f"{NACOS_SERVER}/nacos/v1/cs/configs"
            params = {
                'dataId': data_id,
                'group': GROUP,
                'content': safe_content,
                'namespaceId': NAMESPACE,
                'type': 'json'
            }

            response = requests.post(url, params=params)
            if response.status_code == 200:
                print(f"✅ 配置文件 {file} 已安全加载到 Nacos")
                print(f"  原始文件敏感字段已加密，加密密钥来自环境变量 NACOS_ENCRYPTION_KEY")
            else:
                print(f"❌ 配置文件 {file} 上传失败: {response.text}")

    except Exception as e:
        print(f"加载配置文件到 Nacos 失败: {str(e)}")


if __name__ == "__main__":

    load_config_to_nacos()