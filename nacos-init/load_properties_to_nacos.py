import os
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# Nacos 配置中心地址
NACOS_SERVER = os.getenv('NACOS_SERVER') or'http://localhost:8848'
NAMESPACE = os.getenv('NACOS_NAMESPACE') or'public'
GROUP = os.getenv('NACOS_GROUP') or 'DEFAULT_GROUP'


def load_properties_to_nacos():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        property_files = [f for f in os.listdir(current_dir) if f.endswith('.properties')]

        if not property_files:
            print("未找到任何 .properties 文件")
            return

        for file in property_files:
            file_path = os.path.join(current_dir, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            data_id = file.replace('.properties', '')

            # 调用 Nacos API 上传配置
            url = f"{NACOS_SERVER}/nacos/v1/cs/configs"
            params = {
                'dataId': data_id,
                'group': GROUP,
                'content': content,
                'namespaceId': NAMESPACE,
            }

            response = requests.post(url, params=params)
            print(f"配置文件 {file} 已成功加载到 Nacos: {response.text}")

    except Exception as e:
        print(f"加载配置文件到 Nacos 失败: {str(e)}")


if __name__ == "__main__":
    load_properties_to_nacos()