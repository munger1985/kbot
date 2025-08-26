import os
import sys
# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(current_file))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.nacos_manager import load_config, AppConfig

config = load_config("app_config") # 获取配置

# 获取配置值
if isinstance(config, AppConfig):
    level = config.kbot.log.level
    dir = config.kbot.log.dir
    rotation = config.kbot.log.rotation
    retention = config.kbot.log.retention

print(f"file logger {level}, {dir}, {rotation}, {retention}")


# python -m tests.test_nacos