import configparser
from nacos_manager.manager import nacos_manager # type: ignore

config = nacos_manager.get_config("logger", "DEV_GROUP") # 获取配置
print(config)

# 解析配置内容
config_parser = configparser.ConfigParser()
config_parser.read_string(f"[DEV_GROUP]\n{config}")

# 获取配置值
level = config_parser.get('DEV_GROUP', 'level')
dir = config_parser.get('DEV_GROUP', 'dir')
rotation = config_parser.get('DEV_GROUP', 'rotation')
retention = config_parser.get('DEV_GROUP', 'retention')

print(f"file logger {level}, {dir}, {rotation}, {retention}")