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