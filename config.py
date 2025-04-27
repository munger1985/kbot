

from dynaconf import Dynaconf

# 初始化配置
config = Dynaconf(
    # 配置文件路径，按顺序加载（后面的会覆盖前面的）
    settings_files=['settings.toml', 'lightrag.env', '.secrets.toml'],

    # 启用环境变量支持
    environments=True,
    # 环境变量前缀
    env_prefix="MYAPP_",
    # 加载 .env 文件
    load_dotenv=True,
    # 配置文件中的默认环境
    env="default",
    # 环境变量切换器
    env_switcher="MYAPP_ENV",
    # 配置验证（可选）
    # validators=[
    #     # 确保必要的配置项存在
    #     "verify_required_settings",
    # ],
    # 必要的配置项
    # required_settings=["database.host", "database.name"],
)