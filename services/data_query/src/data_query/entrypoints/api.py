"""Data Query API 进程入口。"""

import sys

import uvicorn

from data_query.bootstrap import create_data_query_api
from data_query.config import get_data_query_settings
from platform_core.platform.port_check import check_port_available


settings = get_data_query_settings()
config = settings.api
app = create_data_query_api(settings)


if __name__ == "__main__":
    if not check_port_available(
        config.service_host,
        config.service_port,
        config.service_name,
    ):
        sys.exit(1)
    uvicorn.run(
        app,
        host=config.service_host,
        port=config.service_port,
        log_config=None,
        loop="asyncio",
    )
