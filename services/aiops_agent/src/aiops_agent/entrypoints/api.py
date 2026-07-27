"""AIOps Internal API 进程入口。"""

import sys

import uvicorn

from aiops_agent.bootstrap import create_aiops_api
from aiops_agent.config import get_aiops_settings
from platform_core.platform.port_check import check_port_available


settings = get_aiops_settings()
config = settings.api
app = create_aiops_api(settings)


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
