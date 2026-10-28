"""敏感API的统一禁止缓存路由；覆盖鉴权、字段校验及已登记应用错误响应。"""

from fastapi.routing import APIRoute
from starlette.datastructures import MutableHeaders


class NoStoreAPIRoute(APIRoute):
    async def handle(self, scope, receive, send):
        async def send_no_store(message):
            if message["type"] == "http.response.start":
                MutableHeaders(scope=message)["Cache-Control"] = "no-store"
            await send(message)

        await super().handle(scope, receive, send_no_store)
