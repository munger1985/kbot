import asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from mcp.server.models import InitializationOptions
import mcp.types as types
from starlette.responses import Response
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.server.sse import SseServerTransport
from mcp_tools import KBSearchTool, KBAskTool
from core.config.settings import get_app_config
from starlette.middleware.cors import CORSMiddleware


# 1. 初始化 Server 实例
server = Server("kbot-mcp-server")
kb_search_tool = KBSearchTool()
kb_ask_tool = KBAskTool()

# 2. 注册工具列表
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    search_schema = kb_search_tool.get_schema()
    ask_schema = kb_ask_tool.get_schema()
    return [
        types.Tool(
            name=kb_search_tool.tool_name,
            description=kb_search_tool.description,
            inputSchema=search_schema
        ),
        types.Tool(
            name=kb_ask_tool.tool_name,
            description=kb_ask_tool.description,
            inputSchema=ask_schema
        )
    ]

# 3. 处理工具调用逻辑
@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent]:
    print(f"DEBUG: MCP Tool Called -> Name: {name}")
    if name == "kbot_search":
        if not arguments:
            raise ValueError("Missing arguments")

        # 如果 schema 里是 'query'，对应的就是 execute 里的 'question'
        results = await kb_search_tool.execute(
            agent_id=arguments.get("agent_id"), # type: ignore
            question=arguments.get("query", "")
        )
        
        # 将结果转为 MCP 要求的格式
        import json
        return [
            types.TextContent(
                type="text",
                text=json.dumps(results, ensure_ascii=False, indent=2)
            )
        ]
    if name == "kbot_ask":
        if not arguments:
            raise ValueError("Missing arguments")

        results = await kb_ask_tool.execute(
            agent_id=arguments.get("agent_id"), # type: ignore
            question=arguments.get("query", "")
        )
        
        # 将结果转为 MCP 要求的格式
        import json
        return [
            types.TextContent(
                type="text",
                text=json.dumps(results, ensure_ascii=False, indent=2)
            )
        ]
    raise ValueError(f"Unknown tool: {name}")

# 4. 创建 SSE 传输实例
sse = SseServerTransport("/messages/")

# 5. 设置路由映射
async def handle_sse(request):
    """处理 SSE 连接"""
    async with sse.connect_sse(
        request.scope, 
        request.receive, 
        request._send
    ) as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="kbot-mcp-service",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
    # 结束后返回一个空的响应，防止 Starlette 报 NoneType 错误
    return Response()

# 6. 组装 Starlette 应用
app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ]
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# 7. 启动并设置端口
if __name__ == "__main__":
    # 在这里设置 host 和 port
    host = get_app_config().service_host
    port = get_app_config().mcp_port
    uvicorn.run(app, host=host, port=port)