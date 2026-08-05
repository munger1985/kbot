"""通过真实 HTTP Server 验证开发日志检索、轮转与脱敏。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import socket
from tempfile import TemporaryDirectory

import aiohttp
from fastapi import FastAPI
import uvicorn

from main_api.api.development_logs import router
from main_api.log_reader import LocalLogSearchService


async def main() -> None:
    with TemporaryDirectory(prefix="kbot-development-logs-") as directory:
        root = Path(directory)
        log_root = root / "logs"
        service_root = log_root / "main_api"
        service_root.mkdir(parents=True)
        topology = root / "topology.toml"
        topology.write_text(
            '[[processes]]\nservice_config = "main_api"\n'
            'service_name = "kbot-main-api"\n',
            encoding="utf-8",
        )
        stamp = datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]
        active = service_root / "runtime.log"
        active.write_text(
            f"{stamp} | ERROR    | [api] main_api.smoke:main:1 - "
            "调用失败 | trace_id=trace-smoke | password=secret-password\n"
            "Traceback (most recent call last):\n"
            "  RuntimeError: smoke Authorization=Bearer secret-token",
            encoding="utf-8",
        )
        app = FastAPI()
        app.state.development_log_search_service = LocalLogSearchService(
            log_root=log_root,
            topology_path=topology,
            max_bytes_per_file=4096,
            max_total_scan_bytes=8192,
        )
        app.include_router(router)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        port = int(listener.getsockname()[1])
        server = uvicorn.Server(uvicorn.Config(
            app, log_level="warning", lifespan="off",
        ))
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            for _ in range(200):
                if server.started:
                    break
                await asyncio.sleep(0.01)
            else:
                raise RuntimeError("开发日志测试服务器未能启动")
            base = f"http://127.0.0.1:{port}/api/v1/development/logs"
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base}/services") as response:
                    assert response.status == 200
                    assert len((await response.json())["services"]) == 1
                async with session.get(
                    f"{base}/events",
                    params={"trace_id": "trace-smoke", "stream": "RUNTIME"},
                ) as response:
                    assert response.status == 200
                    events = (await response.json())["events"]
                    assert len(events) == 1
                    assert "raw" not in events[0]
                    event_id = events[0]["event_id"]
                async with session.get(f"{base}/events/{event_id}") as response:
                    assert response.status == 200
                    raw = (await response.json())["raw"]
                    assert "secret-password" not in raw
                    assert "secret-token" not in raw
                    assert "[REDACTED]" in raw

                active.rename(service_root / "runtime.log.2026-08-05")
                active.write_text(
                    f"{stamp} | INFO     | [api] main_api.smoke:main:2 - "
                    "轮转后事件 | trace_id=trace-rotated",
                    encoding="utf-8",
                )
                async with session.get(
                    f"{base}/events",
                    params={"trace_id": "trace-rotated"},
                ) as response:
                    assert response.status == 200
                    assert (await response.json())["total"] == 1
        finally:
            server.should_exit = True
            await asyncio.wait_for(task, timeout=5)
    print("S5 开发日志 HTTP Smoke 通过：真实文件、轮转、检索和详情脱敏均正常")


if __name__ == "__main__":
    asyncio.run(main())
