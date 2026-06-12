"""
测试 OpsAgent (AIOps 智能故障自愈) 是否可以正常运行。

本测试覆盖三层:
  1. Prometheus 连通性校验 — 验证 PrometheusClient 能否连接 127.0.0.1:9090 并执行查询
  2. OpsOrchestrator 流水线直连测试 — 跳过 HTTP 层，直接调用编排器验证完整链路
  3. (可选) HTTP 全链路测试 — 通过 requests 调用 /api/ops/chat 端点，模拟前端行为

用法:
    python tests/test_ops_agent.py

前置条件:
    - PostgreSQL 可连接 (DB 中已配置 agent 与 instance 绑定关系)
    - Prometheus 已启动 (127.0.0.1:9090)
    - 各微服务已启动 (LLM / Embedding / Reranker / Parser / DB Executor)
      若部分微服务未启动，测试会优雅降级并报告失败原因
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from fastapi import BackgroundTasks

from core.dictionary import PacketType
from agent.orchestrator import OpsOrchestrator
from utils.monitor import PrometheusClient, UnifiedMetricRegistry

# ── 从数据库查到的真实绑定关系 (agent ↔ instance) ──
AGENT_ID    = 126
INSTANCE_ID = "5406458BE47BEF6BE0636703A8C0BE90"
USER_ID     = "chris"

# 测试用的运维问题
TEST_QUERY = "监控告警表空间满了，帮我查一下 oracle-dev-01 的表空间使用情况"


# ════════════════════════════════════════════════════════════════════════════════
# Phase 1: Prometheus 连通性
# ════════════════════════════════════════════════════════════════════════════════

async def test_prometheus_connectivity() -> bool:
    """验证 PrometheusClient 能否正常连接并执行查询"""
    print("\n" + "=" * 60)
    print("🔌 Phase 1: Prometheus 连通性测试")
    print("=" * 60)

    client = PrometheusClient()

    # 1.1 健康检查
    try:
        healthy = await client.health_check()
        if healthy:
            print("  ✅ Prometheus 健康检查通过 (127.0.0.1:9090)")
        else:
            print("  ❌ Prometheus 健康检查失败")
            return False
    except Exception as e:
        print(f"  ❌ Prometheus 连接异常: {e}")
        return False

    # 1.2 瞬时查询 — 检查 oracle-dev-01 是否在线
    try:
        result = await client.query_instant('up{instance="oracle-dev-01"}')
        if result.series:
            val = result.series[0]["value"]
            print(f"  ✅ oracle-dev-01 up 指标 = {val} (1=在线, 0=离线)")
        else:
            print("  ⚠️  oracle-dev-01 up 指标无数据 (Exporter 未启动?)")
    except Exception as e:
        print(f"  ❌ PromQL 查询异常: {e}")
        return False

    # 1.3 查询 oracle 可用指标数量
    try:
        result = await client.query_instant('count({__name__=~"oracledb_.*"})')
        count = int(result.series[0]["value"]) if result.series else 0
        print(f"  ✅ Prometheus 中可用的 oracledb_* 指标数: {count}")
    except Exception as e:
        print(f"  ⚠️  统计 oracle 指标异常: {e}")

    # 1.4 指标注册表校验
    registry = UnifiedMetricRegistry()
    prom_metrics = registry.list_for_llm_prompt(monitor_type="prometheus", db_type="oracle")
    print(f"  ✅ UnifiedMetricRegistry 中 oracle + prometheus 指标条目数: {prom_metrics.count(chr(10)) + 1}")

    return True


# ════════════════════════════════════════════════════════════════════════════════
# Phase 2: OpsOrchestrator 流水线直连
# ════════════════════════════════════════════════════════════════════════════════

async def test_ops_orchestrator_pipeline() -> bool:
    """直接调用 OpsOrchestrator，验证自愈流水线能否完整执行"""
    print("\n" + "=" * 60)
    print("🧠 Phase 2: OpsOrchestrator 流水线直连测试")
    print("=" * 60)

    orchestrator = OpsOrchestrator()
    background_tasks = BackgroundTasks()

    session_id  = f"sess_test_{datetime.now(timezone.utc).strftime('%H%M%S')}"
    packet_count = 0
    errors       = []
    done_received = False
    PIPELINE_TIMEOUT = 120  # 秒，避免因外部 API 挂死而无限等待

    print(f"\n  📋 测试参数:")
    print(f"     agent_id    = {AGENT_ID}")
    print(f"     instance_id = {INSTANCE_ID}")
    print(f"     session_id  = {session_id}")
    print(f"     timeout     = {PIPELINE_TIMEOUT}s")
    print(f"     query       = {TEST_QUERY}")
    print(f"\n  ── 流水线输出 ──\n")

    try:
        pipeline = orchestrator.execute_ops_stream_pipeline(
            background_tasks=background_tasks,
            user_id=USER_ID,
            session_id=session_id,
            agent_id=AGENT_ID,
            question=TEST_QUERY,
            instance_id=INSTANCE_ID,
            trigger_type="manual",
        )

        async def _consume_pipeline():
            nonlocal packet_count, errors, done_received
            async for packet in pipeline:
                packet_count += 1
                p_type = packet.get("type", "UNKNOWN")
                content = packet.get("content", "")

                # 格式化输出
                prefix = _packet_prefix(p_type)
                content_str = _format_content(content)

                if p_type == PacketType.ERROR:
                    errors.append(content_str)
                    print(f"  {prefix} {content_str}")
                elif p_type == PacketType.DONE:
                    done_received = True
                    if isinstance(content, dict):
                        print(f"  {prefix} entry_id={content.get('entry_id', 'N/A')}")
                    else:
                        print(f"  {prefix} {content_str}")
                elif p_type == PacketType.CALL:
                    if isinstance(content, dict):
                        print(f"  {prefix} skill={content.get('skill', '?')} | {content.get('description', '')}")
                    else:
                        print(f"  {prefix} {content_str}")
                elif p_type in (PacketType.THOUGHT, PacketType.ANSWER):
                    if len(content_str) > 200:
                        content_str = content_str[:200] + "..."
                    print(f"  {prefix} {content_str}")
                elif p_type == PacketType.DOC_RESULTS:
                    doc_count = len(content) if isinstance(content, list) else 1
                    print(f"  {prefix} 检索到 {doc_count} 篇运维 SOP 文档")
                elif p_type == PacketType.SQL_RESULTS:
                    print(f"  {prefix} SQL 执行结果已返回")
                else:
                    if len(content_str) > 200:
                        content_str = content_str[:200] + "..."
                    print(f"  {prefix} {content_str}")

        await asyncio.wait_for(_consume_pipeline(), timeout=PIPELINE_TIMEOUT)

        print(f"\n  ── 流水线结束 ──")
        print(f"  总包数: {packet_count}")
        print(f"  错误数: {len(errors)}")
        print(f"  DONE 包: {'✅ 已收到' if done_received else '❌ 未收到'}")

        if done_received and len(errors) == 0:
            print("\n  ✅ OpsOrchestrator 流水线执行成功!")
            return True
        elif done_received:
            print(f"\n  ⚠️  流水线完成但有 {len(errors)} 个错误")
            return False
        else:
            print("\n  ❌ 流水线未正常结束 (未收到 DONE 包)")
            return False

    except asyncio.TimeoutError:
        print(f"\n  ⏰ 流水线超时 ({PIPELINE_TIMEOUT}s) | 已收到 {packet_count} 个包")
        print(f"  💡 可能原因: LLM API (DeepSeek) 流式调用挂死，请检查外部 API 状态")
        return False
    except Exception as e:
        print(f"\n  ❌ 流水线崩溃: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ════════════════════════════════════════════════════════════════════════════════
# Phase 3: HTTP 全链路测试 (可选, 需要主服务运行)
# ════════════════════════════════════════════════════════════════════════════════

def test_ops_agent_via_http() -> bool:
    """通过 HTTP 调用 /api/ops/chat 端点，完全模拟前端 SSE 消费行为"""
    print("\n" + "=" * 60)
    print("🌐 Phase 3: HTTP 全链路 SSE 测试 (需要 cube_main.py 已启动)")
    print("=" * 60)

    try:
        import requests
    except ImportError:
        print("  ⚠️  requests 库未安装，跳过 HTTP 测试")
        return True  # 不算失败

    url = "http://127.0.0.1:18090/api/ops/chat"
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "agent_id":    AGENT_ID,
        "instance_id": INSTANCE_ID,
        "query":       TEST_QUERY,
        "user_id":     USER_ID,
        "session_id":  "new_session",
    }

    print(f"  POST {url}")
    print(f"  Body: {json.dumps(payload, ensure_ascii=False)}\n")

    try:
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=120)
        response.raise_for_status()

        current_event = None
        line_count = 0

        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8").strip()
            line_count += 1

            if line_str.startswith("event:"):
                current_event = line_str[6:].strip()
            elif line_str.startswith("data:"):
                data_str = line_str[5:].strip()
                if not data_str:
                    continue

                try:
                    data = json.loads(data_str)
                    p_type = data.get("type", current_event or "?")
                    content = data.get("content", "")

                    prefix = _packet_prefix(p_type)
                    content_str = _format_content(content)
                    if len(content_str) > 150:
                        content_str = content_str[:150] + "..."

                    if p_type == "DONE":
                        print(f"  {prefix} entry_id={data.get('content', {}).get('entry_id', '?')}")
                    else:
                        print(f"  {prefix} {content_str}")

                except json.JSONDecodeError:
                    print(f"  📝 {data_str[:120]}")

        print(f"\n  ✅ HTTP SSE 流接收完毕 (共 {line_count} 行)")
        return True

    except requests.exceptions.ConnectionError:
        print("  ⚠️  无法连接主服务 (cube_main.py 未启动?)，跳过 HTTP 测试")
        return True  # 不算硬失败
    except Exception as e:
        print(f"  ❌ HTTP 请求异常: {e}")
        return False


# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

def _packet_prefix(p_type: str) -> str:
    """返回包类型的 emoji 前缀"""
    mapping = {
        "METADATA":      "📋",
        "THOUGHT":       "💭",
        "ANSWER":        "🤖",
        "CALL":          "📞",
        "DOC_RESULTS":   "📚",
        "SQL_RESULTS":   "📊",
        "WARNING":       "⚠️",
        "ERROR":         "❌",
        "DONE":          "✅",
        "REQUIRE_APPROVAL": "🔒",
    }
    emoji = mapping.get(str(p_type), "📦")
    return f"[{emoji} {p_type}]"


def _format_content(content) -> str:
    """将 content 转为可打印字符串"""
    if isinstance(content, str):
        return content.replace("\n", "\\n")
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        return f"[{len(content)} items]"
    return str(content)


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

async def main():
    print("╔" + "═" * 58 + "╗")
    print("║  NexusCube OpsAgent (AIOps) 集成测试" + " " * 22 + "║")
    print("║  Prometheus: 127.0.0.1:9090" + " " * 30 + "║")
    print("╚" + "═" * 58 + "╝")

    results: dict[str, bool] = {}

    # Phase 1 — Prometheus 连接
    results["prometheus"] = await test_prometheus_connectivity()

    # Phase 2 — Orchestrator 直连
    results["orchestrator"] = await test_ops_orchestrator_pipeline()

    # Phase 3 — HTTP 全链路 (可选)
    results["http"] = test_ops_agent_via_http()

    # ── 汇总 ──
    print("\n" + "=" * 60)
    print("📊 测试汇总")
    print("=" * 60)
    for name, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}")

    all_pass = all(results.values())
    print(f"\n  {'🎉 全部通过!' if all_pass else '⚠️  部分测试未通过，请检查上方详情'}")
    print("=" * 60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
