#!/usr/bin/env bash
set -uo pipefail

SERVICE_ROOT="$(cd "$(dirname "$0")" && pwd)"
START_SCRIPT="${SERVICE_ROOT}/start_kbot.sh"
STOP_SCRIPT="${SERVICE_ROOT}/stop_kbot.sh"

usage() {
    cat <<'EOF'
用法：./restart_kbot.sh [all]

不传参数或传入 all 时，依次停止并启动全部 KBot 服务。
停止失败时不会继续启动；停止等待和启动就绪检查分别复用
stop_kbot.sh 与 start_kbot.sh。

  --help    显示帮助

示例：
  ./restart_kbot.sh
  ./restart_kbot.sh all
EOF
}

if [ "$#" -gt 1 ]; then
    echo "❌ restart_kbot.sh 目前仅支持重启全部服务。" >&2
    usage >&2
    exit 2
fi

case "${1:-all}" in
    all)
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        echo "❌ 未知服务目标或选项：$1" >&2
        echo "   KBot 当前启动与停止脚本仅支持全部服务。" >&2
        usage >&2
        exit 2
        ;;
esac

echo "正在重启全部 KBot 服务..."

"$STOP_SCRIPT"
stop_status=$?
if [ "$stop_status" -ne 0 ]; then
    echo "❌ 停止 KBot 服务失败，已取消后续启动。" >&2
    exit "$stop_status"
fi

"$START_SCRIPT"
start_status=$?
if [ "$start_status" -ne 0 ]; then
    echo "❌ KBot 服务重新启动失败。" >&2
    exit "$start_status"
fi

echo "🎉 全部 KBot 服务已重启完成。"
