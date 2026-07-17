#!/bin/bash
# ==============================================================================
# decode_filenames.sh - 批量 URL 解码磁盘上的文件名
#
# 用法:
#   # 试运行（仅预览，不实际重命名）
#   ./scripts/decode_filenames.sh --dry-run
#   ./scripts/decode_filenames.sh ./knowledge_base --dry-run
#
#   # 实际执行
#   ./scripts/decode_filenames.sh
#   ./scripts/decode_filenames.sh /home/chris/kbot3/knowledge_base
# ==============================================================================

set -euo pipefail

DRY_RUN=false
KB_ROOT="/home/ubuntu/kbotfiles"

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n)
            DRY_RUN=true
            ;;
        *)
            KB_ROOT="$arg"
            ;;
    esac
done

if [ ! -d "$KB_ROOT" ]; then
    echo "错误: 目录不存在: $KB_ROOT"
    exit 1
fi

# 转换为绝对路径
KB_ROOT=$(realpath "$KB_ROOT")

echo "============================================"
echo "  文件名 URL 解码工具"
echo "  目标目录: $KB_ROOT"
if [ "$DRY_RUN" = true ]; then
    echo "  模式: 试运行 (不会实际重命名)"
else
    echo "  模式: 实际执行"
fi
echo "============================================"
echo ""

# ---------------------------------------------------------------------------
# 查找所有含 URL 编码的文件名 (%XX 格式)
# 使用 null 分隔符安全处理特殊字符文件名
# ---------------------------------------------------------------------------
DECODED=0
SKIPPED=0
ERRORS=0

while IFS= read -r -d '' filepath; do
    dir=$(dirname "$filepath")
    oldname=$(basename "$filepath")

    # 用 Python 解码文件名
    newname=$(python3 -c "
import sys, urllib.parse
name = sys.argv[1]
decoded = urllib.parse.unquote(name)
# 过滤掉文件名中不允许的字符（/ 会被转成路径分隔符）
if decoded != name:
    print(decoded, end='')
" "$oldname" 2>/dev/null) || true

    # 解码后没变化则跳过
    if [ -z "$newname" ] || [ "$oldname" = "$newname" ]; then
        continue
    fi

    newpath="$dir/$newname"

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $oldname -> $newname"
        ((DECODED++)) || true
        continue
    fi

    # 检查目标是否已存在
    if [ -e "$newpath" ]; then
        echo "[SKIP] 目标已存在: $newname"
        ((SKIPPED++)) || true
        continue
    fi

    # 执行重命名
    if mv "$filepath" "$newpath" 2>/dev/null; then
        echo "[OK] $oldname -> $newname"
        ((DECODED++)) || true
    else
        echo "[FAIL] 重命名失败: $oldname"
        ((ERRORS++)) || true
    fi

done < <(find "$KB_ROOT" -type f -name "*%*" -print0 2>/dev/null || true)

echo ""
echo "============================================"
echo "  处理完成: 解码 $DECODED 个, 跳过 $SKIPPED 个, 失败 $ERRORS 个"
if [ "$DRY_RUN" = true ]; then
    echo "  确认无误后运行: ./scripts/decode_filenames.sh $KB_ROOT"
fi
echo "============================================"
