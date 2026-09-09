#!/usr/bin/env bash

# 在 Ubuntu 24.04 开发机安装可供 Model Serving 与 Docling 共用的 OCR 运行时。

set -euo pipefail

KBOT_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONDA_ENV_NAME="${KBOT_CONDA_ENV:-kbot4}"
DOCLING_MODELS_DIR="${KBOT_DOCLING_MODELS_DIR:-/home/chris/models/docling_models}"
INSTALL_SYSTEMD_SERVICE=false
SKIP_RUNTIME_INSTALL=false

usage() {
    echo "Usage: $0 [--install-service] [--skip-runtime-install]" >&2
}

for argument in "$@"; do
    case "$argument" in
        --install-service) INSTALL_SYSTEMD_SERVICE=true ;;
        --skip-runtime-install) SKIP_RUNTIME_INSTALL=true ;;
        *) usage; exit 2 ;;
    esac
done

find_conda() {
    local candidate="${CONDA_EXE:-}"
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    candidate="$(command -v conda || true)"
    if [[ -n "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    for candidate in \
        "$HOME/anaconda3/bin/conda" \
        "$HOME/miniconda3/bin/conda" \
        "/opt/anaconda3/bin/conda" \
        "/opt/miniconda3/bin/conda"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

conda_bin="$(find_conda || true)"
if [[ -z "$conda_bin" ]]; then
    echo "未找到 conda，无法安装与 Docling 共用的 Tesseract 运行时。" >&2
    exit 1
fi
if ! "$conda_bin" env list | awk 'NF > 1 && $1 !~ /^#/ {print $1}' | grep -Fxq "$CONDA_ENV_NAME"; then
    echo "Conda 环境不存在：$CONDA_ENV_NAME" >&2
    exit 1
fi
if [[ ! -d "$DOCLING_MODELS_DIR" ]]; then
    echo "Docling 模型目录不存在：$DOCLING_MODELS_DIR" >&2
    exit 1
fi

cd "$KBOT_SOURCE_ROOT"
if [[ "$SKIP_RUNTIME_INSTALL" == false ]]; then
    echo "安装 Conda OCR 依赖：$CONDA_ENV_NAME"
    "$conda_bin" install -y -n "$CONDA_ENV_NAME" -c conda-forge \
        'onnxruntime=1.29.0=*_cpu' scikit-image python-bidi tesseract tesserocr

    echo "安装 KBot 工作区及 Python OCR 依赖"
    KBOT_CONDA_ENV="$CONDA_ENV_NAME" bash scripts/deployment/install_workspace.sh
else
    echo "按参数跳过依赖安装，仅验证并部署已有 OCR 运行时。"
fi

python_bin="$($conda_bin run -n "$CONDA_ENV_NAME" python -c 'import sys; print(sys.executable)')"
tessdata_dir="$($conda_bin run -n "$CONDA_ENV_NAME" python -c 'import sys; from pathlib import Path; print(Path(sys.prefix) / "share/tessdata")')"
if [[ ! -d "$tessdata_dir" ]]; then
    echo "Tesseract 语言包目录不存在：$tessdata_dir" >&2
    exit 1
fi

echo "验证 Docling 与本地 OCR 依赖"
"$conda_bin" run -n "$CONDA_ENV_NAME" python -c '
import easyocr
import onnxruntime
import tesserocr
from rapidocr import RapidOCR
print("EasyOCR:", easyocr.__version__)
print("ONNX Runtime:", onnxruntime.__version__)
print("Tesseract:", tesserocr.tesseract_version())
print("RapidOCR:", RapidOCR.__module__)
'
"$conda_bin" run -n "$CONDA_ENV_NAME" tesseract --list-langs | grep -Ex 'chi_sim|eng' >/dev/null
KBOT_DOCLING_MODELS_DIR="$DOCLING_MODELS_DIR" "$conda_bin" run -n "$CONDA_ENV_NAME" python -c '
from pathlib import Path
from rapidocr import RapidOCR
import os
root = Path(os.environ["KBOT_DOCLING_MODELS_DIR"])
RapidOCR(params={
    "Det.model_path": str(root / "RapidOcr/onnx/PP-OCRv6/det/PP-OCRv6_det_small.onnx"),
    "Rec.model_path": str(root / "RapidOcr/onnx/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx"),
    "Cls.model_path": str(root / "RapidOcr/onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_infer.onnx"),
    "Global.font_path": str(root / "RapidOcr/fonts/FZYTK.TTF"),
})
print("RapidOCR 本地模型加载验证通过")
'
KBOT_DOCLING_MODELS_DIR="$DOCLING_MODELS_DIR" "$conda_bin" run -n "$CONDA_ENV_NAME" python -c '
from pathlib import Path
import easyocr
import os
model_dir = Path(os.environ["KBOT_DOCLING_MODELS_DIR"]) / "EasyOcr"
easyocr.Reader(["ch_sim", "en"], gpu=False, model_storage_directory=str(model_dir), download_enabled=False, verbose=False)
print("EasyOCR 本地模型加载验证通过")
'

if [[ "$INSTALL_SYSTEMD_SERVICE" == false ]]; then
    echo "OCR 运行时已安装。追加 --install-service 可安装并启动 kbot-model-ocr。"
    exit 0
fi
if [[ ! -f "$KBOT_SOURCE_ROOT/.env" ]]; then
    echo "缺少 $KBOT_SOURCE_ROOT/.env，拒绝创建可能缺少数据库凭据的 systemd 服务。" >&2
    exit 1
fi
if [[ "$(stat -c '%a' "$KBOT_SOURCE_ROOT/.env")" != "600" ]]; then
    echo ".env 必须为 0600，当前权限不符合要求。" >&2
    exit 1
fi

runtime_user="${SUDO_USER:-$(id -un)}"
runtime_group="$(id -gn "$runtime_user")"
unit_candidate="$(mktemp)"
trap 'rm -f "$unit_candidate"' EXIT
sed \
    -e "s|@KBOT_RUNTIME_USER@|$runtime_user|g" \
    -e "s|@KBOT_RUNTIME_GROUP@|$runtime_group|g" \
    -e "s|@KBOT_SOURCE_ROOT@|$KBOT_SOURCE_ROOT|g" \
    -e "s|@KBOT_PYTHON@|$python_bin|g" \
    -e "s|@TESSDATA_PREFIX@|$tessdata_dir/|g" \
    scripts/deployment/systemd/kbot-model-ocr.service.template > "$unit_candidate"

sudo install -m 0644 "$unit_candidate" /etc/systemd/system/kbot-model-ocr.service
sudo systemctl daemon-reload
sudo systemctl enable --now kbot-model-ocr.service
curl --fail --silent --show-error --max-time 10 http://127.0.0.1:18096/health >/dev/null
sudo systemctl is-active --quiet kbot-model-ocr.service
echo "kbot-model-ocr 已启动，健康检查通过：http://127.0.0.1:18096/health"
