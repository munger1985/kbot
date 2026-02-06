
# 在 Ubuntu 环境部署 Kbot3 步骤概述
## 1. Kbot 后台服务器以及系统准备

| 配置项 | 最低配置 | 推荐配置 |
|--------|----------|----------|
| 操作系统 | Ubuntu 22.04+ 64位 | Ubuntu 24.04 LTS 64位 |
| CPU | 8核 | 16核或更高 |
| 内存 | 64GB | 128GB或更高 |
| 显卡 | 无 | 8GB显存（用于加速本地OCR模型和BGE模型） |
| 硬盘 | 200GB | 500GB或更高 |

**说明**：
- 最低配置可满足所有功能运行，但模型需要调用API接口
- 推荐配置（特别是显卡）可显著提升OCR模型的效果
- 推荐使用 Ubuntu 24.04 LTS 版本

## 2. 网络配置和防火墙设置
### 2.1 配置 OCI 网络安全规则

在OCI（Oracle Cloud Infrastructure）上部署时，需要配置网络安全规则

在VCN的Security List中添加以下入站规则：
- 端口 18099: Kbot API服务端口
- 端口 1521: Oracle数据库端口
- 端口 22: SSH远程登录端口
- 协议类型：TCP
- 源CIDR：根据实际需求配置（如 0.0.0.0/0 或特定IP段）

### 2.2 配置 Ubuntu 系统防火墙

Ubuntu系统层面：配置iptables允许端口访问
```bash
# 查看当前iptables规则
sudo iptables -nvL

# 允许Kbot API服务端口（18099）
sudo iptables -I INPUT -p tcp --dport 18099 -j ACCEPT

# 允许Oracle数据库端口（1521）
sudo iptables -I INPUT -p tcp --dport 1521 -j ACCEPT

# 允许SSH端口（22，通常默认已开放）
sudo iptables -I INPUT -p tcp --dport 22 -j ACCEPT

# 保存iptables规则，使配置持久化
sudo sh -c "iptables-save > /etc/iptables/rules.v4"

# Oracle Linux/其他发行版：关闭firewalld防火墙
sudo systemctl stop firewalld
sudo systemctl disable firewalld

# 检查防火墙状态
sudo systemctl status firewalld
```

## 3 安装必要的软件包

### 3.1 安装 Anaconda（Python环境管理）

```bash
# 下载 Anaconda 安装包（版本 2024.10-1，Python 3.12）
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

# 安装 Anaconda
# -b: 批处理模式（无交互）
# -u: 更新已存在的安装
# -p: 指定安装路径
sh Anaconda3-2024.10-1-Linux-x86_64.sh -b -u -p ~/anaconda3

# 初始化 conda（使 conda 命令在新的 shell 会话中生效）
~/anaconda3/bin/conda init bash

# 使配置生效（重新加载 bash 配置）
source ~/.bashrc

# 验证安装
conda --version
python --version

# 创建conda虚拟环境
conda create -n kbot3 python=3.12
conda activate kbot3
```

### 3.2 安装 Git

```bash
# 安装 Git 版本控制系统
sudo apt install git -y

# 验证安装
git --version

# 配置 Git 用户信息（可选，用于代码提交）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```
### 3.3 Kbot3 源代码下载

#### 选项 1：下载指定版本（推荐用于生产环境）

```bash
# 克隆指定版本的 Kbot 代码（例如 v3.1 标签）
git clone https://github.com/munger1985/kbot.git -b v3.1
```

#### 选项 2：下载最新开发版本（推荐用于测试和开发）

```bash
# 克隆最新的开发分支 kbot3
git clone https://github.com/munger1985/kbot.git -b kbot3
```

**说明**：
- `v3.1` 是稳定版本标签，适合生产环境部署
- `kbot3` 是开发分支，包含最新功能和修复，适合测试环境
- 执行 `git branch -a` 可查看所有可用分支和标签
- 执行 `git tag` 可查看所有版本标签

## 4. 安装 Kbot3.0 依赖包

### 4.1 安装 Python 依赖包
```bash
cd kbot3
pip install -r requirements.txt
```
### 4.2 安装 CUDA 和 cuDNN（可选，用于GPU加速）

**重要说明**：如果没有GPU，请跳过本节"。

#### 4.2.1 检查GPU信息

```bash
# 检查 NVIDIA GPU 是否可用
nvidia-smi

# 查看CUDA驱动版本（需要安装NVIDIA驱动）
cat /proc/driver/nvidia/version
```

#### 4.2.2 安装 CUDA Toolkit

**针对 Ubuntu 22.04：**

```bash
# 下载并配置 NVIDIA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 添加 NVIDIA GPG 密钥
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub

# 添加 CUDA 仓库
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
```

**针对 Ubuntu 24.04：**

```bash
# 下载并配置 NVIDIA 仓库
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600

# 添加 NVIDIA GPG 密钥
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub

# 添加 CUDA 仓库
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
```

**安装 CUDA 12.8：**

```bash
# 更新软件包列表
sudo apt-get update

# 安装 CUDA Toolkit 12.8
sudo apt-get install -y cuda-12-8

# 卸载 CUDA（如需要）
# sudo apt-get remove --purge cuda-12-8
```

**配置 CUDA 环境变量：**

```bash
# 临时配置（当前会话有效）
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.8

# 永久配置（添加到 ~/.bashrc）
echo 'export PATH=/usr/local/cuda-12.8/bin:${PATH}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-12.8' >> ~/.bashrc

# 重新加载配置
source ~/.bashrc
```

**验证 CUDA 安装：**

```bash
# 验证 CUDA 版本
nvcc --version

# 验证驱动和运行时
nvidia-smi
```

#### 4.2.3 安装 PyTorch（GPU版本）

**针对 CUDA 12.8：**

```bash
# 安装 PyTorch、TorchVision、TorchAudio（CUDA 12.8版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装 Transformers（Hugging Face）
pip install transformers

# 验证 PyTorch GPU 支持
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('CUDA版本:', torch.version.cuda)"
```

**其他 CUDA 版本对应的 PyTorch 安装：**

| CUDA 版本 | 安装命令 |
|-----------|----------|
| CUDA 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 12.4 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| CPU 版本 | `pip install torch torchvision torchaudio` |

#### 4.2.4 安装 Flash Attention（可选，提升性能）
整个编译过程会持续30-60分钟

```bash
# 安装构建依赖
pip install psutil ninja packaging

# 安装 Flash Attention（不隔离构建环境）
MAX_JOBS=4 pip install flash-attn --no-cache-dir --no-build-isolation
# 验证安装
python -c "import flash_attn; print('Flash Attention 版本:', flash_attn.__version__)"
```
如果使用 ssh 连接服务器超时：
```bash
# Start a new session:
tmux new -s build_flash

# Run your build command (inside the new green bar window):
conda activate kbot3
export TORCH_CUDA_ARCH_LIST="8.6"
MAX_JOBS=4 pip install flash-attn --no-cache-dir --no-build-isolation

# Detach safely: Press Ctrl + B, then let go and press D. You are now back in your main shell, and the build is running in the background. 
# You can even close your laptop or exit SSH.
# Reattach later: To see how it's doing, log back in via SSH and type:
tmux attach -t build_flash
```

**说明**：
- Flash Attention 可以显著提升Transformer模型的训练和推理速度
- 安装需要编译，可能需要较长时间（约10-30分钟）
- 如果编译失败，可以跳过此步骤，不影响基本功能

### 4.3 安装 Tesseract OCR 库（可选，用于图片文字识别）

**说明**：Tesseract OCR 用于从图片中提取文字，支持中文、英文等多种语言。

#### 4.3.1 安装 Tesseract OCR

```bash
# 使用 conda 安装 Tesseract OCR 和 Python 绑定
conda activate kbot3
conda install -c conda-forge tesserocr tesseract

# 或者使用 pip 安装（需要先安装系统依赖）
# sudo apt install tesseract-ocr tesseract-ocr-chi-sim
# pip install tesserocr
```

#### 4.3.2 验证安装

```bash
# 检查 Tesseract 版本
tesseract --version

# 列出已安装的语言
tesseract --list-langs
# 期望输出：List of available languages in "/home/ubuntu/anaconda3/envs/km/share/tessdata/" (125):
# ...
# 
```

#### 4.3.3 配置语言数据路径

```bash
# 获取语言数据包路径（tessdata 目录）
export tessdata_path=$(tesseract --list-langs | head -n1 | perl -ne 'if (/"([^"]+)"/) { print "$1\n"; }')
echo "tessdata_path: ${tessdata_path}"

# 示例输出: /home/ubuntu/miniconda3/envs/kbot3/share/tessdata/
```

**方法 1：通过环境变量配置**

```bash
# 在 ~/.bashrc 中添加环境变量

echo "export TESSDATA_PREFIX=${tessdata_path}" >> ~/.bashrc

# 使配置生效
source ~/.bashrc
```

**方法 2：在 Kbot 配置文件中设置**

```bash
# 在 kbot3 项目根目录的 .env 文件中添加
echo "TESSDATA_PREFIX=\"${tessdata_path}\"" >> .env
```

#### 4.3.4 验证 Python 绑定

```bash
# 测试 tesserocr Python 绑定
python -c "
import tesserocr
api = tesserocr.PyTessBaseAPI(path='${tessdata_path}')
print('支持的语言:', api.GetAvailableLanguages())
api.End()
"
```

#### 4.3.5 支持的语言列表

| 语言 | 代码 | 语言 | 代码 |
|------|------|------|------|
| 简体中文 | chi_sim | 繁体中文 | chi_tra |
| 英文 | eng | 日文 | jpn |
| 韩文 | kor | 阿拉伯文 | ara |
| 法文 | fra | 德文 | deu |
| 西班牙文 | spa | 俄文 | rus |
| ... | ... | ... | ... |

**常用语言代码**：
- `chi_sim`: 简体中文（推荐）
- `chi_tra`: 繁体中文
- `eng`: 英文

#### 4.3.6 测试 OCR 功能

```bash
# 测试图片文字识别（需要准备一张测试图片）
python -c "
import tesserocr
from PIL import Image

# 打开图片
image = Image.open('test_image.jpg')

# 创建 OCR 实例
api = tesserocr.PyTessBaseAPI(path='${tessdata_path}')
api.SetImage(image)

# 设置识别语言（英文+简体中文）
api.SetVariable('tessedit_char_whitelist', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

# 获取文本
text = api.GetUTF8Text()
print('识别结果:')
print(text)

api.End()
"
```

**说明**：
- Tesseract OCR 在处理复杂文档时准确率有限，建议配合 PDF 解析器使用
- 对于高质量 OCR，建议使用云服务（如 Azure OCR、Google Vision API）
- 支持的语言列表超过100种，可根据实际需求选择


## 5.本地模型准备

### 5.1 下载 embedding/rerank 模型 (可选)
**说明**：
- 如果采用本地部署 rerank 和 embedding 模型（本项目默认使用 Qwen3）需要事先下载好。
- 也可以使用其他开源模型，例如 bge-reranker-v2-m3 和 bge-m3 模型。
- 也可以调用 API 实现 rerank 和 embedding。

```bash
pip install modelscope
python docs/install/models/download_qwen_model.py
```
**说明**：
- 下载完成后会输出模型目录，默认在 ~/.modelscope/models/Qwen/目录下
- 可以根据实际情况移动到其他目录，并在应用启动后将模型目录配置到系统中。

### 5.2 下载 tokenizer 模型（用于文本解析时的语义分词）(必须)
```bash
pip install huggingface_hub
python docs/install/models/download_tokenizer_model.py
```
### 5.3 下载 docling 模型（用于文档解析）(必须)
```bash
# 使用 docling-tools 强制下载到物理路径，而不是缓存软链接
mkdir -p ~/cached_models/docling_models
docling-tools models download --all -o ~/cached_models/docling_models
# 然后把 ~/cached_models/docling_models 目录添加到base.toml中的local_artifacts_path
```

### 5.4 下载 EasyOCR 模型（用于图片文字识别）（可选）
```bash
pip install easyocr
python docs/install/models/download_easyocr_model.py
```

## 6 修改 kbot 后台服务配置（可选）
**说明**：
- 如果需要修改 kbot 后台服务的配置（例如端口号，数据库连接等），可以在 configuration/base.toml 文件中进行修改。
- 如果没有 configuration/base.toml 文件，则需要先复制 configuration/example/base.toml.example 文件并重命名为 base.toml。
- 修改后需要重启 kbot 后台服务才能生效。

## 6.启动/停止 kbot 后台服务
```bash
cd /home/ubuntu/kbot3
# 启动 kbot 后台服务
./start_kbot.sh
# 停止 kbot 后台服务
./stop_kbot.sh
# 验证 kbot 后台服务是否启动
curl http://localhost:18099/health
# 输出: {"status":"healthy"}
```

## 7. KBot3 后台接口文档
http://localhost:18099/docs

http://localhost:18099/redoc