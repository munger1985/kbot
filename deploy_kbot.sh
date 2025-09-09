#!/bin/bash

# KBot 3.0 自动化部署脚本
# 适用于 Ubuntu 22.04 或更高版本
# 需要以具有 sudo 权限的用户运行
# 请根据实际情况调整脚本中的路径和配置

set -e  # 遇到错误退出

echo "开始部署 KBot 3.0 ..."

### 1. 系统准备和依赖安装 ###
echo "=== 步骤 1: 系统准备和依赖安装 ==="

# 关闭防火墙（根据实际需求调整）
sudo systemctl stop firewalld 2>/dev/null || true
sudo ufw disable 2>/dev/null || true

# 放开必要端口
sudo iptables -I INPUT -p tcp --dport 8848 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT -p tcp --dport 18099 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT -p tcp --dport 1521 -j ACCEPT 2>/dev/null || true
sudo iptables -I INPUT -p tcp --dport 22 -j ACCEPT 2>/dev/null || true

# 安装必要软件包
sudo apt update
sudo apt install -y wget curl git

# 安装 Anaconda
if [ ! -d "$HOME/anaconda3" ]; then
    echo "安装 Anaconda..."
    wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -u -p ~/anaconda3
    ~/anaconda3/bin/conda init bash
    source ~/.bashrc
else
    echo "Anaconda 已安装"
fi

# 安装 Docker
if ! command -v docker &> /dev/null; then
    echo "安装 Docker..."
    sudo apt install -y docker.io docker-compose
    sudo usermod -aG docker $USER
    sudo systemctl enable docker
    sudo systemctl start docker
else
    echo "Docker 已安装"
fi

### 2. 下载代码和准备环境 ###
echo "=== 步骤 2: 下载代码和准备环境 ==="

if [ ! -d "kbot3" ]; then
    git clone -b kbot3 https://github.com/munger1985/kbot.git kbot3
else
    echo "kbot3 代码已存在，跳过克隆"
fi

cd kbot3

# 创建 conda 环境
if ! conda env list | grep -q "kbot3"; then
    conda create -n kbot3 python=3.12 -y
fi

# 激活环境并安装依赖
eval "$(conda shell.bash hook)"
conda activate kbot3
pip install -r requirements.txt

### 3. 准备模型文件 ###
echo "=== 步骤 3: 准备模型文件 ==="

MODELS_DIR="/home/$(whoami)/Models"
mkdir -p $MODELS_DIR

# 下载 fasttext 模型
if [ ! -f "$MODELS_DIR/cc.zh.300.bin" ]; then
    echo "下载 fasttext 模型..."
    wget -O $MODELS_DIR/cc.zh.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz
    gunzip $MODELS_DIR/cc.zh.300.bin.gz
else
    echo "fasttext 模型已存在"
fi

# 注意：BGE 模型需要手动下载或使用 huggingface-cli
# 这里只是创建目录
mkdir -p $MODELS_DIR/bge-reranker-v2-m3
mkdir -p $MODELS_DIR/bge-m3

echo "请手动下载 BGE 模型到 $MODELS_DIR 目录:"
echo "huggingface-cli download BAAI/bge-reranker-v2-m3 --local-dir $MODELS_DIR/bge-reranker-v2-m3"
echo "huggingface-cli download BAAI/bge-m3 --local-dir $MODELS_DIR/bge-m3"

### 4. 部署 Docker 容器 ###
echo "=== 步骤 4: 部署 Docker 容器 ==="

# 创建数据目录
KBOT_DATA_DIR="/home/$(whoami)/kbot_data"
mkdir -p $KBOT_DATA_DIR/{redis,libreoffice}

# 部署 Redis
if ! docker ps -a | grep -q "kbot-redis"; then
    echo "启动 Redis 容器..."
    sudo docker run --name kbot-redis -d \
      -p 6379:6379 \
      -v $KBOT_DATA_DIR/redis:/data \
      redis:latest \
      redis-server --appendonly yes --requirepass "welcome1"
    docker update --restart unless-stopped kbot-redis
else
    echo "Redis 容器已存在"
fi

# 部署 LibreOffice
if ! docker ps -a | grep -q "libreoffice"; then
    echo "请确保已准备好 libreoffice 镜像 (libreoffice:latest)"
    echo "如果已有镜像文件，使用: docker load -i libre-images.tar"
    echo "然后运行: docker tag [IMAGE_ID] libreoffice:latest"
    echo "按回车继续..." 
    read
    
    sudo docker run --name libreoffice -d \
      -p 9316:9316 \
      -v $KBOT_DATA_DIR/libreoffice:/data \
      libreoffice:latest
else
    echo "LibreOffice 容器已存在"
fi

# 部署 Nacos
if ! docker ps -a | grep -q "nacos"; then
    echo "启动 Nacos 容器..."
    cd core/nacos_manager/nacos-init
    docker-compose up -d
    cd ../../..
    
    # 等待 Nacos 启动
    sleep 30
else
    echo "Nacos 容器已存在"
fi


### 5. 初始化配置文件 ###
echo "=== 步骤 5: 初始化配置文件 ==="

CONFIG_DIR="configuration"
EXAMPLE_APP_CONFIG="$CONFIG_DIR/example/app_config.json.example"
EXAMPLE_DB_CONFIG="$CONFIG_DIR/example/db_config.json.example"
EXAMPLE_MODEL_CONFIG="$CONFIG_DIR/example/model_config.json.example"

APP_CONFIG="$CONFIG_DIR/app_config.json"
DB_CONFIG="$CONFIG_DIR/db_config.json"
MODEL_CONFIG="$CONFIG_DIR/model_config.json"

# 创建配置目录（如果不存在）
mkdir -p $CONFIG_DIR

# 检查示例配置文件是否存在
if [ -f "$EXAMPLE_APP_CONFIG" ]; then
    echo "复制示例配置文件..."
    cp "$EXAMPLE_APP_CONFIG" "$APP_CONFIG"
    
    # 创建文件存储目录
    FILE_STORAGE_DIR="$KBOT_DATA_DIR/files"
    mkdir -p "$FILE_STORAGE_DIR"
    
    # 创建日志目录
    LOGS_DIR="$(pwd)/logs"
    mkdir -p "$LOGS_DIR"
    
    # 更新配置文件路径
    echo "更新配置文件路径..."
    sed -i "s|\"file_storage\": \"/your/path/kbot3/knowledge_base\"|\"file_storage\": \"$FILE_STORAGE_DIR\"|g" "$APP_CONFIG"
    sed -i "s|\"dir\": \"/your/path/kbot3/logs/\"|\"dir\": \"$LOGS_DIR/\"|g" "$APP_CONFIG"
    
    echo "配置文件初始化完成:"
    echo "文件存储路径: $FILE_STORAGE_DIR"
    echo "日志路径: $LOGS_DIR"
else
    echo "警告: 示例配置文件 $EXAMPLE_APP_CONFIG 不存在"
    echo "请手动创建配置文件 $APP_CONFIG"
fi

if [ -f "$EXAMPLE_DB_CONFIG" ]; then
    cp "$EXAMPLE_DB_CONFIG" "$DB_CONFIG"
    echo "请编辑 $DB_CONFIG 以配置数据库连接"
else
    echo "警告: 示例配置文件 $EXAMPLE_DB_CONFIG 不存在"
    echo "请手动创建配置文件 $DB_CONFIG"
fi

if [ -f "$EXAMPLE_MODEL_CONFIG" ]; then
    cp "$EXAMPLE_MODEL_CONFIG" "$MODEL_CONFIG"
    echo "请编辑 $MODEL_CONFIG 以配置模型路径"
else
    echo "警告: 示例配置文件 $EXAMPLE_MODEL_CONFIG 不存在"
    echo "请手动创建配置文件 $MODEL_CONFIG"
fi

# 加载配置到 Nacos
echo "加载配置到 Nacos..."
bash load_config.sh


### 6. 微服务初始化 ###
echo "=== 步骤 6: 微服务初始化 ==="
bash init_microservices.sh

### 7. 数据库初始化 ###
echo "=== 步骤 7: 数据库初始化 ==="

echo "请手动执行以下数据库操作:"
echo "1. 确保 Oracle 23ai 数据库已启用 DRCP"
echo "2. 执行 kbot3/docs/kbot_ddl_v1.0.sql"
echo "3. 执行 kbot3/docs/apex_ui_v1.0.sql"
echo "4. 创建全文检索索引"
echo ""
echo "示例命令:"
echo "sqlplus sys/your_password@host:port/service as sysdba"
echo "SELECT STATUS, MAXSIZE FROM DBA_CPOOL_INFO;"
echo "EXECUTE DBMS_CONNECTION_POOL.START_POOL();"
echo ""
echo "sqlplus kbotui_dev/your_password@host:port/service @kbot_ddl_v1.0.sql"
echo "sqlplus kbotui_dev/your_password@host:port/service @apex_ui_v1.0.sql"

### 8. 启动服务 ###
echo "=== 步骤 8: 启动 KBot 服务 ==="

echo "启动 KBot 后台服务..."
bash start_kbot.sh

echo "=== 部署完成 ==="
echo "请继续完成以下步骤:"
echo "1. 部署 APEX 和 KBot UI"
echo "2. 访问 Nacos 管理界面: http://localhost:8848/nacos/"
echo "3. 配置系统设置和模型参数"
echo "4. 测试完整流程"