#!/bin/bash

# 初始化 conda 环境
eval "$(conda shell.bash hook)"

# 激活 conda 环境
conda activate kbot3

# 加载配置文档到 nacos
cd "$(dirname "$0")" && python core/nacos_manager/load_to_nacos.py
