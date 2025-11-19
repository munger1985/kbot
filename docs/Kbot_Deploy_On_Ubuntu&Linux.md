
# 在Ubuntu境部署Kbot步骤概述
### 1.Kbot后台服务器以及系统准备
```bash
#1.准备Ubuntu 22或者以上版本 64位，或者Linux8或以上，CPU 至少8核心；Memory至少64GB；Disk Storage至少：200GB
# 如果要效果好，推荐至少有一块8G显存的显卡来运行Qwen Embedding 和 rerank 模型
##推荐Ubuntu 22.0464位
#2.安装必要的包以及网络等配置
#在OCI上部署时，需要放开端口，以及关闭防火墙
#在VCN的security list中添加端口（8848、18099、1521、22）
#Ubuntu系统层面放开端口
sudo iptables -nvL
sudo iptables -I INPUT -p tcp --dport 8848 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 18099 -j ACCEPT
#Linux关闭防火墙
systemctl stop firewalld
#3.安装必要的软件包
#3.1.下载conda软件包，并安装
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
sh Anaconda3-2024.10-1-Linux-x86_64.sh -b -u -p ~/anaconda3
~/anaconda3/bin/conda init bash
#3.2.Ubuntu安装docker（27.5.1或以上）
sudo apt install docker -y
sudo apt install docker-compose -y
sudo usermod -aG docker $USER
#3.2.Linux安装docker
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
# 注意：由于 Oracle Linux 与 RHEL/CentOS 的二进制兼容性，我们使用 CentOS 的仓库。
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin --allowerasing
sudo usermod -aG docker $USER
#3.3 git安装
sudo apt install git -y
```

### 2.Kbot代码下载以及依赖包准备
```bash
#1.下载Kbot3.0源代码，Hub提供子版本的Tag
git clone https://github.com/munger1985/kbot.git -b v3.1
#2.创建conda虚拟环境
cd kbot3
conda create -n kbot3 python=3.12
conda activate kbot3
pip install -r requirements.txt
# 2.1 根据cuda版本安装torch和transformers，如果没有GPU则跳过这一步，直接到步骤3，准备Kbot元数据库
# 验证cuda版本
nvcc --version
# 配置nvidia官方源并升级cuda，如果需要
# 添加NVIDIA官方仓库 (适用于Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
# 安装CUDA 13.0
sudo apt-get update
sudo apt-get install cuda-13-0
# 添加NVIDIA官方仓库 (适用于Ubuntu 24.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-ubuntu2404.pin
sudo mv cuda-ubuntu2404.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/ /"
# 安装CUDA 12.8
sudo apt-get update
sudo apt-get install cuda-12-8
#sudo apt-get remove --purge cuda-12-8
# 安装完CUDA后配置环境变量
export PATH=/usr/local/cuda-12.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH}
export CUDA_HOME=/usr/local/cuda-12.8

# 根据cuda版本安装torch和transformers，例如：如果cuda版本为12.8，则安装torch和transformers如下
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers
# 2.2 安装flash-attn
# 安装 flash-attn 所需的构建依赖
pip install psutil ninja packaging
pip install flash-attn --no-build-isolation

```
### 3.准备Kbot元数据库（Oracle23ai的schema）连接信息
```bash
sqlplus kbot_poc/VEctor#_123@10.45.151.152:1521/aipocpdb.databasessubnet.vcnpairs.oraclevcn.com
```

### 4.准备模型
```bash
# 准备开源模型和LLM模型，或者准备LLM模型的api key信息。
#huggingface-cli download  BAAI/bge-reranker-v2-m3 --local-dir /home/opc/Models/bge-reranker-v2-m3
#huggingface-cli download  BAAI/bge-m3 --local-dir /home/opc/Models/bge-m3

# 准备Qwen3 rerank和embedding模型
# 在下载前，请先通过如下命令安装ModelScope
pip install modelscope
# 命令行下载完整模型库
modelscope download --model Qwen/Qwen3-Reranker-4B
modelscope download --model Qwen/Qwen3-Embedding-4B
# 下载完成后，请将模型文件放置到模型目录下，默认在 ~/.modelscope/models/Qwen/目录下
```

### 5.部署Docker容器：
```bash
#1.安装Redis以及启动Redis
#1.1 下载Redis 镜像
sudo docker pull redis:latest
#1.2 启动redis容器
sudo docker run --name kbot-redis -d \
  -p 6379:6379 \
  -v /home/ubuntu/kbot_data/redis:/data \
  redis:latest \
  redis-server --appendonly yes --requirepass "welcome1"
#1.3.设置容器自动重启
docker update --restart unless-stopped kbot-redis #容器名或ID

#2.配置 libreoffice 容器(把word/ppt数据转成pdf)
##方法一：从开发环境导出容器，并导入镜像
#2.1 docker export -o my_nginx.tar 容器名  
#2.2 然后到新的开发机，docker load -i my_nginx.tar
#2.3 再给导入的镜像重新命名即可 docker tag 镜像名 《名字》
#安装示例：
docker load -i libre-images.tar
docker tag f9b5dc8f2fb5 libreoffice
#如果关闭了防火墙，需要重启docker。
sudo systemctl restart docker
#2.4 启动容器
sudo docker run --name libreoffice -d \
  -p 9316:9316 \
  -v /home/opc/kbot_data/libreoffice:/data \
  libreoffice:latest 

##方法二：安装部署libreoffice
#2.1 安装从libreoffice 的docker-compose文件中部署
cd microservices/libreoffice
sudo docker-compose up -d
#2.2 启动容器
sudo docker run --name libreoffice -d \
  -p 9316:9316 \
  -v /home/opc/kbot_data/libreoffice:/data \
  libreoffice:latest 

# 服务端口修改docker-compose.yaml里的端口号。默认为9316
# 容器所在服务器ip和容器端口需要更新到app.properties文件的 libre_host 和 libre_port 两项中


#3 部署elastic search容器（作为向量存储）
#3.1 创建elastic search网络
docker network create elastic
#3.2 创建elastic search存储目录
mkdir -p /home/ubuntu/elastic/eskb
#3.3 修改目录权限
sudo chown -R 1000:1000 /home/ubuntu/elastic/
sudo chmod -R 777 /home/ubuntu/elastic/
#3.4 启动elastic search容器
docker run --name eskb --net elastic \
  -p 9202:9200 \
  -p 9302:9300 \
  -v /home/ubuntu/elastic/eskb:/usr/share/elasticsearch/data \
  -d -m 2GB \
  -e "discovery.type=single-node" \
  elasticsearch:9.1.5

#3.5 修改初始密码，将输出到控制台的密码复制到configuration/db_config.json文件中
docker exec -it eskb /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic

#3.6 复制证书
sudo docker cp eskb:/usr/share/elasticsearch/config/certs/http_ca.crt /home/ubuntu/elastic/eskb/
sudo chown -R ubuntu:ubuntu /home/ubuntu/elastic/eskb/http_ca.crt

#3.7 验证elastic search安装，使用前面修改过的初始密码登录，用户名默认elastic
https://localhost:9202/


#3.配置 nacos 容器
#3.1 创建nacos容器镜像
cd docs/install/nacos-container
docker-compose up -d

#3.2 Kbot 3.0后端的配置文件：
#3.2.1 .env配置
.env
  .env配置示例：
  # NACOS config
  NACOS_SERVER_ADDR="0.0.0.0:8848"
  NACOS_GROUP="dev"
  NACOS_USERNAME="nacos"
  NACOS_PASSWORD="nacos"
  NACOS_ENCRYPTION_KEY="aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789+ab="

  # main app config
  KBOT_SERVICE_NAME="kbot_main"
  KBOT_HOST=0.0.0.0
  KBOT_PORT=18099
  KBOT_IP=64.181.236.219
  KBOT_AUTH_ENCRYPTION_KEY="aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789+ab="
  KBOT_AUTH_EXPIRE_MINUTES=600
cp .env ./microservices/embedding/
cp .env ./microservices/llm/
cp .env ./microservices/reranker/
cp .env ./microservices/vlm/

#3.2.2 app_config.json 配置文件说明：
configuration/app_config.json
配置示例：
{
  "kbot": {
    "service_name": "Kbot_Main",
    "title": "KBOT",
    "description": "KBot API Service",
    "version": "3.0.0",
    "debug": true,
    "file_storage": "/home/ubuntu/kbot_data/files",
    "upload_workers": 10,
    "parser": {
      "max_workers": 2,
      "check_interval": 60
    },
    "log": {
      "level": "DEBUG",
      "dir": "/home/ubuntu/kbot3/logs/",
      "rotation": "100 MB",
      "retention": "10 days"
    }
  },
  "libre": {
    "host": "localhost",
    "port": 9316
  }
}

#3.2.3  db_config.json 配置文件说明：
configuration/db_config.json
配置示例：
{
  "oracle": {
    "username": "kbotui_dev",
    "password": "xxx##",
    "host": "132.145.81.123",
    "port": 1521,
    "service_name": "DB1007_pdb1.regionalpublics.hysunhevcn.oraclevcn.com"
  },
  "redis": {
    "password": "welcome1",
    "host": "localhost",
    "port": 6379,
    "max_connections": 10,
    "socket_connect_timeout": 3,
    "socket_timeout": 5,
    "retry_on_timeout": true,
    "health_check_interval": 30
  },
  "sqlalchemy": {
    "echo": false,
    "pool_size": 10,
    "pool_timeout": 60,
    "max_overflow": 20,
    "pool_pre_ping": true,
    "pool_use_lifo": true,
    "pool_recycle": 1800
  },
  "eslog": {
    "hosts": [
      "https://localhost:9201"
    ],
    "username": "elastic",
    "password": "=5n3exXB4J+3Ays1muHO",
    "ca_certs": "/home/ubuntu/elastic/eslog/http_ca.crt",
    "index": "filebeat-*"
  }
}

#3.2.4  model_config.json 配置文件说明：
configuration/model_config.json
配置示例：
{
  "embed": {
    "service_name": "KBot_Embedding_Service",
    "service_version": "1.0.0",
    "service_host": "0.0.0.0",
    "service_port": 9901,
    "max_tokens": 8192,
    "timeout": 300,
    "max_retries": 3,
    "cache_dir": "/home/ubuntu/kbot_data/cached_models"
  },
  "llm": {
    "service_name": "KBot_LLM_Service",
    "service_version": "1.0.0",
    "service_host": "0.0.0.0",
    "service_port": 9902,
    "max_tokens": 8192,
    "temperature": 0.7,
    "top_p": 1.0,
    "top_k": 0,
    "timeout": 300,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0
  },
  "reranker": {
    "service_name": "KBot_Reranker_Service",
    "service_version": "1.0.0",
    "service_host": "0.0.0.0",
    "service_port": 9903,
    "cache_dir": "/home/ubuntu/kbot_data/cached_models",
    "timeout": 300
  },
  "vlm": {
    "service_name": "KBot_VLM_Service",
    "service_version": "1.0.0",
    "service_host": "0.0.0.0",
    "service_port": 9904,
    "timeout": 300
  },
  "tokenizer": {
    "custom_dict_path": "/home/ubuntu/kbot3/configuration/custom_dict.txt",
    "stop_words_path": "/home/ubuntu/kbot3/configuration/stopwords.txt"
  },
  "prompt": {
    "image2text": "SYSTEM/image2text",
    "summary": "SYSTEM/summary"
  }
}
Kbot的主服务默认端口：18099，Nacos的默认端口：8848

#3.3 然后将配置文档注入nacos，只需要运行一次，之后就不再需要初始化nacos，除非配置有变更需要重新运行下面的命令
./load_config.sh
#3.4 nacos默认管理界面，验证nacos安装
http://localhost:8848/nacos/

```
### 6.部署elk用于收集日志
```bash
cd docs/install/elk-log-container

# 根据实际情况修改.env文件中的配置，如果.env不存在则复制.env.example文件并重命名为.env

# 启动容器集群
./start_elk.sh
```
### 7.初始化Kbot数据库表信息
```bash
#1.在DBA用户创建Kbot元数据库（Oracle23ai的schema），并赋予权限
#在CDB级别开启DRCP 
#sqlplus sys/BotWelcome123## as sysdba;
#如果查询没有返回结果，或者 STATUS不是 ACTIVE，说明 DRCP 未启用。
SELECT STATUS, MAXSIZE FROM DBA_CPOOL_INFO;
#执行下面的脚本，启动DRCP
EXECUTE DBMS_CONNECTION_POOL.START_POOL();
EXECUTE DBMS_CONNECTION_POOL.CONFIGURE_POOL(pool_name => 'SYS_DEFAULT_CONNECTION_POOL',minsize => 4,maxsize => 40,incrsize => 2,inactivity_timeout => 300,max_lifetime_session => 86400);
#或者更详细的配置，例如：
EXECUTE DBMS_CONNECTION_POOL.CONFIGURE_POOL(
    pool_name => 'SYS_DEFAULT_CONNECTION_POOL',
    minsize => 4,
    maxsize => 40,
    inactivity_timeout => 300
);

#2.执行Kbot元数据表DDL脚本
cd kbot3/docs
sqlplus kbotui_dev/BotWelcome123##@132.145.81.123:1521/DB1007_pdb1.regionalpublics.hysunhevcn.oraclevcn.com @kbot_ddl_v1.0.sql
#3.执行apex_ui_v1.0.sql脚本,注意,里面的schema需要更换成自定义创建的schema
#4.创建Oracle 23ai全文检索创建索引
--对CHUNK_DOC字段创建全文检索索引
--先用dba用户赋予kbotui_dev用户可执行权限
grant execute on ctxsys.ctx_ddl to kbotui_dev;

--在普通用户执行如下步骤，创建索引。（中文）
exec ctx_ddl.create_preference('chinese_lexer','chinese_vgram_lexer');
CREATE INDEX IDX_FULLSEARCH_TXT_EMBEDDING ON  KBOT_BIZ_TXT_EMBEDDING("CHUNK_DOC") INDEXTYPE IS "CTXSYS"."CONTEXT" PARAMETERS ('lexer chinese_lexer');

--在普通用户执行如下步骤，创建索引。（英文）
--exec ctx_ddl.create_preference('english_lexer','basic_lexer');
--CREATE INDEX IDX_FULLSEARCH_TXT_EMBEDDING ON  KBOT_BIZ_TXT_EMBEDDING("CHUNK_DOC") INDEXTYPE IS "CTXSYS"."CONTEXT" PARAMETERS ('lexer english_lexer');
```

### 8.前端apex安装以及kbot UI部署
```bash
#1.安装apex
#2.部署Kbot UI到apex中
```

### 9.启动后台服务
```bash
cd /home/ubuntu/kbot3
#5.1启动后台服务
./start_kbot.sh
#5.2停止后台服务
./stop_kbot.sh
```

### 10.apex UI系统基本配置以及验证
```bash
#1.系统设置=》系统配置=〉服务URL
#2.系统设置=》向量DB连接
#3.系统设置=》LLM配置
# 在kbot v3.0.3版本中，需要登录到kbot后台服务器，切换到kbot3环境，并执行：python ./tests/test_model_redis.py，把模型同步到redis中。
# 前端修改的模型，也需要执行这个脚本，同步模型信息到redis中。
# 这个自动同步，后续v3.0.4版本会修复。
#4.提示词管理=》提示词模版
#5.完整流程跑通测试
#创建业务域=》创建知识库=》上传数据=》待提交编辑=》待审批=》查询进度=〉创建智能体=》配置智能体=》Chat=》下载/预览文件
```
