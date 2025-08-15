
# kbot3.0 开发环境：
ssh to a10-kbot-dev(64.181.236.219) via ubuntu
项目根目录 /home/ubuntu/kbot3
```bash
cd /home/ubuntu/kbot3
./start_kbot.sh
./stop_kbot.sh
```
```bash
git pull                      #本地更新代码      
git reset --hard origin/kbot3  # 强制将本地 main 分支重置为远程状态
```

### 后端API文档：
http://64.181.236.219:18099/docs
http://64.181.236.219:18099/redoc

### Kbot3.0前端：
http://132.145.81.123:8080/ords/r/kbotui_dev/km-chat/home     chinase1/chinase1  admin/12345678 
http://132.145.81.123:8080/ords/r/kbotui_dev/ai-platform-portal/home  admin/12345678

### 后端调用示例：
```bash
curl -X POST "http://localhost:8001/embed" -H "Content-Type: application/json" -d '{"model_id": 65, "texts": ["这是一个测试文本11111", "这是另一个 测试文本222222"], "batch_size": 2}'

curl -X POST "http://150.230.37.250:8001/embed" -H "Content-Type: application/json" -d '{"model_id": 65, "texts": ["这是一个测试文本11111", "这是另一个 测试文本222222"], "batch_size": 2}'

curl -X POST "http://150.230.37.250:8002/chat" -H "Content-Type: application/json" -d '{"model_id": 67, "stream": true, "messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]}'

curl -X POST "http://localhost:8002/chat" -H "Content-Type: application/json" -d '{"model_id": 67, "stream": true, "messages": [{"role": "system", "content": "You are a helpful assistant"}, {"role": "user", "content": "Hello"}]}'
```

### Ubuntu 安装 Oracle Instant Client
1.下载安装包
```bash
wget https://download.oracle.com/otn_software/linux/instantclient/2380000/instantclient-basic-linux.x64-23.8.0.25.04.zip
wget https://download.oracle.com/otn_software/linux/instantclient/2380000/instantclient-sqlplus-linux.x64-23.8.0.25.04.zip
```
2.解压安装：
```bash
sudo mkdir -p /opt/oracle
sudo unzip instantclient-basic-linux.x64-*.zip -d /opt/oracle
sudo unzip instantclient-sqlplus-linux.x64-*.zip -d /opt/oracle
```
3.设置环境变量：
```bash
export ORACLE_HOME=/opt/oracle/instantclient_23_8
export LD_LIBRARY_PATH=$ORACLE_HOME:$LD_LIBRARY_PATH
export PATH=$ORACLE_HOME:$PATH
```

### 部署Docker：
```bash
sudo apt install docker
sudo apt  install docker-compose
sudo usermod -aG docker $USER
docker pull redis:latest

# 启动redis容器
docker run --name kbot-redis -d \
  -p 6379:6379 \
  -v /home/ubuntu/kbot_data/redis:/data \
  redis:latest \
  redis-server --appendonly yes --requirepass "welcome1"

# 设置容器自动重启
docker update --restart unless-stopped kbot-redis #容器名或ID
```
### 安装字体，用于libreoffice转PDF时解决中文乱码
```bash
sudo apt-get install fonts-noto-cjk libreoffice
```

### 页码提取工具，用于提取pdf中的页码信息
```bash
sudo apt-get install poppler-utils
```

### 项目初始化时需要到nacos-init目录下初始化nacos的docker容器
```bash
# 首先在home目录下创建nacos/data和nacos/logs
# 然后修改nacos-init/docker-compose.yaml文件中的相关目录
cd nacos-init
docker-compose up -d
```
### 然后将配置文档注入nacos，只需要运行一次，之后就不再需要初始化nacos，除非配置有变更需要重新运行下面的命令
```bash
python nacos-init/load_properties_to_nacos.py
# nacos默认管理界面
http://localhost:8848/nacos/
```

### 本地开发包安装
包位于项目根目录下的shared-libs目录中：
进到shared-libs下的logger_manager和nacos_manager目录中，用下面的命令分别安装
```bash
pip install -e .
```

```sql
select * from KBOT_MD_DOMAIN;
select * from KBOT_MD_KB;
select * from kbot_md_agent;
select * from kbot_md_agent_conf;
select * from KBOT_MD_KB_BATCH;
select * from KBOT_MD_KB_FILES where kb_id = 165;
select * from KBOT_MD_PROMPT;
select embed_id,chunk_metadata,file_id,security_level from KBOT_BIZ_TXT_EMBEDDING 
    where file_id = '744d6f6f-cf5f-47c7-974e-3ca3e2f2ceab'
    and file_id = 'e08c72c5-3789-4cf8-b126-6f09117e0c9c'
    ;
```
