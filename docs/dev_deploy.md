
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
git clone https://github.com/munger1985/kbot.git 
git clone -b kbot3 https://github.com/munger1985/kbot.git
mv kbot kbot3
git reset --hard origin/kbot3  # 强制将本地 main 分支重置为远程状态
```

### kbot3 env
conda create -n kbot3  python=3.12
conda activate kbot3
cd ./kbot3
pip install -r requirements.txt

### 后端API文档：
http://64.181.236.219:18099/docs 
http://64.181.236.219:18099/redoc 
http://64.181.236.219:18099/api/health

### Kbot3.0前端：
http://132.145.81.123:8080/ords/r/kbotui_dev/km-chat/home     chinase1/chinase1  admin/12345678
http://132.145.81.123:8080/ords/r/kbotui_dev/ai-platform-portal/home  admin/12345678

### nacos :
http://64.181.236.219:8848/nacos

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
### 配置 libreoffice 容器(把word/ppt数据转成pdf)
```bash
cd microservices/libreoffice
docker-compose up -d

# 服务端口修改docker-compose.yaml里的端口号。默认为9316
# 容器所在服务器ip和容器端口需要更新到app.properties文件的 libre_host 和 libre_port 两项中
```

### 配置 nacos 容器
```bash
# 首先在home目录下创建nacos/data和nacos/logs
# 然后修改nacos-init/docker-compose.yaml文件中的相关目录
cd nacos-init
docker-compose up -d
```
### 然后将配置文档注入nacos，只需要运行一次，之后就不再需要初始化nacos，除非配置有变更需要重新运行下面的命令
```bash
# 配置文档位于项目根目录的configuration文件夹中
python core/nacos_manager/nacos-init/load_to_nacos.py
# nacos默认管理界面
http://localhost:8848/nacos/
```

### 如果需要本地部署jina reranker模型，需要安装nvidia-cuda-toolkit
```bash
sudo apt install nvidia-cuda-toolkit
```


### 在OCI上部署时，需要放开端口
1.在VCN的security list中添加端口（8848、18099）
2.在系统层面放开端口
```bash
sudo iptables -nvL
sudo iptables -I INPUT -p tcp --dport 8848 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 18099 -j ACCEPT
```


### Oracle 23ai全文检索创建索引
```bash
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