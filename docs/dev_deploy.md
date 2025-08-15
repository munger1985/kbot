
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
http://64.181.236.219:18099/api/health

### Kbot3.0前端：
http://132.145.81.123:8080/ords/r/kbotui_dev/km-chat/home     chinase1/chinase1  admin/12345678
http://132.145.81.123:8080/ords/r/kbotui_dev/ai-platform-portal/home  admin/12345678

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
cd nacos-init
docker-compose up -d
```
### 然后将配置文档注入nacos，只需要运行一次，之后就不再需要初始化nacos，除非配置有变更需要重新运行下面的命令
```bash
python nacos-init/load_properties_to_nacos.py
# nacos默认管理界面
http://localhost:8848/nacos/
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