
# 在Ubuntu境部署Kbot步骤概述
### 1.Kbot后台服务器以及系统准备
```bash
#1.准备Ubuntu 22或者以上版本 64位，或者Linux8或以上，CPU 至少4核心；Memory至少32GB；Disk Storage至少：200GB
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
git clone https://github.com/munger1985/kbot.git -b v3.0.3
#2.创建conda虚拟环境
cd kbot3
conda create -n kbot3 python=3.12
conda activate kbot3
pip install -r requirements.txt
#3.准备Kbot元数据库（Oracle23ai的schema）连接信息
sqlplus kbot_poc/VEctor#_123@10.45.151.152:1521/aipocpdb.databasessubnet.vcnpairs.oraclevcn.com
```

### 3.准备模型
```bash
#1.准备开源模型和LLM模型，或者准备LLM模型的api key信息。
#huggingface-cli download  BAAI/bge-reranker-v2-m3 --local-dir /home/opc/Models/bge-reranker-v2-m3
#huggingface-cli download  BAAI/bge-m3 --local-dir /home/opc/Models/bge-m3
#2.准备同义词模型 fasttext cc.zh.300.bin
# 使用 wget 下载
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz
# 或者使用 curl 下载
curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.bin.gz
# 准备Qwen3 rerank和embedding模型
# 在下载前，请先通过如下命令安装ModelScope
pip install modelscope
# 命令行下载完整模型库
modelscope download --model Qwen/Qwen3-Reranker-4B
modelscope download --model Qwen/Qwen3-Embedding-4B
# 下载完成后，请将模型文件放置到模型目录下，默认在 ~/.modelscope/models/Qwen/目录下
```

### 4.部署Docker容器：
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

#3.配置 nacos 容器
#3.1 创建nacos容器镜像
cd docs/nacos-init
docker-compose up -d

#3.2 Kbot 3.0后端的配置文件：
.env
configuration/app_config.json
configuration/db_config.json
configuration/model_config.json
每个微服务下有个.env，配置了默认值，可以不用修改。
Kbot的主服务默认端口：18099，Nacos的默认端口：8848

#3.3 然后将配置文档注入nacos，只需要运行一次，之后就不再需要初始化nacos，除非配置有变更需要重新运行下面的命令
python load_config.sh
#3.4 nacos默认管理界面，验证nacos安装
http://localhost:8848/nacos/

#4 部署elastic search容器
#4.1 获取elastic search官方镜像
docker pull docker.elastic.co/elasticsearch/elasticsearch:9.1.5
#4.2 创建elastic search网络
docker network create elastic
#4.3 创建elastic search存储目录
mkdir -p /home/ubuntu/elastic/eslog
mkdir -p /home/ubuntu/elastic/eskb
#4.4 启动elastic search容器：eslog容器用于记录全局日志，eskb用于作为kbot的全文索引库
docker run --name eslog --net elastic \
  -p 9201:9200 \
  -p 9301:9300 \
  -v /home/ubuntu/elastic/eslog:/usr/share/elasticsearch/data \
  -d -m 1GB \
  elasticsearch:9.1.5

docker run --name eskb --net elastic \
  -p 9202:9200 \
  -p 9302:9300 \
  -v /home/ubuntu/elastic/eskb:/usr/share/elasticsearch/data \
  -d -m 2GB \
  elasticsearch:9.1.5

#4.5 修改初始密码，将输出到控制台的密码复制到configuration/db_config.json文件中
docker exec -it eslog /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic
docker exec -it eskb /usr/share/elasticsearch/bin/elasticsearch-reset-password -u elastic

#4.5 验证elastic search安装，使用前面修改过的初始密码登录，用户名默认elastic
https://localhost:9201/
https://localhost:9202/

```

### 5.初始化Kbot数据库表信息
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
#3.执行apex_ui_v1.0.sql脚本
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

### 6.前端apex安装以及kbot UI部署
```bash
#1.安装apex
#2.部署Kbot UI到apex中
```

### 7.启动后台服务
```bash
cd /home/ubuntu/kbot3
#5.1启动后台服务
./start_kbot.sh
#5.2停止后台服务
./stop_kbot.sh
```

### 8.apex UI系统基本配置以及验证
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
