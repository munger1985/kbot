
# 在Ubuntu境部署Kbot步骤概述
### 1.在OCI上部署时，需要放开端口，以及关闭防火墙
1.在VCN的security list中添加端口（8848、18099）
2.在系统层面放开端口
```bash
sudo iptables -nvL
sudo iptables -I INPUT -p tcp --dport 8848 -j ACCEPT
sudo iptables -I INPUT -p tcp --dport 18099 -j ACCEPT
```

### 2.部署Docker 以及Redis容器：
```bash
sudo apt install docker
sudo apt  install docker-compose
sudo usermod -aG docker $USER

#2.1 获取Redis容器镜像
sudo docker pull redis:latest
#2.2 启动redis容器
sudo docker run --name kbot-redis -d \
  -p 6379:6379 \
  -v /home/ubuntu/kbot_data/redis:/data \
  redis:latest \
  redis-server --appendonly yes --requirepass "welcome1"

#2.3置容器自动重启
docker update --restart unless-stopped kbot-redis #容器名或ID
```
### 3.配置 libreoffice 容器(把word/ppt数据转成pdf)
```bash
##方法一：从开发环境导出容器，并导入镜像
#3.1 docker export -o my_nginx.tar 容器名  
#3.2 然后到新的开发机，docker load -i my_nginx.tar
#3.3 再给导入的镜像重新命名即可 docker tag 镜像名 《名字》
#安装示例：
docker load -i libre-images.tar
docker tag f9b5dc8f2fb5 libreoffice
#如果关闭了防火墙，需要重启docker。
sudo systemctl restart docker
#3.4 启动容器
sudo docker run --name libreoffice -d \
  -p 9316:9316 \
  -v /home/opc/kbot_data/libreoffice:/data \
  libreoffice:latest 

##方法二：安装部署libreoffice
#3.1 安装从libreoffice 的docker-compose文件中部署
cd microservices/libreoffice
sudo docker-compose up -d
#3.2 启动容器
sudo docker run --name libreoffice -d \
  -p 9316:9316 \
  -v /home/opc/kbot_data/libreoffice:/data \
  libreoffice:latest 

# 服务端口修改docker-compose.yaml里的端口号。默认为9316
# 容器所在服务器ip和容器端口需要更新到app.properties文件的 libre_host 和 libre_port 两项中
```

### 4.配置 nacos 容器
```bash
#4.1 创建nacos容器镜像
cd core/nacos_manager/nacos-init
docker-compose up -d

#4.2 Kbot 3.0后端的配置文件：
.env
configconfiguration/app_config.json
configconfiguration/db_config.json
configconfiguration/model_config.json
每个微服务下有个.env，配置了默认值，可以不用修改。
Kbot的主服务默认端口：18099，Nacos的默认端口：8848

#4.3 然后将配置文档注入nacos，只需要运行一次，之后就不再需要初始化nacos，除非配置有变更需要重新运行下面的命令
python core/nacos_manager/load_to_nacos.py
#4.4 nacos默认管理界面，验证nacos安装
http://localhost:8848/nacos/
```

### 5.初始化Kbot数据库表信息
sqlplus kbot_poc/VEctor#_123@10.45.151.152:1521/aipocpdb.databasessubnet.vcnpairs.oraclevcn.com @kbot_ddl_v1.0.sql

### 6.Oracle 23ai全文检索创建索引
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

### 6.准备模型

#6.1准备开源模型和LLM模型，或者准备LLM模型的api key信息在前端界面中配置。
huggingface-cli download  BAAI/bge-reranker-v2-m3 --local-dir /home/opc/Models/bge-reranker-v2-m3
huggingface-cli download  BAAI/bge-m3 --local-dir /home/opc/Models/bge-m3

#6.2设置OCI GenAI Model
#在Chicago开通GenAI服务，在用户下创建API Key。
update KBOT_MD_MODELS set model_params = '{"temperature":0,"max_tokens":4000,"profile_path":"/home/ubuntu/.oci/config","compartment_id":"ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"}'
where model_id = 101;
update KBOT_MD_MODELS set model_params = '{"config_profile":"DEFAULT","config_file":"~/.oci/config","compartment_id":"ocid1.compartment.oc1..aaaaaaaapw7vdtp4sakhe7zs7tybhtapgc26ga472v62ykdboxxbuo2cad6q"}'
where model_id = 102;
commit;
```

### 7.启动后台服务
```bash
cd /home/ubuntu/kbot3
#7.1启动后台服务
./start_kbot.sh
#7.2停止后台服务
./stop_kbot.sh
```

