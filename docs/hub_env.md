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
https://github.com/munger1985/kbot/tree/kbot3 #下载kbot3.0
git clone https://github.com/munger1985/kbot.git 
https://github.com/munger1985/kbot/releases/tag/v3.2
git clone -b kbot3 https://github.com/munger1985/kbot.git
git clone https://github.com/munger1985/kbot.git -b v3.0.3
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
http://132.145.81.123:8080/ords/r/kbotui_dev/km-chat/home     admin/12345678 chinase1/chinase1 
http://132.145.81.123:8080/ords/r/kbotui_dev/ai-platform-portal/home  admin/12345678


### nacos :
http://64.181.236.219:8848/nacos

### 数据库连接
sqlplus kbotui_dev/BotWelcome123##@132.145.81.123:1521/DB1007_pdb1.regionalpublics.hysunhevcn.oraclevcn.com
sqlplus sys/BotWelcome123##@132.145.81.123:1521/DB1007_pdb1.regionalpublics.hysunhevcn.oraclevcn.com as sysdba;
sys/BotWelcome123##
APEX_PUBLIC_USER

# 在数据库端，使用CDB级别开启DRCP
EXECUTE DBMS_CONNECTION_POOL.START_POOL();
EXECUTE DBMS_CONNECTION_POOL.CONFIGURE_POOL(pool_name => 'SYS_DEFAULT_CONNECTION_POOL',minsize => 4,maxsize => 40,incrsize => 2,inactivity_timeout => 300,max_lifetime_session => 86400);

# ssh穿透访问
ssh -i /Users/zzou/Desktop/Work/Config/my_oci_putty_key.pem -C -v -t -L 127.0.0.1:5601:10.0.81.62:5601 opc@130.61.43.111


# Embedding api
curl -X POST "http://localhost:9201/v1/embeddings" \
-H "Content-Type: application/json" -d '{"model_unique_name": "KBOT114/BGE_M3", "texts": ["这是一个测试文本", "这是另一个测试文本"], "batch_size": 32}'

# 添加文件分块
curl -X 'POST' \
  'http://localhost:8000/api/kb/file/chunk' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "kb_id": 30,
  "file_id": "15c8c426-2f04-4db9-bd2e-82f2288780fa",
  "embed_id": "9d99a0de-e832-440e-ad6b-f8e180a267e2",
  "new_chunk": "abcabc",
  "action": "update"
}'


curl -X 'POST' \
  'http://localhost:8000/api/kb/file/chunk' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "kb_id": 30,
  "file_id": "15c8c426-2f04-4db9-bd2e-82f2288780fa",
  "embed_id": "9d99a0de-e832-440e-ad6b-f8e180a267e2",
  "action": "delete"
}'

curl -X 'POST' \
  'http://localhost:8000/api/security/get_token' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d '{"username":"KBot_UI", "password":"xxxx"}'
