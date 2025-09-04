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
git clone https://github.com/munger1985/kbot.git -b v3.0.1 
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

