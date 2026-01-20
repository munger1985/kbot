
# 在Ubuntu境部署Kbot步骤概述
### 1.Kbot后台服务器以及系统准备
```bash
#1.准备Ubuntu 22或者以上版本 64位，或者Linux8或以上，CPU 至少8核心；Memory至少64GB；Disk Storage至少：200GB
# 如果要效果好，推荐至少有一块8G显存的显卡来运行Qwen Embedding 和 rerank 模型
##推荐Ubuntu 22.0464位
#2.安装必要的包以及网络等配置
#在OCI上部署时，需要放开端口，以及关闭防火墙
#在VCN的security list中添加端口（18099、1521、22）
#Ubuntu系统层面放开端口
sudo iptables -nvL
sudo iptables -I INPUT -p tcp --dport 18099 -j ACCEPT
sudo sh -c "iptables-save > /etc/iptables/rules.v4"
#Linux关闭防火墙
systemctl stop firewalld
#3.安装必要的软件包
#3.1.下载conda软件包，并安装
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
sh Anaconda3-2024.10-1-Linux-x86_64.sh -b -u -p ~/anaconda3
~/anaconda3/bin/conda init bash
#3.2 git安装
sudo apt install git -y
#3.3 安装libreoffice
sudo apt install libreoffice -y
```

### 2.Kbot代码下载以及依赖包准备
```bash
# 1.下载Kbot3.0源代码，Hub提供子版本的Tag
git clone https://github.com/munger1985/kbot.git -b v3.1

# 2 根据GPU版本安装cuda,torch和transformers，如果没有GPU则跳过这一步，直接到步骤3
# 配置nvidia官方源并升级cuda，如果需要
# 添加NVIDIA官方仓库 (适用于Ubuntu 22.04)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
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
# 验证cuda版本
nvcc --version

# 3.创建conda虚拟环境
cd kbot3
conda create -n kbot3 python=3.12
conda activate kbot3
pip install -r requirements.txt
# 根据cuda版本安装torch和transformers，例如：如果cuda版本为12.8，则安装torch和transformers如下
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers
# 2.2 安装flash-attn
# 安装 flash-attn 所需的构建依赖
pip install psutil ninja packaging
pip install flash-attn --no-build-isolation

```

### 3.安装Tesseract OCR引擎（可选）
```bash
conda install -c conda-forge tesserocr tesseract
```

# 验证安装
```bash
tesseract --version
tesseract --list-langs #这个命令会输出语言包安装路径，如/home/chris/miniconda3/envs/cube/share/tessdata/
# 根据语言包输出路径，设置环境变量
# 在.env中修改
TESSDATA_PREFIX=/home/chris/miniconda3/envs/cube/share/tessdata/
# 根据输出的语言包路径，修改python代码中的tesserocr.PyTessBaseAPI(path=)参数，例如：/home/chris/miniconda3/envs/cube/share/tessdata/
python -c "import tesserocr; api = tesserocr.PyTessBaseAPI(path='/home/chris/miniconda3/envs/cube/share/tessdata/'); print('支持的语言:', api.GetAvailableLanguages()); api.End()"

# 支持的语言: ['afr', 'amh', 'ara', 'asm', 'aze', 'aze_cyrl', 'bel', 'ben', 'bod', 'bos', 'bre', 'bul', 'cat', 'ceb', 'ces', 'chi_sim', 'chi_sim_vert', 'chi_tra', 'chi_tra_vert', 'chr', 'cos', 'cym', 'dan', 'deu', 'div', 'dzo', 'ell', 'eng', 'enm', 'epo', 'equ', 'est', 'eus', 'fao', 'fas', 'fil', 'fin', 'fra', 'frk', 'frm', 'fry', 'gla', 'gle', 'glg', 'grc', 'guj', 'hat', 'heb', 'hin', 'hrv', 'hun', 'hye', 'iku', 'ind', 'isl', 'ita', 'ita_old', 'jav', 'jpn', 'jpn_vert', 'kan', 'kat', 'kat_old', 'kaz', 'khm', 'kir', 'kmr', 'kor', 'kor_vert', 'lao', 'lat', 'lav', 'lit', 'ltz', 'mal', 'mar', 'mkd', 'mlt', 'mon', 'mri', 'msa', 'mya', 'nep', 'nld', 'nor', 'oci', 'ori', 'osd', 'pan', 'pol', 'por', 'pus', 'que', 'ron', 'rus', 'san', 'sin', 'slk', 'slv', 'snd', 'spa', 'spa_old', 'sqi', 'srp', 'srp_latn', 'sun', 'swa', 'swe', 'syr', 'tam', 'tat', 'tel', 'tgk', 'tha', 'tir', 'ton', 'tur', 'uig', 'ukr', 'urd', 'uzb', 'uzb_cyrl', 'vie', 'yid', 'yor']
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

### 5.部署elk用于收集日志（可选）
```bash
cd docs/install/elk-log-container

# 根据实际情况修改.env文件中的配置，如果.env不存在则复制.env.example文件并重命名为.env

# 启动容器集群
./start_elk.sh
```

### 6.初始化Kbot数据库表信息
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

### 7.前端apex安装以及kbot UI部署
```bash
#1.安装apex
#2.部署Kbot UI到apex中
```

### 8.启动后台服务
```bash
cd /home/ubuntu/kbot3
#5.1启动后台服务
./start_kbot.sh
#5.2停止后台服务
./stop_kbot.sh
```

### 9.apex UI系统基本配置以及验证
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
