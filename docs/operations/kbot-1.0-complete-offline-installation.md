# KBot 1.0 完全离线安装手册

本文用于从一个 KBot 离线归档，在不连接 Git、PyPI 或 OCIR 的目标机上完成 KBot 1.0
安装、Oracle 空库初始化、首次账号初始化、启动和验收。目标机仍须能访问目标 Oracle PDB；
需要调用外部模型 Provider 的业务功能，还须能访问对应 Provider，或改用内网模型服务。

## 1. 交付物与版本

安装人员只需要取得以下三个文件：

```text
kbot-1.0-<tool-revision>-linux-amd64.tar.zst
kbot-1.0-<tool-revision>-linux-amd64.tar.zst.sha256
KBot-1.0-Complete-Offline-Installation.md
```

离线归档固定信息：

| 项目 | 值 |
|---|---|
| 平台 | `linux/amd64` |
| KBot 镜像 | 10 个正式镜像 |
| 部署工具修订 | 以包内 `bundle-manifest.json` 为准 |
| 归档字节数 | 以 Object Storage 对象元数据为准 |
| 归档 SHA-256 | 以同名 `.sha256` 文件为准 |

归档包含 10 个 KBot 镜像、镜像锁、部署工具、Oracle 初始化器、静态 UI、Docling 2.114.0
当前运行路径所需的 Layout、TableFormer、EasyOCR 模型、逐文件模型清单和内部
`SHA256SUMS`。不包含当前代码未调用的 Docling VLM、公式、图表和其他 OCR 模型，也不
包含 Oracle 密码、KBot 主密钥、真实模型 Provider 凭据、用户数据或 Prometheus、Loki 等
第三方观测镜像。以后启用新的 Docling 能力时，必须发布包含对应模型的新离线包。

部署全程不需要 OCIR 用户名或 Auth Token。`load-images` 从包内归档导入镜像，
`deploy` 固定使用 `--pull never`。

## 2. 安装前检查

### 2.1 宿主机基线

目标机必须预先具备：

- Linux x86_64；不支持 ARM/Ampere；
- Python 3.11 或更高版本；
- Docker Engine；
- Docker Compose v2 插件；
- GNU tar、zstd、sha256sum 和 curl；
- 执行用户可运行 Docker；
- 到 Oracle PDB 的网络连通性。

安装前执行：

```bash
uname -m
python3 --version
docker version
docker compose version
tar --version
zstd --version
df -h
```

`uname -m` 必须返回 `x86_64`。如果 Docker、Compose 或 Python 尚未安装，应由操作系统
管理员另行准备匹配目标发行版的离线依赖；KBot 核心包不安装宿主机软件。

未经压测的起始资源建议：

| 场景 | 建议起点 |
|---|---|
| 外部模型 Provider、轻量业务 | 4 OCPU、32 GB 内存、150 GB 磁盘 |
| 三个 App 全开、OCR/CPU Torch | 8 OCPU、64 GB 内存、300 GB 磁盘 |

只需对用户入口开放 `ui_port`，默认是 TCP `8080`。KBot 内部服务端口不得向宿主机网络
发布。目标机到 Oracle 通常需要 TCP `1521`。

### 2.2 Oracle PDB 和 Schema

使用专用 PDB 和专用 KBot Schema。下列 SQL 由 DBA 连接到目标 PDB 后按实际命名、容量
和密码策略调整执行：

```sql
CREATE TABLESPACE KBOT_DATA
  DATAFILE SIZE 5G
  AUTOEXTEND ON NEXT 256M
  MAXSIZE UNLIMITED
  SEGMENT SPACE MANAGEMENT AUTO;

CREATE USER KBOT_APP IDENTIFIED BY "<由DBA安全录入的密码>"
  DEFAULT TABLESPACE KBOT_DATA
  TEMPORARY TABLESPACE TEMP;

GRANT CREATE SESSION, CREATE TABLE, CREATE VIEW TO KBOT_APP;
ALTER USER KBOT_APP QUOTA UNLIMITED ON KBOT_DATA;
GRANT EXECUTE ON SYS.DBMS_ALERT TO KBOT_APP;
```

如果数据库未启用 Oracle Managed Files，`DATAFILE` 必须由 DBA 指定合规路径。如果创建
表空间时报 `ORA-28361: Master key not yet set`，先由 DBA 为该 PDB 建立 TDE Master Key；
不要绕过 TDE，也不要在部署记录中保存 Wallet 密码。

默认表空间必须是 ASSM，不能使用 `SYSTEM`。初始化前须建立数据库恢复点，并以 KBot
Schema 用户确认当前 Schema 没有 KBot 对象：

```sql
SELECT COUNT(*) AS KBOT_OBJECT_COUNT
FROM USER_OBJECTS
WHERE OBJECT_NAME LIKE 'KBOT\_%' ESCAPE '\';
```

结果必须为 `0`。首次 App 范围一旦用于建库，就不能再向同一 Schema 追加其他 App。

## 3. 校验并解压交付包

把归档和本手册复制到目标机的版本目录，例如 `/opt/kbot/releases`。先验证外层归档：

```bash
cd /opt/kbot/releases
sha256sum --check kbot-1.0-<tool-revision>-linux-amd64.tar.zst.sha256
```

必须看到：

```text
kbot-1.0-<tool-revision>-linux-amd64.tar.zst: OK
```

解压并校验包内所有受控文件：

```bash
tar --zstd -xf kbot-1.0-<tool-revision>-linux-amd64.tar.zst
cd kbot-1.0
chmod +x kbot-offline
./kbot-offline verify-package
```

任何 `FAILED`、缺失文件或 SHA-256 不匹配都必须停止部署并重新取得归档，不能跳过校验。

## 4. 导入 10 个离线镜像

```bash
./kbot-offline load-images
./kbot-offline verify-images
```

`load-images` 会再次校验离线包，然后执行本地 `docker image load`。`verify-images` 将实际
Image ID 与包内清单逐一比对，必须显示 10 个镜像全部通过。此步骤不访问 OCIR。

不要执行 `docker compose pull`，也不要自行更改包内 Registry 或镜像标签。

## 5. 创建部署配置

首次执行：

```bash
./kbot-offline prepare
chmod 600 runtime/deployment.ini
```

编辑 `runtime/deployment.ini`。以下是三个 App 全开的示例；尖括号内容必须替换：

```ini
[deployment]
project_name = kbot-prod
registry = nrt.ocir.io/sehubjapacprod/kbot
image_version = 1.0
ui_port = 8080
public_base_url = http://<用户实际访问的IP或域名>:8080
embedding_dimension = 2048

[apps]
knowledge_retrieval = true
media_studio = true
aiops = true

[database]
host = <Oracle地址>
port = 1521
service_name = <PDB Service Name>
username = <KBot Schema用户名>

[paths]
data_dir = /srv/kbot/data
log_dir = /srv/kbot/log

[secrets]
oracle_password_file = secrets/oracle_password
master_key_file = secrets/master_key

[aiops]
agent_execution_enabled = false
mutation_enabled = false

[observability]
enabled = false
```

注意：

- `registry` 和 `image_version` 保持包内默认值；它们标识本地导入镜像，不代表需要访问 OCIR；
- `public_base_url` 必须是浏览器实际使用的完整 HTTP(S) 地址，不能在远程部署时保留
  `127.0.0.1`，末尾不要加 `/`；
- `project_name` 只能包含小写字母、数字、下划线和连字符；
- 只启用本次建库需要的 App；如果将来可能使用三个 App，应在空库初始化前全部启用；
- 核心离线包没有第三方观测镜像，因此保持 `observability.enabled = false`；
- AIOps 自动执行和变更能力默认关闭，完成只读诊断与审批验收后再单独评审开启。

准备数据和日志目录：

```bash
sudo mkdir -p /srv/kbot/data /srv/kbot/log
sudo chown -R "$(id -u):$(id -g)" /srv/kbot
chmod 750 /srv/kbot/data /srv/kbot/log
```

如果使用其他目录，以上命令和 `deployment.ini` 必须保持一致。

## 6. 安装离线 Docling 模型资产

Knowledge Core 文档解析需要包内 Docling 2.114.0 运行模型。先完成第 5 节配置，工具将按
`data_dir` 安装到 `<data_dir>/models/docling_models`：

```bash
./kbot-offline install-models
./kbot-offline verify-models --full
```

安装器先验证模型归档和逐文件清单，再解压到同文件系统临时目录，完整校验后原子安装。
目标目录已有不同内容时会拒绝覆盖；不要手工混用其他 Docling 版本。完整校验覆盖：

- Layout：`docling-project--docling-layout-heron`；
- TableFormer：`docling-project--docling-models`；
- EasyOCR：`EasyOcr`。

默认 OCR 使用镜像内的 Tesseract `chi_sim`、`eng` 语言包；EasyOCR 模型用于部署时切换
OCR 引擎。生成图片裁剪不等同于图片分类或图片描述，因此当前包不携带相关 VLM。

## 7. 交互准备 Secret

```bash
./kbot-offline prepare-secrets
```

工具会无回显要求输入并确认 Oracle Schema 密码，并首次生成至少 32 字节的 KBot 主密钥：

```text
runtime/secrets/oracle_password
runtime/secrets/master_key
```

两者权限应为 `0600`。数据库密码、主密钥、API Key 和 Token 不得写入命令行、Shell
历史、安装手册、工单或聊天记录。

立即把 `master_key` 备份到受控 Secret 系统和受控离线介质。升级、迁移和恢复必须使用
同一主密钥；丢失后，数据库中已有托管凭据无法解密。

重复执行 `prepare-secrets` 默认保留现有 Secret。除非明确进行数据库密码轮换，不要使用
`--replace-oracle-password`；该参数也不会轮换主密钥。

## 8. 静态校验与渲染

```bash
./kbot-offline validate
./kbot-offline render
```

检查生成结果：

```bash
sed -n '1p' installation/generated/deployment/config/runtime-config.js
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  config --quiet
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  config --services
```

`runtime-config.js` 的 `mainApiBaseUrl` 必须等于 `public_base_url`，不能是空字符串。此时
不要启动业务容器，先完成空库初始化。

## 9. Oracle 空库初始化

### 9.1 只解析计划

```bash
./kbot-offline installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py \
  --config /opt/kbot/configuration/oracle_schema_services.ini \
  --dry-run
```

这一步不连接或修改数据库。核对服务选择后，由 DBA 再次确认目标 PDB、Schema、ASSM
表空间、额度、`DBMS_ALERT` 授权、空库和恢复点。

### 9.2 执行规范初始化

```bash
./kbot-offline installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py \
  --config /opt/kbot/configuration/oracle_schema_services.ini
```

该命令创建所选服务的表、视图、索引、Prompt Catalog、默认 Domain、平台 `ADMIN` 用户
及其 `platform_admin` 角色。它不会创建真实模型、业务 Agent、知识资产、会话或 AIOps
Target。

Oracle DDL 会隐式提交。如果中途失败，不得用零散 SQL 补表或直接重跑。应保存错误证据，
修复根因，删除并重建空白 Schema，再从头初始化。

### 9.3 校验平台基础数据

```bash
./kbot-offline installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py \
  --check-foundation
```

必须确认默认 Domain、`ADMIN` 和 `platform_admin` 均校验通过。

## 10. 首次初始化所选 App

只运行 `deployment.ini` 中已启用 App 的命令。

### 10.1 Knowledge Retrieval

```bash
./kbot-offline installer python \
  /opt/kbot/scripts/db/initialize_product_app.py \
  knowledge_retrieval --show-initial-password
```

创建 `knowledge_retrieval_portal` 和 `knowledgeadmin`。只有首次创建凭据时才显示初始密码；
日常复查改用 `--check-only`，不要使用 `--show-initial-password`。

### 10.2 Media Studio

```bash
./kbot-offline installer python \
  /opt/kbot/scripts/db/initialize_product_app.py \
  media_studio --show-initial-password
```

创建 `media_studio_portal` 和 `mediaadmin`。日常复查改用 `--check-only`。

### 10.3 AIOps

```bash
./kbot-offline installer python \
  /opt/kbot/scripts/db/seed_aiops_inspection_templates.py

./kbot-offline installer python \
  /opt/kbot/scripts/db/initialize_aiops.py
```

创建默认巡检模板、`aiops_portal`、`aiopsadmin`、默认运维知识库、
`operations-manuals` 和 `diagnosis-cases`。

警告：AIOps 写初始化会恢复固定初始密码，不能作为日常健康检查重复执行。日常只读复查：

```bash
./kbot-offline installer python \
  /opt/kbot/scripts/db/initialize_aiops.py --check-only
```

## 11. 在受限网络中启动 KBot

所有数据库和 App 初始化完成后，先确认 `ui_port` 仅允许部署管理网访问，再执行：

```bash
./kbot-offline deploy
./kbot-offline status
```

部署器重新渲染配置并执行 `docker compose up --detach --remove-orphans --pull never`。
镜像缺失或被替换时会直接失败，不会访问 Registry。

初始密码轮换完成前，不得向普通用户网段或公网开放入口。

## 12. 技术检查

### 12.1 容器与入口

```bash
./kbot-offline status
curl -fsS http://127.0.0.1:8080/healthz
curl -fsS http://127.0.0.1:8080/ui/runtime-config.js
```

如果修改了默认 `ui_port`，把以上 `8080` 替换为实际端口。

确认：

- 所有预期容器都处于 `running`，没有持续重启；
- `/healthz` 返回成功；
- `runtime-config.js` 的 `mainApiBaseUrl` 非空且等于外部入口；
- 浏览器打开的协议、主机和端口与 `public_base_url` 完全一致。

检查 Main API 自身就绪状态：

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  exec -T main-api \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:18099/readyz', timeout=15).read().decode())"
```

检查最近日志：

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  logs --since 10m --tail 200
```

不得存在持续的 `ORA-`、Secret 读取失败、配置缺失、模型目录缺失或无限重试。

## 13. 首次账号交接与密码轮换

服务通过上述技术检查后，由部署人员记录适用于已启用 App 的初始账号，并立即完成密码
轮换。KBot 1.0 当前固定初始凭据如下：

| 范围 | 用户 | 初始密码 | 是否按 App 启用 |
|---|---|---|---|
| 平台 | `ADMIN` | `Admin@2026!` | 始终创建 |
| Knowledge Retrieval | `knowledgeadmin` | `KnowledgeAdmin@2026!` | 仅启用该 App |
| Media Studio | `mediaadmin` | `MediaAdmin@2026!` | 仅启用该 App |
| AIOps | `aiopsadmin` | `AIOpsAdmin@2026!` | 仅启用该 App |

默认密码是公开的部署引导值，不是生产 Secret。首次轮换前，只允许部署管理网访问入口。
新密码至少 12 位，并同时包含大小写字母、数字和特殊字符。安装证据只记录“已轮换”，不得
保存新密码。

如果没有平台管理 UI，可使用以下 Python 标准库脚本安全交互修改密码。它不会把密码写入
Shell 历史；每个账号分别执行一次：

```bash
python3 - <<'PY'
from getpass import getpass
import json
import urllib.error
import urllib.request

entries = {
    "platform": ("/api/v1/auth/platform/login", "/api/v1/auth/password"),
    "knowledge": (
        "/api/v1/apps/knowledge-retrieval/auth/login",
        "/api/v1/apps/knowledge-retrieval/auth/password",
    ),
    "media": (
        "/api/v1/apps/media-studio/auth/login",
        "/api/v1/apps/media-studio/auth/password",
    ),
    "aiops": (
        "/api/v1/apps/aiops/auth/login",
        "/api/v1/apps/aiops/auth/password",
    ),
}

base_url = input("KBot URL，例如 http://10.0.0.10:8080: ").strip().rstrip("/")
entry = input("入口 platform/knowledge/media/aiops: ").strip()
if entry not in entries:
    raise SystemExit("入口必须是 platform、knowledge、media 或 aiops")
user_id = input("用户 ID: ").strip()
current_password = getpass("当前密码: ")
new_password = getpass("新密码: ")
if new_password != getpass("再次输入新密码: "):
    raise SystemExit("两次输入的新密码不一致")
login_path, password_path = entries[entry]

def post(path, payload, token=""):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise SystemExit(error.read().decode("utf-8", errors="replace")) from error

session = post(login_path, {"user_id": user_id, "password": current_password})
post(
    password_path,
    {"current_password": current_password, "new_password": new_password},
    session["access_token"],
)
print("密码修改成功；旧会话已失效，请用新密码重新登录验证")
PY
```

所有已启用 App 的初始管理员均用新密码重新登录成功后，才可以按正式网络策略开放入口。

## 14. App 与业务验收

### 14.1 App 页面

只检查已启用的 App：

```text
Knowledge Retrieval: /ui/knowledge_retrieval/login.html
Media Studio:        /ui/media_studio/login.html
AIOps:               /ui/aiops/login.html
```

确认各初始管理员能够使用轮换后的新密码登录。HTTP 200 只证明路由存在，还要检查响应
内容和实际业务结果。

### 14.2 业务验收

初始化器不会预置真实模型。技术验收后按以下顺序配置：

1. 录入模型 Provider 和模型；
2. 配置 LLM、Embedding、OCR、VLM 或图片生成绑定；
3. 创建业务 Domain、用户、成员和角色；
4. Knowledge Retrieval 完成小文件上传、解析、索引和问答；
5. Media Studio 完成一次生成；
6. AIOps 配置监控源、Target、知识库和 Agent，并先完成只读诊断；
7. 需要观测栈时另行取得第三方观测离线包，不能让核心包在线拉取。

完全隔离互联网时，外部 SaaS 模型不可用；必须提供目标网络内可达的模型服务和相应模型
权重。KBot 核心包不包含通用 LLM/VLM 大模型权重。

## 15. 备份和恢复

上线前至少备份：

- Oracle PDB/RMAN 恢复点；
- `data_dir`；
- `runtime/secrets/master_key`；
- `runtime/deployment.ini`；
- 本离线归档及其 SHA-256；
- 初始化和验收记录，但不包含明文 Secret；
- 网络、DNS、TLS 和入口配置。

重新部署容器可以重复执行 `deploy`，但不得重复执行空库 DDL或 AIOps 写初始化。恢复必须
同时使用匹配版本的镜像、Oracle 数据、业务数据目录和原 KBot 主密钥。只回退镜像而不恢复
不兼容数据库结构不是有效回滚。

## 16. 常见故障

### `mainApiBaseUrl` 为空

说明使用了旧部署工具或错误归档。重新核对随包 `.sha256`，执行 `verify-package`，重新
渲染并重建 UI；不要只手工修改容器内文件。

### 部署尝试访问 OCIR

只使用 `./kbot-offline load-images` 和 `./kbot-offline deploy`。不要执行
`docker compose pull`，也不要移除 `--pull never`。

### `ORA-28361: Master key not yet set`

目标 PDB 尚未建立 TDE Master Key。由 DBA 修复 Wallet/TDE 后重新创建业务表空间；KBot
安装器不能也不应绕过数据库加密策略。

### 缺少 `CREATE TABLE`、`CREATE VIEW` 或表空间额度

由 PDB 管理员补齐最小权限、默认 ASSM 表空间和 QUOTA，再在仍为空的 Schema 上重新执行
预检。不要授予不必要的系统权限。

### Knowledge Core 报 `DBMS_ALERT` 权限错误

由 PDB 管理员执行：

```sql
GRANT EXECUTE ON SYS.DBMS_ALERT TO <KBOT_SCHEMA>;
```

### 初始化器拒绝非空 Schema

这是防覆盖保护。不要强行绕过。确认是否选错 Schema；若是失败的首次初始化，保存证据后
由 DBA 重建空白 Schema，再从头执行。

### 文档解析尝试联网下载模型

重新执行第 6 节的 `verify-models --full`。若提示安装回执、目录或文件摘要不匹配，重新
取得同版本归档；不要允许容器联网补下载，也不要在原目录中混放其他版本模型。

### AIOps 管理员密码意外恢复

写初始化器被重复执行。以后仅使用 `initialize_aiops.py --check-only` 做健康复查，并立即再次
轮换 AIOps 管理员密码。

### UI 页面可以打开但登录失败

依次检查 `runtime-config.js`、同源 `/api/` 路由、Main API `/readyz`、数据库连接和账号状态。
不能把所有 HTTP 错误都归因于 Nginx。

## 17. 完成标准

只有同时满足以下条件才能标记部署完成：

- 外层归档、包内文件、10 个镜像和 Docling 模型完整校验全部通过；
- Oracle 初始化及平台基础数据校验通过；
- 所选 App 首次初始化通过；
- 所有预期容器稳定运行；
- `/healthz`、Main API `/readyz` 和所选 App 登录通过；
- `mainApiBaseUrl` 正确；
- 初始管理员密码全部轮换；
- KBot 主密钥和数据库恢复点已备份；
- 至少一个已启用 App 完成真实业务冒烟；
- 安装证据中没有明文密码、Token、完整 DSN 或主密钥。
