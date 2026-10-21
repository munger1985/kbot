# KBot 1.0 OCIR 生产部署与初始化方案

## 文档状态

本文定义 KBot 正式容器版本 `1.0` 的生产部署、空库初始化、验收、备份和回滚方案。
镜像已经发布，部署尚未执行。正式执行前必须先完成本文的 Digest 锁定门禁；在门禁完成前，
不得把当前标签式 `kbot-deploy` 当作不可变生产部署入口。

发布事实：

- Registry：`nrt.ocir.io/sehubjapacprod/kbot`；
- 平台：`linux/amd64`；
- 镜像源码 Commit：`497be08edc55d458582665482a50c57133ada79e`；
- 正式标签：`1.0`；
- 10 个正式镜像及 Digest 以
  [`installation/README.md`](../../installation/README.md#正式版本-10) 为准；
- 试发布标签 `4.0.0-e1b511db` 已删除，不是可用回滚版本。

## 目标架构

第一阶段采用仓库现有工具支持的单机 Docker Compose 交付，Oracle 作为外部受管数据库。
高可用、多节点共享对象存储和 Kubernetes 不属于本次 1.0 部署范围。

```text
Internet
   │ HTTPS 443
OCI Load Balancer / WAF
   │ HTTP 8080，仅允许负载均衡子网访问
KBot Compute，Linux x86_64
   ├─ kbot-ui → Main API
   ├─ 8 个业务服务镜像
   ├─ kbot-installer 工具容器
   └─ 本机持久化数据、日志与 Secret
        ├─ OCIR nrt.ocir.io，只读拉取
        ├─ Oracle 26ai PDB，TCP 1521
        └─ 模型 Provider、OEM 和监控源
```

只有 `kbot-ui` 发布宿主机端口。Main API 和其他内部服务只在 Compose 网络内互访。
OEM 只在 AIOps App 内配置，不写入部署 INI 或 Compose。

## 阶段一：部署前门禁

### 1. Digest 锁定

当前 `deployment.ini` 只接受 `image_version = 1.0`，在线发布器会把服务渲染为
`<registry>/<image>:1.0`。仓库已保存受版本控制的正式镜像锁定清单：

```text
installation/releases/1.0-images.json
```

该清单与 `installation/releases/1.0-docling-models.json` 共同约束完整脱机包：制作时
核对 Registry Digest，并打包当前解析路径使用的 Layout、TableFormer、EasyOCR；目标机
导入后核对 Image ID、逐文件校验模型，并以 `--pull never` 启动。在线部署入口的 Compose
Digest 引用和拉取门禁仍需按下列要求补齐，不能因为已有离线门禁而视为完成：

锁定能力必须满足：

1. 清单包含 Registry、版本、镜像名、Digest、源码 Commit 和平台；
2. 发布器验证所选业务镜像、`kbot-ui` 和 `kbot-installer` 均有唯一 Digest；
3. Compose 镜像引用渲染为 `<registry>/<image>@sha256:<digest>`；
4. 缺失、重复、非法或与 `1.0` 发布记录不一致时拒绝部署；
5. `installation-input.json` 记录锁文件路径、Hash 和最终镜像引用；
6. `kbot-deploy` 在 `up` 前显式拉取镜像并核对远端 Digest；
7. 发布工具离线契约测试覆盖以上行为。

门禁完成后，本文后续命令中的 `compose.yaml` 才是正式可执行制品。门禁完成前只允许渲染
和检查，不允许对生产主机执行 `up`。

### 2. App 范围一次性确认

`platform_core` 始终初始化，其余 Schema 范围由 `[apps]` 一次确定：

| App | 额外业务服务 |
|---|---|
| Knowledge Retrieval | Knowledge Core、Knowledge Retrieval App、Agent Runtime、Data Query |
| Media Studio | Media Studio App |
| AIOps | Knowledge Core、Agent Runtime、Data Query、AIOps Agent |

如果三个 App 后续都可能使用，首次部署应全部选择。初始化器拒绝向已有 KBot Schema
追加另一组服务对象；改变范围时必须准备新的空白 Schema。

### 3. 变更窗口与责任人

执行前明确：

- OCI 网络、Compute、Load Balancer 和 IAM 负责人；
- Oracle PDB、Schema、授权、备份和恢复负责人；
- KBot 配置、Secret、初始化和应用验收负责人；
- DNS/TLS 切换窗口及失败回退决定人；
- 发布证据保存位置。

## 阶段二：OCI 基础设施

### 1. Compute 基线

部署节点必须是 `linux/amd64`，不得使用 OCI Ampere ARM。可按以下未压测起始值申请，
上线前再根据真实并发、模型位置和资料规模压测调整：

| 场景 | 起始资源建议 |
|---|---|
| 外部模型 Provider、轻量业务 | 4 OCPU、32 GB 内存、150 GB 磁盘 |
| 三个 App 全开、OCR/CPU Torch | 8 OCPU、64 GB 内存、300 GB 磁盘 |
| 同机启用完整观测栈 | 8–12 OCPU、64–96 GB 内存，并单独规划日志容量 |

主机安装 Docker Engine、Docker Compose v2、Git 和 Python 3.10+。建议使用独立路径：

```text
/opt/kbot/releases/1.0       版本化部署工具
/srv/kbot/data               业务文件和模型数据
/srv/kbot/log                应用日志
/etc/kbot/secrets            Oracle 密码和 KBot 主密钥
```

Docker 日志必须配置轮转；系统盘、业务数据盘和日志盘分别设置容量告警。

### 2. 网络与入口

- Compute 放在私有子网，通过 OCI Bastion 或受限管理网访问；
- Load Balancer/WAF 终止 TLS，对 Compute 使用 `8080`；
- `22` 只允许 Bastion 或管理网段；
- `8080` 只允许 Load Balancer 子网；
- Compute 到 Oracle PDB 开放 `1521`；
- 出站 `443` 允许访问 OCIR 和已批准的模型 Provider；
- 不向宿主机发布任何 KBot 内部服务端口；
- Load Balancer 存活检查使用 `/healthz`，发布验收另行检查 Main API `/readyz`。

### 3. OCIR 拉取权限

部署身份只授予目标 Compartment 内 OCIR Repository 的读取权限，不授予推送和删除权限。
使用 OCI Auth Token 交互登录：

```bash
docker login nrt.ocir.io
```

用户名和 Auth Token 不写入脚本、Git、部署文档或 Shell 历史。登录后先拉取并检查
`kbot-installer` 和 `kbot-ui`，确认 DNS、TLS、IAM 和网络链路正常。

## 阶段三：Oracle 准备

使用独立 Oracle 26ai PDB 和独立 KBot Schema。Schema 用户至少需要：

- `CREATE SESSION`；
- `CREATE TABLE`；
- `CREATE VIEW`；
- 专用默认表空间上的足够 `QUOTA`。

默认表空间必须启用 `SEGMENT SPACE MANAGEMENT AUTO`，不得使用 `SYSTEM` 作为业务默认
表空间。启用 Knowledge Core 时，由 PDB 管理员对实际 Schema 用户执行：

```sql
GRANT EXECUTE ON SYS.DBMS_ALERT TO <KBOT_SCHEMA>;
```

执行初始化前，由 DBA 确认当前用户下没有任何 `KBOT_%` 表或视图，并完成初始化前备份点。
应用初始化器也会再次检查空库、权限、表空间和 `DBMS_ALERT`，检查不通过时不会执行第一条 DDL。

## 阶段四：部署配置与 Secret

在版本目录创建部署配置：

```bash
cp installation/dev/deployment.ini.example installation/dev/deployment.ini
chmod 600 installation/dev/deployment.ini
```

三个 App 全开时的核心配置如下，尖括号内容必须替换：

```ini
[deployment]
project_name = kbot4-prod
registry = nrt.ocir.io/sehubjapacprod/kbot
image_version = 1.0
ui_port = 8080
public_base_url = https://<正式域名>
embedding_dimension = 2048

[apps]
knowledge_retrieval = true
media_studio = true
aiops = true

[database]
host = <Oracle私网地址>
port = 1521
service_name = <PDB Service Name>
username = <KBOT Schema>

[paths]
data_dir = /srv/kbot/data
log_dir = /srv/kbot/log

[secrets]
oracle_password_file = /etc/kbot/secrets/oracle_password
master_key_file = /etc/kbot/secrets/master_key

[aiops]
agent_execution_enabled = false
mutation_enabled = false

[observability]
enabled = false
```

首次准备 Secret：

```bash
installation/dev/kbot-release prepare-secrets \
  --config installation/dev/deployment.ini
```

Oracle 密码通过终端无回显输入。KBot 主密钥只在首次部署生成，之后升级、回滚和迁移必须
恢复同一个文件；重新生成会导致已有托管凭据无法解密。主密钥必须立即保存到 OCI Vault
或等价受控 Secret 系统，并保留受控离线备份。

## 阶段五：渲染、拉取与静态预检

Digest 门禁完成后执行：

```bash
installation/dev/kbot-release validate \
  --deployment-config installation/dev/deployment.ini

installation/dev/kbot-release render-deployment

docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  config --quiet

docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  pull
```

检查生成的 `installation-input.json`、App 范围、服务数量、外部 URL、数据路径、Secret
权限和所有镜像 Digest。此时不要启动业务容器；空库应先由 Installer 完成初始化，避免服务
因缺表反复重启。

## 阶段六：空白 Oracle Schema 初始化

以下命令均通过 `kbot-installer:1.0` 执行，并复用最终运行配置和 Compose Secret。

### 1. 解析 DDL 计划

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py \
  --config /opt/kbot/configuration/oracle_schema_services.ini \
  --dry-run
```

这一步只解析服务选择、DDL 和 Prompt Catalog，不连接或修改数据库。

### 2. 执行规范 Schema 初始化

DBA 再次确认 PDB、Schema、空库、授权和恢复点后执行：

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py \
  --config /opt/kbot/configuration/oracle_schema_services.ini
```

该命令创建所选服务的规范对象、Prompt Catalog、默认 Domain、平台 `ADMIN` 用户和
`platform_admin` 权限。它不会创建模型、业务 Agent、知识资产、会话或真实 AIOps Target。

Oracle DDL 会隐式提交。如果中途失败，不得用零散 SQL 手工补表，也不得直接重跑；应保存
失败证据、修复规范源、删除并重建空白 Schema，再从头初始化。

### 3. 校验平台基础数据

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py --check-foundation
```

## 阶段七：App 首次初始化

只执行已选择 App 的命令。所有初始管理员密码必须在开放公网入口前轮换。

### Knowledge Retrieval

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/initialize_product_app.py \
  knowledge_retrieval --show-initial-password
```

创建 `knowledge_retrieval_portal` Domain 和 `knowledgeadmin`。重复执行不会重置已有密码；
日常复查使用同一命令加 `--check-only`，不加 `--show-initial-password`。

### Media Studio

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/initialize_product_app.py \
  media_studio --show-initial-password
```

创建 `media_studio_portal` Domain 和 `mediaadmin`。日常复查使用 `--check-only`。

### AIOps

先写入默认数据库巡检模板：

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/seed_aiops_inspection_templates.py
```

再创建 AIOps 固定业务资源：

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/initialize_aiops.py
```

该命令创建 `aiops_portal`、`aiopsadmin`、默认运维知识库、`operations-manuals` 和
`diagnosis-cases`。AIOps 写初始化会恢复固定初始密码，生产环境完成首次初始化后不得把它
当作健康检查重复执行。日常只读复查使用：

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  --profile tools run --rm installer \
  python /opt/kbot/scripts/db/initialize_aiops.py --check-only
```

## 阶段八：启动与技术验收

初始化全部通过后，执行已具备 Digest 锁定能力的无参数入口：

```bash
installation/dev/kbot-deploy
```

部署器必须重新渲染出相同的 Digest 引用，先拉取并校验，再执行
`up --detach --remove-orphans`。验收至少包括：

```bash
docker compose \
  --project-directory installation/generated/deployment \
  -f installation/generated/deployment/compose.yaml \
  ps

curl -fsS http://127.0.0.1:8080/healthz
```

还要逐项确认：

1. 预期容器全部运行且没有持续重启；
2. 实际 RepoDigest 与 `1.0` 锁定清单完全一致；
3. Main API `/readyz` 的数据库检查为 `ok`；
4. 最近日志没有 `ORA-`、Secret 读取失败、配置缺失或持续重试；
5. 平台及所选 App 管理员能够登录；
6. 所有初始密码已经轮换；
7. `/srv/kbot/data` 和 `/srv/kbot/log` 权限及剩余空间正常；
8. 小文件上传、解析、索引链路成功；
9. 测试 LLM 和 Embedding 调用成功；
10. SSE/流式回答通过 Load Balancer，不被代理缓冲或提前断开。

当前 Compose 没有容器级 Healthcheck，因此“容器显示 Up”不能作为部署成功证据。

## 阶段九：业务初始化顺序

技术验收后按以下顺序完成业务配置：

1. 轮换平台及各 App 初始管理员密码；
2. 在 Model Serving 中录入 Provider 和模型，初始化器不会预置模型；
3. 配置 LLM、Embedding、OCR、VLM 或图片生成模型绑定；
4. 创建正式业务 Domain、用户、成员和角色；
5. Knowledge Retrieval 创建知识库，完成测试资料的上传、审批、发布和问答；
6. Media Studio 配置生成模型绑定并完成一次生成；
7. AIOps 配置监控源、Target、知识库和 Agent；
8. OEM 只在 AIOps App 内接入；
9. `agent_execution_enabled` 和 `mutation_enabled` 保持关闭，完成只读诊断、审批和审计
   验收后再单独评审开启；
10. KBot 核心服务验收稳定后，再启用 Prometheus、Loki、Alloy 和 Exporter 观测栈。

AIOps 观测栈保持独立 Compose Project，仍由统一 `deployment.ini` 映射到既有
`scripts/aiops-stack` 零参数入口。每个容器只挂载自身需要的 Secret。

## 阶段十：备份、恢复与回滚

上线前完成并验证以下备份：

- Oracle RMAN 或受管数据库恢复点；
- `/srv/kbot/data`；
- KBot 主密钥；
- `deployment.ini`；
- `1.0` 镜像 Digest 锁定清单；
- Load Balancer、DNS、TLS、NSG 和 IAM 配置；
- 部署证据与初始化输出，但不包含明文 Secret。

旧试发布镜像已删除，因此不存在旧镜像回滚路径。当前 1.0 的恢复方式是重新部署相同
Digest，并恢复相匹配的 Oracle、数据目录和主密钥备份。仅回退镜像、不回退不兼容的 Schema
或数据不是有效回滚。

后续正式版本至少保留前一个已验收版本的 Digest，完成新版本稳定性观察和恢复演练后再决定
是否清理。

## 执行证据清单

部署完成后归档：

- 部署工具 Git Commit 与干净工作区证明；
- Digest 锁文件及其 Hash；
- `installation-input.json`；
- `docker compose config` 校验结果；
- 所有运行容器的镜像 Digest；
- Oracle PDB、Schema、对象数和初始化检查结果；
- App `--check-only` 结果；
- `/healthz`、`/readyz` 和关键业务冒烟结果；
- 初始密码已轮换的审计事实，不记录新密码；
- 备份/恢复点标识；
- DNS/TLS 切换和回退结论。

只有上述证据完整，才能把状态标记为“1.0 生产部署完成”。
