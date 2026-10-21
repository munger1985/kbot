# KBot 1.0 脱机部署工具包

离线包由联网发布机生成，目标机不需要 Git、Python 第三方包或 OCIR 凭据，只需要
Linux x86_64、Python 3.11+、Docker Engine 和 Docker Compose v2。

面向安装人员的完整宿主机、Oracle、初始化、启动、改密、验收和恢复流程见
[KBot 1.0 完全离线安装手册](../../docs/operations/kbot-1.0-complete-offline-installation.md)。

离线包包含 10 个 KBot 正式镜像、Docling 2.114.0 的 Layout、TableFormer、EasyOCR 运行
模型、逐文件模型清单、镜像锁定信息、自包含部署工具和 SHA-256 清单；不包含 Oracle
密码、KBot 主密钥或其他环境 Secret。AIOps App 属于 KBot 核心镜像，可以离线部署；
Prometheus、Loki 等第三方观测组件不在该核心包中。

## 在联网发布机制作

先登录 OCIR并拉取正式版全部镜像，然后在干净的发布工具 Commit 上执行：

```bash
installation/dev/kbot-release export-offline \
  --config installation/releases/1.0.ini \
  --docling-models-dir /srv/release-assets/docling_models \
  --output installation/generated/offline/kbot-1.0
```

制作器会逐个确认本地镜像的 RepoDigest 与 `installation/releases/1.0-images.json`
一致，并按 `installation/releases/1.0-docling-models.json` 校验和打包当前运行路径使用的
三个 Docling 模型。生成目录整体复制到移动介质；不要只复制镜像或模型归档。

## 在脱机目标机准备

```bash
cd kbot-1.0
./kbot-offline verify-package
./kbot-offline load-images
./kbot-offline prepare
vi runtime/deployment.ini
./kbot-offline install-models
./kbot-offline verify-models --full
./kbot-offline prepare-secrets
./kbot-offline validate
./kbot-offline render
```

`prepare-secrets` 无回显录入 Oracle Schema 密码，并首次生成稳定的 KBot 主密钥。立即对
`runtime/secrets/master_key` 做受控备份；以后升级必须继续使用同一个文件。

`install-models` 将模型原子安装到 `<data_dir>/models/docling_models`，逐文件校验大小和
SHA-256，并拒绝覆盖不匹配的现有目录。日常 `deploy` 会执行快速回执校验；介质验收和故障
排查使用 `verify-models --full` 重新读取全部模型文件。

## 空白 Schema 初始化

先只解析计划，不连接数据库：

```bash
./kbot-offline installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py \
  --config /opt/kbot/configuration/oracle_schema_services.ini --dry-run
```

DBA 确认 PDB、Schema、授权、空库和恢复点后，去掉 `--dry-run` 执行规范初始化，再校验：

```bash
./kbot-offline installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py \
  --config /opt/kbot/configuration/oracle_schema_services.ini

./kbot-offline installer \
  python /opt/kbot/scripts/db/apply_oracle_schema.py --check-foundation
```

随后只对 `runtime/deployment.ini` 中已选择的 App 执行对应首次初始化命令：

```bash
# Knowledge Retrieval
./kbot-offline installer python \
  /opt/kbot/scripts/db/initialize_product_app.py \
  knowledge_retrieval --show-initial-password

# Media Studio
./kbot-offline installer python \
  /opt/kbot/scripts/db/initialize_product_app.py \
  media_studio --show-initial-password

# AIOps
./kbot-offline installer python \
  /opt/kbot/scripts/db/seed_aiops_inspection_templates.py
./kbot-offline installer python \
  /opt/kbot/scripts/db/initialize_aiops.py
```

Oracle DDL 会隐式提交。初始化中途失败时保存证据并重建空白 Schema，不要用零散 SQL
修补或直接重放。

## 启动与检查

```bash
./kbot-offline deploy
./kbot-offline status
curl -fsS http://127.0.0.1:8080/healthz
```

离线入口始终使用 `--pull never`，缺少或被替换的镜像会直接失败，不会访问 Registry。
