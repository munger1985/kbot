# KBot AIOps观测栈自动化部署资产

本目录是生产自动化部署包的受控资产目录，不是用户直接执行入口。

- 唯一入口：`scripts/aiops-stack`
- 唯一用户配置：`var/aiops-stack/aiops-stack.ini`
- 生产部署说明：`docs/operations/aiops-observability-production-deployment.md`
- 人工安装与运维：`docs/operations/aiops-observability-manual-deployment.md`

`compose.yaml`只定义稳定的中心组件和主机采集组件。数据库Exporter、Oracle Alert
Collector、逐目标Secret、Volume和KBot Webhook签名桥由入口脚本生成到
`var/aiops-stack/generated/compose.generated.yaml`。不要手工维护生成文件，也不要从
本目录直接执行裸`docker compose up`。

OEM不属于此安装包，部署完成后在AIOps App内配置。
