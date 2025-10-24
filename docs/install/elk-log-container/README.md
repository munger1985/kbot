# ELK Log Container

这是一个基于 Docker 的 ELK（Elasticsearch, Logstash, Kibana）日志收集与分析的解决方案，用于集中管理和分析应用程序日志。

## 文件说明

- `.env`: 环境变量配置文件，包含 Elasticsearch 和 Kibana 的版本、密码、端口等设置。
- `.env.example`: 环境变量配置示例文件。
- `docker-compose.es.yml`: Elasticsearch 的 Docker Compose 配置文件。
- `docker-compose.kf.yml`: Kibana 和 Filebeat 的 Docker Compose 配置文件。
- `filebeat.yml`: Filebeat 的配置文件，用于收集和转发日志到 Elasticsearch。
- `start_elk.sh`: 启动 ELK 栈的脚本。
- `stop_elk.sh`: 停止 ELK 栈的脚本。

## 快速开始

1. **配置环境变量**
   - 复制 `.env.example` 为 `.env` 并修改其中的配置（如密码、端口等）。

2. **启动 ELK 栈**
   - 运行以下命令启动服务：
     ```bash
     ./start_elk.sh
     ```

3. **停止 ELK 栈**
   - 运行以下命令停止服务：
     ```bash
     ./stop_elk.sh
     ```

## 访问服务

- **Elasticsearch**: `http://localhost:9201`
- **Kibana**: `http://localhost:5601`

## 日志收集

Filebeat 配置为从 `${KB_LOGS}` 目录收集日志文件，并将其发送到 Elasticsearch。日志索引格式为 `filebeat-kblogs-%{+yyyy.MM.dd}`。

## 注意事项

- 确保 Docker 和 Docker Compose 已安装并运行。
- 修改 `.env` 文件中的 `ELASTIC_PASSWORD` 为强密码。
- 首次启动时，脚本会自动生成 Kibana 服务账户令牌并更新到 `.env` 文件中。