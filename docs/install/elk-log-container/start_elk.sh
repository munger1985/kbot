#!/bin/bash

set -e  # 遇到错误立即退出

echo "=== 开始部署 ELK 栈 ==="

# 检查 .env 文件
if [ ! -f .env ]; then
    echo "错误: 找不到 .env 文件"
    exit 1
fi

# 加载环境变量
# source .env

echo "1. 启动 Elasticsearch..."
docker-compose -f docker-compose.es.yml up -d

echo "等待 Elasticsearch 启动..."
# 简单可靠的等待方式
echo "等待 Elasticsearch 启动（最多等待60秒）..."

counter=0
max_wait=60  # 最多等待60秒

while [ $counter -lt $max_wait ]; do
    # 检查容器是否在运行
    CONTAINER_STATE=$(docker inspect kbot-eslog --format='{{.State.Status}}' 2>/dev/null || echo "not_found")
    
    if [ "$CONTAINER_STATE" = "not_found" ]; then
        echo "错误: Elasticsearch 容器不存在"
        docker-compose -f docker-compose.es.yml logs kbot-eslog
        exit 1
    fi
    
    if [ "$CONTAINER_STATE" != "running" ]; then
        echo "Elasticsearch 容器状态: $CONTAINER_STATE，等待..."
        sleep 5
        counter=$((counter + 5))
        continue
    fi
    
    # 检查 Elasticsearch 是否响应
    if curl -s -u "elastic:${ELASTIC_PASSWORD}" "http://localhost:${ES_PORT:-9201}/" > /dev/null 2>&1; then
        echo "Elasticsearch 已启动并响应"
        break
    fi
    
    echo "等待 Elasticsearch 启动... ($((counter + 1))秒)"
    sleep 5
    counter=$((counter + 5))
done

if [ $counter -ge $max_wait ]; then
    echo "警告: Elasticsearch 启动超时，但继续执行..."
fi

echo "2. 生成 Kibana 服务账户令牌..."
# 删除可能存在的旧令牌
docker exec kbot-eslog bin/elasticsearch-service-tokens delete elastic/kibana kbot-token 2>/dev/null || true

# 创建新令牌
echo "创建服务账户令牌..."
TOKEN_OUTPUT=$(docker exec kbot-eslog bin/elasticsearch-service-tokens create elastic/kibana kbot-token 2>&1)

# 从输出中提取令牌
TOKEN=$(echo "$TOKEN_OUTPUT" | grep -oE '[A-Za-z0-9_-]{20,}' | head -1)

if [ -z "$TOKEN" ]; then
    # 如果上面的方法失败，尝试直接列出令牌
    echo "尝试通过列表获取令牌..."
    TOKEN=$(docker exec kbot-eslog bin/elasticsearch-service-tokens list | grep kbot-token | awk '{print $NF}')
fi

if [ -z "$TOKEN" ]; then
    echo "错误: 无法提取服务账户令牌"
    echo "原始输出: $TOKEN_OUTPUT"
    echo "令牌列表:"
    docker exec kbot-eslog bin/elasticsearch-service-tokens list
    exit 1
fi

echo "生成的令牌: $TOKEN"

# 更新 .env 文件中的令牌
if grep -q "KIBANA_SERVICE_ACCOUNT_TOKEN" .env; then
    sed -i.bak "s/KIBANA_SERVICE_ACCOUNT_TOKEN=.*/KIBANA_SERVICE_ACCOUNT_TOKEN=$TOKEN/" .env
    rm -f .env.bak 2>/dev/null || true
else
    echo "KIBANA_SERVICE_ACCOUNT_TOKEN=$TOKEN" >> .env
fi

# 重新加载环境变量以获取新令牌
source .env

echo "3. 启动 Kibana 和 Filebeat..."
docker-compose -f docker-compose.kf.yml up -d

echo "4. 等待服务启动..."
sleep 15

echo "=== 部署完成 ==="
echo "Elasticsearch: http://localhost:${ES_PORT:-9201}"
echo "Kibana: http://localhost:${KIBANA_PORT:-5601}"
echo ""
echo "检查服务状态:"
docker-compose -f docker-compose.kf.yml ps
echo ""
echo "查看 Elasticsearch 健康状态:"
curl -s -u "elastic:${ELASTIC_PASSWORD}" "http://localhost:${ES_PORT:-9201}/_cluster/health" | python3 -m json.tool 2>/dev/null || \
curl -s -u "elastic:${ELASTIC_PASSWORD}" "http://localhost:${ES_PORT:-9201}/_cluster/health"