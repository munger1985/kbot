#!/bin/bash

echo "停止 ELK 栈..."
docker-compose -f docker-compose.kf.yml down
docker-compose -f docker-compose.es.yml down

echo "所有服务已停止"