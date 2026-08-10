# Oracle Linux Nginx HTTP 代理 APEX/ORDS 简明步骤

## 1. 访问结构

```text
浏览器：http://146.56.158.44/
  ├── /api/*        → 140.238.44.208:18099（KBot Main API）
  └── /ords/*、/i/* → 127.0.0.1:8080（ORDS/APEX）
```

该方案不使用 HTTPS，不需要证书，也不需要客户端安装本地 CA。

## 2. 确认两个后端正常

在 `146.56.158.44` 上执行：

```bash
curl -i http://127.0.0.1:8080/ords/
curl -i http://140.238.44.208:18099/healthz
```

两条命令都必须能够收到 HTTP 响应，才能继续配置 Nginx。

## 3. 完整替换 Nginx 配置

备份现有 HTTPS 配置：

```bash
sudo cp -a /etc/nginx/conf.d/kbot.conf \
  /etc/nginx/conf.d/kbot.conf.https-backup
```

编辑 `/etc/nginx/conf.d/kbot.conf`，删除原内容并完整替换为：

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

upstream kbot_ords {
    server 127.0.0.1:8080;
    keepalive 16;
}

upstream kbot_main_api {
    server 140.238.44.208:18099;
    keepalive 32;
}

server {
    listen 80;
    server_name 146.56.158.44;

    client_max_body_size 100m;

    access_log /var/log/nginx/kbot_access.log;
    error_log  /var/log/nginx/kbot_error.log warn;

    location ^~ /api/ {
        proxy_pass http://kbot_main_api;

        proxy_http_version 1.1;
        proxy_set_header Host 140.238.44.208:18099;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port 80;
        proxy_set_header X-Forwarded-Proto http;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_connect_timeout 10s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;
        proxy_buffering off;
        proxy_cache off;
    }

    location = /healthz {
        proxy_pass http://kbot_main_api/healthz;
        proxy_set_header Host 140.238.44.208:18099;
    }

    location = /readyz {
        proxy_pass http://kbot_main_api/readyz;
        proxy_set_header Host 140.238.44.208:18099;
    }

    location / {
        proxy_pass http://kbot_ords;

        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port 80;
        proxy_set_header X-Forwarded-Proto http;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        proxy_redirect http://127.0.0.1:8080/ http://146.56.158.44/;
        proxy_redirect http://146.56.158.44:8080/ http://146.56.158.44/;
    }
}
```

此处的 `location /` 会同时代理 `/ords/` 和 `/i/`，不需要分别编写两套 ORDS 配置。

## 4. 启动并验证

```bash
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl status nginx --no-pager
```

开放 HTTP：

```bash
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --reload
sudo firewall-cmd --list-services
```

OCI NSG 或 Security List 还需要允许 TCP 80 入站。

本机验证：

```bash
curl -i http://127.0.0.1/healthz \
  -H 'Host: 146.56.158.44'

curl -I http://127.0.0.1/ords/ \
  -H 'Host: 146.56.158.44'
```

浏览器访问：

```text
http://146.56.158.44/
```

根路径会原样代理到 `http://127.0.0.1:8080/`，因此行为与原来的 `http://146.56.158.44:8080/` 一致。

如果 ORDS 的根路径不是目标 APEX 应用，可在 Nginx 中额外加入精确跳转。将 `<APEX_APPLICATION_PATH>` 替换为浏览器直接访问 8080 时目标应用的实际路径：

```nginx
location = / {
    return 302 <APEX_APPLICATION_PATH>;
}
```

例如 APEX Friendly URL 可能类似 `/ords/r/<workspace>/<application>/home`，旧式 URL 可能类似 `/ords/f?p=<APP_ID>:1`；必须以现有应用实际 URL 为准，不要固定跳转到 ORDS 落地页 `/ords/`。

## 5. APEX 与 KBot 配置

APEX 浏览器端的 Main API 地址使用相对路径：

```javascript
fetch("/api/v1/conversations?limit=200")
```

不要直接请求 `http://140.238.44.208:18099`。

KBot `configuration/kbot.toml` 设置：

```toml
api_allowed_origins = ["http://146.56.158.44"]
```

修改 KBot 配置后重启 KBot。

ORDS 不需要设置 `security.httpsHeaderCheck`。即使此前设置过，该检查在请求头不是 `X-Forwarded-Proto: https` 时也不会把 HTTP 请求识别成 HTTPS。

## 6. 故障定位

如果 Nginx 返回 502：

```bash
curl -i http://127.0.0.1:8080/ords/
getsebool httpd_can_network_connect
sudo tail -n 100 /var/log/nginx/kbot_error.log
```

SELinux 必须允许 Nginx 连接后端：

```bash
sudo setsebool -P httpd_can_network_connect 1
```

如果 firewalld 显示 `http` 已开放，但公网请求仍未出现在 Nginx 日志中，检查是否还有独立的 legacy iptables 规则：

```bash
sudo iptables -L INPUT -n -v --line-numbers
```

如果末尾存在拒绝全部流量的规则，而它前面只有 SSH 放行规则，需要在该拒绝规则之前允许 TCP 80。例如拒绝规则当前编号为 5：

```bash
sudo iptables -I INPUT 5 \
  -p tcp \
  -m conntrack --ctstate NEW \
  --dport 80 \
  -j ACCEPT
```

立即验证公网访问。确认成功后，检查规则由哪个服务管理，再选择持久化方式：

```bash
sudo systemctl is-active iptables
sudo systemctl is-enabled iptables
sudo systemctl is-active firewalld
```

如果 `iptables` 服务处于启用状态，保存当前规则：

```bash
sudo iptables-save | sudo tee /etc/sysconfig/iptables >/dev/null
```

不要在未确认 SSH 放行规则和远程控制台可用前直接停用防火墙服务，以免中断远程连接。长期应只保留一套防火墙管理机制。

如果 APEX 页面出现但没有样式，检查静态资源：

```bash
curl -I http://127.0.0.1:8080/i/
curl -I http://146.56.158.44/i/
```

如果浏览器仍访问 8080 或 18099，清除缓存，并检查 APEX 中硬编码的绝对 URL。

## 7. 安全说明

HTTP 不提供传输加密。客户端到 Nginx 之间的登录信息、API Key 和业务数据可能被网络中的其他设备读取或修改。该方案仅适用于测试环境或有网络隔离的受控环境。
