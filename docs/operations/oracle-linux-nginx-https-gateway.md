# Oracle Linux Nginx HTTPS 统一入口实施手册

## 1. 目标与适用范围

本文用于在前端服务器 `146.56.158.44` 上安装 Nginx，以 HTTPS 统一代理现有前端和 KBot Main API。

实施后的访问链路如下：

```text
浏览器
  │ HTTPS 443
  ▼
146.56.158.44 / Nginx
  ├── /api/* ──→ 140.238.44.208:18099（KBot Main API）
  └── 其他路径 ─→ 127.0.0.1:8080（现有前端）
```

最终浏览器统一访问：

```text
https://146.56.158.44/
```

现有前端继续监听 `127.0.0.1:8080` 或 `0.0.0.0:8080`，不需要改为 443。Nginx 负责 TLS 终止和请求转发。

如果仅供少量受控终端使用，可以创建本地 CA，并用它签发包含 IP SAN 的服务器证书；客户端需要通过操作系统或企业设备策略信任一次该 CA。

如果需要让任意浏览器直接信任，并且暂时没有域名，应优先申请 Let’s Encrypt 公网 IP 证书。Let’s Encrypt 已在 2026 年正式开放 IPv4/IPv6 地址证书；该证书有效期约六天，因此必须使用 Certbot 5.4 或更高版本配置自动续期。本文的本地 CA 步骤作为受控环境和公网 IP 证书申请前的备用方案。

## 2. 安全边界

- 本地 CA 私钥 `/etc/nginx/ssl/kbot-local-ca.key` 不得离开 `146.56.158.44`，也不得提交到代码仓库。
- 客户端只允许安装 `/etc/nginx/ssl/kbot-local-ca.crt`。
- `140.238.44.208:18099` 应仅允许来源 `146.56.158.44/32`，不得继续向全网开放。
- Nginx 与 KBot 之间当前仍通过 HTTP 传输。生产环境应优先改用 OCI 私网地址、VPN 或服务间 TLS。
- 使用自建 CA 只适合受控客户端。获得域名后，应替换为受公共 CA 信任的正式证书。
- 面向非受控客户端时，使用 Let’s Encrypt IP 证书或域名证书，避免要求访问者手工导入私有 CA。

## 2.1 仅使用 HTTP 的简化方案

如果目标只是通过 Nginx 统一访问 APEX 和 KBot，不要求 HTTPS，可以跳过本文的证书生成、443 和客户端 CA 信任步骤。浏览器统一访问：

```text
http://146.56.158.44/
```

Nginx 使用以下 HTTP 配置：

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

upstream kbot_frontend {
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
        proxy_pass http://kbot_frontend;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port 80;
        proxy_set_header X-Forwarded-Proto http;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;
        proxy_redirect http://146.56.158.44:8080/ http://146.56.158.44/;
    }
}
```

前端仍必须把 Main API 地址改为同源相对路径 `/api/v1/...`。KBot 的 `api_allowed_origins` 相应设置为：

```toml
api_allowed_origins = ["http://146.56.158.44"]
```

应用配置后执行：

```bash
sudo nginx -t
sudo systemctl restart nginx
curl -i http://127.0.0.1/healthz -H 'Host: 146.56.158.44'
```

该方案不提供传输加密，登录信息、API Key 和业务数据会以明文 HTTP 在客户端与 Nginx 之间传输，因此只建议用于测试环境或受保护的内网。

## 3. 实施前检查

登录前端服务器 `146.56.158.44`，确认系统版本、SELinux 状态和端口占用：

```bash
cat /etc/oracle-release
getenforce
sudo ss -ltnp | grep -E ':(443|8080)\b'
```

确认前端服务在本机正常：

```bash
curl -I http://127.0.0.1:8080/
```

确认前端服务器可以访问 KBot Main API：

```bash
curl -i --connect-timeout 5 \
  http://140.238.44.208:18099/healthz
```

必须返回 `HTTP/1.1 200 OK`。如果连接失败，先在 `140.238.44.208` 所属 OCI NSG 或 Security List 中增加以下入站规则：

```text
来源：146.56.158.44/32
协议：TCP
端口：18099
```

同时检查 `140.238.44.208` 的主机防火墙是否允许该来源。

## 4. 安装 Nginx 与系统依赖

Oracle Linux 8/9 使用 `dnf`：

```bash
sudo dnf install -y \
  nginx \
  openssl \
  firewalld \
  policycoreutils-python-utils
```

启用 firewalld：

```bash
sudo systemctl enable --now firewalld
```

注册 Nginx 开机启动，但先不要启动服务：

```bash
sudo systemctl enable nginx
```

## 5. 配置 SELinux

Oracle Linux 默认启用 SELinux。Nginx 需要主动连接本机前端和远端 KBot，因此必须允许 HTTP 服务发起网络连接：

```bash
sudo setsebool -P httpd_can_network_connect 1
```

验证：

```bash
getsebool httpd_can_network_connect
```

预期输出：

```text
httpd_can_network_connect --> on
```

如果没有开启，该配置通常会表现为 Nginx 返回 502，SELinux 审计日志中出现 AVC 拒绝。

## 6. 创建本地 CA

创建证书目录：

```bash
sudo mkdir -p /etc/nginx/ssl
cd /etc/nginx/ssl
```

生成本地 CA 私钥：

```bash
sudo openssl genrsa \
  -out /etc/nginx/ssl/kbot-local-ca.key \
  4096
```

创建独立的 `/etc/nginx/ssl/kbot-local-ca.cnf`。Oracle Linux 的 OpenSSL 1.1.1 仍要求配置中存在 `distinguished_name`，因此不能使用空配置文件：

```ini
[req]
distinguished_name = dn
prompt = no
x509_extensions = v3_ca

[dn]
CN = KBot Local Root CA

[v3_ca]
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid:always
basicConstraints = critical,CA:TRUE,pathlen:0
keyUsage = critical,keyCertSign,cRLSign
```

生成有效期十年的本地 CA 证书：

```bash
sudo openssl req \
  -x509 \
  -new \
  -sha256 \
  -config /etc/nginx/ssl/kbot-local-ca.cnf \
  -key /etc/nginx/ssl/kbot-local-ca.key \
  -days 3650 \
  -set_serial 1 \
  -out /etc/nginx/ssl/kbot-local-ca.crt \
  -extensions v3_ca
```

独立配置用于避免 Oracle Linux 的系统 `openssl.cnf` 自动添加 CA 扩展后，再与命令行扩展形成重复。重复的 `Basic Constraints` 等扩展会让 OpenSSL 1.1.1 无法将证书识别为有效 CA。

签发服务器证书前先验证根 CA：

```bash
sudo openssl verify \
  -CAfile /etc/nginx/ssl/kbot-local-ca.crt \
  /etc/nginx/ssl/kbot-local-ca.crt
```

必须输出：

```text
/etc/nginx/ssl/kbot-local-ca.crt: OK
```

## 7. 签发 IP 服务器证书

生成服务器私钥：

```bash
sudo openssl genrsa \
  -out /etc/nginx/ssl/kbot-server.key \
  2048
```

生成证书签名请求：

```bash
sudo openssl req \
  -new \
  -key /etc/nginx/ssl/kbot-server.key \
  -out /etc/nginx/ssl/kbot-server.csr \
  -subj "/CN=146.56.158.44"
```

创建 `/etc/nginx/ssl/kbot-server.ext`：

```ini
[server_cert]
subjectAltName=IP:146.56.158.44
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
```

使用本地 CA 签发证书：

```bash
sudo openssl x509 \
  -req \
  -in /etc/nginx/ssl/kbot-server.csr \
  -CA /etc/nginx/ssl/kbot-local-ca.crt \
  -CAkey /etc/nginx/ssl/kbot-local-ca.key \
  -CAcreateserial \
  -out /etc/nginx/ssl/kbot-server.crt \
  -days 825 \
  -sha256 \
  -extfile /etc/nginx/ssl/kbot-server.ext \
  -extensions server_cert
```

设置所有权、权限和 SELinux 文件上下文：

```bash
sudo chown root:root /etc/nginx/ssl/*
sudo chmod 600 /etc/nginx/ssl/*.key
sudo chmod 644 /etc/nginx/ssl/*.crt
sudo restorecon -Rv /etc/nginx/ssl
```

验证证书：

```bash
openssl x509 \
  -in /etc/nginx/ssl/kbot-server.crt \
  -noout \
  -subject \
  -issuer \
  -dates \
  -ext subjectAltName
```

输出必须包含：

```text
IP Address:146.56.158.44
```

只设置 `CN` 而没有 IP SAN 的证书会被现代浏览器判定为地址不匹配。

## 8. 配置 Nginx 统一入口

备份主配置：

```bash
sudo cp -a /etc/nginx/nginx.conf \
  /etc/nginx/nginx.conf.before-kbot
```

创建 `/etc/nginx/conf.d/kbot.conf`：

```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

upstream kbot_frontend {
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

    return 301 https://146.56.158.44$request_uri;
}

server {
    listen 443 ssl http2;
    server_name 146.56.158.44;

    ssl_certificate     /etc/nginx/ssl/kbot-server.crt;
    ssl_certificate_key /etc/nginx/ssl/kbot-server.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_session_cache shared:KBOT_SSL:10m;
    ssl_session_timeout 1d;

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
        proxy_set_header X-Forwarded-Port 443;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_connect_timeout 10s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;

        # Agent SSE、长轮询和流式响应不能被代理缓存或聚合。
        proxy_buffering off;
        proxy_cache off;
    }

    location = /healthz {
        proxy_pass http://kbot_main_api/healthz;
        proxy_set_header Host 140.238.44.208:18099;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
    }

    location = /readyz {
        proxy_pass http://kbot_main_api/readyz;
        proxy_set_header Host 140.238.44.208:18099;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-Proto https;
    }

    location / {
        proxy_pass http://kbot_frontend;

        proxy_http_version 1.1;
        proxy_set_header Host 146.56.158.44:8080;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port 443;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection $connection_upgrade;

        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        proxy_redirect http://146.56.158.44:8080/ https://146.56.158.44/;
    }
}
```

`proxy_pass` 后没有附加 URI，因此 `/api/v1/...` 会原样转发到 KBot Main API。

不要在使用本地证书期间启用 HSTS。HSTS 会让浏览器强制使用 HTTPS，并可能在证书信任配置错误时增加恢复难度。

### 8.1 APEX/ORDS 专用代理配置

如果 `127.0.0.1:8080` 运行的是 ORDS Standalone，应显式代理 ORDS 默认上下文 `/ords/` 和 APEX 默认静态资源上下文 `/i/`。将上面 HTTPS `server` 中原来的 `location /` 替换为：

```nginx
    location = / {
        return 302 /ords/;
    }

    location = /ords {
        return 301 /ords/;
    }

    location ^~ /ords/ {
        proxy_pass http://kbot_frontend;

        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port 443;
        proxy_set_header X-Forwarded-Proto https;

        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;

        proxy_redirect http://127.0.0.1:8080/ https://146.56.158.44/;
        proxy_redirect http://146.56.158.44:8080/ https://146.56.158.44/;
    }

    location ^~ /i/ {
        proxy_pass http://kbot_frontend;

        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Host $host;
        proxy_set_header X-Forwarded-Port 443;
        proxy_set_header X-Forwarded-Proto https;

        proxy_connect_timeout 10s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
```

不要给这两个 `proxy_pass` 添加结尾 `/`；这样 `/ords/...` 和 `/i/...` 会按原 URI 转发给 ORDS。

在 ORDS 配置目录中设置反向代理 HTTPS 检查：

```bash
ords --config <ORDS_CONFIG_PATH> \
  config set security.httpsHeaderCheck \
  'X-Forwarded-Proto: https'
```

然后重启 ORDS。可以通过以下命令确认 ORDS 的启动命令和配置目录：

```bash
sudo systemctl cat ords
ps -ef | grep '[o]rds'
```

确认 ORDS Standalone 配置：

```bash
ords --config <ORDS_CONFIG_PATH> \
  config get standalone.context.path

ords --config <ORDS_CONFIG_PATH> \
  config get standalone.static.path

ords --config <ORDS_CONFIG_PATH> \
  config get security.httpsHeaderCheck
```

默认应用上下文应为 `/ords`。使用 APEX 时，`standalone.static.path` 必须指向与数据库 APEX 版本匹配的 `apex/images` 目录；静态资源的默认 URL 上下文是 `/i`。

分层验证：

```bash
curl -i http://127.0.0.1:8080/ords/
curl -I http://127.0.0.1:8080/i/

curl --cacert /etc/nginx/ssl/kbot-local-ca.crt \
  --resolve 146.56.158.44:443:127.0.0.1 \
  -i https://146.56.158.44/ords/
```

- 第一条失败：ORDS 未监听 8080，或实际 context path 不是 `/ords`；
- `/ords/` 成功但 `/i/` 失败：APEX images 路径未配置或不匹配；
- 后端直连成功但 HTTPS 返回 502：检查 SELinux `httpd_can_network_connect` 和 Nginx 错误日志；
- 跳转到 `http://...:8080`：检查代理头和 `security.httpsHeaderCheck`；
- 页面存在但 CSS、图标全部丢失：重点检查 `/i/` 与 `standalone.static.path`。

## 9. 检查并启动 Nginx

检查配置语法：

```bash
sudo nginx -t
```

预期输出：

```text
syntax is ok
test is successful
```

启动 Nginx：

```bash
sudo systemctl restart nginx
sudo systemctl status nginx --no-pager
```

查看服务和错误日志：

```bash
sudo journalctl -u nginx -n 100 --no-pager
sudo tail -n 100 /var/log/nginx/kbot_error.log
```

## 10. 配置 firewalld 和 OCI 入站规则

开放 HTTP 和 HTTPS：

```bash
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --reload
sudo firewall-cmd --list-all
```

在 `146.56.158.44` 所属 OCI NSG 或 Security List 中开放：

```text
TCP 443
TCP 80
```

443 应优先只允许办公网或指定客户端 IP。

完成 HTTPS 验证后，检查 8080 是否对公网开放：

```bash
sudo firewall-cmd --list-ports
```

如果存在 `8080/tcp`，并且已经确认外部不再直接访问旧入口，可以执行：

```bash
sudo firewall-cmd --permanent --remove-port=8080/tcp
sudo firewall-cmd --reload
```

不要停止前端 8080 进程。Nginx 仍需通过本机访问该进程。

## 11. 修改前端 Main API 地址

本节修改对象是运行在 `146.56.158.44:8080` 上的 APEX 应用及其浏览器端静态资源，**不是** `140.238.44.208` 上的 Python Main API。Python Main API 仍监听 `18099`，由 Nginx 转发请求。

按实际配置方式选择修改位置：

- 如果 API 地址来自 APEX Application Item 或 Substitution String，在 APEX 应用的 Shared Components 中修改该配置值；
- 如果地址硬编码在 `p21-chat-workspace.js`，修改源 JS 后重新上传或替换 APEX 的 Static Application File，并更新引用版本号；
- 如果页面通过部署时生成的 JavaScript 配置对象取值，修改 `146.56.158.44` 上对应的前端部署配置。

HTTPS 页面不能继续调用：

```text
http://140.238.44.208:18099/api/v1/...
```

否则浏览器会以 Mixed Content 拒绝请求。

前端应使用同源相对路径：

```javascript
fetch("/api/v1/conversations?limit=200")
```

如果前端统一维护 API Base URL，设置为：

```javascript
const apiBaseUrl = window.location.origin;
```

如果地址由 APEX Application Item、Substitution String 或部署参数管理，将其设置为：

```text
https://146.56.158.44
```

修改后清除浏览器缓存，或提升 `p21-chat-workspace.js` 等静态资源的版本号。浏览器开发者工具中不应再出现对 `http://140.238.44.208:18099` 的直接请求。

## 12. 修改 KBot Origin 配置

登录 `140.238.44.208`：

```bash
cd /home/ubuntu/kbot4.0
vi configuration/kbot.toml
```

设置：

```toml
api_allowed_origins = ["https://146.56.158.44"]
```

重启 KBot：

```bash
./stop_kbot.sh
./start_kbot.sh
```

统一入口属于同源访问，正常浏览器请求不再依赖 CORS。保留该 Origin 有助于受控的直接跨域测试。

## 13. 服务端验证

在 `146.56.158.44` 上使用本地 CA 验证 HTTPS。OCI 实例通常不能通过自己的公网 IP 完成回环访问，因此使用 `--resolve` 将证书中的公网 IP 临时解析到本机：

```bash
curl --cacert /etc/nginx/ssl/kbot-local-ca.crt \
  --resolve 146.56.158.44:443:127.0.0.1 \
  -i https://146.56.158.44/healthz
```

验证前端：

```bash
curl --cacert /etc/nginx/ssl/kbot-local-ca.crt \
  --resolve 146.56.158.44:443:127.0.0.1 \
  -I https://146.56.158.44/
```

验证 API 代理：

```bash
curl --cacert /etc/nginx/ssl/kbot-local-ca.crt \
  --resolve 146.56.158.44:443:127.0.0.1 \
  -i 'https://146.56.158.44/api/v1/conversations?limit=200'
```

未提供认证头时，最后一条请求可能返回 400、401 或 403。这说明代理已转发到应用；不得出现 502、连接超时或证书地址错误。

完成本机验证后，还必须从另一台客户端直接访问 `https://146.56.158.44/`，验证 OCI NSG/Security List 和主机防火墙的公网入站链路。

## 14. 客户端信任本地 CA

只向客户端分发：

```text
kbot-local-ca.crt
```

不得分发：

```text
kbot-local-ca.key
```

### 14.1 Windows

使用管理员权限执行：

```powershell
certutil -addstore -f ROOT kbot-local-ca.crt
```

关闭并重新打开浏览器。

### 14.2 Oracle Linux 或 RHEL 客户端

```bash
sudo cp kbot-local-ca.crt \
  /etc/pki/ca-trust/source/anchors/
sudo update-ca-trust
```

### 14.3 Ubuntu 客户端

```bash
sudo cp kbot-local-ca.crt \
  /usr/local/share/ca-certificates/
sudo update-ca-certificates
```

### 14.4 macOS

```bash
sudo security add-trusted-cert \
  -d \
  -r trustRoot \
  -k /Library/Keychains/System.keychain \
  kbot-local-ca.crt
```

Firefox 如果未使用操作系统证书库，需要在“设置 → 隐私与安全 → 证书 → 查看证书 → 证书颁发机构”中单独导入。

如果不导入本地 CA，只能在浏览器证书警告页选择继续访问。该方式只是临时绕过，每台客户端都需要单独处理。

## 15. 故障排查

### 15.1 Nginx 返回 502

分别测试两个上游：

```bash
curl -I http://127.0.0.1:8080/
curl -i http://140.238.44.208:18099/healthz
```

检查 SELinux：

```bash
getsebool httpd_can_network_connect
sudo ausearch -m AVC -ts recent
```

检查 Nginx 错误：

```bash
sudo tail -f /var/log/nginx/kbot_error.log
```

### 15.2 浏览器仍请求 18099

说明前端 API Base URL 或浏览器缓存尚未更新。正确请求必须是：

```text
https://146.56.158.44/api/v1/...
```

不得再出现：

```text
http://140.238.44.208:18099/api/v1/...
```

### 15.3 证书地址不匹配

```bash
openssl x509 \
  -in /etc/nginx/ssl/kbot-server.crt \
  -noout \
  -ext subjectAltName
```

必须包含：

```text
IP Address:146.56.158.44
```

### 15.4 前端反复跳转

查看访问日志：

```bash
sudo tail -f /var/log/nginx/kbot_access.log
```

确认前端能够识别以下代理头：

```text
X-Forwarded-Proto: https
X-Forwarded-Port: 443
```

## 16. 回滚

如果 Nginx 配置导致统一入口不可用，禁用 KBot 配置：

```bash
sudo mv /etc/nginx/conf.d/kbot.conf \
  /etc/nginx/conf.d/kbot.conf.disabled
sudo nginx -t
sudo systemctl restart nginx
```

临时恢复旧入口：

```text
http://146.56.158.44:8080/
```

该方案没有停止或迁移现有前端进程，因此禁用 Nginx 配置后可以直接使用原入口。

## 17. 正式上线建议

完成临时 IP HTTPS 验证后，建议继续执行：

1. 申请正式域名并解析到 `146.56.158.44`；
2. 使用 ACME/Let's Encrypt 或企业 CA 替换本地证书；
3. 将前端 8080 限制为仅本机访问；
4. 将 KBot 18099 限制为仅 `146.56.158.44/32` 或私网访问；
5. 使用 OCI 私网、VPN 或服务间 TLS 保护 Nginx 到 KBot 的链路；
6. 正式证书和跳转验证完成后，再评估启用 HSTS。
