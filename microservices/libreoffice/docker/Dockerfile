FROM ubuntu:22.04

# 设置中文环境
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai \
    LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8

# 使用阿里云镜像并安装依赖
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    locales \
    fonts-noto-cjk-extra \
    ttf-mscorefonts-installer \
    libreoffice \
    libreoffice-l10n-zh-cn \
    poppler-utils \
    ghostscript \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 配置中文locale
RUN locale-gen zh_CN.UTF-8

# 安装Python依赖
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# 复制应用代码
COPY . .

# 创建字体缓存
RUN fc-cache -fv

# 创建非root用户
RUN useradd -m -u 1000 serviceuser && \
    mkdir -p /tmp/conversions && \
    chown -R serviceuser:serviceuser /app /tmp

USER serviceuser
EXPOSE 9316
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9316"]