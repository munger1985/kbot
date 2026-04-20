FROM python:3.12-slim

RUN apt update && apt install -y wget && \
    apt install -y unzip && \
    apt install -y procps && \
    mkdir -p /usr/share/binfmts/


# ========== 新增：安装 Tesseract OCR 系统依赖 ==========
RUN apt update && apt install -y \
    # Tesseract 本体 + 中文语言包（解析中文 PDF 必须）
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    # tesserocr 编译依赖
    libtesseract-dev \
    libleptonica-dev \
    gcc \
    g++ \
    curl \
    # 其他 Docling 依赖（如 PDF 解析）
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN wget -nc https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.deb && \
    apt install -y ./jdk-21_linux-x64_bin.deb && \
    java -version && \
    rm -f ./jdk-21_linux-x64_bin.deb

RUN wget -nc https://download.oracle.com/otn_software/java/sqldeveloper/sqlcl-25.3.2.317.1117.zip && \
    unzip -o sqlcl-*.zip && \
    ln -s $PWD/sqlcl/bin/sql /usr/local/bin/sql && \
    sql -V && \
    rm -f ./sqlcl-*.zip

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements (if any) and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install tesserocr easyocr

# Copy application code
COPY . .

COPY .env.example .env

EXPOSE 8000
EXPOSE 18099

# Define default command
#CMD ["python", "kbot_main.py"]
#CMD ["/bin/bash", "start_kbot.sh"]
#ENTRYPOINT echo "conn -save aireport -savepwd $DB_USER/$DB_PWD@//$DB_HOST:$DB_PORT/$DB_SERVICE" | sql /nolog; python -u kbot_main.py
#ENTRYPOINT python -u kbot_main.py

ENTRYPOINT ["/bin/bash", "-c", "/app/start_kbot.sh && tail -f /dev/null"]
