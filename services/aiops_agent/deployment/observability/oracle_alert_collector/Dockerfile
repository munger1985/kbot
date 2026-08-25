FROM python:3.12.11-slim@sha256:47ae396f09c1303b8653019811a8498470603d7ffefc29cb07c88f1f8cb3d19f

WORKDIR /app
RUN useradd --system --uid 10001 --home-dir /nonexistent collector
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --requirement /app/requirements.txt
COPY collector.py healthcheck.py /app/
RUN mkdir -p /var/lib/kbot/oracle-alert \
    && chown -R collector:collector /var/lib/kbot/oracle-alert
USER collector
ENTRYPOINT ["python", "/app/collector.py"]
