FROM python:3.12.11-slim@sha256:47ae396f09c1303b8653019811a8498470603d7ffefc29cb07c88f1f8cb3d19f

WORKDIR /app
COPY signer.py healthcheck.py /app/

USER 10001:10001
EXPOSE 8080
CMD ["python", "/app/signer.py"]
