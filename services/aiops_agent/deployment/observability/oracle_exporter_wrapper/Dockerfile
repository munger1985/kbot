ARG ORACLE_EXPORTER_IMAGE=ghcr.io/iamseth/oracledb_exporter:0.6.0@sha256:bc43d9d2262b4b149aec1658df520baa1eb590d8a0ec09bbba6a3dca8526dcbf
FROM ${ORACLE_EXPORTER_IMAGE}

USER root
COPY entrypoint.sh /usr/local/bin/kbot-oracle-exporter-entrypoint
RUN chmod 0555 /usr/local/bin/kbot-oracle-exporter-entrypoint
USER 1000
ENTRYPOINT ["/usr/local/bin/kbot-oracle-exporter-entrypoint"]
