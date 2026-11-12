"""隔离 SIM PDU 的管理可达性端点；不模拟供电健康、电流、电压或硬件状态。"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import threading
import time

import switch_simulator as snmp


STARTED = time.monotonic()


def mib():
    # RFC 5612 文档专用 enterprise，不冒充厂商型号以骗过驱动识别。
    values = {"1.3.6.1.2.1.1.1.0": snmp.tlv(4, b"SIM Exadata PDU management reachability only"),
              "1.3.6.1.2.1.1.2.0": snmp.oid("1.3.6.1.4.1.32473.1"),
              "1.3.6.1.2.1.1.3.0": snmp.integer(int((time.monotonic() - STARTED) * 100), 0x43),
              "1.3.6.1.2.1.1.5.0": snmp.tlv(4, b"AIOPS_SIM_EXADATA_PDU01")}
    return {tuple(map(int, key.split("."))): value for key, value in values.items()}


def response(packet, community):
    return snmp.response(packet, community, data=mib())


class ManagementHandler(BaseHTTPRequestHandler):
    server_version = "AIOps-SIM-PDU/1.0"
    sys_version = ""

    def do_GET(self):
        self._read_only(False)

    def do_HEAD(self):
        self._read_only(True)

    def _read_only(self, head):
        if self.path != "/":
            self.send_error(404)
            return
        body = json.dumps({"simulation": True, "target": "AIOPS_SIM_EXADATA_PDU01",
                           "scope": "MANAGEMENT_REACHABILITY_ONLY", "power_health": None}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def log_message(self, *arguments):
        # 不保留请求正文、鉴权头或 URL。
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--community-file", type=Path, required=True)
    args = parser.parse_args()
    community = args.community_file.read_bytes().strip()
    if not 16 <= len(community) <= 128:
        raise SystemExit("模拟器私有配置无效")
    server = HTTPServer(("0.0.0.0", 80), ManagementHandler)
    server.timeout = 3
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
        channel.bind(("0.0.0.0", 1161))
        print(json.dumps({"simulation": True, "scope": "MANAGEMENT_REACHABILITY_ONLY",
                          "http_port": 80, "snmp_port": 1161, "writable": False}), flush=True)
        while True:
            packet, peer = channel.recvfrom(snmp.MAX_PACKET + 1)
            try:
                result = response(packet, community)
                if result is not None:
                    channel.sendto(result, peer)
            except (ValueError, IndexError, OverflowError):
                continue


if __name__ == "__main__":
    main()
