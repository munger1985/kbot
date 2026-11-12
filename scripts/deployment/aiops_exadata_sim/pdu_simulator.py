"""隔离 SIM PDU：原厂增强 MIB 模块/三相遥测和阈值，不伪造整机健康。"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import socket
import threading
import time

import switch_simulator as snmp


STARTED = time.monotonic()
SCENARIO = "NORMAL"
SCENARIOS = ("NORMAL", "WARNING", "CRITICAL", "DISABLED")


def mib(now=None, scenario="NORMAL"):
    if scenario not in SCENARIOS:
        raise ValueError("未知的 PDU 模拟场景")
    elapsed = max(0, int((time.monotonic() if now is None else now) - STARTED))
    # 原厂 KnuerrV2DriverFactory 精确匹配此协议标识；名称、序列号仍明确为 SIM。
    values = {"1.3.6.1.2.1.1.1.0": snmp.tlv(4, b"Oracle RMII PDU system"),
              "1.3.6.1.2.1.1.2.0": snmp.oid("1.3.6.1.4.1.2769.1.4"),
              "1.3.6.1.2.1.1.3.0": snmp.integer(int((time.monotonic() - STARTED) * 100), 0x43),
              "1.3.6.1.2.1.1.5.0": snmp.tlv(4, b"AIOPS_SIM_EXADATA_PDU01")}
    base = "1.3.6.1.4.1.2769.1.4."
    for key, value in {"1.1.0": b"AIOPS_SIM_EXADATA_PDU01", "1.10.0": b"SIM-PDU",
                       "1.11.0": b"SIM-EXADATA-PDU-0001", "1.21.0": b"SIM-1.0",
                       "1.22.0": b"SIM", "1.23.0": b"02:00:00:00:00:10"}.items():
        values[base + key] = snmp.tlv(4, value)
    # EnhancedMibSnmpCommunicatorFactory$8：有功功率 W；能耗原始整数除以 10 得 kWh。
    # 数值范围参考真实快照，时间是本次采集；不回填客户历史。
    # 阈值仅为 SIM 调试配置，不声称来自客户硬件或适用于真实 PDU。
    # 原厂从原始电流/阈值计算 0/1/2；此处不提供任何 OEM 等级或健康 OID。
    currents = {"NORMAL": (270, 271, 272), "WARNING": (1300, 271, 272),
                "CRITICAL": (1500, 271, 272), "DISABLED": (1500, 271, 272)}[scenario]
    voltage = 2300
    apparent = [current * voltage // 1000 for current in currents]
    powers = [value * 98 // 100 for value in apparent]
    energy = 13046 + elapsed * sum(powers) // 360000
    # getFirstModuleNumber=0，但 PduEnhancedOids 对 SNMP 模块索引加 1。
    # NUMBER_OF_PHASES 是总相数 / 模块数，而非逐模块标量。
    values[base + "5.1.2.0"] = snmp.integer(sum(powers))
    values[base + "5.1.5.0"] = snmp.integer(energy)
    values[base + "5.2.1.0"] = snmp.integer(1)
    values[base + "5.3.1.0"] = snmp.integer(3)
    values[base + "2.2.0"] = snmp.integer(1)
    module_values = {3: sum(powers), 8: energy, 11: max(currents) - min(currents),
                     23: 0 if scenario == "DISABLED" else 200}
    for column, value in module_values.items():
        values[base + f"5.2.2.1.{column}.1"] = snmp.integer(value)
    for phase, (current, power, va) in enumerate(zip(currents, powers, apparent), 1):
        # 电流/阈值 ÷100 A，电压 ÷10 V，功率原值 W/VA，能耗 ÷10 kWh。
        phase_values = {3: current, 4: voltage, 5: power, 6: va,
                        10: energy // 3 + (phase <= energy % 3),
                        22: 0 if scenario == "DISABLED" else 1000,
                        23: 0 if scenario == "DISABLED" else 1200,
                        24: 0 if scenario == "DISABLED" else 1400}
        for column, value in phase_values.items():
            values[base + f"5.3.2.1.{column}.1.{phase}"] = snmp.integer(value)
    return {tuple(map(int, key.split("."))): value for key, value in values.items()}


def response(packet, community):
    return snmp.response(packet, community, data=mib(scenario=SCENARIO))


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
                           "scope": "MANAGEMENT_AND_POWER_TELEMETRY", "power_health": None,
                           "scenario": SCENARIO, "modules": 1, "phases": 3}).encode()
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
    global SCENARIO
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--community-file", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="NORMAL")
    args = parser.parse_args()
    SCENARIO = args.scenario
    community = args.community_file.read_bytes().strip()
    if not 16 <= len(community) <= 128:
        raise SystemExit("模拟器私有配置无效")
    server = HTTPServer(("0.0.0.0", 80), ManagementHandler)
    server.timeout = 3
    threading.Thread(target=server.serve_forever, daemon=True).start()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
        channel.bind(("0.0.0.0", 1161))
        print(json.dumps({"simulation": True, "scope": "MANAGEMENT_AND_POWER_TELEMETRY",
                          "scenario": SCENARIO, "modules": 1, "phases": 3,
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
