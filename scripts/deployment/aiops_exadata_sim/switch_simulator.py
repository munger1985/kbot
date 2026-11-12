"""隔离实验网中的只读 SNMPv1/v2c 交换机夹具；数据由原厂 SI 插件采集。

MIB-II/IF-MIB 和 CISCO-PFC-EXT-MIB 映射到真实导出的 PortId、Priority
维度。仅生成 SIM 身份及合成计数，不实现 OEM 接口，不生成 OEM 历史或速率。
"""

import argparse
from bisect import bisect_right
import hmac
import json
from pathlib import Path
import socket
import time


EPOCH = 1767225600
MAX_PACKET = 16384
MAX_BINDINGS = 64
MAX_REPETITIONS = 16
PORTS = {1: "Ethernet1/1", 2: "Ethernet1/2"}


def tlv(tag, value):
    length = len(value)
    encoded = bytes([length]) if length < 128 else bytes([0x80 | ((length.bit_length() + 7) // 8)]) + length.to_bytes((length.bit_length() + 7) // 8, "big")
    return bytes([tag]) + encoded + value


def integer(value, tag=2):
    raw = value.to_bytes(max(1, (value.bit_length() + 8) // 8), "big", signed=True)
    return tlv(tag, raw)


def oid(value):
    parts = tuple(int(n) for n in value.split(".")) if isinstance(value, str) else value
    if len(parts) < 2 or parts[0] not in (0, 1, 2) or parts[1] < 0 or (parts[0] < 2 and parts[1] > 39):
        raise ValueError("OID 无效")
    raw = bytearray()
    for part in (40 * parts[0] + parts[1], *parts[2:]):
        if not 0 <= part <= 2 ** 63 - 1:
            raise ValueError("OID 超限")
        encoded = [part & 127]
        while part > 127:
            part >>= 7
            encoded.append((part & 127) | 128)
        raw.extend(reversed(encoded))
    return tlv(6, bytes(raw))


def read_tlv(data, offset=0):
    if offset + 2 > len(data):
        raise ValueError("BER 截断")
    tag, length = data[offset:offset + 2]
    offset += 2
    if length & 128:
        size = length & 127
        if not 1 <= size <= 3 or offset + size > len(data):
            raise ValueError("BER 长度无效")
        length = int.from_bytes(data[offset:offset + size], "big")
        offset += size
    end = offset + length
    if end > len(data):
        raise ValueError("BER 内容截断")
    return tag, data[offset:end], end


def fields(data):
    result, offset = [], 0
    while offset < len(data):
        tag, value, offset = read_tlv(data, offset)
        result.append((tag, value))
        if len(result) > MAX_BINDINGS:
            raise ValueError("BER 字段超限")
    return result


def decode_oid(raw):
    values, part = [], 0
    for index, byte in enumerate(raw):
        part = (part << 7) | (byte & 127)
        if part > 2 ** 63 - 1:
            raise ValueError("OID 超限")
        if not byte & 128:
            values.append(part)
            part = 0
        elif index == len(raw) - 1:
            raise ValueError("OID 截断")
    if not values or len(values) > 64:
        raise ValueError("OID 无效")
    first = min(2, values[0] // 40)
    return (first, values[0] - first * 40, *values[1:])


def mib(now):
    """原始计数按时间增长；速率和 PFC Delta 留给原厂插件计算。"""
    elapsed = max(0, int(now) - EPOCH)
    result = {}
    def put(name, value):
        result[tuple(int(part) for part in name.split("."))] = value
    def text(value):
        return tlv(4, value.encode())
    def counter(value, bits=32):
        return integer(value % 2 ** bits, 0x46 if bits == 64 else 0x41)
    system = "1.3.6.1.2.1.1."
    for column, value in {
        1: text("SIM - Cisco NX-OS compatible Exadata lab switch; synthetic data"),
        2: oid("1.3.6.1.4.1.9.12.3.1.3.1955"),
        3: integer((elapsed * 100) % 2 ** 32, 0x43),
        4: text("SIM lab"), 5: text("AIOPS_SIM_EXADATA_SWITCH01"),
        6: text("ISOLATED_SIM_NETWORK"), 7: integer(72),
    }.items():
        put(system + str(column) + ".0", value)
    put("1.3.6.1.2.1.2.1.0", integer(len(PORTS)))
    # 原厂以 ENTITY-MIB 的机箱厂家/型号/序列号生成目标关联身份。
    # 只有一个明确 SIM 的机箱，不虚构硬件传感器或告警状态。
    chassis = {1: integer(1), 2: text("SIM Exadata lab chassis"), 3: oid("0.0"),
        4: integer(0), 5: integer(3), 6: integer(-1),
        7: text("SIM Switch System AIOPS_SIM_EXADATA_SWITCH01"), 8: text("SIM"), 9: text("SIM"),
        10: text("SIM NX-OS"), 11: text("SIM-EXADATA-SWITCH-0001"),
        12: text("Cisco"), 13: text("C9336C-FX2"),
        14: text("SIM"), 15: text("SIM"), 16: integer(2)}
    for column, value in chassis.items():
        put(f"1.3.6.1.2.1.47.1.1.1.1.{column}.1", value)
    # 原厂端口配置采集要求 VTP VLAN 目录。只声明一个明确 SIM 的以太网 VLAN；
    # 不捏造 TokenRing/FDDI 属性、远端邻居、传感器或额外性能值。
    vlan = {2: integer(1), 3: integer(1), 4: text("SIM_VLAN1"),
            5: integer(9000), 6: integer(100001, 0x42),
            17: tlv(4, b"\x00"), 18: integer(0)}  # 无 VTP 扩展功能、无 SVI 接口。
    for column, value in vlan.items():
        put(f"1.3.6.1.4.1.9.9.46.1.3.1.1.{column}.1.1", value)
    # 提供真实的后继 MIB 标量，让原厂表遍历可越过 PFC 子树正常结束。
    # 这些是协议身份，不是以零值填充未知硬件指标。
    put("1.3.6.1.6.3.10.2.1.1.0", tlv(4, bytes([128, 0, 0, 9, 4]) + b"SIM-EXADATA-SWITCH-0001"))
    put("1.3.6.1.6.3.10.2.1.2.0", integer(1))
    put("1.3.6.1.6.3.10.2.1.3.0", integer(elapsed))
    put("1.3.6.1.6.3.10.2.1.4.0", integer(MAX_PACKET))
    for index, name in PORTS.items():
        incoming = elapsed * (125000000 + index * 1000000)
        outgoing = elapsed * (95000000 + index * 1000000)
        columns = {
            1: integer(index), 2: text(name), 3: integer(6), 4: integer(9000),
            5: integer(2 ** 32 - 1, 0x42), 6: tlv(4, bytes([2, 0, 0, 0, 0, index])),
            7: integer(1), 8: integer(1), 9: integer(0, 0x43),
            10: counter(incoming), 11: counter(elapsed * 10000),
            12: counter(elapsed * 10), 13: counter(elapsed // 3600),
            14: counter(elapsed // 7200), 15: counter(0),
            16: counter(outgoing), 17: counter(elapsed * 8000),
            18: counter(elapsed * 8), 19: counter(elapsed // 5400),
            20: counter(elapsed // 10800), 21: integer(0, 0x42),
            22: oid("0.0"),
        }
        for column, value in columns.items():
            put(f"1.3.6.1.2.1.2.2.1.{column}.{index}", value)
        high_capacity = {1: text(name), 6: counter(incoming, 64),
            7: counter(elapsed * 10000, 64), 8: counter(elapsed * 8, 64),
            9: counter(elapsed * 2, 64), 10: counter(outgoing, 64),
            11: counter(elapsed * 8000, 64), 12: counter(elapsed * 6, 64),
            13: counter(elapsed * 2, 64), 14: integer(2), 15: integer(100000, 0x42),
            16: integer(1), 17: integer(1), 18: text("SIM Exadata interconnect"),
            19: integer(0, 0x43)}
        for column, value in high_capacity.items():
            put(f"1.3.6.1.2.1.31.1.1.1.{column}.{index}", value)
        # 原厂 SI 24.1.1 CiscoMibSupport 使用的 PFC OID。
        for column, multiplier in ((1, 2), (2, 3)):
            put(f"1.3.6.1.4.1.9.9.813.1.1.1.{column}.{index}", counter(elapsed * multiplier * 8, 64))
        for priority in range(8):
            for column, multiplier in ((2, 2), (3, 3)):
                put(f"1.3.6.1.4.1.9.9.813.1.2.1.{column}.{index}.{priority}", counter(elapsed * multiplier, 64))
    return result


def response(packet, community, *, now=None):
    """拒绝错误认证、写操作、过大报文及未实现版本，不返回伪造的健康零值。"""
    if not 0 < len(packet) <= MAX_PACKET:
        return None
    tag, payload, end = read_tlv(packet)
    if tag != 0x30 or end != len(packet):
        raise ValueError("SNMP 消息无效")
    message = fields(payload)
    if len(message) != 3 or message[0][0] != 2 or message[1][0] != 4:
        raise ValueError("SNMP 消息字段无效")
    version = int.from_bytes(message[0][1], "big", signed=True)
    wire_community = message[1][1]
    authenticated = hmac.compare_digest(wire_community, community)
    vlan_context = hmac.compare_digest(wire_community, community + b"@1")
    if version not in (0, 1) or not (authenticated or vlan_context):
        return None
    operation, raw = message[2]
    if operation not in (0xA0, 0xA1, 0xA3, 0xA5) or (version == 0 and operation == 0xA5):
        return None
    pdu = fields(raw)
    if len(pdu) != 4 or [item[0] for item in pdu] != [2, 2, 2, 0x30]:
        raise ValueError("SNMP PDU 无效")
    if not 1 <= len(pdu[0][1]) <= 4:
        raise ValueError("请求 ID 无效")
    bindings = []
    for tag, value in fields(pdu[3][1]):
        binding = fields(value)
        if tag != 0x30 or len(binding) != 2 or binding[0][0] != 6:
            raise ValueError("SNMP varbind 无效")
        bindings.append(decode_oid(binding[0][1]))
    data = mib(time.time() if now is None else now)
    ordered = sorted(data)
    values, status, error_index = [], 0, 0
    def append(key, next_key=False):
        nonlocal status, error_index
        position = bisect_right(ordered, key)
        resolved = ordered[position] if next_key and position < len(ordered) else key
        value = data.get(resolved) if not next_key or position < len(ordered) else None
        if value is None:
            if version == 0:
                status, error_index = 2, len(values) + 1
            value = tlv(0x82 if next_key else 0x80, b"") if version else tlv(5, b"")
        values.append(tlv(0x30, oid(resolved) + value))
        return resolved
    if operation == 0xA3:
        status, error_index = (4 if version == 0 else 6), (1 if bindings else 0)
        values = [tlv(0x30, oid(key) + tlv(5, b"")) for key in bindings]
    elif operation == 0xA5:
        nonrepeat = max(0, min(len(bindings), int.from_bytes(pdu[1][1], "big", signed=True)))
        repeat = max(0, min(MAX_REPETITIONS, int.from_bytes(pdu[2][1], "big", signed=True)))
        for key in bindings[:nonrepeat]:
            append(key, True)
        cursors = bindings[nonrepeat:]
        for _ in range(repeat):
            if len(values) + len(cursors) > MAX_BINDINGS:
                break
            cursors = [append(key, True) for key in cursors]
    else:
        for key in bindings:
            append(key, operation == 0xA1)
            if status:
                break
    if status and version == 0:
        values = [tlv(0x30, oid(key) + tlv(5, b"")) for key in bindings]
    result = tlv(0xA2, tlv(2, pdu[0][1]) + integer(status) + integer(error_index) + tlv(0x30, b"".join(values)))
    return tlv(0x30, integer(version) + tlv(4, wire_community) + result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--community-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=1161)
    arguments = parser.parse_args()
    community = arguments.community_file.read_bytes().strip()
    if not 16 <= len(community) <= 128 or not 1024 <= arguments.port <= 65535:
        raise SystemExit("模拟器私有配置无效")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
        channel.bind(("0.0.0.0", arguments.port))
        print(json.dumps({"simulation": True, "protocol": "SNMPv1/v2c", "port": arguments.port,
                          "ports": list(PORTS.values()), "writable": False}), flush=True)
        while True:
            packet, peer = channel.recvfrom(MAX_PACKET + 1)
            try:
                result = response(packet, community)
                if result is not None:
                    channel.sendto(result, peer)
            except (ValueError, IndexError, OverflowError):
                # 不记录 community、原始报文或认证信息。
                continue


if __name__ == "__main__":
    main()
