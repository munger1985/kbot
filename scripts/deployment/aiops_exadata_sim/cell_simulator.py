"""仅供隔离 OEM 调试：生成原生 CellCLI XML，不模拟 OEM API。"""

from datetime import datetime, timezone
import math
import re
import time
import xml.etree.ElementTree as ET


CELL = "sim-cell01"
DISKS = (("CD_00_sim_cell01", "HardDisk"), ("FD_00_sim_cell01", "FlashDisk"))


def values_at(stamp):
    """字段映射来自 XA 24.1.1 原厂采集脚本；所有数值为合成调试数据。"""
    wave = math.sin(stamp / 300)
    cell = {
        "CL_CPUT": round(38 + 12 * wave, 2), "CL_MEMUT": round(61 + 4 * wave, 2),
        "N_HCA_MB_RCV_SEC": round(128 + 16 * wave, 2),
        "N_HCA_MB_TRANS_SEC": round(96 + 12 * wave, 2),
        "IORM_MODE": 1, "CL_RUNQ": round(2 + wave, 2),
    }
    result = [("CELL", CELL, key, value) for key, value in cell.items()]
    for name, kind in DISKS:
        scale = 1 if kind == "HardDisk" else 2
        disk = {
            "CD_IO_BY_R_LG_SEC": 48 * scale, "CD_IO_BY_R_SM_SEC": 12 * scale,
            "CD_IO_BY_W_LG_SEC": 24 * scale, "CD_IO_BY_W_SM_SEC": 8 * scale,
            "CD_IO_RQ_R_LG_SEC": 180 * scale, "CD_IO_RQ_R_SM_SEC": 1200 * scale,
            "CD_IO_RQ_W_LG_SEC": 90 * scale, "CD_IO_RQ_W_SM_SEC": 600 * scale,
            "CD_IO_TM_R_LG_RQ": 800 / scale, "CD_IO_TM_R_SM_RQ": 400 / scale,
            "CD_IO_TM_W_LG_RQ": 700 / scale, "CD_IO_TM_W_SM_RQ": 350 / scale,
            "CD_IO_LOAD": 2, "CD_IO_UTIL": 24,
        }
        result.extend(("CELLDISK", name, key, value) for key, value in disk.items())
    flash = {
        "FC_IO_BY_R_SEC": 512, "FC_IO_BYKEEP_R_SEC": 128,
        "FC_IO_RQ_R_SEC": 10000, "FC_IO_RQKEEP_R_SEC": 2000,
        "FC_BY_USED": 1024 ** 4, "FC_BYKEEP_USED": 128 * 1024 ** 3,
    }
    result.extend(("FLASHCACHE", "sim_flashcache", key, value) for key, value in flash.items())
    smart = {"SIO_IO_SI_SV_SEC": 800, "SIO_IO_RD_HD_SEC": 96,
             "SIO_IO_RD_FC_SEC": 400, "SIO_IO_PA_TH_SEC": 1200}
    result.extend(("SMARTIO", "SMARTIO", key, value) for key, value in smart.items())
    return result


def element(parent, tag, fields):
    node = ET.SubElement(parent, tag)
    for key, value in fields.items():
        ET.SubElement(node, key).text = str(value)


def fault(message):
    root = ET.Element("fault")
    ET.SubElement(root, "errorDescription").text = message
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
def render(command, now=None):
    """只允许已实现的 LIST 子集；未知对象/字段明确失败，不返回伪造零值。"""
    now = int(time.time() if now is None else now)
    statements = [part.strip() for part in command.split(";") if part.strip()]
    if not statements or len(statements) > 8 or len(command) > 16384:
        return fault("SIM-001: 请求为空或超出模拟器预算")
    root = ET.Element("cli-output", {"simulation": "true"})
    ET.SubElement(root, "context", {"cell": CELL, "realm": "SIMULATION"})
    for statement in statements:
        match = re.fullmatch(r"list\s+(cell|celldisk|physicaldisk|metriccurrent|metrichistory)(?:\s+(.*))?",
                             statement, re.IGNORECASE)
        if not match:
            return fault("SIM-002: 模拟器不支持此命令；仅允许只读 LIST 子集")
        kind, tail = match[1].lower(), match[2] or ""
        if kind == "cell":
            fields = {"name": CELL, "id": "SIM-CELL-0001", "releaseVersion": "23.1.13.0.0",
                      "interconnect1": "ib0", "makeModel": "Oracle Exadata Storage Server X9M",
                      "cellsrvStatus": "running", "msStatus": "running", "rsStatus": "running",
                      "temperatureReading": 32, "iormBoost": 0, "locatorLEDStatus": "off"}
            if tail.lower().startswith("attributes "):
                names = [name.strip() for name in tail[len("attributes "):].split(",")]
                if any(name not in fields for name in names):
                    return fault("SIM-003: 未实现的 Cell 属性")
                fields = {name: fields[name] for name in names}
            elif tail and tail.lower() != "detail":
                return fault("SIM-004: 未实现的 Cell 查询条件")
            element(root, "cell", fields)
        elif kind in ("celldisk", "physicaldisk"):
            for index, (name, disk_type) in enumerate(DISKS):
                if kind == "physicaldisk":
                    name = "0:0" if index == 0 else "FLASH_1_0"
                fields = {"name": name, "diskType": disk_type, "status": "normal",
                          "size": "10T" if index == 0 else "3T"}
                if tail.lower().startswith("attributes "):
                    names = [name.strip() for name in tail[len("attributes "):].split(",")]
                    if any(name not in fields for name in names):
                        return fault("SIM-005: 未实现的磁盘属性")
                    fields = {name: fields[name] for name in names}
                elif tail and tail.lower() != "detail":
                    return fault("SIM-006: 未实现的磁盘查询条件")
                element(root, kind, fields)
        else:
            stamps = [now]
            requested = None
            if kind == "metrichistory":
                interval = re.findall(r"collectionTime\s*([<>])\s*'([^']+)'", tail, re.IGNORECASE)
                if len(interval) != 2 or {operator for operator, _ in interval} != {"<", ">"}:
                    return fault("SIM-007: 历史查询必须限定起止时间")
                try:
                    limits = {operator: int(datetime.fromisoformat(value).timestamp())
                              for operator, value in interval}
                except ValueError:
                    return fault("SIM-008: 时间格式无效")
                if limits["<"] <= limits[">"] or limits["<"] - limits[">"] > 3601:
                    return fault("SIM-009: 历史范围必须在一小时内")
                stamps = list(range((limits[">"] // 60 + 1) * 60, limits["<"], 60))
                prefix = re.split(r"\s+where\s+", tail, maxsplit=1, flags=re.IGNORECASE)[0]
                if prefix.strip():
                    requested = {name.strip() for name in prefix.split(",")}
            object_filter = re.search(r"objectType\s*=\s*'([A-Z]+)'", tail, re.IGNORECASE)
            for stamp in stamps:
                for object_type, name, metric, value in values_at(stamp):
                    if requested is not None and metric not in requested:
                        continue
                    if object_filter and object_type != object_filter[1].upper():
                        continue
                    fields = {"name": metric, "metricType": "Instantaneous", "metricObjectName": name,
                              "metricValue": value, "collectionTime": stamp * 1000,
                              "objectType": object_type}
                    element(root, kind, fields)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
