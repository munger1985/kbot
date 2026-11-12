"""仅供隔离 OEM 调试：生成原生 CellCLI XML，不模拟 OEM API。"""

from datetime import datetime, timezone
import math
import re
import time
import xml.etree.ElementTree as ET


CELL = "sim-cell01"
DISKS = (("CD_00_sim_cell01", "HardDisk"), ("FD_00_sim_cell01", "FlashDisk"))
EPOCH = 1794355200  # 固定的模拟计数器起点；所有累计量随采集时间单调增长。
CUMULATIVE = frozenset({
    "FC_IO_BY_R", "FC_IO_BY_R_MISS", "FC_IO_RQKEEP_R", "FC_IO_RQKEEP_R_MISS",
    "FC_IO_RQ_R", "FC_IO_RQ_R_MISS", "FC_IO_RQ_W_FIRST", "FC_IO_RQ_W_OVERWRITE",
    "FC_IO_RQ_W_SKIP", "FC_IO_RQ_R_DW", "FC_IO_BY_R_DW", "FC_IO_BY_R_MISS_DW",
    "FL_ACTUAL_OUTLIERS", "FL_PREVENTED_OUTLIERS", "FL_IO_W_SKIP_BUSY",
    "FL_IO_W_SKIP_NO_BUFFER", "FL_IO_W_SKIP_LARGE", "FL_IO_W",
})


def values_at(stamp):
    """字段映射来自 XA 24.1.1 原厂采集脚本；所有数值为合成调试数据。"""
    wave = math.sin(stamp / 300)
    elapsed = max(1, stamp - EPOCH)
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
        "FC_IO_BYKEEP_R": elapsed * 128, "FC_IO_BY_R": elapsed * 512,
        "FC_IO_BY_R_MISS": elapsed * 32,
        "FC_IO_RQKEEP_R": elapsed * 2000, "FC_IO_RQKEEP_R_MISS": elapsed * 20,
        "FC_IO_RQ_R": elapsed * 10000, "FC_IO_RQ_R_MISS": elapsed * 200,
        "FC_IO_RQ_W_FIRST": elapsed * 500, "FC_IO_RQ_W_FIRST_SEC": 500,
        "FC_IO_RQ_W_OVERWRITE": elapsed * 100, "FC_IO_RQ_W_OVERWRITE_SEC": 100,
        "FC_IO_RQ_DISK_WRITE_SEC": 450, "FC_IO_RQ_W_SKIP": elapsed * 5,
        "FC_IO_RQ_R_DW": elapsed * 100, "FC_IO_BY_R_DW": elapsed * 16,
        "FC_IO_BY_R_MISS_DW": elapsed * 2, "FC_BYKEEP_OVERWR_SEC": 4,
        "FC_IO_RQ_R_MISS_SEC": 200, "FC_IO_RQKEEP_R_MISS_SEC": 20,
        "FC_IO_RQ_W_POPULATE_SEC": 500,
    }
    result.extend(("FLASHCACHE", "sim_flashcache", key, value) for key, value in flash.items())
    flashlog = {
        "FL_ACTUAL_OUTLIERS": elapsed, "FL_PREVENTED_OUTLIERS": elapsed * 3,
        "FL_BY_KEEP": 1024 ** 3, "FL_DISK_IO_ERRS": 0,
        "FL_EFFICIENCY_PERCENTAGE": 98, "FL_EFFICIENCY_PERCENTAGE_HOUR": 97,
        "FL_FLASH_IO_ERRS": 0, "FL_IO_DB_BY_W_SEC": 32, "FL_IO_FL_BY_W_SEC": 30,
        "FL_IO_W": elapsed * 800, "FL_IO_W_SKIP_BUSY": elapsed * 2,
        "FL_IO_W_SKIP_BUSY_MIN": 2, "FL_IO_W_SKIP_LARGE": elapsed,
        "FL_IO_W_SKIP_NO_BUFFER": elapsed,
    }
    result.extend(("FLASHLOG", "sim_flashlog", key, value) for key, value in flashlog.items())
    smart = {"SIO_IO_SI_SV_SEC": 800, "SIO_IO_RD_HD_SEC": 96,
             "SIO_IO_RD_FC_SEC": 400, "SIO_IO_PA_TH_SEC": 1200}
    result.extend(("SMARTIO", "SMARTIO", key, value) for key, value in smart.items())
    # DB/PDB/消费组三种原生对象分开建模，绝不使用 IORM_MODE 冒充其 IO 指标。
    for prefix, kind, name, scale in (
        ("DB", "IORM_DATABASE", "simdb", 3),
        ("PDB", "IORM_PLUGGABLE_DATABASE", "simdb.simpdb", 2),
        ("CG", "IORM_CONSUMER_GROUP", "simdb.simcg", 1),
    ):
        iorm = {
            "IO_RQ_LG_SEC": 120 * scale, "IO_RQ_SM_SEC": 600 * scale,
            "IO_WT_LG": elapsed * scale * 200, "IO_WT_SM": elapsed * scale * 100,
            "IO_UTIL_LG": 8 * scale, "IO_UTIL_SM": 10 * scale,
            "IO_WT_LG_RQ": 2, "IO_WT_SM_RQ": 1,
            "IO_BY_SEC": 24 * scale, "IO_LOAD": scale,
            "IO_TM_LG": elapsed * scale * 800, "IO_TM_LG_RQ": 4,
            "IO_TM_SM": elapsed * scale * 600, "IO_TM_SM_RQ": 2,
            "FD_IO_RQ_LG_SEC": 240 * scale, "FD_IO_RQ_SM_SEC": 1200 * scale,
            "FD_IO_UTIL": 7 * scale, "FD_IO_BY_SEC": 64 * scale,
            "FD_IO_WT_LG_RQ": 0.4, "FD_IO_WT_SM_RQ": 0.2,
            "FD_IO_TM": elapsed * scale * 300, "FD_IO_TM_RQ": 0.5,
            "FD_IO_TM_SM_RQ": 0.3, "FC_BY_ALLOCATED": scale * 128 * 1024 ** 3,
        }
        result.extend((kind, name, prefix + "_" + key, value) for key, value in iorm.items())
    # 原厂互联脚本要求 HOST_INTERCONNECT 的 Cumulative 计数器，而非 CELL 的 HCA 速率。
    interconnect = {"N_MB_SENT": elapsed * 96, "N_MB_DROP": elapsed,
                    "N_MB_RESENT": elapsed * 2, "N_MB_RECEIVED": elapsed * 128,
                    "N_MB_RDMA_DROP": elapsed, "N_RDMA_RETRY_TM": elapsed * 3}
    result.extend(("HOST_INTERCONNECT", "simdb:ib0", key, value) for key, value in interconnect.items())
    result.extend(("CELL_FILESYSTEM", name, "FS_USE", value)
                  for name, value in (("/", round(42 + wave, 2)), ("/opt/oracle", round(61 + wave, 2))))
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
    ET.SubElement(root, "context", {"cell": CELL, "realm": "SIMULATION",
                                  "ossStartTimestamp": str(EPOCH * 1000),
                                  "iormResetTimestamp": str(EPOCH * 1000)})
    for statement in statements:
        match = re.fullmatch(r"list\s+(cell|celldisk|physicaldisk|flashcache|griddisk|metriccurrent|metrichistory)(?:\s+(.*))?",
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
                          "size": "10240G" if index == 0 else "3072G",
                          "freeSpace": "2048G" if index == 0 else "1024G",
                          "physicalSize": "10240G" if index == 0 else "3072G",
                          "slotNumber": index, "id": "SIM-DISK-" + str(index),
                          "physicalSerial": "SIM-SERIAL-" + str(index),
                          "makeModel": "SIM - Exadata disk", "physicalInterface": "SAS"}
                if tail.lower().startswith("attributes "):
                    names = [name.strip() for name in tail[len("attributes "):].split(",")]
                    if any(name not in fields for name in names):
                        return fault("SIM-005: 未实现的磁盘属性")
                    fields = {name: fields[name] for name in names}
                elif tail and tail.lower() != "detail":
                    return fault("SIM-006: 未实现的磁盘查询条件")
                element(root, kind, fields)
        elif kind in ("flashcache", "griddisk"):
            fields = ({"name": "sim_flashcache", "status": "normal", "size": "2048G",
                       "cellDisk": DISKS[1][0]} if kind == "flashcache" else
                      {"name": "SIMDATA_" + DISKS[0][0], "status": "active", "size": "8192G",
                       "cellDisk": DISKS[0][0], "asmName": "SIMDATA01",
                       "asmDiskGroupName": "SIMDATA", "sparse": "FALSE"})
            if tail.lower().startswith("attributes "):
                names = [name.strip() for name in tail[len("attributes "):].split(",")]
                if any(name not in fields for name in names):
                    return fault("SIM-012: 未实现的容量属性")
                fields = {name: fields[name] for name in names}
            elif tail and tail.lower() != "detail":
                return fault("SIM-013: 未实现的容量查询条件")
            element(root, kind, fields)
        else:
            stamps = [now]
            requested = None
            if kind == "metriccurrent":
                prefix = re.split(r"\s*(?:where|attributes)\s+", tail, maxsplit=1, flags=re.IGNORECASE)[0]
                if prefix.strip():
                    requested = {name.strip() for name in prefix.split(",")}
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
            object_filter = re.search(r"objectType\s*=\s*'([A-Z_]+)'", tail, re.IGNORECASE)
            for stamp in stamps:
                for object_type, name, metric, value in values_at(stamp):
                    if requested is not None and metric not in requested:
                        continue
                    if object_filter and object_type != object_filter[1].upper():
                        continue
                    fields = {"name": metric, "metricType": "Instantaneous", "metricObjectName": name,
                              "metricValue": value, "collectionTime": stamp * 1000,
                              "objectType": object_type}
                    if metric in CUMULATIVE or (object_type.startswith("IORM_") and
                            re.fullmatch(r"(?:DB|PDB|CG)_(?:IO_(?:WT|TM)_(?:LG|SM)|FD_IO_TM)", metric)):
                        fields["metricType"] = "Cumulative"
                    if object_type == "HOST_INTERCONNECT":
                        # 此原厂旧解析器在 metricValue 结束时取时间；必须先给 collectionTime。
                        fields = {"name": metric, "metricType": "Cumulative", "metricObjectName": name,
                                  "collectionTime": stamp * 1000, "metricValue": value,
                                  "objectType": object_type}
                    element(root, kind, fields)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)
