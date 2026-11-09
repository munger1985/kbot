"""核验显式登记的数据库路径映射；只读挂载元数据，不访问 Docker API 或数据库凭据。"""

import json
import os
from pathlib import Path
import posixpath
import re
import stat as file_stat
import time


def clean_path(value):
    if not isinstance(value, str) or not value.startswith("/") or "\x00" in value or "\\" in value:
        raise ValueError("路径必须为绝对 Unix 路径")
    if ".." in value.split("/"):
        raise ValueError("路径不得包含上级目录")
    return posixpath.normpath(value)


def decode_mount(value):
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value)


def parse_mounts(text):
    mounts = []
    for line in text.splitlines():
        left, right = line.split(" - ", 1)
        fields, device = left.split(), right.split()
        mounts.append({
            "root": decode_mount(fields[3]),
            "mountpoint": decode_mount(fields[4]), "device": decode_mount(device[1]),
            "fstype": device[0], "device_number": fields[2],
        })
    return mounts


def contains(parent, child):
    return parent == child or child.startswith(parent.rstrip("/") + "/")


def quote(value):
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def container_namespaces(mappings, *, proc=Path("/host-proc")):
    """只读取登记容器的 cgroup 与 mountinfo；不读环境、命令行或进程根目录。"""
    wanted = {item["container_id"] for item in mappings}
    found = {identity: set() for identity in wanted}
    pids = sorted((item for item in proc.iterdir() if item.name.isdecimal()), key=lambda item: int(item.name))
    if len(pids) > 8192:
        raise ValueError("主机进程枚举超出有界范围")
    for path in pids:
        try:
            cgroup = (path / "cgroup").read_text()
            matches = [identity for identity in wanted if identity in cgroup]
            if not matches:
                continue
            mountinfo = (path / "mountinfo").read_text()
            for identity in matches:
                found[identity].add(mountinfo)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return {identity: parse_mounts(next(iter(values))) for identity, values in found.items() if len(values) == 1}


def verify_container_mount(mapping, host_mount, namespaces):
    """核对当前容器命名空间的设备及文件系统内路径，不能只核对主机目录存在。"""
    database = clean_path(mapping["database_path"])
    host = clean_path(mapping["host_path"])
    mounts = namespaces.get(mapping["container_id"], ())
    candidates = [item for item in mounts if contains(item["mountpoint"], database)]
    longest = max((len(item["mountpoint"]) for item in candidates), default=-1)
    candidates = [item for item in candidates if len(item["mountpoint"]) == longest]
    if len(candidates) != 1:
        raise ValueError("当前容器路径不可核实")
    mounted = candidates[0]
    database_root = posixpath.normpath(posixpath.join(mounted["root"], posixpath.relpath(database, mounted["mountpoint"])))
    host_root = posixpath.normpath(posixpath.join(host_mount["root"], posixpath.relpath(host, host_mount["mountpoint"])))
    if mounted["device_number"] != host_mount["device_number"] or database_root != host_root:
        raise ValueError("当前容器挂载与登记路径不一致")
    # 未登记的嵌套挂载不能沿用父路径映射，避免把另一块盘归到父盘。
    if any(item["mountpoint"] != database and contains(database, item["mountpoint"]) for item in mounts):
        raise ValueError("数据库路径含嵌套挂载，需要分别登记")


def collect(config, mount_text, *, stamp, stat=os.stat, lstat=os.lstat, namespaces=None):
    """按目录边界选择最长挂载前缀，同时枚举根路径下的独立子挂载。"""
    mounts = parse_mounts(mount_text)
    if namespaces is None:
        namespaces = container_namespaces(config["mappings"])
    lines = [
        "# HELP aiops_database_storage_mount_info 当前数据库路径映射的核验时间（Unix秒）",
        "# TYPE aiops_database_storage_mount_info gauge",
    ]
    emitted = set()
    errors = 0
    for mapping in config["mappings"]:
        try:
            host = clean_path(mapping["host_path"])
            database = clean_path(mapping["database_path"])
            # 不猜测符号链接目标；部署时需登记真实路径。
            parts = host.strip("/").split("/") if host != "/" else []
            for index in range(1, len(parts) + 1):
                candidate = "/host/" + "/".join(parts[:index])
                # 仅挂载映射根的侧车没有所有父路径，父路径不存在可忽略。
                try:
                    if file_stat.S_ISLNK(lstat(candidate).st_mode):
                        raise ValueError("登记路径包含符号链接")
                except FileNotFoundError:
                    if index == len(parts):
                        raise
            info = stat("/host" + host)
            candidates = [item for item in mounts if contains(item["mountpoint"], host)]
            longest = max((len(item["mountpoint"]) for item in candidates), default=-1)
            candidates = [item for item in candidates if len(item["mountpoint"]) == longest]
            if len(candidates) != 1:
                raise ValueError("承载挂载点缺失或不唯一")
            selected = candidates[0]
            verify_container_mount(mapping, selected, namespaces)
            if selected["device_number"] != f"{os.major(info.st_dev)}:{os.minor(info.st_dev)}":
                raise ValueError("登记路径与当前主机设备不一致")
            applicable = [(selected, database, host)]
            for mount in mounts:
                if mount["mountpoint"] != host and contains(host, mount["mountpoint"]):
                    suffix = posixpath.relpath(mount["mountpoint"], host)
                    applicable.append((mount, posixpath.join(database, suffix), mount["mountpoint"]))
            for mount, db_path, host_path in applicable:
                if mount["fstype"] in {"tmpfs", "overlay", "squashfs", "proc", "sysfs", "devtmpfs"}:
                    continue
                labels = {
                    "database_target_key": mapping["database_target_key"],
                    "server_uuid": mapping["server_uuid"],
                    "database_path": db_path, "host_path": host_path,
                    **{key: mount[key] for key in ("mountpoint", "device", "fstype")},
                }
                key = tuple(sorted(labels.items()))
                if key in emitted:
                    continue
                if len(emitted) >= 128:
                    raise ValueError("路径映射超过有界采集范围")
                emitted.add(key)
                lines.append("aiops_database_storage_mount_info{" +
                    ",".join(f"{key}={quote(value)}" for key, value in sorted(labels.items())) +
                    "} " + str(int(stamp)))
        except (OSError, ValueError, KeyError):
            errors += 1
    if errors:
        # 任一登记映射未核实就不发布本轮映射，避免截断目录冒充完整。
        return "\n".join(lines[:2]) + "\n", errors
    return "\n".join(lines) + "\n", 0


def main():
    while True:
        try:
            config = json.loads(Path("/config/mappings.json").read_text())
            payload, errors = collect(config, Path("/host-mountinfo").read_text(), stamp=time.time())
        except (OSError, ValueError, KeyError):
            payload, errors = "", 1
        output = Path("/output/storage-paths.prom")
        temporary = output.with_suffix(".tmp")
        temporary.write_text(payload)
        temporary.chmod(0o644)
        temporary.replace(output)
        if errors:
            print("数据库路径映射未核验，通过空映射保留真实缺口", flush=True)
        time.sleep(30)


if __name__ == "__main__":
    main()
