"""数据库路径只连接同实例、同绑定、当前核验的承载文件系统。"""

from decimal import Decimal
import posixpath

from aiops_agent.domain.evidence.observations import parse_numeric_sample


def _path(value, *, base=None):
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return None
    if not value.startswith("/"):
        if base is None:
            return None
        value = posixpath.join(base, value)
    return posixpath.normpath(value)


def _contains(parent, child):
    return parent == child or child.startswith(parent.rstrip("/") + "/")


def storage_capacity_resolution(*, task_frame, monitoring, evidence):
    """返回逐路径证据定位；不跨口径算值，也不把未匹配主机盘交给模型任选。"""
    if task_frame.get("visualization_profile_id") != "storage.trend":
        return None
    bindings = {
        item["binding_id"]: item for item in monitoring.get("bindings", ())
        if item.get("purpose") == "STORAGE_CAPACITY"
    }
    result = {
        "status": "UNAVAILABLE", "volumes": [], "unresolved_paths": [],
        "boundaries": (
            "数据库逻辑用量与文件系统用量分别描述，不混合计算。",
            "当前映射不证明历史数据路径不变；文件系统预测使用该卷自己的历史，并包含其他共享写入。",
            "仅 volumes 内已定位卷可用于本数据库承载卷判断；其余主机文件系统不得任意归因。",
            "未显式请求预测时不把预计写满天数为空列为问题。",
        ),
    }
    locations = [fact for fact in evidence if fact.tool_id == "db.storage.locations"
        and fact.trust_level == "SOURCE_VERIFIED" and fact.followup_run_id is None]
    if not locations:
        result["reason"] = "本轮未取得可核验的数据库文件路径；已安排的路径工具以实际回执为准"
        return result
    paths, identities = [], set()
    truncated = False
    for fact in locations:
        truncated |= fact.truncated
        names = [column["name"] for column in fact.columns]
        records = [dict(zip(names, row)) for row in fact.rows]
        datadirs = {_path(row.get("file_path")) for row in records if row.get("location_kind") == "DATA_DIRECTORY"}
        datadirs.discard(None)
        base = next(iter(datadirs)) if len(datadirs) == 1 else None
        for index, row in enumerate(records):
            identity = row.get("server_uuid")
            if isinstance(identity, str) and identity:
                identities.add(identity)
            path = _path(row.get("file_path"), base=base)
            if path and identity:
                paths.append((path, identity, fact.evidence_ref, index))
            else:
                result["unresolved_paths"].append({"reason": "路径或实例身份无效", "evidence_ref": fact.evidence_ref, "row_index": index})
    if len(identities) != 1:
        result["reason"] = "数据库路径实例身份缺失或冲突"
        return result
    maps, series_rows = [], []
    for fact in evidence:
        if fact.tool_id != "metric.query_range" or fact.trust_level != "SOURCE_VERIFIED" or fact.truncated:
            continue
        names = [column["name"] for column in fact.columns]
        for trace in fact.series_computations:
            binding = bindings.get(str(trace.binding_id))
            if binding is None or str(trace.source_id) != str(binding["source"]["source_id"]):
                continue
            row = dict(zip(names, fact.rows[trace.row_index]))
            dimensions = dict(trace.dimensions)
            entry = (fact, trace, row, dimensions, binding)
            if trace.metric_code == "host.storage.mount_info":
                stamp = parse_numeric_sample(row.get("latest"))
                end = Decimal(str(trace.window_end.timestamp()))
                if stamp is not None and not trace.quality_flags and end - 300 <= stamp <= end + 30:
                    maps.append(entry)
            else:
                series_rows.append(entry)
    volumes = {}
    for path, identity, ref, index in paths:
        candidates = []
        for fact, trace, row, dims, binding in maps:
            prefix = _path(dims.get("database_path"))
            if (dims.get("server_uuid") != identity
                or dims.get("database_target_key") not in binding["storage_context"]["database_locator_keys"]
                or dims.get("target_key") != binding["source_locator_key"]
                or prefix is None or not _contains(prefix, path)):
                continue
            if not all(dims.get(key) for key in ("mountpoint", "device", "fstype", "host_path")):
                continue
            candidates.append((len(prefix), fact, trace, dims, binding))
        longest = max((item[0] for item in candidates), default=-1)
        candidates = [item for item in candidates if item[0] == longest]
        keys = {(item[2].binding_id, *(item[3][key] for key in ("database_path", "host_path", "mountpoint", "device", "fstype"))) for item in candidates}
        if len(keys) != 1:
            result["unresolved_paths"].append({"path": path, "evidence_ref": ref, "row_index": index,
                "reason": "承载路径映射缺失、过期或有歧义"})
            continue
        _, mapping_fact, mapping_trace, dims, binding = candidates[0]
        key = (binding["binding_id"], dims["mountpoint"], dims["device"], dims["fstype"])
        if key not in volumes:
            metrics = {}
            ambiguous = False
            for fact, trace, row, fs, current in series_rows:
                if (current["binding_id"] != binding["binding_id"] or
                    any(fs.get(name) != dims[name] for name in ("target_key", "mountpoint", "device", "fstype"))):
                    continue
                if trace.metric_code in metrics:
                    ambiguous = True
                metrics[trace.metric_code] = {
                    "evidence_ref": fact.evidence_ref, "row_index": trace.row_index,
                    "latest": row.get("latest"), "unit": row.get("unit"),
                    "change_per_day": row.get("change_per_day"),
                    "estimated_days_to_limit": row.get("estimated_days_to_limit"),
                    "forecast_confidence": row.get("forecast_confidence"),
                    "quality_flags": trace.quality_flags,
                }
            if ambiguous:
                result["unresolved_paths"].append({"path": path, "reason": "同卷指标系列不唯一，未任选一条"})
                continue
            volumes[key] = {
                "host_name": binding["resource_name"], "host_resource_id": binding["resource_id"],
                "mountpoint": dims["mountpoint"], "device": dims["device"], "fstype": dims["fstype"],
                "mapping_evidence_ref": mapping_fact.evidence_ref, "mapping_row_index": mapping_trace.row_index,
                "paths": [], "metrics": metrics,
            }
        volumes[key]["paths"].append({"path": path, "evidence_ref": ref, "row_index": index})
    result["volumes"] = list(volumes.values())
    if not paths:
        result["reason"] = "路径工具没有返回有效路径"
    elif volumes:
        result["status"] = "PARTIAL" if truncated or result["unresolved_paths"] else "RESOLVED"
    result["path_enumeration_truncated"] = truncated
    return result
