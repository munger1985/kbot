INDEX_ROLES = {"embedding": "TEXT", "visual_embedding": "IMAGE"}


def index_rebuild_batches(configuration):
    """整库批次与单文件批次分开；旧的库级单文件记录仍归原文件。"""
    bulk = configuration.get("index_rebuild")
    files = dict(configuration.get("file_index_rebuilds", {}))
    if bulk and bulk.get("scope") == "FILE":
        files.setdefault(next(iter(bulk["members"])), bulk)
        bulk = None
    return bulk, files


def active_rebuild(batch):
    return bool(batch and not batch.get("terminal_status"))


def index_batch_progress(manifest, *, validated):
    """进度只读代次清单里的事务已提交批次，不暴露制品引用。"""
    committed = len(manifest.get("batches", {}))
    planned = manifest.get("batch_count" if validated else "planned_batch_count")
    total = planned if type(planned) is int and planned > 0 and planned >= committed else None
    return {"committed_batches": committed, "total_batches": total,
            "validated": validated and total == committed}


class LibraryConflict(ValueError):
    """可披露的知识库治理冲突；不附带其他主体或依赖标识。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)
