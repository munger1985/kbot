# dao/data_services/kbot_md_sys_conf_pb2.pyi
from typing import Any, List, Optional

class KbotMdSysConf:
    conf_id: str
    # 添加其他字段的类型提示
    def __init__(self, *, conf_id: str = ...) -> None: ...
    
class KbotMdSysConfList:
    configs: List[KbotMdSysConf]
    def __init__(self, *, configs: Optional[List[KbotMdSysConf]] = ...) -> None: ...

class DeleteResponse:
    success: bool
    def __init__(self, *, success: bool = ...) -> None: ...

class Empty:
    def __init__(self) -> None: ...