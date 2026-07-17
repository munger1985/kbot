import sys
import shutil
import re
import zipfile
import importlib
import inspect
from pathlib import Path
from loguru import logger
from typing import Any, Type, AsyncGenerator
from .skill_registry import SkillMetadata, SkillParam
from core.dictionary import PacketType
from agent.common import ContextMemory
from skills.base import SkillDomain, SkillRunMode


class SkillManager:
    _instance = None

    # IntentType → skill category 映射：确保 LLM 规划时能看到正确的技能
    # None 表示不过滤，展示全部技能
    INTENT_CATEGORY_MAP: dict[str, list[str] | None] = {
        "chitchat": ["general"],
        "knowledge_query": ["knowledge_retrieval", "cognitive_reasoning"],
        "data_analysis": ["data_visualization", "cognitive_reasoning"],
        "task_execution": None,
        "complex_hybrid": None,
        "ambiguous": ["general"],
    }

    def __new__(cls, skills_dir: str = "skills/skill_libs"):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, skills_dir: str = "skills/skill_libs"):
        """
        初始化动态自治技能管理器（全面兼容大模型 Tool Naming 规范与社区既有老技能协议）。
        :param skills_dir: 技能包存放的根目录
        """
        if self._initialized:
            return
        
        self._skills: dict[str, SkillMetadata] = {}
        self._skills_dir = Path(skills_dir)
        # 确保技能根目录存在
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始自动扫描注册
        self.auto_discover_skills()
        self._initialized = True

    def _normalize_skill_name(self, name: str) -> str:
        """
        工业级清洗：强制将技能名称收拢为符合 Anthropic 协议的小写字母、数字和连字符标准。
        例如: "AskDataSkill" -> "ask-data-skill", "ask_data_skill" -> "ask-data-skill"
        """
        # 处理驼峰命名，在大小写交界处加连字符
        s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1-\2', name)
        s2 = re.sub(r'([a-z0-9])([A-Z])', r'\1-\2', s1).lower()
        # 将所有的下划线及特殊符号全部替换为连字符
        cleaned = re.sub(r'[^a-z0-9\-]', '-', s2)
        # 去除两端连续的连字符
        return re.sub(r'-+', '-', cleaned).strip('-')

    def _wrap_community_skill(self, original_class: Type[Any], skill_name: str) -> Type[Any]:
        """
        【动态适配器黑魔法】：
        当社区或上传的老技能没有继承 BaseSkill 或未实现标准的 run_stream 时，
        此适配器会自动探测其既有入口，实现智能参数拦截、流式包伪装以及无感知多态调用。
        """
        # 实例化一个临时对象用来判定成员函数
        try:
            instance_mock = original_class()
        except Exception:
            return original_class  # 如果构造函数需要复杂传参，交由底层自行报错，不拦截

        possible_methods = ["run_stream", "run", "execute", "_run", "_arun", "call"]
        target_method_name = None
        
        for method_name in possible_methods:
            if hasattr(instance_mock, method_name) and callable(getattr(instance_mock, method_name)):
                target_method_name = method_name
                break
                
        # 情况 1：如果已经是原生高标准的 run_stream 异步生成器，且继承了规范，直接原样放行
        if target_method_name == "run_stream" and inspect.isasyncgenfunction(getattr(instance_mock, target_method_name)):
            return original_class

        # 情况 2：触发老代码降维打法，动态生成一个符合系统规范的桩类（Stub Class）
        actual_method_name = target_method_name
        if not actual_method_name:
            logger.warning(f"组件类 [{original_class.__name__}] 未包含任何已知执行入口，将原样暴露以依赖底层抛错机制。")
            return original_class

        logger.info(f"⚡ 成功为社区老技能 [{original_class.__name__}] 注入智能流化适配器管道。")

        class AdaptedSkill:
            def __init__(self, *args, **kwargs):
                self._original_instance = original_class(*args, **kwargs)
                # 反向同步原作者可能在初始化时定义的各类变量与属性
                self.__dict__.update(self._original_instance.__dict__)

            async def run_stream(self, context: ContextMemory, **kwargs) -> AsyncGenerator[dict[str, Any], None]:
                yield {"type": PacketType.THOUGHT, "content": f"正在调遣社区生态组件 [{original_class.__name__}]...\n"}
                
                try:
                    current_execution = context.get("current_execution") or {}
                    actual_func = getattr(self._original_instance, actual_method_name)
                    
                    # 智能化分析社区这个老函数到底需要什么入参
                    sig = inspect.signature(actual_func)
                    injected_args = {}
                    
                    # 遍历对方的入参诉求，从当前系统上下文及总线中进行提取
                    for param_name in sig.parameters:
                        if param_name in ["self"]:
                            continue
                        if param_name == "context":
                            injected_args["context"] = context
                            continue
                            
                        # 核心两级反转：优先取已经用 Runtime 洗白过的变量值，次之看全局变量池，最后看流式透传 kwargs
                        val = (
                            context.get("variables", {}).get(param_name) 
                            or kwargs.get(param_name)
                        )
                        # 如果是 query/text 且依然为空，直接用 resolved_input 作为终极兜底填充
                        if val is None and param_name in ["query", "text", "question", "prompt"]:
                            val = current_execution.get("resolved_input")
                            
                        injected_args[param_name] = val

                    # 执行核心探测拦截
                    if inspect.isasyncgenfunction(actual_func):
                        # 兼容有写 run_stream 但没用最新总线机制的异步生成器
                        async for pack in actual_func(**injected_args):
                            yield pack
                        return
                    elif inspect.iscoroutinefunction(actual_func):
                        # 传统标准异步普通函数 (async def run)
                        result = await actual_func(**injected_args)
                    else:
                        # 传统同步普通函数 (def run)
                        result = actual_func(**injected_args)
                    
                    # 将社区老插件返回的各种原生数据体（dict, list, str 等）统一规整包装成总线认识的 DONE 消息体
                    yield {"type": PacketType.DONE, "content": result}
                    
                except Exception as e:
                    logger.exception(f"社区组件内部逻辑执行崩溃: {e}")
                    yield {"type": PacketType.ERROR, "content": f"社区外部组件执行失败: {str(e)}"}

        # 保持包装后的类名可读性
        AdaptedSkill.__name__ = original_class.__name__
        return AdaptedSkill

    def auto_discover_skills(self):
        """
        分布式自动化扫描注册：扫描每个独立技能包文件夹下的 skill.md 进行解析和类绑定。
        """
        new_skills = {}
        for md_path in self._skills_dir.glob("*/skill.md"):
            try:
                package_dir = md_path.parent
                package_name = package_dir.name  # 例如: "ask-data-skill"
                
                # 1. 解析说明书 skill.md
                meta_data = self._parse_skill_md(md_path)
                if not meta_data:
                    continue
                
                # 2. 强行对齐大模型标准的规范小写连字符名字
                standard_name = self._normalize_skill_name(meta_data["name"])
                
                # 3. 动态加载类
                class_name = meta_data["name"]
                impl_class = self._load_class_from_package(package_name, class_name)
                
                if not impl_class:
                    continue

                # 4. 关键卡点：为该类套上智能流化适配器防护罩，对齐框架标准
                adapted_impl_class = self._wrap_community_skill(impl_class, standard_name)

                # 5. 组装强类型元数据，将清洗后的规范名称作为唯一主键
                new_skills[standard_name] = SkillMetadata(
                    name=standard_name,
                    description=meta_data["description"],
                    category=meta_data["category"],
                    domain=meta_data.get("domain", SkillDomain.BUSINESS),
                    run_mode=meta_data.get("run_mode", SkillRunMode.READ_ONLY),
                    usage_example=meta_data["usage_example"],
                    params=meta_data["params"],
                    implementation_class=adapted_impl_class
                )
                logger.info(f"分布式自治技能 [ {standard_name} ] 注册激活！类别: {meta_data['category']}")

            except Exception as e:
                logger.error(f"扫描解析技能包 {md_path.parent.name} 失败: {e}", exc_info=True)

        self._skills = new_skills
        logger.success(f"全局自治域技能库构建完毕，共激活 {len(self._skills)} 个组件")

    def _parse_skill_md(self, md_path: Path) -> dict[str, Any] | None:
        """
        高度兼容的 Markdown 解析器：提取 Front Matter 元数据和参数定义
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()

            match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", content, re.DOTALL)
            if not match:
                logger.warning(f"{md_path} 格式不规范，缺少 Front Matter")
                return None

            front_matter_raw = match.group(1)
            body_raw = match.group(2)

            meta = {}
            for line in front_matter_raw.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()

            meta.setdefault("category", "general")
            meta.setdefault("description", "暂无描述")
            meta.setdefault("usage_example", "暂无示例")
            # 解析 domain 和 run_mode（运维Agent 移植需要）
            raw_domain = meta.get("domain", "business").lower()
            raw_run_mode = meta.get("run_mode", "read_only").lower()
            try:
                skill_domain = SkillDomain(raw_domain)
            except ValueError:
                skill_domain = SkillDomain.BUSINESS
            try:
                skill_run_mode = SkillRunMode(raw_run_mode)
            except ValueError:
                skill_run_mode = SkillRunMode.READ_ONLY
            meta["domain"] = skill_domain
            meta["run_mode"] = skill_run_mode

            params = []
            param_matches = re.findall(r"\*\s+(\w+)\s+\((\w+),\s*(必填|可选)\):\s*(.*)", body_raw)
            for p_name, p_type, p_req, p_desc in param_matches:
                params.append(SkillParam(
                    name=p_name,
                    param_type=p_type,
                    description=p_desc.strip(),
                    required=(p_req == "必填")
                ))

            meta["params"] = params
            return meta
        except Exception as e:
            logger.error(f"解析说明文件描述失败 {md_path}: {e}")
            return None

    def _load_class_from_package(self, package_name: str, class_name: str) -> Type[Any] | None:
        """从自治技能包内动态导入目标业务类，具备智能模糊兼容性"""
        try:
            module_path = f"skills.skill_libs.{package_name}"
            module = importlib.import_module(module_path)
            
            # 优先精准匹配类名
            if hasattr(module, class_name):
                return getattr(module, class_name)
            
            # 【老代码兼容】：如果写了驼峰的类名，但在 md 中写了小写，进行全局扫描
            for attr in dir(module):
                if attr.lower() == class_name.lower().replace("-", "").replace("_", ""):
                    return getattr(module, attr)
            
            logger.error(f"在模块 {module_path} 中未找到导出的类 {class_name}，请确认 __init__.py")
            return None
        except Exception as e:
            logger.error(f"动态构建类加载通道失败 {package_name} -> {class_name}: {e}")
            return None

    def register_uploaded_skill(self, temp_zip_path: Path) -> dict[str, Any]:
        """
        工业安全级：接收前端上传的 zip 包，强制将临时沙箱文件夹和落地包目录统一清洗为
        标准小写连字符形式，保证文件系统与大模型 API 工具链的一致性。
        """
        # 清洗上传包基础文件前缀，强转小写及连字符，防止 Python 动态 import 时路径崩溃
        sanitized_stem = self._normalize_skill_name(temp_zip_path.stem).replace("-", "_")
        extract_dir = self._skills_dir / f"tmp_{sanitized_stem}"
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
            md_files = [f for f in extract_dir.rglob("*") if f.name.lower() == "skill.md"]
            if not md_files:
                raise ValueError("非法扩展包：压缩包内未检测到任何标准的 skill.md 自治描述文件")
            
            raw_md_path = md_files[0]
            raw_package_dir = raw_md_path.parent
            
            # 读取新上传包中 skill.md 定义的名称
            parsed_meta = self._parse_skill_md(raw_md_path)
            if not parsed_meta or "name" not in parsed_meta:
                raise ValueError("解包失败：无法正常解析上传组件内 skill.md 的 Front Matter 元数据")
            
            # 【大模型标准对齐】：文件夹命名与规范完全咬死，由于 Python 模块名不能带连字符，包目录用下划线风格
            package_dir_name = self._normalize_skill_name(parsed_meta["name"]).replace("-", "_")
            final_dest_dir = self._skills_dir / package_dir_name
            
            # 强制规范说明书文件名为标准小写
            if raw_md_path.name != "skill.md":
                raw_md_path.rename(raw_md_path.with_name("skill.md"))
                
            # 热替换机制
            if final_dest_dir.exists():
                logger.warning(f"检测到已存在的技能包目录 [ {package_dir_name} ]，执行热重载全量覆盖。")
                shutil.rmtree(final_dest_dir)
                
            # 移动沙箱中的真实业务包到技能库
            shutil.move(str(raw_package_dir), str(final_dest_dir))
            
            # 清理 Python sys.modules 模块依赖缓存，防止不释放旧代码导致的热重载失效
            module_path = f"skills.skill_libs.{package_dir_name}"
            if module_path in sys.modules:
                del sys.modules[module_path]
            invalid_keys = [k for k in sys.modules.keys() if k.startswith(f"{module_path}.")]
            for k in invalid_keys:
                del sys.modules[k]
                
            # 驱动重新全盘加载
            self.auto_discover_skills()
            
            # 获取清洗后的标准大模型 Tool Name 键
            standard_skill_key = self._normalize_skill_name(parsed_meta["name"])
            
            if standard_skill_key in self._skills:
                return {"success": True, "message": f"全兼容社区组件 [{standard_skill_key}] 热装配就绪！"}
            else:
                raise RuntimeError("模块代码文件已被解包落地，但系统在二次静态扫描时未能绑定成功。")
                
        except Exception as e:
            logger.error(f"热注册扩展插件管道遭受重创: {e}")
            return {"success": False, "error": str(e)}
            
        finally:
            if extract_dir.exists():
                shutil.rmtree(extract_dir)

    def get_skill_list_for_planner(self, category_filter: str | None = None, domain_filter: SkillDomain | None = None) -> str:
        """生成给 Planner 看的协议描述（支持工具动态裁剪），输出大模型极度亲和的标准的连字符键名。

        Args:
            category_filter: 按 skill.md 中的 category 字段过滤
            domain_filter: 按 SkillDomain 枚举过滤（运维Agent使用 OPS 域隔离）
        """
        if not self._skills:
            return "当前系统未配置任何业务技能，请使用通用知识回答。"

        # 将 IntentType 映射到 skill category 列表
        allowed_categories: list[str] | None = None
        if category_filter:
            allowed_categories = self.INTENT_CATEGORY_MAP.get(category_filter)

        segments = []
        for name, meta in self._skills.items():
            # 领域过滤（运维Agent 的 OPS 域隔离）
            if domain_filter is not None:
                skill_domain = getattr(meta, "domain", SkillDomain.BUSINESS)
                if skill_domain != domain_filter:
                    continue

            # 类别过滤：将 IntentType 映射到 skill category
            skill_category = getattr(meta, "category", "general")
            if allowed_categories is not None:
                # "general" 类别始终保留（兜底用），其他类别需在允许列表中
                if skill_category != "general" and skill_category not in allowed_categories:
                    continue

            # 安全检查：输出技能运行模式（供 OpsOrchestrator 安全门禁使用）
            skill_run_mode = getattr(meta, "run_mode", SkillRunMode.READ_ONLY)
            mode_tag = ""
            if skill_run_mode == SkillRunMode.MUTATION:
                mode_tag = " ⚠️[高危变更操作·需审批]"

            param_details = [
                f"{p.name}({p.param_type}, {'必填' if p.required else '可选'}): {p.description}"
                for p in meta.params
            ]
            p_str = " | ".join(param_details) if param_details else "无参数"

            block = (
                f"### 技能: `{name}`{mode_tag}\n"
                f"- **功能**: {meta.description}\n"
                f"- **输入参数**: [{p_str}]\n"
                f"- **场景示例**: {meta.usage_example}"
            )
            segments.append(block)

        if not segments:
            filter_desc = f" [{domain_filter.value}] 域" if domain_filter else ""
            return f"当前{filter_desc}下无特定专属工具，请使用通用推理直接回答。"

        return "\n\n".join(segments)

    def get_skill_instance(self, name: str):
        """
        根据名称获取技能实例。
        修复逻辑：从 Metadata 中提取 implementation_class 并实例化
        """
        target_meta = None

        # 1. 尝试直接匹配
        if name in self._skills:
            target_meta = self._skills[name]

        # 2. 归一化匹配（保留你之前的逻辑）
        if not target_meta:
            def normalize(s: str):
                s = s.lower().replace("-", "").replace("_", "")
                if s.endswith("skill"):
                    s = s[:-5]
                return s

            target_norm = normalize(name)
            for reg_name, meta in self._skills.items():
                if normalize(reg_name) == target_norm:
                    target_meta = meta
                    break

        # 3. 核心修复：执行实例化
        if target_meta:
            try:
                # 从元数据中获取被 _wrap_community_skill 包装过的类并实例化
                instance = target_meta.implementation_class()
                return instance
            except Exception as e:
                logger.error(f"实例化技能类 {target_meta.name} 失败: {e}")
                raise RuntimeError(f"技能 {name} 实例化崩溃")

        # 4. 兜底报错
        logger.error(f"匹配失败！请求名称: '{name}'")
        logger.error(f"当前已注册技能: {list(self._skills.keys())}")
        raise ValueError(f"无法定位并初始化该技能组件: {name}")