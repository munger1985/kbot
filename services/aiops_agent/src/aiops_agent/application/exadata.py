"""有界的 OEM Exadata 快照监控与资源级只读诊断。"""

import asyncio
from decimal import Decimal
from datetime import UTC, datetime, timedelta

from platform_core.identity import uuid7
from platform_core.contracts.aiops.exadata import (
    ExadataCandidate, ExadataDiscovery, ExadataView, ExadataSample, ExadataGap,
    ExadataQuery, ExadataFinding, ExadataReport,
)
from aiops_agent.entities.exadata import ExadataReportEntity
from aiops_agent.ports.exadata import ExadataMetricRead
from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found, state_conflict, validation_failed
from aiops_agent.application.configuration.exadata_binding import exadata_source, scope_args
from aiops_agent.application.configuration.common import add_configuration_event
from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from .exadata_catalog import load_catalog


class ExadataOEMService:
    def __init__(self, configuration):
        self.configuration = configuration
        self.version, self.digest, self.metrics, self.profiles, self.diagnosis_profiles = load_catalog()

    async def discover(self, *, scope, source_id):
        async with self.configuration._uow_factory() as uow:
            source, adapter = await exadata_source(self.configuration, uow, scope, source_id)
            version = int(source.row_version)
            rows = await adapter.discover_exadata()
            items = []
            for root, members, digest in rows:
                token = self.configuration._instance_candidate_refs.encode(source_id=source_id,
                    actor_id=scope.actor_id, purpose="candidate", **scope_args(scope),
                    value={"candidate_kind": "EXADATA", "source_version": version,
                        "root_id": root["target_id"], "inventory_sha256": digest})
                items.append(ExadataCandidate(candidate_ref=token, locator_key=root["target_id"],
                    display_name=root["display_name"], members=tuple(members)))
            return ExadataDiscovery(items=tuple(items))

    async def _freeze(self, uow, scope, system_id, resource_id, source_id):
        system = await uow.business_systems.get_scoped(business_system_id=system_id, lock=True, **scope_args(scope))
        resource = await uow.business_systems.get_resource_scoped(resource_id=resource_id, lock=True, **scope_args(scope))
        if system is None or resource is None:
            raise resource_not_found("Business System Resource")
        rows = await uow.business_systems.list_system_resource_rows(system_id)
        if not any(link.resource_id == resource_id and link.status == "ACTIVE" for link, *_ in rows):
            raise resource_not_found("Business System Resource")
        if system.status != "ENABLED" or resource.lifecycle_status != "ACTIVE":
            raise state_conflict("只有已上线系统的活跃资源可读取 Exadata 指标")
        if resource.resource_type_key != "EXADATA":
            raise validation_failed("Exadata 专用入口不能读取其他类型资源")
        source, adapter = await exadata_source(self.configuration, uow, scope, source_id)
        bindings = await uow.business_systems.list_resource_source_bindings(resource_id)
        binding = next((item for item in bindings if item.diagnostic_source_id == source_id), None)
        if binding is None:
            raise resource_not_found("Exadata OEM Binding")
        locator = binding.source_locator_json or {}
        if locator.get("resource_scope") != "EXADATA" or locator.get("target_type") != "oracle_dbmachine":
            raise state_conflict("Exadata 整机绑定身份无效")
        inventory = await uow.exadata.inventory(binding.resource_source_binding_id)
        if inventory is None:
            raise state_conflict("Exadata 尚未确认成员，请重新绑定")
        frozen = {"binding_id": binding.resource_source_binding_id, "binding_version": int(binding.row_version),
            "source_version": int(source.row_version), "resource_version": int(resource.row_version),
            "system_version": int(system.row_version), "root_id": binding.source_locator_key,
            "inventory_sha256": inventory.inventory_sha256, "inventory": dict(inventory.members_json)}
        return frozen, adapter

    async def _snapshot(self, scope, system_id, resource_id, source_id):
        async with self.configuration._uow_factory() as uow:
            return await self._freeze(uow, scope, system_id, resource_id, source_id)

    async def _validate_live_inventory(self, adapter, frozen):
        _, _, digest = await adapter.exadata_inventory(frozen["root_id"])
        if digest != frozen["inventory_sha256"]:
            raise state_conflict("OEM 整机成员已变化，必须重新发现并确认")

    async def collect(self, *, scope, system_id, resource_id, source_id, query: ExadataQuery, _diagnosis=False,
                      _metric_codes: tuple[str, ...] | None = None, _component_names: tuple[str, ...] = ()):
        # 聊天只能缩小已审核场景；公共页面合同与原有诊断选集不变。
        profiles = self.diagnosis_profiles if _diagnosis else self.profiles
        profile = next(item for item in profiles if item["id"] == query.scene)
        codes = profile["metric_codes"] if _metric_codes is None else _metric_codes
        if _metric_codes is not None and (_diagnosis or not codes or len(set(codes)) != len(codes)
                                         or any(code not in profile["metric_codes"] for code in codes)):
            raise validation_failed("聊天指标必须属于所选审核场景，不能修改诊断指标选集")
        frozen, adapter = await self._snapshot(scope, system_id, resource_id, source_id)
        window_end = datetime.now(UTC) if query.mode == "HISTORY" else None
        window_start = window_end - timedelta(minutes=query.window_minutes) if window_end else None
        samples, gaps = [], []
        members = frozen["inventory"]["members"]
        components = [frozen["inventory"]["root"], *members]
        if query.component_id:
            components = [item for item in components if item["target_id"] == query.component_id]
            if not components:
                raise validation_failed("所选组件不属于已确认的 Exadata 整机")
        if _component_names:
            matched = []
            for name in _component_names:
                matches = [item for item in components
                           if name.casefold() in {item["target_id"].casefold(), item["display_name"].casefold()}]
                if len(matches) != 1:
                    raise validation_failed("指定组件不属于已确认整机，或组件名称不唯一；请提供完整组件名")
                if matches[0] not in matched:
                    matched.append(matches[0])
            components = matched
        selected = [self.metrics[code] for code in codes]
        reads = [(item, metric) for metric in selected for item in components if item["target_type"] == metric["target_type"]]
        if not reads:
            gaps.append(ExadataGap(code="NO_APPLICABLE_METRICS", detail="此场景没有与已确认组件类型匹配的审核指标"))
        if query.offset > len(reads):
            raise validation_failed("Exadata 指标分页已超出当前成员目录，请从首页重读")
        end = min(query.offset + query.limit, len(reads))
        next_offset = end if end < len(reads) else None
        if query.offset or next_offset is not None:
            gaps.append(ExadataGap(code="QUERY_PAGE_PARTIAL", detail="本次只读取场景的一个有界分页，不代表整机场景完整覆盖"))
        try:
            async with asyncio.timeout(55):
                await self._validate_live_inventory(adapter, frozen)
                try:
                    async with asyncio.timeout(30):
                        for member, definition in reads[query.offset:end]:
                            try:
                                read = adapter.exadata_history if query.mode == "HISTORY" else adapter.exadata_latest
                                result = await read(ExadataMetricRead(source_locator_key=member["target_id"],
                                    window_start=window_start, window_end=window_end), definition, member["target_type"])
                            except DiagnosticSourceAdapterError as exc:
                                gaps.append(ExadataGap(code=exc.code, component_id=member["target_id"],
                                    metric_code=definition["code"], detail=(
                                        "OEM 原厂指标接口读取失败；不是 OEM 返回空采样"
                                        if exc.code == "SOURCE_PROVIDER_ERROR" else
                                        "OEM 指标读取失败或在线元数据与审核定义不匹配")))
                                continue
                            if not result["points"]:
                                gaps.append(ExadataGap(code="NO_DATA", component_id=member["target_id"],
                                    metric_code=definition["code"], detail="OEM 未返回采样，不能判定健康"))
                            for (when, value), dimensions in zip(result["points"], result["dimensions"], strict=True):
                                quality = ("TRUNCATED" if result["truncated"] else "EMPTY" if value in (None, "") else
                                    "STALE" if query.mode == "LATEST_SNAPSHOT" and
                                    (result["captured_at"] - when).total_seconds() > definition["max_age_seconds"] else "OBSERVED")
                                samples.append(ExadataSample(component_id=member["target_id"], metric_code=definition["code"],
                                    selector=definition["selector"], observed_at=when, value=value, dimensions=dimensions,
                                    unit=result["unit"], native_value_type=result["native_value_type"], quality=quality,
                                    response_sha256=result["response_sha256"]))
                            if result["truncated"]:
                                gaps.append(ExadataGap(code="SOURCE_TRUNCATED", component_id=member["target_id"],
                                    metric_code=definition["code"], detail="组件维度超出读取预算，不得视为完整覆盖"))
                except TimeoutError:
                    gaps.append(ExadataGap(code="COLLECTION_TIMEOUT", detail="采集达到 30 秒预算，部分指标尚未读取"))
                await self._validate_live_inventory(adapter, frozen)
        except TimeoutError as exc:
            raise AIOpsApplicationError(code="EXADATA_IDENTITY_TIMEOUT", message="整机身份复核超时，拒绝返回未核验数据",
                status_code=504, retryable=True) from exc
        current, _ = await self._snapshot(scope, system_id, resource_id, source_id)
        if current != frozen:
            raise state_conflict("采集期间资源、来源或绑定版本变化，请重试")
        gaps.append(ExadataGap(code="THRESHOLDS_UNVERIFIED", detail="未读取可验证的 OEM 在线阈值；仅按已审核字典解释组件状态，不据此判整机健康"))
        if query.mode == "LATEST_SNAPSHOT":
            gaps.append(ExadataGap(code="HISTORY_UNAVAILABLE", detail="本次仅为最新快照，不生成趋势、增量或跨组件均值"))
        else:
            gaps.append(ExadataGap(code="HISTORY_RESOLUTION_PROVIDER_MANAGED", detail="OEM 按实际留存选择原始/小时/日聚合；不据此推断精确增量或原始执行明细，文本状态不回退快照"))
        gaps.append(ExadataGap(code="MEMBERSHIP_SCOPE", detail="成员仅含 OEM 显式包含关系及 GUID 目录复核的整机直接组件和 Grid 下 Cell；不推断其他间接组件归属"))
        return ExadataView(resource_id=resource_id, binding_id=frozen["binding_id"],
            binding_version=frozen["binding_version"], source_version=frozen["source_version"],
            catalog_version=self.version, catalog_sha256=self.digest, captured_at=datetime.now(UTC),
            scene=query.scene, members=tuple([frozen["inventory"]["root"], *members]),
            query_mode=query.mode, window_start=window_start, window_end=window_end,
            history_resolution="PROVIDER_MANAGED" if query.mode == "HISTORY" else None,
            samples=tuple(samples), definitions=tuple({metric["code"]: metric for _, metric in reads[query.offset:end]}.values()),
            gaps=tuple(gaps), partial=True, total_queries=len(reads), next_offset=next_offset)

    async def diagnose(self, *, scope, system_id, resource_id, source_id, query, idempotency_key):
        # 单次有界资源取证，不投递数据库 Worker，不产生硬件变更动作。
        if not idempotency_key or len(idempotency_key) > 256:
            raise validation_failed("只读诊断必须提供不超过 256 字符的 Idempotency-Key")
        view = await self.collect(scope=scope, system_id=system_id, resource_id=resource_id, source_id=source_id,
            query=query, _diagnosis=True)
        findings = [ExadataFinding(code=gap.code, severity="INCONCLUSIVE", component_id=gap.component_id,
            metric_code=gap.metric_code, detail=gap.detail) for gap in view.gaps]
        for index, sample in enumerate(view.samples):
            if sample.quality != "OBSERVED":
                findings.append(ExadataFinding(code=sample.quality, severity="INCONCLUSIVE",
                    component_id=sample.component_id, metric_code=sample.metric_code, evidence_indexes=(index,),
                    detail="采样为空、过旧或截断，不能据此证明健康"))
        groups = {}
        for index, sample in enumerate(view.samples):
            if (sample.native_value_type == "NUMBER" and sample.quality == "OBSERVED" and sample.unit
                    and not self.metrics[sample.metric_code].get("status")):
                groups.setdefault((sample.metric_code, sample.unit), []).append((Decimal(sample.value), index))
        for (code, unit), values in groups.items():
            low, high = min(values), max(values)
            findings.append(ExadataFinding(code="OBSERVED_RANGE_IN_PAGE", severity="INCONCLUSIVE", metric_code=code,
                evidence_indexes=tuple(dict.fromkeys((low[1], high[1]))),
                detail=f"本页同指标、同原生单位的最小值 {low[0]}，最大值 {high[0]} {unit}；仅定位观察样本，不代表历史变化或故障阈值"))
        report = ExadataReport(report_id=uuid7(), resource_id=resource_id, status="INCONCLUSIVE",
            findings=tuple(findings), evidence=view, created_at=datetime.now(UTC))
        async def handler(uow, now):
            frozen, _ = await self._freeze(uow, scope, system_id, resource_id, source_id)
            if (frozen["binding_id"], frozen["binding_version"], frozen["source_version"]) != (
                    view.binding_id, view.binding_version, view.source_version):
                raise state_conflict("保存报告前 Exadata 绑定已变化")
            await uow.exadata.add_report(ExadataReportEntity(report_id=report.report_id, resource_id=resource_id,
                binding_id=view.binding_id, report_json=report.model_dump(mode="json"),
                created_by=scope.actor_id, created_at=now))
            await add_configuration_event(uow=uow, scope=scope, aggregate_type="EXADATA_REPORT", aggregate_id=report.report_id,
                event_type="EXADATA_READONLY_DIAGNOSIS_CREATED", row_version=1, details={"resource_id": str(resource_id)})
            return report
        return await self.configuration._idempotent(scope=scope, operation="EXADATA_READONLY_DIAGNOSIS",
            parent_resource=f"{system_id}:{resource_id}:{source_id}", idempotency_key=idempotency_key,
            payload=query.model_dump(mode="json"), response_type=ExadataReport, handler=handler)

    async def get_report(self, *, scope, system_id, resource_id, source_id, report_id):
        async with self.configuration._uow_factory() as uow:
            frozen, _ = await self._freeze(uow, scope, system_id, resource_id, source_id)
            row = await uow.exadata.report(report_id, resource_id, frozen["binding_id"])
            if row is None:
                raise resource_not_found("Exadata Report")
            return ExadataReport.model_validate(row.report_json)
