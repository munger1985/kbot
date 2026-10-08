"""把可信结构化数据确定性编译为语义图表合同。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from math import ceil
from typing import Any

from platform_core.contracts.visualization import (
    ChartAxis,
    ChartPoint,
    ChartSeries,
    ChartSpec,
)


class ChartSkill:
    """共享 Chart Skill；不生成代码，也不重新计算业务结论。"""

    _maximum_points_per_series = 240

    @classmethod
    def time_series(
        cls,
        *,
        title: str,
        unit: str,
        series: Sequence[Mapping[str, Any]],
        source_ids: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> ChartSpec:
        display_unit, divisor = cls._display_unit(unit)
        compiled: list[ChartSeries] = []
        for index, item in enumerate(series):
            raw_points = tuple(
                sorted(
                    [
                        (str(point["x"]), float(point["y"]) / divisor)
                        for point in item.get("points", ())
                        if cls._is_number(point.get("y"))
                    ],
                    key=lambda point: point[0],
                )
            )
            if not raw_points:
                continue
            points = cls._downsample(raw_points)
            compiled.append(
                ChartSeries(
                    series_key=str(item.get("series_key") or f"series-{index + 1}"),
                    name=str(item.get("name") or f"序列 {index + 1}"),
                    unit=display_unit,
                    points=tuple(
                        ChartPoint(
                            x=x,
                            y=round(y, 4),
                            display_value=cls._format_value(y, display_unit),
                        )
                        for x, y in points
                    ),
                )
            )
        if not compiled:
            raise ValueError("趋势图缺少可用数值点")
        return ChartSpec(
            chart_type="LINE",
            intent="TREND",
            title=title,
            layout="SMALL_MULTIPLES" if len(compiled) > 1 else "OVERLAY",
            x_axis=ChartAxis(axis_type="TIME", label="时间"),
            y_axis=ChartAxis(axis_type="VALUE", label="数值", unit=display_unit),
            series=tuple(compiled),
            source_ids=tuple(dict.fromkeys(str(item) for item in source_ids)),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def capacity(
        cls,
        *,
        title: str,
        rows: Sequence[Mapping[str, Any]],
        source_ids: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> ChartSpec:
        used_points: list[ChartPoint] = []
        free_points: list[ChartPoint] = []
        expandable_points: list[ChartPoint] = []
        for row in rows:
            name = str(row.get("tablespace_name") or "").strip()
            used = cls._number(row.get("used_mb"))
            allocated = cls._number(row.get("allocated_mb"))
            maximum = cls._number(row.get("maximum_mb"))
            if not name or used is None or allocated is None or not maximum:
                continue
            current_free = max(0.0, allocated - used)
            expandable = max(0.0, maximum - allocated)
            common = {
                "used_mib": round(used, 2),
                "allocated_mib": round(allocated, 2),
                "maximum_mib": round(maximum, 2),
                "allocated_used_percent": round(used / allocated * 100, 2)
                if allocated
                else None,
            }
            used_points.append(
                ChartPoint(
                    x=name,
                    y=round(used / maximum * 100, 4),
                    display_value=f"{used:.2f} MiB",
                    metadata=common,
                )
            )
            free_points.append(
                ChartPoint(
                    x=name,
                    y=round(current_free / maximum * 100, 4),
                    display_value=f"{current_free:.2f} MiB",
                    metadata=common,
                )
            )
            expandable_points.append(
                ChartPoint(
                    x=name,
                    y=round(expandable / maximum * 100, 4),
                    display_value=f"{expandable:.2f} MiB",
                    metadata=common,
                )
            )
        if not used_points:
            raise ValueError("容量图缺少可用容量行")
        return ChartSpec(
            chart_type="STACKED_BAR",
            intent="CAPACITY",
            title=title,
            layout="NORMALIZED",
            x_axis=ChartAxis(axis_type="CATEGORY", label="表空间"),
            y_axis=ChartAxis(axis_type="VALUE", label="最大容量占比", unit="%"),
            series=(
                ChartSeries(
                    series_key="used",
                    name="已用",
                    unit="%",
                    stack="capacity",
                    points=tuple(used_points),
                ),
                ChartSeries(
                    series_key="current_free",
                    name="当前已分配余量",
                    unit="%",
                    stack="capacity",
                    points=tuple(free_points),
                ),
                ChartSeries(
                    series_key="expandable",
                    name="可扩展余量",
                    unit="%",
                    stack="capacity",
                    points=tuple(expandable_points),
                ),
            ),
            source_ids=tuple(dict.fromkeys(str(item) for item in source_ids)),
            metadata={"value_unit": "MiB", **dict(metadata or {})},
        )

    @classmethod
    def tabular(
        cls,
        *,
        title: str,
        columns: Sequence[Mapping[str, Any]],
        rows: Sequence[Mapping[str, Any]],
        source_ids: Sequence[str],
    ) -> ChartSpec:
        if not rows:
            raise ValueError("图表数据不能为空")
        names = [str(item.get("name") or item.get("key") or "") for item in columns]
        dimension = next(
            (
                name
                for name in names
                if name
                and cls._column_kind(columns, name)
                in {"STRING", "DATETIME"}
            ),
            None,
        )
        numeric = [
            name
            for name in names
            if name
            and cls._column_kind(columns, name)
            in {"INTEGER", "DECIMAL", "NUMBER"}
        ][:4]
        if dimension is None:
            dimension = next(
                (
                    name
                    for name in names
                    if name and isinstance(rows[0].get(name), str)
                ),
                "row",
            )
        if not numeric:
            numeric = [
                key
                for key, value in rows[0].items()
                if cls._is_number(value)
            ][:4]
        if not numeric:
            raise ValueError("图表数据缺少数值列")
        is_time = any(
            str(item.get("name") or item.get("key") or "") == dimension
            and str(item.get("logical_type") or "").upper() == "DATETIME"
            for item in columns
        )
        if not is_time and dimension != "row":
            dimension_values = [
                row.get(dimension)
                for row in rows[:10]
                if row.get(dimension) is not None
            ]
            is_time = bool(dimension_values) and all(
                cls._looks_datetime(value) for value in dimension_values
            )
        series_items = []
        for metric in numeric:
            points = []
            for index, row in enumerate(rows[:1000]):
                value = row.get(metric)
                if not cls._is_number(value):
                    continue
                points.append(
                    ChartPoint(
                        x=str(row.get(dimension, index + 1)),
                        y=float(value),
                        display_value=cls._format_value(float(value), None),
                    )
                )
            if points:
                series_items.append(
                    ChartSeries(
                        series_key=metric,
                        name=metric,
                        points=tuple(points),
                    )
                )
        if not series_items:
            raise ValueError("图表数据缺少可用数值")
        if is_time:
            normalized = [
                {
                    "series_key": item.series_key,
                    "name": item.name,
                    "points": tuple({"x": p.x, "y": p.y} for p in item.points),
                }
                for item in series_items
            ]
            return cls.time_series(
                title=title,
                unit="",
                series=normalized,
                source_ids=source_ids,
            )
        return ChartSpec(
            chart_type="BAR",
            intent="COMPARISON",
            title=title,
            x_axis=ChartAxis(axis_type="CATEGORY", label=dimension),
            y_axis=ChartAxis(axis_type="VALUE", label="数值"),
            series=tuple(series_items),
            source_ids=tuple(dict.fromkeys(str(item) for item in source_ids)),
        )

    @classmethod
    def _downsample(
        cls, points: Sequence[tuple[str, float]]
    ) -> tuple[tuple[str, float], ...]:
        limit = cls._maximum_points_per_series
        if len(points) <= limit:
            return tuple(points)
        interior = points[1:-1]
        bucket_count = max(1, (limit - 2) // 2)
        bucket_size = ceil(len(interior) / bucket_count)
        selected: list[tuple[str, float]] = [points[0]]
        for start in range(0, len(interior), bucket_size):
            bucket = list(interior[start : start + bucket_size])
            extrema = {
                min(range(len(bucket)), key=lambda i: bucket[i][1]),
                max(range(len(bucket)), key=lambda i: bucket[i][1]),
            }
            selected.extend(bucket[index] for index in sorted(extrema))
        selected.append(points[-1])
        if len(selected) > limit:
            return tuple(selected[: limit - 1] + [selected[-1]])
        return tuple(selected)

    @staticmethod
    def _display_unit(unit: str) -> tuple[str | None, float]:
        normalized = unit.strip().casefold()
        if normalized in {"byte", "bytes", "b"}:
            return "MiB", 1024 * 1024
        if normalized in {"percent", "percentage", "%"}:
            return "%", 1
        return (unit.strip() or None), 1

    @staticmethod
    def _column_kind(columns: Sequence[Mapping[str, Any]], name: str) -> str:
        item = next(
            (
                column
                for column in columns
                if str(column.get("name") or column.get("key") or "") == name
            ),
            {},
        )
        return str(item.get("logical_type") or item.get("type") or "").upper()

    @staticmethod
    def _number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    @staticmethod
    def _is_number(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float))

    @staticmethod
    def _looks_datetime(value: Any) -> bool:
        if not isinstance(value, str):
            return False
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return True

    @staticmethod
    def _format_value(value: float, unit: str | None) -> str:
        suffix = f" {unit}" if unit else ""
        if abs(value) >= 100:
            return f"{value:,.1f}{suffix}"
        return f"{value:,.2f}{suffix}"
