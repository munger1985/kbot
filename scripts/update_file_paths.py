#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库文件路径迁移脚本

功能：
  1. 从 kbot_md_kb_files 表中读取 file_path 和 file_name
  2. URL 解码 file_name
  3. 将 file_path 替换为项目当前的文件保存路径结构: {file_storage}/{kb_id}/{batch}/{file_name}

用法：
  # 试运行（仅预览，不实际更新）
  python scripts/update_file_paths.py --dry-run

  # 实际执行
  python scripts/update_file_paths.py

  # 指定配置环境
  ENVIRONMENT=production python scripts/update_file_paths.py --dry-run
"""

import asyncio
import argparse
import os
import sys
import urllib.parse
from pathlib import Path
from typing import List, Tuple

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.database.oracle import get_session
from core.config.settings import get_app_config


# 目标表的列
TABLE_NAME = "kbot_md_kb_files"
FILE_PATH_COL = "file_path"
FILE_NAME_COL = "file_name"
KB_ID_COL = "kb_id"
BATCH_COL = "batch"


def build_new_path(file_storage: str, kb_id: int, batch: str, filename: str) -> str:
    """根据项目路径结构构造新的文件路径"""
    return str(Path(file_storage).resolve() / str(kb_id) / batch / filename)


async def query_records(session: AsyncSession) -> List[dict]:
    """查询所有文件记录"""
    query = text(f"""
        SELECT file_id, {KB_ID_COL}, {BATCH_COL}, {FILE_PATH_COL}, {FILE_NAME_COL}
        FROM {TABLE_NAME}
    """)
    result = await session.execute(query)
    rows = result.fetchall()
    records = [
        {
            "file_id": row[0],
            "kb_id": row[1],
            "batch": row[2],
            "old_path": row[3],
            "old_name": row[4],
        }
        for row in rows
    ]
    logger.info(f"查询到 {len(records)} 条记录")
    return records


def compute_updates(records: List[dict], file_storage: str) -> List[dict]:
    """计算需要更新的记录"""
    updates = []
    skipped = 0

    for rec in records:
        old_name = rec["old_name"] or ""
        old_path = rec["old_path"] or ""

        # URL 解码文件名
        decoded_name = urllib.parse.unquote(old_name)

        # 构造新路径
        new_path = build_new_path(
            file_storage=file_storage,
            kb_id=rec["kb_id"],
            batch=rec["batch"],
            filename=decoded_name,
        )

        # 判断是否需要更新
        name_changed = decoded_name != old_name
        path_changed = new_path != old_path

        if name_changed or path_changed:
            updates.append({
                "file_id": rec["file_id"],
                "kb_id": rec["kb_id"],
                "batch": rec["batch"],
                "old_path": old_path,
                "new_path": new_path,
                "old_name": old_name,
                "new_name": decoded_name,
                "name_changed": name_changed,
                "path_changed": path_changed,
            })
        else:
            skipped += 1

    logger.info(f"需要更新 {len(updates)} 条，无需变更 {skipped} 条")
    return updates


def print_dry_run(updates: List[dict], file_storage: str):
    """试运行模式：打印变更预览"""
    print(f"\n{'='*80}")
    print(f"  试运行模式 - 文件存储根目录: {file_storage}")
    print(f"{'='*80}")

    if not updates:
        print("  没有需要更新的记录。")
        return

    # 按 kb_id 分组统计
    from collections import Counter
    kb_counter = Counter(u["kb_id"] for u in updates)

    print(f"\n  总计需要更新 {len(updates)} 条记录:")
    print(f"  {'知识库ID':<12} {'记录数':<8}")
    print(f"  {'-'*20}")
    for kb_id, count in sorted(kb_counter.items()):
        print(f"  {kb_id:<12} {count:<8}")

    print(f"\n  变更详情 (前 20 条):")
    print(f"  {'file_id':<38} {'变更类型':<12} {'旧值':<50} {'新值':<50}")
    print(f"  {'-'*160}")

    for u in updates[:20]:
        if u["name_changed"] and u["path_changed"]:
            change_type = "name+path"
            old_val = u["old_name"]
            new_val = u["new_name"]
        elif u["path_changed"]:
            change_type = "path"
            old_val = u["old_path"]
            new_val = u["new_path"]
        else:
            change_type = "name"
            old_val = u["old_name"]
            new_val = u["new_name"]

        # 截断过长的值
        old_display = old_val if len(old_val) <= 48 else "..." + old_val[-45:]
        new_display = new_val if len(new_val) <= 48 else "..." + new_val[-45:]

        print(f"  {u['file_id']:<38} {change_type:<12} {old_display:<50} {new_display:<50}")

    if len(updates) > 20:
        print(f"  ... 还有 {len(updates) - 20} 条记录未显示")

    print(f"\n  {'='*80}")
    print(f"  确认执行？运行: python scripts/update_file_paths.py")
    print(f"  {'='*80}\n")


async def do_update(session: AsyncSession, updates: List[dict], batch_size: int = 500):
    """批量执行更新"""
    total = len(updates)
    updated = 0
    update_stmt = text(
        f"UPDATE {TABLE_NAME} SET {FILE_PATH_COL} = :new_path, {FILE_NAME_COL} = :new_name WHERE file_id = :file_id"
    )

    for i in range(0, total, batch_size):
        batch = updates[i : i + batch_size]
        for u in batch:
            await session.execute(
                update_stmt,
                {"new_path": u["new_path"], "new_name": u["new_name"], "file_id": u["file_id"]},
            )

        await session.commit()
        updated += len(batch)
        logger.info(f"已提交 {updated}/{total} 条")

    logger.info(f"更新完成，共更新 {updated} 条记录")


async def main():
    parser = argparse.ArgumentParser(description="迁移文件路径到项目当前存储路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="试运行模式，仅预览变更不实际写入",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="每批提交的记录数 (默认 500)",
    )
    args = parser.parse_args()

    # 加载配置
    file_storage = get_app_config().file_storage
    logger.info(f"文件存储根目录: {file_storage}")

    # 查询所有文件记录
    async with get_session() as session:
        records = await query_records(session)

    if not records:
        logger.info("数据库中没有文件记录")
        return

    # 计算需要更新的记录
    updates = compute_updates(records, file_storage)

    # 试运行模式
    if args.dry_run:
        print_dry_run(updates, file_storage)
        return

    if not updates:
        logger.info("没有需要更新的记录")
        return

    # 实际更新
    logger.info(f"开始更新 {len(updates)} 条记录...")
    async with get_session() as session:
        await do_update(session, updates, batch_size=args.batch_size)

    logger.info("全部完成！")


if __name__ == "__main__":
    asyncio.run(main())
