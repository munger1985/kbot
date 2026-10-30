"""以隔离 SQLite 实际落库验证来源复合主键，不代替生产 PG/Oracle 验收。"""

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from aiops_agent.entities import ReportSourceEntity


def assert_report_sources_persist(sources):
    engine = create_engine("sqlite:///:memory:")
    try:
        ReportSourceEntity.__table__.create(engine)
        with Session(engine) as session:
            session.add_all(sources)
            session.flush()
            assert session.scalar(select(func.count()).select_from(ReportSourceEntity)) == len(sources)
    finally:
        engine.dispose()
