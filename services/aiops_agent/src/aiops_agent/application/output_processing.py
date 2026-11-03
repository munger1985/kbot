"""输出任务按业务类型分派，报告与案例仍共用同一个 Worker。"""

from aiops_agent.application.errors import AIOpsApplicationError


class OutputJobProcessor:
    def __init__(self, *, case_processor, report_processor):
        self.case_processor = case_processor
        self.report_processor = report_processor

    async def advance(self, execution, *, authority, checkpoint, record_receipt):
        processor = {"CASE_EXTRACTION": self.case_processor,
            "REPORT_GENERATION": self.report_processor}.get(execution.job_type)
        if processor is None:
            raise AIOpsApplicationError(code="AIOPS_OUTPUT_JOB_KIND_UNSUPPORTED",
                message="该输出任务类型未配置处理器", status_code=409)
        return await processor.advance(execution, authority=authority,
            checkpoint=checkpoint, record_receipt=record_receipt)
