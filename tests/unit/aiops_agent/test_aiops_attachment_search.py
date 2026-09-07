"""用户附件受控检索的安全边界测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aiops_agent.adapters.conversation_uploads import LocalConversationUploadStore
from aiops_agent.application.conversation_inputs import ConversationInputResolver
from aiops_agent.application.investigation.query_freezing import (
    prepare_attachment_searches,
)
from aiops_agent.application.investigation.reasoner import (
    InvestigationPlanValidationError,
)
from aiops_agent.workers.attachment_handlers import AttachmentSearchHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from aiops_agent.playbooks import PlaybookRegistry
from aiops_agent.tools import InvestigationTaskCompiler
from platform_core.contracts.aiops.playbooks import DbaPlaybookPlan
from platform_core.contracts.aiops import (
    InvestigationAction,
    InvestigationPlan,
    InvestigationPlanningOutput,
    InputMaterial,
    MaterialKind,
    TaskFrame,
    TurnInputEnvelope,
)


async def _chunks(value: bytes):
    yield value


def _investigation(action: InvestigationAction):
    return InvestigationPlanningOutput(
        input_envelope=TurnInputEnvelope(
            materials=(
                InputMaterial(
                    item_no=1,
                    material_kind=MaterialKind.DATABASE_LOG,
                    summary="用户上传的追踪日志",
                    confidence=1,
                    contains_user_evidence=True,
                ),
            ),
            explicit_question="检查上传日志",
        ),
        task_frame=TaskFrame(
            objectives=("DIAGNOSE",),
            problem_statement="检查上传日志",
            success_criteria=("给出证据",),
        ),
        plan=InvestigationPlan(revision_no=1, actions=(action,)),
    )


class AttachmentSearchTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = LocalConversationUploadStore(
            Path(self.temporary.name), max_bytes=4096
        )
        stored = await self.store.store(
            domain_id=7,
            actor_id="dba-1",
            file_name="prod_ora_123.trc",
            media_type="text/plain",
            chunks=_chunks(
                b"begin\nORA-00600: internal error\nincident 123\nend\n"
            ),
        )
        resolver = ConversationInputResolver(
            upload_store=self.store, max_extracted_chars=10
        )
        _, uploads = await resolver.resolve(
            domain_id=7,
            actor_id="dba-1",
            content=({"upload_id": stored.upload_id},),
            image_capabilities={},
        )
        self.upload = uploads[0]

    async def asyncTearDown(self):
        self.temporary.cleanup()

    async def test_search_uses_literal_term_and_returns_line_context(self):
        action = InvestigationAction(
            action_id="a1",
            tool_id="artifact.search",
            question="定位 ORA-00600",
            expected_evidence_kind="USER_FILE",
            measurement_semantics="NOT_APPLICABLE",
            input={
                "upload_id": self.upload.upload_id,
                "terms": ["ORA-00600"],
                "context_lines": 1,
            },
        )
        frozen, searches = prepare_attachment_searches(
            _investigation(action), (self.upload,)
        )
        self.assertEqual("ORA-00600", frozen.plan.actions[0].input["terms"][0])
        result = await AttachmentSearchHandler(upload_store=self.store).execute(
            TaskExecutionContext(
                run_id="run-1", task_id="task-1", task_key="attachment:a1",
                target_id="target-1", agent_id="agent-1", trigger_type="CHAT",
                trace_id="trace-1", attempt=1, deadline_at=None,
                plan_snapshot={"attachment_search": list(searches)},
                policy_snapshot={}, input_artifacts=(),
            )
        )
        self.assertEqual("ATTACHMENT_EVIDENCE_SET.v1", result.schema_version)
        self.assertIn("2: ORA-00600: internal error", result.matches[0].text)
        self.assertIn("3: incident 123", result.matches[0].text)

    async def test_rejects_foreign_upload_and_control_characters(self):
        foreign = InvestigationAction(
            action_id="a1", tool_id="artifact.search", question="x",
            expected_evidence_kind="USER_FILE",
            measurement_semantics="NOT_APPLICABLE",
            input={"upload_id": "foreign", "terms": ["ORA"], "context_lines": 1},
        )
        with self.assertRaises(InvestigationPlanValidationError):
            prepare_attachment_searches(_investigation(foreign), (self.upload,))
        unsafe = foreign.model_copy(update={
            "input": {
                "upload_id": self.upload.upload_id,
                "terms": ["ORA\n--glob=*"],
                "context_lines": 1,
            }
        })
        with self.assertRaises(InvestigationPlanValidationError):
            prepare_attachment_searches(_investigation(unsafe), (self.upload,))

    async def test_compiler_creates_dedicated_attachment_evidence_task(self):
        action = InvestigationAction(
            action_id="a1", tool_id="artifact.search", question="x",
            expected_evidence_kind="USER_FILE",
            measurement_semantics="NOT_APPLICABLE",
            input={
                "upload_id": self.upload.upload_id,
                "terms": ["ORA-00600"],
                "context_lines": 1,
            },
        )
        registry = PlaybookRegistry.load()
        compiled = InvestigationTaskCompiler(registry).compile(
            DbaPlaybookPlan(catalog_hash=registry.catalog_hash, items=()),
            investigation_actions=(action,),
        )
        self.assertEqual(("attachment:a1",), compiled.attachment_search_task_keys)
        task = next(item for item in compiled.tasks if item.task_key == "attachment:a1")
        self.assertEqual("evidence.attachment-search", task.handler_id)
        self.assertIn("attachment:a1", next(
            item for item in compiled.tasks if item.task_key == "evidence:assess"
        ).depends_on)

    async def test_assessment_uses_attachment_evidence_as_user_provided(self):
        artifact = {
            "artifact_id": "artifact-1",
            "schema_version": "ATTACHMENT_EVIDENCE_SET.v1",
            "payload": {
                "schema_version": "ATTACHMENT_EVIDENCE_SET.v1",
                "upload_id": self.upload.upload_id,
                "file_name": self.upload.file_name,
                "content_hash": self.upload.searchable_content_hash,
                "query_terms": ["ORA-00600"],
                "matches": [{
                    "line_start": 2,
                    "line_end": 2,
                    "text": "2: ORA-00600: internal error",
                    "match_terms": ["ORA-00600"],
                }],
                "truncated": False,
                "query_fingerprint": "a" * 64,
            },
        }
        assessment = await DbaEvidenceAssessmentHandler().execute(
            TaskExecutionContext(
                run_id="run-1", task_id="task-1", task_key="evidence:assess",
                target_id="target-1", agent_id="agent-1", trigger_type="CHAT",
                trace_id="trace-1", attempt=1, deadline_at=None,
                plan_snapshot={"answer_context": {}}, policy_snapshot={},
                input_artifacts=(artifact,),
            )
        )
        self.assertEqual(1, len(assessment.evidence))
        self.assertEqual("USER_PROVIDED", assessment.evidence[0].trust_level)
        self.assertEqual("artifact.search", assessment.evidence[0].tool_id)


if __name__ == "__main__":
    unittest.main()
