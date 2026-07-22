"""Lease transitions are pure and testable without an Oracle worker queue."""
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from knowledge_core.domain.parse_tasks import ParseLeaseError, ParseTaskClaim, claim_job, verify_lease


class ParseTaskRulesTest(unittest.TestCase):
    def _job(self):
        return SimpleNamespace(job_status="PENDING", available_at=datetime(2026, 1, 1, tzinfo=timezone.utc), attempt_count=0, row_version=1, started_at=None, input_fingerprint="hash")

    def test_claim_and_verify_current_lease(self):
        now = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
        job = self._job()
        lease_until = claim_job(job, ParseTaskClaim("parser-a", 1, 60), now)
        self.assertEqual("RUNNING", job.job_status)
        self.assertEqual(1, job.attempt_count)
        self.assertEqual(now + timedelta(seconds=60), lease_until)
        verify_lease(job, worker_id="parser-a", input_fingerprint="hash", now=now + timedelta(seconds=1))

    def test_rejects_expired_or_wrong_input_callback(self):
        now = datetime(2026, 1, 1, 1, tzinfo=timezone.utc)
        job = self._job()
        claim_job(job, ParseTaskClaim("parser-a", 1, 30), now)
        with self.assertRaisesRegex(ParseLeaseError, "JOB_STALE"):
            verify_lease(job, worker_id="parser-a", input_fingerprint="old", now=now)
        with self.assertRaisesRegex(ParseLeaseError, "JOB_LEASE_INVALID"):
            verify_lease(job, worker_id="parser-a", input_fingerprint="hash", now=now + timedelta(seconds=31))


if __name__ == "__main__":
    unittest.main()
