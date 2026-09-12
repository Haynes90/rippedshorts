import sqlite3
import tempfile
import unittest
from pathlib import Path

import durable_jobs


class DurableJobsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "jobs.db"

    def tearDown(self):
        self.temp.cleanup()

    def test_only_one_live_owner_can_claim(self):
        self.assertTrue(durable_jobs.claim(self.path, "job-1", "process"))
        original = durable_jobs.OWNER
        durable_jobs.OWNER = "second-worker"
        try:
            self.assertFalse(durable_jobs.claim(self.path, "job-1", "process"))
        finally:
            durable_jobs.OWNER = original

    def test_released_processing_job_can_resume(self):
        self.assertTrue(durable_jobs.claim(self.path, "job-2", "process"))
        durable_jobs.finish(self.path, "job-2", "process")
        self.assertTrue(durable_jobs.claim(self.path, "job-2", "process"))

    def test_completed_handoff_is_exactly_once(self):
        self.assertTrue(durable_jobs.claim(self.path, "job-3", "schedule_handoff"))
        durable_jobs.finish(
            self.path, "job-3", "schedule_handoff", complete=True
        )
        self.assertFalse(
            durable_jobs.claim(self.path, "job-3", "schedule_handoff")
        )

    def test_schema_includes_outbox(self):
        with sqlite3.connect(self.path) as db:
            durable_jobs.ensure_schema(db)
            names = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        self.assertIn("durable_job_leases", names)
        self.assertIn("durable_outbox", names)


if __name__ == "__main__":
    unittest.main()
