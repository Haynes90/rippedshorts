import ast
from pathlib import Path
import unittest


SOURCE = (Path(__file__).parent / "audio_master_handoff.py").read_text()
TREE = ast.parse(SOURCE)
FUNCTIONS = {
    node.name: node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name in {"normalize_sermon"}
}
namespace = {}
exec(compile(ast.Module(body=list(FUNCTIONS.values()), type_ignores=[]), "handoff-functions", "exec"), namespace)
normalize_sermon = namespace["normalize_sermon"]


class TimelineTests(unittest.TestCase):
    def test_original_service_timeline_is_filtered_and_rebased(self):
        segments = [
            {"start": 90, "end": 99, "text": "before"},
            {"start": 100, "end": 110, "text": "opening"},
            {"start": 125, "end": 140, "text": "message"},
            {"start": 171, "end": 180, "text": "after"},
        ]
        manifest = {"sermon_start_seconds": 100, "sermon_end_seconds": 170, "transcript_timeline": "original_service"}
        result = normalize_sermon(segments, manifest)
        self.assertEqual([item["text"] for item in result], ["opening", "message"])
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[1]["start"], 25)

    def test_empty_sermon_range_is_retryable_error(self):
        with self.assertRaisesRegex(RuntimeError, "no segments"):
            normalize_sermon(
                [{"start": 10, "end": 20, "text": "outside"}],
                {"sermon_start_seconds": 100, "sermon_end_seconds": 200, "transcript_timeline": "original_service"},
            )

    def test_contract_has_idempotent_primary_key_and_duplicate_response(self):
        self.assertIn("source_job_id TEXT PRIMARY KEY", SOURCE)
        self.assertIn('{"status": "duplicate"', SOURCE)


if __name__ == "__main__":
    unittest.main()
