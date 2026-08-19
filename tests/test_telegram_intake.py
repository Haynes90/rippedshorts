import ast
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "telegram_intake.py").read_text()
TREE = ast.parse(SOURCE)
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name in {"parse_request", "validate_complete_candidates"}}
namespace = {"re": __import__("re"), "Any": object, "Literal": __import__("typing").Literal}
exec("YOUTUBE_RE = re.compile(r'https?://(?:www\\.)?(?:youtube\\.com/(?:watch\\?[^\\s]*v=|shorts/)|youtu\\.be/)([A-Za-z0-9_-]{6,20})', re.I)\nDRIVE_RE = re.compile(r'https?://drive\\.google\\.com/(?:file/d/|open\\?id=|uc\\?(?:[^\\s]*&)?id=)([A-Za-z0-9_-]+)', re.I)", namespace)
exec(compile(ast.Module(body=[FUNCTIONS["parse_request"]], type_ignores=[]), "telegram-functions", "exec"), namespace)
parse_request = namespace["parse_request"]
exec(compile(ast.Module(body=[FUNCTIONS["validate_complete_candidates"]], type_ignores=[]), "candidate-functions", "exec"), namespace)
validate_complete_candidates = namespace["validate_complete_candidates"]


class TelegramParsingTests(unittest.TestCase):
    def test_plain_youtube_link_defaults_to_shorts(self):
        result = parse_request("https://youtu.be/abcdefghijk")
        self.assertEqual(result["source_kind"], "youtube")
        self.assertEqual(result["mode"], "shorts")

    def test_drive_short_request(self):
        result = parse_request("Find short highlights https://drive.google.com/file/d/abc_DEF-123/view")
        self.assertEqual(result["source_kind"], "drive")
        self.assertEqual(result["mode"], "shorts")

    def test_drive_video_and_transcript_are_retained(self):
        result = parse_request("Process https://drive.google.com/file/d/video123/view transcript https://drive.google.com/file/d/transcript456/view")
        self.assertEqual(result["drive_ids"], ["video123", "transcript456"])

    def test_missing_link_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "YouTube link"):
            parse_request("find clips")

    def test_security_and_idempotency_contracts_exist(self):
        self.assertIn("TELEGRAM_ALLOWED_CHAT_IDS", SOURCE)
        self.assertIn("TELEGRAM_WEBHOOK_SECRET", SOURCE)
        self.assertIn("update_id TEXT UNIQUE", SOURCE)

    def test_complete_candidate_must_use_transcript_boundaries(self):
        transcript = [
            {"start": 0, "end": 4, "text": "This is a complete sentence."},
            {"start": 4, "end": 9, "text": "This is the complete payoff."},
        ]
        payload = {"segments": [
            {"start": 0, "end": 9, "duration": 9, "transcript": "This is a complete sentence. This is the complete payoff."},
            {"start": 1, "end": 8, "duration": 7, "transcript": "is a complete sentence This is the complete"},
        ]}
        result = validate_complete_candidates(payload, transcript)
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(len(result["validation_rejections"]), 1)

    def test_existing_clipmaster_chat_variable_is_reused(self):
        self.assertIn('os.getenv("TELEGRAM_CHAT_ID"', SOURCE)


if __name__ == "__main__":
    unittest.main()
