import ast
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "telegram_intake.py").read_text()
TREE = ast.parse(SOURCE)
FUNCTIONS = {
    node.name: node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {
        "parse_request",
        "validate_complete_candidates",
        "_build_contiguous_topic_segments",
    }
}
namespace = {
    "re": __import__("re"),
    "os": __import__("os"),
    "Any": object,
    "Literal": __import__("typing").Literal,
}
exec("YOUTUBE_RE = re.compile(r'https?://(?:(?:www\\.|m\\.)?youtube\\.com/(?:watch\\?[^\\s]*v=|shorts/|live/|embed/)|youtu\\.be/)([A-Za-z0-9_-]{6,20})', re.I)\nDRIVE_RE = re.compile(r'https?://drive\\.google\\.com/(?:file/d/|open\\?id=|uc\\?(?:[^\\s]*&)?id=)([A-Za-z0-9_-]+)', re.I)", namespace)
exec(compile(ast.Module(body=[FUNCTIONS["parse_request"]], type_ignores=[]), "telegram-functions", "exec"), namespace)
parse_request = namespace["parse_request"]
exec(compile(ast.Module(body=[FUNCTIONS["validate_complete_candidates"]], type_ignores=[]), "candidate-functions", "exec"), namespace)
validate_complete_candidates = namespace["validate_complete_candidates"]
exec(
    compile(
        ast.Module(
            body=[FUNCTIONS["_build_contiguous_topic_segments"]],
            type_ignores=[],
        ),
        "topic-functions",
        "exec",
    ),
    namespace,
)
build_topic_highlights = namespace["_build_contiguous_topic_segments"]


class TelegramParsingTests(unittest.TestCase):
    def test_plain_youtube_link_defaults_to_both_editorial_lanes(self):
        result = parse_request("https://youtu.be/abcdefghijk")
        self.assertEqual(result["source_kind"], "youtube")
        self.assertEqual(result["mode"], "both")

    def test_youtube_live_link_is_accepted(self):
        result = parse_request("https://www.youtube.com/live/abcdefghijk?si=share")
        self.assertEqual(result["source_kind"], "youtube")
        self.assertEqual(result["video_id"], "abcdefghijk")

    def test_mobile_youtube_link_is_accepted(self):
        result = parse_request("https://m.youtube.com/watch?v=abcdefghijk")
        self.assertEqual(result["video_id"], "abcdefghijk")

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

    def test_clipmaster_is_only_telegram_webhook_owner(self):
        self.assertIn('SERVICE_ROLE", ""', SOURCE)
        self.assertIn('!= "clip_master"', SOURCE)
        self.assertIn("RIPPED_SHORTS_INTERNAL_URL", SOURCE)
        self.assertIn("RIPPED_SHORTS_SHARED_SECRET", SOURCE)
        self.assertIn('/api/ripped-shorts/intake', SOURCE)

    def test_internal_intake_is_observable_and_trusts_authenticated_gateway(self):
        self.assertIn("trusted_source=True", SOURCE)
        self.assertIn("Ripped Shorts intake result", SOURCE)
        self.assertIn("Ripped Shorts job queued", SOURCE)
        self.assertIn("response.status_code = 202", SOURCE)

    def test_processing_failures_are_written_to_railway_logs(self):
        process_source = ast.get_source_segment(
            SOURCE,
            next(
                node
                for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == "_process"
            ),
        )
        self.assertIn("Ripped Shorts job starting", process_source)
        self.assertIn("logger.exception", process_source)

    def test_analysis_does_not_render_before_approval(self):
        process_source = ast.get_source_segment(SOURCE, next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_process"))
        self.assertIn('"awaiting_review"', process_source)
        self.assertNotIn("attach_clip_assets", process_source)
        render_source = ast.get_source_segment(SOURCE, next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "_render_approved"))
        self.assertIn("attach_clip_assets", render_source)

    def test_complete_candidate_validation(self):
        transcript = [
            {"start": 0, "end": 4, "text": "This is a complete sentence."},
            {"start": 4, "end": 9, "text": "This is the complete payoff."},
        ]
        payload = {"segments": [
            {"start": 0, "end": 9, "duration": 9, "transcript": "This is a complete sentence. This is the complete payoff."},
            {"start": 1, "end": 8, "duration": 7, "transcript": "is a complete sentence This is"},
        ]}
        result = validate_complete_candidates(payload, transcript)
        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(len(result["validation_rejections"]), 1)

    def test_16_9_highlights_are_selected_not_full_timeline_chunks(self):
        transcript = [
            {"start": 0, "end": 60, "duration": 60, "text": "Intro."},
            {"start": 60, "end": 180, "duration": 120, "text": "Weak material."},
            {"start": 180, "end": 360, "duration": 180, "text": "A complete strong story."},
            {"start": 360, "end": 540, "duration": 180, "text": "Unselected closing material."},
        ]
        suggestions = [
            {
                "start": 180,
                "end": 360,
                "title": "Strong Story",
                "summary": "A complete story",
                "highlight_type": "story",
                "reason": "Standalone and compelling",
                "score": 95,
            }
        ]
        result = build_topic_highlights(transcript, suggestions)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["start"], 180)
        self.assertEqual(result[0]["end"], 360)
        self.assertNotEqual(result[0]["start"], 0)
        self.assertNotEqual(result[0]["end"], 540)

    def test_16_9_waits_for_explicit_shorts_confirmation(self):
        process_source = ast.get_source_segment(
            SOURCE,
            next(
                node
                for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == "_process"
            ),
        )
        self.assertIn('row["mode"] == "topics"', process_source)
        self.assertNotIn('row["mode"] in {"topics", "both"}', process_source)
        self.assertIn("rs:shorts_confirm:", SOURCE)
        self.assertIn("shorts_confirmed_at", SOURCE)
        self.assertIn("Please approve or reject every 9:16 Short", SOURCE)

    def test_existing_clipmaster_chat_is_reused(self):
        self.assertIn('os.getenv("TELEGRAM_CHAT_ID"', SOURCE)

    def test_rapid_approvals_use_bounded_concurrent_render_pool(self):
        self.assertIn("RIPPED_SHORTS_RENDER_WORKERS", SOURCE)
        self.assertIn("RENDER_EXECUTOR.submit(_render_approved", SOURCE)
        self.assertIn('"status": "queued"', SOURCE)

    def test_candidate_decisions_are_logged_without_user_scores(self):
        self.assertIn("def _log_candidate_decision", SOURCE)
        self.assertIn('RIPPED_LOG_SHEET_TAB', SOURCE)
        self.assertIn('"Ripped Shorts"', SOURCE)
        log_function = ast.get_source_segment(
            SOURCE,
            next(
                node
                for node in TREE.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_log_candidate_decision"
            ),
        )
        self.assertNotIn('candidate.get("score"', log_function)

    def test_concurrent_review_updates_reload_latest_state(self):
        self.assertIn("latest_state", SOURCE)
        self.assertIn("render_failed", SOURCE)


if __name__ == "__main__":
    unittest.main()
