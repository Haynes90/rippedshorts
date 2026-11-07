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
        "_sentence_complete_candidate",
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
exec(compile(ast.Module(body=[FUNCTIONS["_sentence_complete_candidate"]], type_ignores=[]), "sentence-functions", "exec"), namespace)
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

    def test_normal_short_wording_still_defaults_to_both(self):
        result = parse_request("Find short highlights https://drive.google.com/file/d/abc_DEF-123/view")
        self.assertEqual(result["source_kind"], "drive")
        self.assertEqual(result["mode"], "both")

    def test_ripped_shorts_name_does_not_suppress_16_9(self):
        result = parse_request("Ripped Shorts TCB clip https://youtu.be/abcdefghijk")
        self.assertEqual(result["mode"], "both")

    def test_shorts_only_requires_explicit_override(self):
        result = parse_request("shorts only https://youtu.be/abcdefghijk")
        self.assertEqual(result["mode"], "shorts")

    def test_16_9_only_requires_explicit_override(self):
        result = parse_request("16:9 only https://youtu.be/abcdefghijk")
        self.assertEqual(result["mode"], "topics")

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

    def test_dedicated_webhook_acknowledges_before_processing(self):
        self.assertIn(
            '@router.post("/api/ripped-shorts/telegram/webhook", status_code=202)',
            SOURCE,
        )
        self.assertIn("_process_ripped_telegram_update_after_ack", SOURCE)
        self.assertIn(
            "background_tasks.add_task(_process_ripped_telegram_update_after_ack, update)",
            SOURCE,
        )
        webhook_start = SOURCE.index(
            '@router.post("/api/ripped-shorts/telegram/webhook", status_code=202)'
        )
        webhook_end = SOURCE.index(
            "_RIPPED_WEBHOOK_WATCHDOG_STARTED", webhook_start
        )
        webhook_source = SOURCE[webhook_start:webhook_end]
        self.assertNotIn("_accept_update(update, background_tasks", webhook_source)

    def test_webhook_get_is_a_health_check_not_an_update(self):
        self.assertIn(
            '@router.get("/api/ripped-shorts/telegram/webhook")',
            SOURCE,
        )
        self.assertIn('"delivery_method": "POST"', SOURCE)

    def test_shared_group_retry_accepts_bot_username(self):
        self.assertIn('(?:@rippedshortsbot)?', SOURCE)
        self.assertIn('"retry_source_missing"', SOURCE)

    def test_webhook_monitor_distinguishes_stale_errors_and_group_privacy(self):
        import telegram_intake as tg,time
        from unittest.mock import patch,Mock
        now=int(time.time()); info={'url':'https://fixture.test/webhook','last_error_message':'old failure','last_error_date':now-3600}
        def api(token,method,*args):
            return {'result':{'username':'rippedshortsbot','id':1,'can_read_all_group_messages':False}} if method=='getMe' else {'result':info}
        with patch.object(tg,'_ripped_bot_token',return_value='fixture-only'),patch.object(tg,'_ripped_webhook_url',return_value=info['url']),patch.object(tg,'_telegram_api',side_effect=api),patch.object(tg,'logger') as log,patch.dict('os.environ',{'TELEGRAM_CHAT_ID':'','TELEGRAM_GROUP_CHAT_ID':'','Telegram_Group_Chat_ID':''}):
            tg._ensure_ripped_telegram_webhook()
            self.assertTrue(any('historical' in call.args[0] for call in log.info.call_args_list))
            self.assertTrue(any('cannot receive plain group messages' in call.args[0] for call in log.warning.call_args_list))
            self.assertFalse(any('recent delivery error' in call.args[0] for call in log.warning.call_args_list))
            info['last_error_date']=now
            log.reset_mock();tg._ensure_ripped_telegram_webhook()
            self.assertTrue(any('recent delivery error' in call.args[0] for call in log.warning.call_args_list))

    def test_reliability_features_are_integrated(self):
        reliability = (
            Path(__file__).resolve().parents[1] / "workflow_reliability.py"
        ).read_text()
        self.assertIn("Workflow Jobs", reliability)
        self.assertIn("def upsert_job", reliability)
        self.assertIn("def latest_incomplete", reliability)
        self.assertIn("def classify_error", reliability)
        self.assertIn("def readiness_snapshot", reliability)
        self.assertIn("/resume(?:@rippedshortsbot)?", SOURCE)
        self.assertIn("def _update_status_card", SOURCE)
        self.assertIn("editMessageText", SOURCE)
        self.assertIn("TELEGRAM_CANDIDATES_PER_PAGE", SOURCE)
        self.assertIn("/api/ripped-shorts/readiness", SOURCE)
        self.assertIn("/api/ripped-shorts/smoke-test", SOURCE)
        self.assertIn("RIPPED_SHORTS_READINESS", SOURCE)

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
        import telegram_intake as tg
        state={'candidate_reviews':{},'topic_reviews':{}}
        _,markup=tg._short_control_payload('existing',{'mode':'both'},state)
        buttons=[b for row in markup['inline_keyboard'] for b in row]
        self.assertTrue(any(b.get('callback_data')=='rs:shorts_confirm:existing' for b in buttons))
        self.assertNotIn('shorts_confirmed_at',state)
        state['shorts_confirmed_at']='human-confirmed'
        _,after=tg._short_control_payload('existing',{'mode':'both'},state)
        self.assertFalse(any(b.get('callback_data')=='rs:shorts_confirm:existing' for row in after['inline_keyboard'] for b in row))

    def test_completed_short_selection_hands_off_to_schedule_master(self):
        self.assertIn("def _handoff_shorts_to_schedule_master", SOURCE)
        self.assertIn('f"{target}/schedule/intake"', SOURCE)
        self.assertIn("SCHEDULE_MASTER_INTERNAL_URL", SOURCE)
        self.assertIn("SCHEDULE_MASTER_URL", SOURCE)
        self.assertIn('"shorts_status": "accepted"', SOURCE)
        self.assertIn("Schedule Master received", SOURCE)
        self.assertIn('"copy_approved": True', SOURCE)
        self.assertIn('_persist_schedule_outbox(payload, "ACCEPTED")', SOURCE)
        self.assertIn('_persist_schedule_outbox(payload, "RETRY")', SOURCE)

    def test_short_review_uses_one_reusable_page_and_control_card(self):
        self.assertIn("def _refresh_short_review_message", SOURCE)
        self.assertIn('"short_review_message_id"', SOURCE)
        self.assertIn('"short_control_message_id"', SOURCE)
        self.assertIn("rs:shorts_page:", SOURCE)
        self.assertIn("Rendering continues in the background", SOURCE)

    def test_edited_caption_stays_visible_as_final_reviewed_copy(self):
        from acceptance_fixtures import copy_review_fixture
        with copy_review_fixture() as f:
            f.reply();state=f.read()
            text,_=f.tg._copy_review_payload('first',state['copy_drafts'],state['copy_review_index'])
            self.assertIn('EDITED FINAL COPY',text)
            self.assertIn('My exact edited wording\n\nSecond paragraph.',text)
            self.assertEqual(state['copy_review_index'],0)

    def test_copy_reply_uses_durable_prompt_session_before_group_filter(self):
        self.assertIn("CREATE TABLE IF NOT EXISTS telegram_copy_inputs", SOURCE)
        self.assertIn("WHERE prompt_message_id=? AND chat_id=? AND user_id=?", SOURCE)
        self.assertIn("INSERT OR REPLACE INTO telegram_copy_inputs", SOURCE)
        self.assertIn("COPY_REPLY_APPLIED", SOURCE)

    def test_copy_reply_accepts_review_card_or_single_pending_edit(self):
        self.assertIn("candidate_matches = []", SOURCE)
        self.assertIn("reply_to_message_id in {prompt_id, review_id}", SOURCE)
        self.assertIn("len(candidate_matches) == 1", SOURCE)

    def test_copy_edit_is_checkpointed_before_success_confirmation(self):
        self.assertIn("def _checkpoint_copy_edit(", SOURCE)
        self.assertIn("COPY_EDIT_CHECKPOINTED", SOURCE)
        self.assertIn("for checkpoint_attempt in range(1, 4)", SOURCE)
        self.assertIn('"confirmation": "saved_and_checkpointed"', SOURCE)
        self.assertIn("durable copy sheet is ready for Schedule Master", SOURCE)

    def test_copy_checkpoint_failure_keeps_same_edit_route_open(self):
        self.assertIn('latest_state["awaiting_copy_input"] = waiting', SOURCE)
        self.assertIn('"status": "copy_checkpoint_failed"', SOURCE)
        self.assertIn("You are still on this same caption", SOURCE)

    def test_schedule_handoff_prefers_durable_caption_checkpoint(self):
        import telegram_intake as tg
        from unittest.mock import patch
        rows=[{'request_id':'r','asset_id':'a','decision':'EDITED','recorded_at':'2026-10-31','final_caption':'Exact owner wording','ai_caption':'AI draft'}, {'request_id':'other','asset_id':'a','decision':'EDITED','final_caption':'Wrong request'}]
        with patch.object(tg,'get_rows',return_value=rows):
            resolved=tg._caption_learning_checkpoint_map('r')
        self.assertEqual(resolved['a']['social_caption'],'Exact owner wording')

    def test_copy_reply_always_confirms_saved_edit(self):
        self.assertIn("confirmation_done = False", SOURCE)
        self.assertIn("reply_to_message_id", SOURCE)
        self.assertIn('"confirmation": "saved_and_checkpointed"', SOURCE)
        self.assertIn("last_copy_edit_saved_at", SOURCE)

    def test_copy_reply_is_bound_to_exact_prompt_and_preserves_route(self):
        from acceptance_fixtures import copy_review_fixture
        with copy_review_fixture() as f:
            result=f.reply('100')
            self.assertEqual(result['status'],'copy_updated')
            self.assertEqual(f.read()['copy_drafts'][0]['social_caption'],'My exact edited wording\n\nSecond paragraph.')
            self.assertEqual(f.read('second')['copy_drafts'][0]['social_caption'],'Original second')
            self.assertEqual(f.read()['route_source'],'owner')
            self.assertEqual(f.checkpoints[0][1]['copy_drafts'][0]['copy_status'],'edited')

    def test_copy_reply_updates_existing_prompt_instead_of_sending_status_spam(self):
        from acceptance_fixtures import copy_review_fixture
        with copy_review_fixture() as f:
            f.reply()
            methods=[call.args[0] for call in f.telegram.call_args_list]
            self.assertNotIn('sendMessage',methods)
            self.assertGreaterEqual(methods.count('editMessageText'),2)
            self.assertTrue(any(call.args[1].get('message_id')=='100' for call in f.telegram.call_args_list))

    def test_copy_review_uses_one_reusable_message(self):
        self.assertIn("def _refresh_copy_review", SOURCE)
        self.assertIn('"copy_review_message_id"', SOURCE)
        self.assertIn("rs:copy_page:", SOURCE)
        self.assertIn("✅ Finish & Schedule", SOURCE)

    def test_16_9_runs_semantic_fallback_when_long_pass_is_empty(self):
        self.assertIn("semantic_fallback: bool = False", SOURCE)
        self.assertIn('TOPIC_FALLBACK_MIN_SECONDS", "180"', SOURCE)
        self.assertIn("questions with their full answers", SOURCE)
        self.assertIn("follow-up exchanges that stay on one subject", SOURCE)
        self.assertIn("topic discussions", SOURCE)
        self.assertIn("stories with setup and payoff", SOURCE)
        self.assertIn("This is a minimum only, never a target or maximum", SOURCE)
        self.assertIn("5, 8, 12, or more minutes", SOURCE)
        self.assertIn("semantic_fallback=True", SOURCE)
        self.assertIn('selection_mode = "semantic_fallback"', SOURCE)

    def test_16_9_builder_really_uses_fallback_minimum_at_runtime(self):
        transcript = [
            {"start": 0, "end": 50, "text": "Question and setup."},
            {"start": 50, "end": 100, "text": "Complete answer and payoff."},
        ]
        suggestions = [{
            "start": 0,
            "end": 100,
            "title": "Q&A",
            "summary": "A complete question and answer.",
            "highlight_type": "question_answer",
            "reason": "Standalone exchange",
            "score": 90,
        }]
        self.assertEqual(build_topic_highlights(transcript, suggestions), [])
        recovered = build_topic_highlights(
            transcript,
            suggestions,
            minimum_seconds=90,
        )
        self.assertEqual(recovered, [])  # A 100-second fallback is below the established minimum.

    def test_16_9_builder_keeps_three_minute_floor_in_fallback(self):
        self.assertIn("minimum_seconds: float | None = None", SOURCE)
        self.assertIn("max(180.0, float(minimum_seconds))", SOURCE)
        self.assertIn('TOPIC_SEGMENT_MIN_SECONDS", "180"', SOURCE)

    def test_topic_review_uses_one_reusable_message(self):
        self.assertIn("def _refresh_topic_review", SOURCE)
        self.assertIn('"topic_review_message_id"', SOURCE)
        self.assertIn("rs:topic_page:", SOURCE)

    def test_schedule_now_waits_for_both_render_lanes(self):
        self.assertIn("rs:schedule_now:", SOURCE)
        self.assertIn('"schedule_requested_at"', SOURCE)
        notify_source = ast.get_source_segment(
            SOURCE,
            next(
                node
                for node in TREE.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_notify_render_queue_complete"
            ),
        )
        self.assertIn('state.get("candidate_reviews")', notify_source)
        self.assertIn('state.get("topic_reviews")', notify_source)
        self.assertIn('status in {"queued", "rendering"}', notify_source)
        self.assertIn("if active:", notify_source)

    def test_schedule_handoff_includes_short_and_highlight_assets(self):
        handoff_source = ast.get_source_segment(
            SOURCE,
            next(
                node
                for node in TREE.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_handoff_shorts_to_schedule_master"
            ),
        )
        self.assertIn('"asset_type": "9:16_SHORT"', handoff_source)
        self.assertIn('"asset_type": "16:9_HIGHLIGHT"', handoff_source)
        self.assertIn('review.get("segment_url")', handoff_source)

    def test_use_existing_rebuilds_assets_from_drive(self):
        self.assertIn("list_existing_rendered_assets", SOURCE)
        self.assertIn('"recovered_renders_from_drive": True', SOURCE)
        self.assertIn('"status": "rendered"', SOURCE)
        self.assertIn('"segment_url": asset["segment_url"]', SOURCE)

    def test_schedule_handoff_generates_copy(self):
        self.assertIn("def _generate_schedule_copy", SOURCE)
        self.assertIn('"social_caption"', SOURCE)
        self.assertIn('"video_title"', SOURCE)
        self.assertIn('"video_description"', SOURCE)
        self.assertIn("BRAND RULE", SOURCE)

    def test_short_caption_keeps_hashtags_separate_and_clean(self):
        from acceptance_fixtures import copy_review_fixture
        with copy_review_fixture() as f:
            f.reply();draft=f.read()['copy_drafts'][0]
            self.assertEqual(draft['hashtags'],'#Plants')
            self.assertNotIn('#Plants',draft['social_caption'])
            self.assertEqual(f.tg._normalize_hashtags('["#Plants", "#Seeds"]'),'#Plants #Seeds')
            text,_=f.tg._copy_review_payload('first',[draft],0)
            self.assertIn(draft['social_caption'],text)
            self.assertNotIn('Hook:',text)

    def test_schedule_now_requires_copy_review(self):
        self.assertIn('"copy_review_requested_at"', SOURCE)
        self.assertIn("rs:copy_edit_caption:", SOURCE)
        self.assertIn("rs:copy_edit_title:", SOURCE)
        self.assertIn("rs:copy_edit_description:", SOURCE)
        self.assertIn("rs:copy_finish:", SOURCE)

    def test_copy_edits_are_saved_as_learning_examples(self):
        self.assertIn("Caption Learning", SOURCE)
        self.assertIn('"EDITED" if edited else "ACCEPTED"', SOURCE)
        self.assertIn('"user_edited"] = True', SOURCE)
        self.assertIn("PAST AI-TO-FINAL EXAMPLES", SOURCE)

    def test_existing_clipmaster_chat_is_reused(self):
        self.assertIn('os.getenv("TELEGRAM_CHAT_ID"', SOURCE)

    def test_rapid_approvals_use_bounded_concurrent_render_pool(self):
        self.assertIn("RIPPED_SHORTS_RENDER_WORKERS", SOURCE)
        self.assertIn("RENDER_EXECUTOR.submit(renderer, request_id, index, chat_id, render_ticket=ticket)", SOURCE)
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

    def test_status_card_merges_into_latest_state(self):
        status_source = ast.get_source_segment(
            SOURCE,
            next(
                node
                for node in TREE.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_update_status_card"
            ),
        )
        self.assertIn("SELECT status, state_json", status_source)
        self.assertIn('latest_state["status_message_id"]', status_source)
        self.assertNotIn("(json.dumps(state), now(), request_id)", status_source)


if __name__ == "__main__":
    unittest.main()
