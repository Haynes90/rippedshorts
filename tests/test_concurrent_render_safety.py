from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_clip_output_names_use_youtube_id_and_candidate_number():
    source = (ROOT / "main.py").read_text()
    telegram_source = (ROOT / "telegram_intake.py").read_text()
    assert 'clip_name = f"{video_id}_Clip_{candidate_number}.mp4"' in source
    assert 'f".render-{uuid.uuid4().hex}-{clip_name}"' in source
    assert 'payload_candidate["candidate_number"] = int(' in telegram_source


def test_render_failure_includes_ffmpeg_diagnostics():
    source = (ROOT / "main.py").read_text()
    assert "FFmpeg render failed with code" in source
    assert "(completed.stderr or completed.stdout)[-3000:]" in source


def test_selector_uses_quality_without_quota_and_collects_learning_metadata():
    from main import openai_clip_prompt
    prompt = openai_clip_prompt([], None)
    assert "Do not force a clip count" in prompt
    assert '"content_type"' in prompt
    assert '"keywords"' in prompt


def test_approvals_use_bounded_pool_and_are_idempotent():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "RIPPED_SHORTS_RENDER_WORKERS" in source
    assert "RENDER_EXECUTOR.submit(renderer, request_id, index, chat_id, render_ticket=ticket)" in source
    assert 'existing_status in {"queued", "rendering", "rendered", "render_failed"}' in source
    assert '"status": "rendering"' in source


def test_learning_log_uses_decisions_without_user_score():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "RIPPED_LOG_SHEET_TAB" in source
    assert '"approved"' in source
    assert '"rejected"' in source
    assert 'candidate.get("score")' not in source


def test_telegram_updates_workspace_instead_of_spamming_render_messages():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "def _refresh_short_review_message" in source
    assert "def _send_short_confirmation" in source
    assert "def _refresh_topic_review" in source
    assert "Current Ripped Shorts render queue complete" in source
    assert "drive.google.com/drive/folders/" in source


def test_telegram_reports_dynamic_aggregate_progress():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "Progress: {processed}/{total} processed ({percent}%)" in source
    assert "{rendering} rendering | {queued} queued" in source
    assert "{rendered} rendered | {failed} failed" in source


def test_restart_recovers_only_approved_unfinished_sheet_clips():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "def _approved_clip_history_from_sheet" in source
    assert 'str(padded[2]).strip() != video_id' in source
    assert 'str(padded[15]).strip().lower() != "approved"' in source
    assert 'is_rendered = render_status == "rendered" and bool(clip_url)' in source
    assert "recovered_approvals_from_sheet" in source
    assert "Rendering only those shorts now" in source


def test_saved_source_analysis_does_not_repeat_model_selection():
    from acceptance_fixtures import intelligence_db
    import production_intelligence as pi
    from unittest.mock import Mock
    with intelligence_db():
        analyze=Mock(return_value={'segments':[{'start':10,'end':30,'asset_id':'existing'}]})
        first=pi.cached_analysis('source:immutable-transcript:9:16',analyze)
        second=pi.cached_analysis('source:immutable-transcript:9:16',analyze)
        assert first==second and second['segments'][0]['asset_id']=='existing'
        assert analyze.call_count==1


def test_reused_video_render_creates_output_directory():
    source = (ROOT / "main.py").read_text()
    assert 'workdir.mkdir(parents=True, exist_ok=True)' in source


def test_selector_uses_quality_and_allows_distinct_overlap():
    prompt_source = (ROOT / "main.py").read_text()
    intake_source = (ROOT / "telegram_intake.py").read_text()
    selector_source = (ROOT / "source_ingestion.py").read_text()
    assert "Do not force a clip count" in prompt_source
    assert "return 20" not in prompt_source
    assert "Clips MAY overlap in time" in prompt_source
    assert "allow_overlap=True" in intake_source
    assert "normalized_text" in selector_source


def test_quality_selector_does_not_pad_to_a_review_quota():
    # Approved quality-over-quota policy replaces forced replenishment to ten.
    from main import openai_clip_prompt
    from source_ingestion import select_non_overlapping
    prompt=openai_clip_prompt([],None)
    assert 'Do not force a clip count' in prompt and 'full transcript' in prompt.lower()
    clips=[{'start':0,'end':20,'score':95,'transcript':'One strong complete thought.'}]
    assert select_non_overlapping(clips,limit=20)==clips
    assert select_non_overlapping([],limit=20)==[]


def test_telegram_uses_short_not_candidate_label():
    import telegram_intake as tg
    state={'result':{'segments':[{'start':10,'end':30,'duration':20,'candidate_number':9,'transcript':'Approved full thought.'}]},'candidate_reviews':{}}
    text,markup=tg._short_review_page_payload('existing',state,1)
    assert 'SHORT 9' in text and 'Approved full thought.' in text
    buttons=[b for row in markup['inline_keyboard'] for b in row]
    assert any(b.get('callback_data')=='rs:approve:existing:0' for b in buttons)
    assert state['candidate_reviews']=={}


def test_opencv_speaker_tracking_and_center_fallback():
    source = (ROOT / "main.py").read_text()
    docker = (ROOT / "Dockerfile").read_text()
    assert "opencv-data" in docker
    assert "def _estimate_speaker_track" in source
    assert "motion_score * 3.0" in source
    assert "smooths noisy analysis samples only" in source
    assert "target_center = previous_center" in source
    assert "SPEAKER_SWITCH_CONFIRMATIONS" in source
    assert "def _stepped_crop_expression" in source


def test_repeat_video_prompts_for_rerip_choice():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "awaiting_rerip_choice" in source
    assert "Do you want to re-rip the Shorts" in source
    assert "rs:rerip:{request_id}" in source
    assert "rs:reuse:{request_id}" in source


def test_short_selection_uses_sheet_approval_learning():
    from acceptance_fixtures import learning_rows
    with learning_rows() as (tg,service):
        prompt=tg._short_learning_prompt('source')
    assert 'Approved complete useful explanation.' in prompt
    assert 'Rejected weak setup.' in prompt
    assert 'complete standalone thought' in prompt and 'payoff' in prompt
    assert service.spreadsheets.return_value.values.return_value.get.call_args.kwargs['range'].endswith('!A:U')


def test_rerip_is_fresh_but_may_reselect_strong_approvals():
    source = (ROOT / "telegram_intake.py").read_text()
    assert "needed = 20 if force_rerip" in source
    assert "old approvals are learning" in source
    assert "Previously approved Shorts may be selected again" in source
    assert 'item.get("decision") == "rejected"' in source


def test_opus_style_framing_holds_then_snaps_at_natural_boundaries():
    source = (ROOT / "main.py").read_text()
    assert "def _detect_audio_pause_boundaries" in source
    assert "silencedetect=noise=-35dB:d=0.18" in source
    assert "def _plan_framing_sections" in source
    assert 'SPEAKER_REFRAME_SECONDS", "4.0"' in source
    assert 'SPEAKER_REFRAME_MIN_SECONDS", "3.0"' in source
    assert 'SPEAKER_REFRAME_MAX_SECONDS", "7.0"' in source
    assert "from statistics import median" in source
    assert "SPEAKER_REFRAME_THRESHOLD" in source
    assert "interpolated =" not in source
    assert 'expression = (' in source
    assert 'f"if(lt(t,{next_timestamp:.3f})' in source


def test_tracker_holds_on_loss_and_requires_consistent_speaker_switch():
    source = (ROOT / "main.py").read_text()
    assert "if lost_samples <= 2:" in source
    assert "pending_switch_count >= switch_confirmations" in source
    assert 'SPEAKER_SWITCH_DISTANCE", "0.18"' in source
    assert 'SPEAKER_SWITCH_CONFIRMATIONS", "3"' in source


def test_plain_link_starts_shorts_and_horizontal_topic_lanes():
    import telegram_intake as tg
    import production_intelligence as pi
    from threading import Event
    assert tg.parse_request('https://youtu.be/abcdefghijk')['mode']=='both'
    a,b=Event(),Event()
    def short():
        a.set();assert b.wait(2);return {'shorts':[1,2]}
    def highlight():
        b.set();assert a.wait(2);return {'highlights':[3]}
    futures=pi.parallel_analyses(short,highlight)
    assert futures['9:16'].result(timeout=3)=={'shorts':[1,2]}
    assert futures['16:9'].result(timeout=3)=={'highlights':[3]}


def test_semantic_highlights_leave_weak_gaps_and_keep_minimum():
    # Semantic selections may leave weak intervals uncovered; no arbitrary full-source slices.
    import telegram_intake as tg
    transcript=[{'start':0,'end':240,'text':'Complete first explanation.'},{'start':240,'end':400,'text':'Weak interlude.'},{'start':400,'end':650,'text':'Complete second discussion.'}]
    selected=tg._build_contiguous_topic_segments(transcript,[{'start':0,'end':240,'score':90},{'start':400,'end':650,'score':95},{'start':240,'end':400,'score':99}])
    assert [(x['start'],x['end']) for x in selected]==[(0,240),(400,650)]
    assert all(x['duration']>=180 for x in selected)


def test_horizontal_segments_have_separate_review_and_renderer():
    intake = (ROOT / "telegram_intake.py").read_text()
    main = (ROOT / "main.py").read_text()
    assert "Approve 16:9" in intake
    assert "rs:topic_approve:" in intake
    assert "def _render_topic_approved" in intake
    assert "def create_topic_segment_file" in main
    assert 'f"{video_id}_Segment_{segment_number}.mp4"' in main
    assert "attach_topic_segment_asset" in main


def test_shorts_expand_to_complete_sentence_boundaries():
    import telegram_intake as tg
    segments=[{'start':0,'end':15,'text':'A setup that continues'},{'start':15,'end':30,'text':'and ends with the full payoff.'}]
    result=tg.validate_complete_candidates({'segments':[{'start':15,'end':30,'transcript':'and ends with the full payoff.'}]},segments)
    assert len(result['segments'])==1
    assert result['segments'][0]['start']==0 and result['segments'][0]['end']==30
    assert result['segments'][0]['transcript']=='A setup that continues and ends with the full payoff.'


def test_topic_coherence_precedes_eight_minute_reference():
    import telegram_intake as tg
    segments=[{'start':0,'end':900,'text':'A complete long explanation.'}]
    result=tg._build_contiguous_topic_segments(segments,[{'start':0,'end':900,'score':95,'title':'Full explanation'}])
    assert len(result)==1 and result[0]['duration']==900
    assert result[0]['transcript']==segments[0]['text']
    assert tg._build_contiguous_topic_segments([{'start':0,'end':100,'text':'Too short.'}],[{'start':0,'end':100}],minimum_seconds=90)==[]
