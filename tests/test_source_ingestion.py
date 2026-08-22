from pathlib import Path

from source_ingestion import parse_timed_transcript, restrict_to_boundary, select_non_overlapping


def test_parses_audio_master_transcript_format():
    segments = parse_timed_transcript(
        "[00:10 | start=10.00s | dur=2.50s] First complete sentence.\n"
        "[00:12 | start=12.50s | dur=3.00s] Second complete sentence."
    )
    assert segments == [
        {"start": 10.0, "duration": 2.5, "text": "First complete sentence."},
        {"start": 12.5, "duration": 3.0, "text": "Second complete sentence."},
    ]


def test_restricts_transcript_to_approved_sermon_boundary():
    segments = [
        {"start": 5, "duration": 4, "text": "intro"},
        {"start": 10, "duration": 5, "text": "sermon"},
        {"start": 20, "duration": 6, "text": "sermon end"},
        {"start": 30, "duration": 4, "text": "outro"},
    ]
    assert [item["text"] for item in restrict_to_boundary(segments, {"start": 10, "end": 26})] == [
        "sermon",
        "sermon end",
    ]


def test_selects_highest_scoring_non_overlapping_clips():
    clips = [
        {"start": 0, "end": 30, "score": 95},
        {"start": 10, "end": 25, "score": 90},
        {"start": 35, "end": 60, "score": 85},
        {"start": 70, "end": 90, "score": 80},
    ]
    selected = select_non_overlapping(clips, limit=20)
    assert [(item["start"], item["end"]) for item in selected] == [(0.0, 30.0), (35.0, 60.0), (70.0, 90.0)]


def test_telegram_path_does_not_use_rapidapi_downloader():
    source = (Path(__file__).resolve().parents[1] / "telegram_intake.py").read_text(encoding="utf-8")
    assert "download_youtube_resilient" in source
    assert "main.download_youtube_video" not in source
    assert "main.get_transcript" not in source


def test_prompt_requires_distinct_non_overlapping_complete_thoughts():
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "Clips MUST NOT overlap in time" in source
    assert "Prefer fewer excellent complete thoughts" in source
    assert "beginning, middle, and end" in source
