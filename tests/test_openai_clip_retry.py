from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


def test_full_transcript_selection_has_long_configurable_timeout():
    assert 'OPENAI_CLIP_TIMEOUT_SECONDS' in SOURCE
    assert 'os.getenv("OPENAI_CLIP_TIMEOUT_SECONDS", "600")' in SOURCE
    assert 'timeout=(15, OPENAI_CLIP_TIMEOUT_SECONDS)' in SOURCE


def test_full_transcript_selection_retries_transient_failures():
    assert 'OPENAI_CLIP_ATTEMPTS' in SOURCE
    assert 'requests.ReadTimeout' in SOURCE
    assert 'requests.ConnectionError' in SOURCE
    assert 'resp.status_code not in {408, 409, 429}' in SOURCE
    assert 'len(transcript_segments)' in SOURCE
