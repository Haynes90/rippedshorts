from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "source_ingestion.py").read_text(encoding="utf-8")


def test_accepts_audio_master_cookie_variable_names():
    assert 'os.getenv("YOUTUBE_COOKIES_BASE64")' in SOURCE
    assert 'os.getenv("YTDLP_COOKIE_FILE")' in SOURCE
    assert 'os.getenv("YTDLP_COOKIES_FILE")' in SOURCE
    assert 'os.getenv("YOUTUBE_COOKIES_FILE")' in SOURCE


def test_base64_secret_is_materialized_for_ytdlp():
    assert "base64.b64decode(cookies_base64, validate=True)" in SOURCE
    assert 'workdir / "youtube-cookies.txt"' in SOURCE
    assert 'options["cookiefile"] = cookie_file' in SOURCE


def test_missing_cookie_path_falls_back_to_base64_secret():
    assert "not Path(cookie_file).is_file()" in SOURCE
