from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "source_ingestion.py").read_text(encoding="utf-8")


def test_existing_rapidapi_video_downloader_is_in_source_race():
    assert "youtube-video-fast-downloader-24-7.p.rapidapi.com" in SOURCE
    assert '"/download_video/{video_id}"' in SOURCE
    assert '"name": "rapidapi_video_download"' in SOURCE
    assert '"provider": "rapidapi"' in SOURCE
    assert "params={\"quality\": quality}" in SOURCE


def test_rapidapi_file_url_is_polled_until_ready():
    assert 'payload.get("file")' in SOURCE
    assert "file_response.status_code == 404" in SOURCE
    assert "YOUTUBE_DL_READY_TIMEOUT_SECONDS" in SOURCE


def test_race_dispatches_rapidapi_and_ytdlp():
    assert "def _run_source_profile(" in SOURCE
    assert 'profile.get("provider") == "rapidapi"' in SOURCE
    assert "ThreadPoolExecutor(max_workers=2" in SOURCE
    assert "_run_source_profile" in SOURCE


def test_source_winner_is_logged_and_durably_recorded():
    # Existing durable workflow ledger replaces the obsolete per-workdir JSONL filename.
    import source_ingestion as si,workflow_reliability
    from unittest.mock import patch
    from tempfile import TemporaryDirectory
    from pathlib import Path
    with TemporaryDirectory() as folder:
        media=Path(folder)/'retained.mp4';media.write_bytes(b'retained-test-media')
        with patch.object(workflow_reliability,'record_source_winner') as record,patch('builtins.print') as output:
            si._log_source_winner(video_id='source',profile={'provider':'yt-dlp','name':'cached'},path=media,workdir=Path(folder))
        assert record.call_args.kwargs['video_id']=='source'
        assert record.call_args.kwargs['source_bytes']==len(b'retained-test-media')
        assert record.call_args.kwargs['provider']=='yt-dlp' and record.call_args.kwargs['profile']=='cached'
        assert 'RIPPED_SOURCE_WINNER' in output.call_args_list[0].args[0]
