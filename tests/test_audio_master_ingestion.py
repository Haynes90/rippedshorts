from pathlib import Path


INGEST = (Path(__file__).resolve().parents[1] / "source_ingestion.py").read_text(encoding="utf-8")
TELEGRAM = (Path(__file__).resolve().parents[1] / "telegram_intake.py").read_text(encoding="utf-8")


def test_missing_assets_are_delegated_to_audio_master():
    import source_ingestion as si
    from unittest.mock import Mock,patch
    from types import SimpleNamespace
    from contextlib import ExitStack
    post=Mock(return_value=SimpleNamespace(status_code=202,json=lambda:{'job_id':'retained-job'}))
    get=Mock(side_effect=[SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:{'status':'processing','transcript':{'drive_files':{'json':{'id':'transcript'}}},'source_video':{'status':'running'}}),SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:{'status':'completed','transcript':{'drive_files':{'json':{'id':'transcript'}}},'source_video':{'status':'ready','drive_file_id':'video'}})])
    with patch.dict('os.environ',{'AUDIO_MASTER_INTERNAL_URL':'https://audio.test','AUDIO_MASTER_INGEST_SECRET':'fixture-only'}),patch.object(si.requests,'post',post),patch.object(si.requests,'get',get),patch.object(si.time,'sleep'):
        result=si.ingest_with_audio_master('source','https://youtube.test/source')
    assert result['source_video']['drive_file_id']=='video' and get.call_count==2
    assert post.call_count==1 and post.call_args.kwargs['json']['video_id']=='source'
    assert post.call_args.kwargs['headers']=={'x-ripped-shorts-ingest-secret':'fixture-only'}


def test_audio_master_client_uses_protected_dedicated_routes():
    assert 'f"{base}/api/ripped-shorts/ingest"' in INGEST
    assert 'f"{base}/api/ripped-shorts/ingest/{job_id}"' in INGEST
    assert '"x-ripped-shorts-ingest-secret": secret' in INGEST


def test_audio_master_client_requires_transcript_and_source_video():
    assert "transcript_ready and source_ready" in INGEST
    assert "Audio Master ingestion timed out" in INGEST


def test_audio_master_configuration_names_are_explicit():
    assert 'os.getenv("AUDIO_MASTER_INTERNAL_URL")' in INGEST
    assert 'os.getenv("AUDIO_MASTER_INGEST_SECRET")' in INGEST
    assert 'os.getenv("AUDIO_MASTER_WEBHOOK_SECRET")' in INGEST
