"""Isolated authenticated-retained-source tests; no real jobs or media downloads."""
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import source_ingestion as source

VIDEO = '9mW0bhmUaw0'
URL = 'https://www.youtube.com/watch?v=' + VIDEO


@pytest.fixture
def retained(tmp_path, monkeypatch):
    path = tmp_path / (VIDEO + '-source.mp4')
    path.write_bytes(b'complete verified video fixture')
    document = {'video_id': VIDEO, 'speech_timing_basis': 'provider_segment_timing',
                'segments': [{'text':'A complete explanation.', 'start':0, 'end':30},
                             {'text':'Another complete thought.', 'start':31, 'end':59}]}
    evidence = {'job_id':'audio-job', 'video_id':VIDEO, 'video_url':URL,
        'workflow_runs':{'TW-request':{'request_id':'request', 'source_video_id':VIDEO,
                                     'brand_id':'TDOG', 'source_channel_id':'confirmed-channel'}},
        'source_video':{'status':'ready', 'drive_file_id':'media', 'validation':{
            'video':True, 'audio':True, 'duration_matched':True, 'duration':60, 'expected_duration':60}},
        'transcript':{'segments_count':2, 'drive_files':{'json':{'file_id':'transcript'}}},
        'checkpoints':{k:{'status':'PASSED'} for k in ('MEDIA_READY','TRANSCRIPT_READY')}}
    metadata = {'id':'media', 'mimeType':'video/mp4', 'size':str(path.stat().st_size),
                'md5Checksum':hashlib.md5(path.read_bytes(),usedforsecurity=False).hexdigest()}
    response = SimpleNamespace(status_code=200, json=lambda:evidence)
    get = Mock(return_value=response)
    monkeypatch.setenv('AUDIO_MASTER_INTERNAL_URL','https://audio.example.test')
    monkeypatch.setenv('AUDIO_MASTER_INGEST_SECRET','isolated-secret')
    monkeypatch.setenv('FFPROBE_BINARY','ffprobe')
    monkeypatch.setenv('FFMPEG_BINARY','ffmpeg')
    monkeypatch.setattr(source.requests,'get',get)
    forbidden = Mock(side_effect=AssertionError('No new intake, source download, ASR or YouTube extraction'))
    monkeypatch.setattr(source.requests,'post',forbidden)
    monkeypatch.setattr(source,'download_drive_file',forbidden)
    monkeypatch.setattr(source,'YoutubeDL',forbidden)
    class Files:
        def get(self, **kw):
            assert kw['fileId']=='media'
            return SimpleNamespace(execute=lambda:metadata)
        def get_media(self, **kw):
            # Only transcript JSON may be read; never download the source video.
            assert kw['fileId']=='transcript'
            return SimpleNamespace(execute=lambda:json.dumps(document).encode())
    monkeypatch.setattr(source,'drive_service',lambda:SimpleNamespace(files=Files))
    probe = {'format':{'duration':'60'}, 'streams':[{'codec_type':'video'},{'codec_type':'audio'}]}
    def run(args, **kw):
        return SimpleNamespace(stdout=json.dumps(probe), stderr='', returncode=0)
    decoder = Mock(side_effect=run)
    monkeypatch.setattr(source.subprocess,'run',decoder)
    return SimpleNamespace(path=path, evidence=evidence, document=document, metadata=metadata,
                           probe=probe, response=response, get=get, forbidden=forbidden, decoder=decoder)


def verify(f, **kw):
    return source.verify_youtube_source(f.path, VIDEO, URL, request_id='request',
        audio_master_job_id='audio-job', brand_id='TDOG', **kw)


def test_valid_retained_source_reuses_verified_transcript_without_youtube_or_download(retained):
    before=retained.path.stat().st_mtime_ns
    receipt=verify(retained)
    assert receipt['verification_method']=='authenticated_audio_master_retained'
    assert receipt['source_id']==VIDEO and receipt['request_id']=='request'
    assert receipt['audio_master_job_id']=='audio-job'
    assert receipt['streams_decoded'] and receipt['transcript_segments']==2
    assert receipt['timing_basis']=='provider_segment_timing'
    assert json.loads(Path(receipt['transcript_path']).read_text())==retained.document
    assert retained.path.stat().st_mtime_ns==before
    assert retained.get.call_count==1
    assert retained.get.call_args.args[0].endswith('/ingest/audio-job')
    assert retained.decoder.call_count==2
    assert '-xerror' in retained.decoder.call_args.args[0]
    retained.forbidden.assert_not_called()
    assert 'isolated-secret' not in json.dumps(receipt)


@pytest.mark.parametrize('mismatch',['job','video','url','request','brand','transcript','drive'])
def test_mismatched_identity_holds_instead_of_trusting_cached_file(retained,mismatch):
    if mismatch=='job':retained.evidence['job_id']='different'
    if mismatch=='video':retained.evidence['video_id']='different'
    if mismatch=='url':retained.evidence['video_url']='https://youtu.be/different'
    if mismatch=='request':retained.evidence['workflow_runs']['TW-request']['request_id']='different'
    if mismatch=='brand':retained.evidence['workflow_runs']['TW-request']['brand_id']='AGAPE'
    if mismatch=='transcript':retained.document['video_id']='different'
    if mismatch=='drive':retained.metadata['id']='different'
    with pytest.raises(ValueError,match='mismatch'):
        verify(retained)
    retained.forbidden.assert_not_called()


def test_missing_media_stops_without_any_acquisition(retained):
    retained.path.unlink()
    with pytest.raises(FileNotFoundError):verify(retained)
    retained.get.assert_not_called()
    retained.forbidden.assert_not_called()


@pytest.mark.parametrize('kind',['partial','same_size'])
def test_corrupted_or_incomplete_cached_media_is_not_certified(retained,kind):
    retained.path.write_bytes(b'partial' if kind=='partial' else b'x'*retained.path.stat().st_size)
    with pytest.raises(ValueError,match='checksum/size'):verify(retained)
    retained.decoder.assert_not_called()
    retained.forbidden.assert_not_called()


@pytest.mark.parametrize('bad',['duration','nan','zero','audio','video'])
def test_bad_duration_or_missing_streams_hold(retained,bad):
    if bad in {'duration','nan','zero'}:
        retained.probe['format']['duration']={'duration':'30','nan':'NaN','zero':'0'}[bad]
    else:retained.probe['streams']=[s for s in retained.probe['streams'] if s['codec_type']!=bad]
    with pytest.raises(ValueError,match='incomplete|streams'):verify(retained)
    retained.forbidden.assert_not_called()


def test_corrupt_frames_are_detected_even_with_matching_checksum_and_probe(retained):
    probe=retained.decoder.side_effect
    def decode(args,**kw):
        if args[0]=='ffmpeg':raise subprocess.CalledProcessError(1,args)
        return probe(args,**kw)
    retained.decoder.side_effect=decode
    with pytest.raises(subprocess.CalledProcessError):verify(retained)
    retained.forbidden.assert_not_called()


@pytest.mark.parametrize('bad',['interval','count'])
def test_invalid_or_incomplete_transcript_is_not_reused(retained,bad):
    if bad=='interval':retained.document['segments'][1]['end']=90
    else:retained.evidence['transcript']['segments_count']=3
    with pytest.raises(ValueError):verify(retained)
    retained.forbidden.assert_not_called()


@pytest.mark.parametrize('field',['duration','expected_duration'])
def test_invalid_authenticated_duration_cannot_certify_cache(retained,field):
    retained.evidence['source_video']['validation'][field]=float('nan')
    with pytest.raises(ValueError,match='authenticated original source duration'):verify(retained)
    retained.decoder.assert_not_called()


@pytest.mark.parametrize('missing',['correlation','checksum','checkpoint','validation'])
def test_insufficient_retained_evidence_falls_back_to_normal_verification(retained,monkeypatch,missing):
    if missing=='correlation':retained.evidence['workflow_runs']={}
    if missing=='checksum':retained.metadata.pop('md5Checksum')
    if missing=='checkpoint':retained.evidence['checkpoints']['MEDIA_READY']['status']='RUNNING'
    if missing=='validation':retained.evidence['source_video']['validation'].pop('duration_matched')
    extract=Mock(return_value={'id':VIDEO,'duration':60,'channel_id':'live-channel'})
    class Youtube:
        def __init__(self,*a,**k):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        extract_info=extract
    monkeypatch.setattr(source,'YoutubeDL',Youtube)
    assert verify(retained)['verification_method']=='youtube_metadata'
    extract.assert_called_once_with(URL,download=False)


@pytest.mark.parametrize('status',[401,403])
def test_authentication_failure_is_not_bypassed(retained,status):
    retained.response.status_code=status
    with pytest.raises(RuntimeError,match=str(status)):verify(retained)
    retained.forbidden.assert_not_called()


def test_forged_retained_receipt_does_not_replace_live_authenticated_binding(retained):
    receipt=verify(retained)
    receipt['source_sha256']='forged'
    (retained.path.parent/(VIDEO+'-source-validation.json')).write_text(json.dumps(receipt))
    retained.path.write_bytes(b'other equally untrusted contents')
    with pytest.raises(ValueError,match='checksum/size'):verify(retained)
    assert retained.get.call_count==2


@pytest.mark.parametrize('measured_words',[False,True])
def test_existing_process_reuses_authenticated_assets_before_candidate_generation(retained,monkeypatch,measured_words):
    import ast
    import logging
    import os
    import threading
    root=retained.path.parent
    if measured_words:
        retained.document['words']=[{'word':'Measured.', 'start':0,'end':1}]
    work=root/'telegram-request';work.mkdir()
    retained.path=retained.path.rename(work/retained.path.name)
    state={'parsed':{'source_kind':'youtube','video_id':VIDEO,'source_value':URL},
           'show_id':'TDOG','vid_title':'Saved source title','retained_audio_master_job_id':'audio-job'}
    row={'request_id':'request','status':'error','state_json':json.dumps(state),'chat_id':'chat',
         'mode':'topics','source_kind':'youtube','updated_at':'2026-10-30T00:00:00Z'}
    class DB:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,*a):return self
        def fetchone(self):return row
        def fetchall(self):return []
    saved=[];topics=Mock()
    env=dict(logger=logging.getLogger('retained-test'),os=os,json=json,Path=Path,
        _ensure_storage_headroom=lambda:{'free_bytes':2**40},_LOCK=threading.RLock(),_telegram_db=DB,
        _TERMINAL_JOB_STAGES={'complete','cancelled'},_resume_saved_copy_review=lambda _:False,
        _save=lambda rid,status,s:saved.append((status,dict(s))),send=Mock(),SOURCE_DIR=root,
        _youtube_vid_title=retained.forbidden,reuse_from_drive=retained.forbidden,
        _reusable_youtube_job=lambda _:None,download_youtube_resilient=retained.forbidden,
        ingest_with_audio_master=retained.forbidden,_transcribe=retained.forbidden,
        _segments=lambda p:json.loads(p.read_text())['segments'],_process_topics=topics,now=lambda:'now')
    tree=ast.parse((Path(__file__).resolve().parents[1]/'telegram_intake.py').read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_process')
    node.decorator_list=[]
    exec(compile(ast.Module(body=[node],type_ignores=[]),'retained-process','exec'),env)
    env['_process']('request')
    assert topics.call_count==1,saved
    assert len(topics.call_args.args[5])==2
    final=next(s for _,s in saved if s.get('stage')=='source_validated')
    assert final['transcript_evidence']['audio_master_job_id']=='audio-job'
    assert final['transcript_evidence']['timed_segments']==2
    assert final['source_validation']['verification_method']=='authenticated_audio_master_retained'
    assert final['transcript_evidence']['word_count']==int(measured_words)
    retained.forbidden.assert_not_called()
