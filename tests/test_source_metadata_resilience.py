"""Provider/restart fault injection; no actual acquisition, ASR or media processing."""
import ast
import copy
import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import source_ingestion as source

VIDEO='Vcxu4BOjb6A';URL='https://www.youtube.com/watch?v='+VIDEO


@pytest.fixture
def media(tmp_path,monkeypatch):
    path=tmp_path/(VIDEO+'-source.mp4');path.write_bytes(b'isolated complete source')
    monkeypatch.setenv('GOOGLE_API','private-test-key')
    monkeypatch.delenv('AUDIO_MASTER_INTERNAL_URL',raising=False)
    body={'items':[{'id':VIDEO,'snippet':{'channelId':'real-source-channel','title':'Actual source title','liveBroadcastContent':'none'},
                    'contentDetails':{'duration':'PT1M'}}]}
    response=SimpleNamespace(status_code=200,json=lambda:body)
    getter=Mock(return_value=response);monkeypatch.setattr(source.requests,'get',getter)
    extractor=Mock(side_effect=AssertionError('No extraction needed for official verified metadata'))
    monkeypatch.setattr(source,'YoutubeDL',extractor)
    probe=Mock(return_value=SimpleNamespace(stdout=json.dumps({'format':{'duration':60},'streams':[{'codec_type':'video'},{'codec_type':'audio'}]})))
    monkeypatch.setattr(source.subprocess,'run',probe)
    return SimpleNamespace(path=path,body=body,response=response,get=getter,extractor=extractor,probe=probe)


def test_official_metadata_is_primary_and_exact_verified_file_is_reused(media):
    first=source.verify_youtube_source(media.path,VIDEO,URL,request_id='request')
    assert first['metadata_provider']=='youtube_data_api' and first['source_channel_id']=='real-source-channel'
    assert first['expected_duration']==60 and first['request_id']=='request'
    assert 'private-test-key' not in json.dumps(first)
    assert source.verify_youtube_source(media.path,VIDEO,URL,request_id='request')==first
    assert media.get.call_count==media.probe.call_count==1
    media.extractor.assert_not_called()


@pytest.mark.parametrize('bad',['missing','wrong_id','channel','duration','live','schema','snippet'])
def test_required_identity_and_duration_cannot_be_bypassed(media,bad):
    item=media.body['items'][0]
    if bad=='missing':media.body['items']=[]
    if bad=='wrong_id':item['id']='other-video'
    if bad=='channel':item['snippet'].pop('channelId')
    if bad=='duration':item['contentDetails']['duration']='unknown'
    if bad=='live':item['snippet']['liveBroadcastContent']='live'
    if bad=='schema':media.body['items']='not a video collection'
    if bad=='snippet':item['snippet']='not a metadata object'
    with pytest.raises(ValueError):source.verify_youtube_source(media.path,VIDEO,URL)
    media.extractor.assert_not_called()
    assert not media.path.with_name(VIDEO+'-source-validation.json').exists()


def test_missing_optional_title_does_not_block_verified_media(media):
    media.body['items'][0]['snippet'].pop('title')
    assert source.verify_youtube_source(media.path,VIDEO,URL)['status']=='verified'


@pytest.mark.parametrize('failure',[403,429,503,'timeout'])
def test_provider_rejection_uses_one_existing_authorized_alternative(media,monkeypatch,failure):
    if failure=='timeout':media.get.side_effect=requests.Timeout('private-test-key must not be logged')
    else:media.response.status_code=failure
    extract=Mock(return_value={'id':VIDEO,'duration':60,'channel_id':'real-source-channel'})
    class Youtube:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        extract_info=extract
    monkeypatch.setattr(source,'YoutubeDL',Youtube)
    assert source.verify_youtube_source(media.path,VIDEO,URL)['metadata_provider']=='yt_dlp'
    media.get.assert_called_once();extract.assert_called_once()


def test_no_safe_metadata_preserves_source_without_secrets_or_retry_loop(media,monkeypatch):
    media.response.status_code=503
    monkeypatch.setattr(source,'YoutubeDL',Mock(side_effect=RuntimeError('private-test-key')))
    before=media.path.read_bytes()
    with pytest.raises(RuntimeError,match='SOURCE_METADATA_UNAVAILABLE') as exc:
        source.verify_youtube_source(media.path,VIDEO,URL)
    assert 'private-test-key' not in str(exc.value)
    assert media.path.read_bytes()==before and media.get.call_count==1
    from workflow_reliability import classify_error
    assert classify_error(exc.value)=='SOURCE_VERIFICATION'


def test_changed_or_incomplete_source_is_revalidated_and_held(media):
    source.verify_youtube_source(media.path,VIDEO,URL)
    media.path.write_bytes(b'partial')
    media.probe.return_value.stdout=json.dumps({'format':{'duration':10},'streams':[{'codec_type':'video'},{'codec_type':'audio'}]})
    with pytest.raises(ValueError,match='incomplete'):source.verify_youtube_source(media.path,VIDEO,URL)
    assert media.get.call_count==2


def process_fixture(media,monkeypatch,*,saved=True):
    root=media.path.parent;work=root/'telegram-request';work.mkdir();media.path=media.path.rename(work/media.path.name)
    transcript=[{'start':0,'end':60,'duration':60,'text':'A complete explanation.','speaker':'guest',
                 'words':[{'word':'Explanation.','start':1,'end':2}]}]
    proof=source.verify_youtube_source(media.path,VIDEO,URL,request_id='request')
    state={'parsed':{'video_id':VIDEO,'source_value':URL},'show_id':'TCB','video_path':str(media.path),
           'source_validation':proof,'topic_source_segments':transcript if saved else [],'vid_title':'Owner title'}
    row={'request_id':'request','status':'error','state_json':json.dumps(state),'chat_id':'chat','mode':'topics','source_kind':'youtube','updated_at':'now'}
    class DB:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,*a):return self
        def fetchone(self):return row
        def fetchall(self):return []
    forbidden=Mock(side_effect=AssertionError('Do not redownload, re-ingest, reanalyze or redo paid ASR'))
    saved_states=[];topics=Mock();transcriber=forbidden if saved else Mock(return_value=transcript)
    env=dict(Path=Path,json=json,logger=logging.getLogger('checkpoint-test'),_ensure_storage_headroom=lambda:{'free_bytes':10**10},
        os=os,_LOCK=threading.RLock(),_telegram_db=DB,_TERMINAL_JOB_STAGES={'complete'},_resume_saved_copy_review=lambda _:False,
        _save=lambda rid,status,s:saved_states.append((status,copy.deepcopy(s))),send=Mock(),SOURCE_DIR=root,
        _youtube_vid_title=forbidden,reuse_from_drive=forbidden,_reusable_youtube_job=lambda _:None,
        download_youtube_resilient=forbidden,ingest_with_audio_master=forbidden,_transcribe=transcriber,
        _process_topics=topics,now=lambda:'now',requests=requests)
    node=next(n for n in ast.parse((Path(__file__).parents[1]/'telegram_intake.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='_process')
    node.decorator_list=[];exec(compile(ast.Module(body=[node],type_ignores=[]),'checkpoint-process','exec'),env)
    return env,media,state,row,saved_states,topics,transcriber,transcript


@pytest.mark.parametrize('saved',[True,False])
def test_worker_restart_resumes_verified_checkpoint_without_source_reacquisition(media,monkeypatch,saved):
    env,media,state,row,states,topics,transcriber,transcript=process_fixture(media,monkeypatch,saved=saved)
    env['_process']('request')
    assert topics.call_count==1,states
    assert topics.call_args.args[5]==transcript
    checkpoint=next(s for _,s in states if s.get('stage')=='source_validated')
    assert checkpoint['source_validation']['source_sha256']==hashlib.sha256(media.path.read_bytes()).hexdigest()
    assert checkpoint['topic_source_segments']==transcript
    assert transcriber.call_count==int(not saved)


def test_failed_later_analysis_does_not_erase_saved_transcript(media,monkeypatch):
    env,media,state,row,states,topics,transcriber,transcript=process_fixture(media,monkeypatch)
    topics.side_effect=RuntimeError('Temporary analysis outage')
    env['_process']('request')
    assert any(s.get('topic_source_segments')==transcript and s.get('source_validation',{}).get('status')=='verified' for _,s in states)
    transcriber.assert_not_called()


def test_mismatched_checkpoint_cannot_transfer_a_transcript_to_other_media(media,monkeypatch):
    env,media,state,row,states,topics,transcriber,transcript=process_fixture(media,monkeypatch)
    state['source_validation']['source_sha256']='other-source-revision';row['state_json']=json.dumps(state)
    env['_process']('request')
    topics.assert_not_called();transcriber.assert_not_called()
    assert media.path.exists()
    assert any('source revision mismatch' in str(s.get('error')) for _,s in states)


def test_optional_source_title_failure_uses_verified_metadata_title(media,monkeypatch):
    env,media,state,row,states,topics,transcriber,transcript=process_fixture(media,monkeypatch)
    state.pop('vid_title');row['state_json']=json.dumps(state)
    env['_youtube_vid_title']=Mock(side_effect=requests.Timeout('title provider unavailable'))
    env['_process']('request')
    assert topics.call_count==1
    assert next(s for _,s in states if s.get('stage')=='source_validated')['vid_title']=='Actual source title'


def test_plain_notification_failure_is_observable_but_cannot_abort_processing(monkeypatch,caplog):
    import telegram_intake as tg
    monkeypatch.setattr(tg,'telegram',Mock(side_effect=requests.HTTPError('https://api.telegram.org/botSECRET/sendMessage')))
    tg.send('owner','Processing continues')
    assert 'RIPPED_NOTIFICATION_FAILED' in caplog.text and 'SECRET' not in caplog.text
