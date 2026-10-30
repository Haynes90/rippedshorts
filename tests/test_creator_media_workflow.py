import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import creator_approval as creators
import caption_review_web as web
import source_ingestion as source
import clip_completion as speech
import ast
import os
import logging
import threading
import sys


@pytest.fixture
def directory(tmp_path,monkeypatch):
    path=tmp_path/'isolated.db'
    def factory():
        db=sqlite3.connect(path);db.row_factory=sqlite3.Row
        return db
    monkeypatch.setattr(creators,'connect',factory)
    monkeypatch.setattr(creators,'DEFAULTS_PATH',tmp_path/'no-defaults.json')
    monkeypatch.setattr(web,'connect',factory)
    return factory


def entry(**kw):
    return dict(display_name='Confirmed Fixture',instagram_handle='fixture_verified',intent='invite',**kw)


def test_tcb_channel_suggestion_requires_explicit_confirmed_association(directory):
    creators.confirm_batch('one','TCB','video','channel',[entry()], 'reviewer',True)
    assert creators.suggestions('TCB','channel')==[]
    creators.confirm_batch('two','TCB','video2','channel',[entry(confirm_source_association=True)],'reviewer',True)
    assert creators.suggestions('TCB','channel')[0]['instagram_handle']=='fixture_verified'
    assert creators.suggestions('TCB','other')==[]
    assert creators.suggestions('TDOG','channel')==[]


def test_tdog_hosts_and_returning_guests_are_durable_and_not_automatically_invited(directory):
    selected=creators.confirm_batch('one','TDOG','video','channel',
        [entry(recurring_host=True),dict(display_name='Guest',instagram_handle='fixture_guest',intent='none')], 'human',True)
    suggestions=creators.suggestions('TDOG','different-channel')
    assert len(suggestions)==2 and suggestions[0]['recurring_host']
    assert creators.approved_batch('one')==selected
    assert 'invitation_status' not in selected['creators'][0]


def test_owner_confirmed_tdog_standing_rule_requires_both_hosts(directory,monkeypatch):
    monkeypatch.setattr(creators,'DEFAULTS_PATH',Path(creators.__file__).with_name('creator_defaults.json'))
    hosts=creators.suggestions('TDOG','channel')
    assert {h['instagram_handle'] for h in hosts}=={'andrethefarmer','urbangardeningwithgray'}
    with pytest.raises(ValueError,match='recurring hosts'):
        creators.confirm_batch('tdog','TDOG','v','channel',[],'human',True)
    approved=creators.confirm_batch('tdog','TDOG','v','channel',
        [dict(h,intent='invite') for h in hosts],'human',True)
    assert all(c['standing_invite_authorized'] for c in approved['creators'])
    assert creators.suggestions('TCB','channel')==[]


def test_idempotent_batch_and_explicit_empty_selection(directory):
    first=creators.confirm_batch('one','TCB','v','c',[entry()],'human',True)
    assert creators.confirm_batch('one','TCB','v','c',[entry()],'human',True)==first
    assert creators.confirm_batch('none','TCB','v','c',[],'human',True)['creators']==[]
    with pytest.raises(ValueError):creators.confirm_batch('bad','TCB','v','c',[entry()],'human',False)


def test_batch_cannot_approve_more_than_three_native_collaborators(directory):
    people=[dict(display_name=f'Person {i}',instagram_handle=f'person_{i}',intent='invite') for i in range(4)]
    with pytest.raises(ValueError,match='at most 3'):creators.confirm_batch('too-many','TCB','v','c',people,'human',True)
    assert creators.approved_batch('too-many') is None


def test_one_website_section_and_server_gate_preserve_copy(directory,monkeypatch):
    state={'show_id':'TCB','youtube_channel_id':'channel','parsed':{'video_id':'video'},
           'copy_drafts':[{'asset_id':'one:short:1','asset_type':'9:16_SHORT','social_caption':'My exact words','copy_status':'edited'}]}
    with directory() as db:
        db.execute('CREATE TABLE telegram_requests(request_id TEXT PRIMARY KEY,status TEXT,state_json TEXT,updated_at TEXT)')
        db.execute('INSERT INTO telegram_requests VALUES(?,?,?,?)',('one','review',json.dumps(state),'now'))
    monkeypatch.setattr(web,'_session',lambda _:dict(request_id='one',expires_at='2027-01-01'))
    def row(_):
        with directory() as db:raw=dict(db.execute('SELECT * FROM telegram_requests').fetchone())
        return {**raw,'user_id':'human','chat_id':'fixture'}
    monkeypatch.setattr(web,'_request_row',row)
    app=FastAPI();app.include_router(web.router);client=TestClient(app)
    assert client.post('/api/ripped-shorts/review/token/finish').status_code==409
    saved=client.post('/api/ripped-shorts/review/token/creators',json={'confirmed':True,'entries':[entry()]})
    assert saved.status_code==200
    payload=web._payload_for_session('token')
    html=web._page(payload,'token')
    assert html.count('id="creatorReview"')==1
    assert html.index('id="creatorReview"')<html.index('id="assets"')
    assert payload['items'][0]['social_caption']=='My exact words'
    assert creators.approved_batch('one')['creators'][0]['intent']=='invite'


def test_exact_artifact_ids_avoid_search_and_repeat_download(tmp_path,monkeypatch):
    calls=[]
    def download(file_id,path):
        calls.append(file_id)
        if file_id=='json':path.write_text(json.dumps({'video_id':'v','segments':[{'start':0,'end':2,'text':'Complete.'}]}))
        else:path.write_bytes(b'video')
        return path
    monkeypatch.setattr(source,'download_drive_file',download)
    monkeypatch.setattr(source,'_has_usable_audio',lambda _:True)
    result={'transcript':{'drive_files':{'json':{'file_id':'json'}}},'source_video':{'drive_file_id':'video'}}
    assert source.reuse_audio_master_result(result,'v',tmp_path)['segments'][0]['text']=='Complete.'
    source.reuse_audio_master_result(result,'v',tmp_path)
    assert calls==['json','video']


def test_spoken_edge_mismatch_holds_render_and_removes_output(tmp_path,monkeypatch):
    output=tmp_path/'render.mp4'
    monkeypatch.setattr(speech,'media_duration',lambda _:10)
    monkeypatch.setattr(speech,'review_thought',lambda *a:dict(complete_start=True,complete_end=True,complete_thought=True))
    monkeypatch.setattr(speech,'transcribe_window',lambda *a:[{'word':'Different.','start':0,'end':2}])
    def renderer(*args):output.write_bytes(b'video')
    with pytest.raises(speech.CompletionReviewRequired,match='speech'):
        speech.render_complete_clip('source',{'start':0,'end':10,'transcript':'The approved complete thought.'},output,renderer,'9:16')
    assert not output.exists()


def test_punctuation_alone_cannot_approve_an_incomplete_thought(tmp_path,monkeypatch):
    monkeypatch.setattr(speech,'media_duration',lambda _:10)
    monkeypatch.setattr(speech,'review_thought',lambda *a:dict(complete_start=True,complete_end=True,complete_thought=False))
    renderer=Mock()
    with pytest.raises(speech.CompletionReviewRequired,match='complete thought'):
        speech.render_complete_clip('source',{'start':0,'end':10,'transcript':'But what happened next?'},tmp_path/'output',renderer,'16:9')
    renderer.assert_not_called()


def test_complete_source_validation_is_reusable_and_rejects_partial_video(tmp_path,monkeypatch):
    path=tmp_path/'source.mp4';path.write_bytes(b'full-video')
    class Youtube:
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def extract_info(self,*a,**kw):return {'id':'source','duration':60}
    monkeypatch.setattr(source,'YoutubeDL',Youtube)
    probe=Mock(return_value=SimpleNamespace(stdout=json.dumps({'format':{'duration':60},'streams':[{'codec_type':'video'},{'codec_type':'audio'}]})))
    monkeypatch.setattr(source.subprocess,'run',probe)
    assert source.verify_youtube_source(path,'source','https://fixture.test')['status']=='verified'
    source.verify_youtube_source(path,'source','https://fixture.test')
    assert probe.call_count==1
    path.write_bytes(b'partial')
    probe.return_value.stdout=json.dumps({'format':{'duration':10},'streams':[{'codec_type':'video'},{'codec_type':'audio'}]})
    with pytest.raises(ValueError,match='incomplete'):source.verify_youtube_source(path,'source','https://fixture.test')


def test_actual_handoff_preserves_approved_creator_snapshot_and_copy(directory,monkeypatch):
    approved=creators.confirm_batch('fixture-request','TCB','fixture-video','fixture-channel',[entry()], 'human',True)
    state=dict(show_id='TCB',youtube_channel_id='fixture-channel',parsed=dict(video_id='fixture-video',source_value='https://example.test/source'),
        result=dict(segments=[dict(candidate_number=1,transcript='Complete thought.',duration=30)]),
        candidate_reviews={'0':dict(status='rendered',clip_url='https://example.test/short.mp4')},
        copy_drafts=[dict(asset_id='fixture-request:short:1',social_caption='Stale AI',copy_status='edited')])
    class DB:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def execute(self,sql,args):
            if sql.startswith('UPDATE'):state.clear();state.update(json.loads(args[0]))
            return self
        def fetchone(self):return {'state_json':json.dumps(state)}
    post=Mock(return_value=SimpleNamespace(status_code=200,json=lambda:{'accepted_assets':3}))
    persist=Mock()
    env=dict(os=os,json=json,logger=logging.getLogger('fixture'),_LOCK=threading.RLock(),_telegram_db=DB,
        requests=SimpleNamespace(post=post),send=Mock(),now=lambda:'2026-10-29T12:00:00Z',
        _state_vid_title=lambda s:'Fixture source',_persist_schedule_outbox=persist,
        _caption_learning_checkpoint_map=lambda _: {'fixture-request:short:1':{'social_caption':'My approved exact wording','copy_status':'edited'}})
    monkeypatch.setenv('SCHEDULE_MASTER_URL','https://isolated.test')
    tree=ast.parse((Path(__file__).resolve().parents[1]/'telegram_intake.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_handoff_shorts_to_schedule_master')
    fn.decorator_list=[];exec(compile(ast.Module(body=[fn],type_ignores=[]),'handoff','exec'),env)
    assert env[fn.name]('fixture-request','chat')
    payload=post.call_args.kwargs['json']
    assert payload['creator_approval']==approved
    assert payload['assets'][0]['social_caption']=='My approved exact wording'
    assert persist.call_args_list[0].args[0]['creator_approval']==approved
    assert env[fn.name]('fixture-request','chat') is False
    # Export a synthetic producer fixture for the separate downstream contract test.
    assert payload['copy_approved'] is True


def test_authentication_error_stops_local_transcription_without_three_retries(monkeypatch):
    def extract(args):
        Path(args[-1]).write_bytes(b'isolated-audio')
        return SimpleNamespace(stdout='',stderr='')
    monkeypatch.setattr(speech,'_run',extract)
    post=Mock(return_value=SimpleNamespace(status_code=401))
    monkeypatch.setattr(speech.requests,'post',post)
    sleep=Mock();monkeypatch.setattr(speech.time,'sleep',sleep)
    monkeypatch.setenv('OPENAI_API_KEY','test-only')
    with pytest.raises(speech.CompletionReviewRequired,match='AUTHENTICATION_BLOCKED'):
        speech.transcribe_window('fixture',0,10)
    assert post.call_count==1
    sleep.assert_not_called()


def test_ledger_later_stage_preserves_existing_source_acquisition_columns(monkeypatch):
    import workflow_reliability as ledger
    from unittest.mock import MagicMock
    sheets=MagicMock();values=sheets.spreadsheets().values()
    monkeypatch.setattr(ledger,'_ensure_tab',lambda *args:None)
    old=['job']+['']*17+['provider','profile','when',123]
    values.get().execute.return_value={'values':[ledger.HEADERS,old]}
    monkeypatch.setattr(ledger,'_services',lambda:(None,None,sheets))
    ledger.upsert_job('fixture','job','review',{'stage':'review'})
    assert values.update.call_args.kwargs['body']['values'][0][18:]==old[18:]


def test_acquisition_diagnostic_survives_independent_transcription_failure(tmp_path):
    stored={'state':{'parsed':{'video_id':'video','source_value':'https://example.test/source'}}}
    class DB:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def execute(self,*a):return self
        def fetchall(self):return []
        def fetchone(self):return dict(state_json=json.dumps(stored['state']),chat_id='fixture',status='processing',source_kind='youtube',updated_at='now')
    def save(request,status,state):stored['state']=json.loads(json.dumps(state))
    def fail(message):
        def run(*a,**kw):raise RuntimeError(message)
        return run
    env=dict(os=os,json=json,Path=Path,logger=logging.getLogger('test-acquisition'),
        _ensure_storage_headroom=lambda:{'free_bytes':10**10},_LOCK=threading.RLock(),_telegram_db=DB,
        _TERMINAL_JOB_STAGES=set(),_resume_saved_copy_review=lambda _:False,SOURCE_DIR=tmp_path,
        _save=save,send=Mock(),_youtube_vid_title=lambda _: 'Source title',
        reuse_from_drive=lambda *a:{'video_path':tmp_path/'video.mp4','segments':[]},
        _reusable_youtube_job=lambda _:None,ingest_with_audio_master=fail('Audio Master authentication blocked'),
        _transcribe=fail('Independent transcription blocked'))
    tree=ast.parse((Path(__file__).resolve().parents[1]/'telegram_intake.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_process')
    fn.decorator_list=[];exec(compile(ast.Module(body=[fn],type_ignores=[]),'intake','exec'),env)
    env['_process']('fixture')
    assert stored['state']['audio_master_ingest']['error']=='Audio Master authentication blocked'
    assert stored['state']['error']=='Independent transcription blocked'


def test_failed_transcript_keeps_exact_completed_source_artifact(monkeypatch):
    monkeypatch.setenv('AUDIO_MASTER_INTERNAL_URL','https://isolated.test')
    monkeypatch.setenv('AUDIO_MASTER_INGEST_SECRET','test-only')
    accepted=SimpleNamespace(status_code=202,json=lambda:{'job_id':'one'})
    partial={'job_id':'one','status':'configuration_blocked','error_class':'TRANSCRIPTION_AUTHENTICATION_BLOCKED',
             'source_video':{'drive_file_id':'retained-video'}}
    monkeypatch.setattr(source.requests,'post',lambda *a,**kw:accepted)
    monkeypatch.setattr(source.requests,'get',lambda *a,**kw:SimpleNamespace(status_code=200,json=lambda:partial))
    with pytest.raises(source.AudioMasterIngestError) as result:
        source.ingest_with_audio_master('v','https://example.test/v')
    assert result.value.result['source_video']['drive_file_id']=='retained-video'


def test_transcription_failure_waits_for_parallel_source_instead_of_downloading_twice(monkeypatch):
    monkeypatch.setenv('AUDIO_MASTER_INTERNAL_URL','https://isolated.test')
    monkeypatch.setenv('AUDIO_MASTER_INGEST_SECRET','test-only')
    responses=iter([{'job_id':'one','status':'configuration_blocked','source_video':{'status':'downloading'}},
                    {'job_id':'one','status':'configuration_blocked','source_video':{'status':'ready','drive_file_id':'retained'}}])
    monkeypatch.setattr(source.requests,'post',lambda *a,**kw:SimpleNamespace(status_code=202,json=lambda:{'job_id':'one'}))
    monkeypatch.setattr(source.requests,'get',lambda *a,**kw:SimpleNamespace(status_code=200,json=lambda:next(responses)))
    sleep=Mock();monkeypatch.setattr(source.time,'sleep',sleep)
    with pytest.raises(source.AudioMasterIngestError) as result:source.ingest_with_audio_master('v','https://example.test/v')
    assert sleep.call_count==1
    assert result.value.result['source_video']['drive_file_id']=='retained'


def test_outbox_preserves_creator_metadata_and_unowned_sheet_columns(monkeypatch):
    from unittest.mock import MagicMock
    sheets=MagicMock();values=sheets.spreadsheets().values()
    values.get().execute.return_value={'values':[['asset_id'],['asset']]}
    monkeypatch.setitem(sys.modules,'main',SimpleNamespace(get_google_services=lambda:(None,None,sheets)))
    monkeypatch.setitem(sys.modules,'google_sheets',SimpleNamespace(get_or_create_headers=lambda s,t,h:h+['custom_formula']))
    env=dict(Any=object,now=lambda:'now',json=json,RIPPED_LOG_SHEET_ID='isolated',SCHEDULE_OUTBOX_TAB='Outbox',logger=logging.getLogger('outbox'),_normalize_hashtags=lambda value:value or '')
    tree=ast.parse((Path(__file__).resolve().parents[1]/'telegram_intake.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_persist_schedule_outbox')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'outbox','exec'),env)
    # Existing asset is found through the actual asset_id header position.
    headers=['recorded_at','request_id','asset_id','brand_id','youtube_video_id','youtube_channel_id','source_url','source_title',
      'asset_type','candidate_number','drive_url','transcript','duration_seconds','main_theme','keywords','social_caption',
      'video_title','video_description','hashtags','copy_source','handoff_status','creator_approval_json','workflow_context_json','copy_status','custom_formula']
    values.get().execute.return_value={'values':[headers,['','','asset']+['']*21+['=KEEP()']]}
    env[fn.name]({'request_id':'fixture','show_id':'TCB','creator_approval':{'approval_revision':1},'assets':[{'asset_id':'asset'}]})
    data=values.batchUpdate.call_args.kwargs['body']['data']
    assert not any("!Y" in item['range'] for item in data)
    assert any(item['range'].endswith('!V2') and json.loads(item['values'][0][0])['approval_revision']==1 for item in data)
