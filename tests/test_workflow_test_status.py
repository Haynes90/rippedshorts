import copy
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from workflow_test_status import project, journal, copy_revision


@pytest.fixture
def complete():
    asset='request:short:1'
    creator=dict(request_id='request',approved_at='2026-10-29T12:00:00Z',approval_revision=1,
        creators=[dict(instagram_handle='confirmed_host',intent='invite')])
    draft=dict(asset_id=asset,asset_type='9:16_SHORT',social_caption='Exact human words',copy_status='approved')
    state=dict(show_id='TCB',youtube_channel_id='source-channel',parsed=dict(video_id='video'),
        source_validation=dict(status='verified',duration=100),transcript_evidence=dict(timed_segments=2,speech_evidence=True),
        result={'segments':[dict(candidate_number=1,completion_check={k:True for k in ('media_duration','video_stream','audio_stream','format_valid','export_edges')})]},
        candidate_reviews={'0':dict(status='rendered',clip_url='https://fixture.invalid/video',user_id='human',reviewed_at='2026-10-29T12:00:00Z')},
        creator_approval=creator,copy_drafts=[draft],copy_review_completed_at='2026-10-29T12:00:00Z')
    response=dict(request_id='request',test_run_id='TW-request',asset_group_ids=[asset],delivery_ids=['ig','fb','yt'],
        approved_caption_revision=copy_revision([draft]),approved_collaborator_revision=1,
        publishing_globally_paused=True,evidence_mode='REAL',delivery_validation=dict(status='PASSED',asset_group_ids=[asset],evidence_mode='REAL'))
    return state,creator,response


def test_complete_real_internal_checks_require_separate_publication_approval(complete):
    state,creator,response=complete
    report=project('request',state,creator,response)
    assert report['overall_status']=='READY_FOR_LIVE_CANARY'
    assert report['public_publication_complete'] is False
    assert report['real_platform_acknowledgment'] is False
    assert report['publishing_globally_paused'] is True
    assert report['instagram_collaborators_in_approved_payload']


def test_simulated_completion_can_never_be_a_real_canary(complete):
    state,creator,response=complete
    state['workflow_test']={'evidence_mode':'SIMULATED'}
    report=project('request',state,creator,response)
    assert report['overall_status']=='INTERNAL_SIMULATION_PASSED'
    assert report['checkpoints']['READY_FOR_LIVE_CANARY']['status']=='BLOCKED'


def test_failed_transcription_preserves_media_and_recovery_history(complete):
    state,creator,response=complete
    state['transcript_evidence']={}
    state['audio_master_ingest']={'job_id':'audio','error_class':'TRANSCRIPTION_AUTHENTICATION_BLOCKED'}
    journal('request',state,creator)
    assert state['workflow_test']['checkpoints']['MEDIA_READY']['status']=='PASSED'
    assert state['workflow_test']['checkpoints']['TRANSCRIPT_READY']['status']=='BLOCKED'
    state['transcript_evidence']={'timed_segments':2,'speech_evidence':True}
    journal('request',state,creator)
    assert state['workflow_test']['checkpoints']['TRANSCRIPT_READY']['status']=='PASSED'
    assert [e['status'] for e in state['workflow_test']['history'] if e['stage']=='TRANSCRIPT_READY']==['BLOCKED','PASSED']


@pytest.mark.parametrize('field',['media_duration','audio_stream','format_valid','export_edges'])
def test_incomplete_render_proof_blocks_completion(complete,field):
    state,creator,response=complete
    state['result']['segments'][0]['completion_check'][field]=False
    assert project('request',state,creator,response)['overall_status']!='READY_FOR_LIVE_CANARY'


def test_pending_approval_missing_creators_and_changed_revision_are_not_success(complete):
    state,creator,response=complete
    state.pop('creator_approval')
    state['copy_drafts'][0]['social_caption']='Changed after acknowledgment'
    result=project('request',state,None,response)
    assert result['checkpoints']['CREATORS_APPROVED']['status']=='WAITING_FOR_HUMAN'
    assert result['checkpoints']['SCHEDULER_ACCEPTED']['status']=='BLOCKED'


def test_authenticated_status_get_is_read_only_and_does_not_claim_live_publication(tmp_path,monkeypatch,complete):
    import caption_review_web as web
    import audio_master_handoff as storage
    path=tmp_path/'existing.db';monkeypatch.setattr(storage,'DB_PATH',path)
    state,_,_=complete
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE copy_review_sessions(token TEXT,request_id TEXT,expires_at TEXT,last_opened_at TEXT)')
        db.execute('CREATE TABLE telegram_requests(request_id TEXT,state_json TEXT)')
        db.execute('INSERT INTO copy_review_sessions VALUES(?,?,?,?)',('token','request',(datetime.now(timezone.utc)+timedelta(days=1)).isoformat(),'unchanged'))
        db.execute('INSERT INTO telegram_requests VALUES(?,?)',('request',json.dumps(state)))
    before=path.read_bytes()
    app=FastAPI();app.include_router(web.router);client=TestClient(app)
    result=client.get('/api/ripped-shorts/review/token/test-status')
    assert result.status_code==200
    assert result.json()['real_platform_acknowledgment'] is False
    assert client.get('/api/ripped-shorts/review/wrong/test-status').status_code==404
    assert path.read_bytes()==before


def test_submitted_caption_cannot_change_after_batch_handoff(monkeypatch,complete):
    import caption_review_web as web
    state,_,_=complete
    state['schedule_master']={'shorts_status':'accepted'}
    monkeypatch.setattr(web,'_session',lambda _:dict(request_id='request'))
    monkeypatch.setattr(web,'_request_row',lambda _:dict(state_json=json.dumps(state)))
    app=FastAPI();app.include_router(web.router);client=TestClient(app)
    response=client.post('/api/ripped-shorts/review/token/asset/0',json={'social_caption':'replacement'})
    assert response.status_code==409
    assert state['copy_drafts'][0]['social_caption']=='Exact human words'


def test_service_authenticated_status_works_before_review_session_and_preserves_database(tmp_path,monkeypatch):
    import caption_review_web as web
    import audio_master_handoff as storage
    path=tmp_path/'early.db';monkeypatch.setattr(storage,'DB_PATH',path)
    monkeypatch.setenv('SCHEDULE_MASTER_SHARED_SECRET','test-only')
    state=dict(show_id='TCB',parsed={'video_id':'video'},source_validation={'status':'verified'},
        audio_master_ingest={'job_id':'audio','error_class':'TRANSCRIPTION_AUTHENTICATION_BLOCKED'})
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE telegram_requests(request_id TEXT,state_json TEXT)')
        db.execute('INSERT INTO telegram_requests VALUES(?,?)',('early',json.dumps(state)))
    before=path.read_bytes();app=FastAPI();app.include_router(web.router);client=TestClient(app)
    assert client.get('/api/ripped-shorts/test-runs/early').status_code==401
    result=client.get('/api/ripped-shorts/test-runs/early',headers={'x-schedule-master-secret':'test-only'})
    assert result.status_code==200
    assert result.json()['checkpoints']['TRANSCRIPT_READY']['status']=='BLOCKED'
    assert result.json()['checkpoints']['MEDIA_READY']['status']=='PASSED'
    assert path.read_bytes()==before
