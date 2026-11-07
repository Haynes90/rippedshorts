"""Actual SQLite/queue/renderer/review paths; all external operations isolated."""
from copy import deepcopy
from pathlib import Path
import json
import sqlite3
from unittest.mock import Mock

import pytest

import render_recovery as recovery


@pytest.fixture
def batch(tmp_path, monkeypatch):
    import audio_master_handoff as storage
    import telegram_intake as tg
    import clip_review_state
    path=tmp_path/'batch.db'
    monkeypatch.setattr(storage,'DB_PATH',path)
    monkeypatch.setattr(storage,'DATA_DIR',tmp_path)
    monkeypatch.setattr(tg,'DB_PATH',path)
    monkeypatch.setattr(tg,'send',Mock())
    monkeypatch.setattr(tg,'_save',Mock(side_effect=AssertionError('Retry must not replace selection state')))
    monkeypatch.setattr(tg,'_approved_clip_history_from_sheet',Mock(side_effect=AssertionError('No video-wide Sheet history')))
    monkeypatch.setattr(tg,'_active_request_for_video',Mock(side_effect=AssertionError('No request redirection')))
    revisions={}
    monkeypatch.setattr(clip_review_state,'prepare_segment',lambda rid,aid,c:({**c,'framing_mode':None},revisions.get(aid,{})))
    proof={k:True for k in recovery.PROOF}
    def clip(number,lane):
        return dict(candidate_id=f'candidate-{lane}-{number}',candidate_number=number,
                    start=number*100.,end=number*100.+(30 if lane=='short' else 200),
                    duration=30 if lane=='short' else 200,transcript='A verified complete thought.',title='Fixture')
    state=dict(vid_title='Isolated verified source',show_id='TDOG',parsed={'video_id':'same-source'},source_validation={'source_sha256':'verified-source'},
        result={'segments':[clip(n,'short') for n in (12,13,14,15)]},
        candidate_reviews={str(i):{'status':'render_failed','render_error':'Actual exported speech does not match approved boundary words',
                                  'render_diagnostics':{'stage':'export_boundary_alignment'},'user_id':'owner','reviewed_at':'approved'} for i in range(4)},
        topic_result={'segments':[clip(n,'highlight') for n in (1,2)]},
        topic_reviews={str(i):{'status':'render_failed','render_error':'Export ends too close to speech','user_id':'owner'} for i in range(2)},
        creator_approval={'approved_at':'existing','revision':'frozen-creators'},
        copy_drafts=[{'asset_id':'current:short:13','social_caption':'Human wording','copy_status':'edited'}],
        website_clip_reviews={'current:short:13':{'choice':'approve','version':2}},schedule_master={})
    state['result']['segments'][1]['completion_check']=proof
    state['candidate_reviews']['1'].update(status='rendered',clip_url='https://drive.google.com/file/d/verified-fixture/view',receipt='existing-success')
    with tg._telegram_db() as db:
        for rid,payload in [('current',state),('older',{**deepcopy(state),'result':{'segments':[clip(9,'short')]},'candidate_reviews':{'0':{'status':'queued','user_id':'owner'}}})]:
            db.execute('INSERT INTO telegram_requests VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                       (rid,rid,'chat','owner','awaiting_review','both','youtube','fixture',json.dumps(payload),'created','updated'))
    def read(rid='current'):
        with tg._telegram_db() as db:
            return json.loads(db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',(rid,)).fetchone()[0])
    def write(value,rid='current'):
        with tg._telegram_db() as db:
            db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id=?',(json.dumps(value),rid))
    return dict(tg=tg,read=read,write=write,state=state,path=path,revisions=revisions,proof=proof,tmp=tmp_path)


def identities(state):
    return [c['candidate_id'] for key in ('result','topic_result') for c in state[key]['segments']]


def test_all_five_validation_holds_and_success_remain_selected_without_sheet_or_redirect(batch,monkeypatch):
    tg=batch['tg']; before=batch['read'](); other=batch['read']('older')
    executor=Mock(); monkeypatch.setattr(tg,'RENDER_EXECUTOR',executor)
    response=tg._queue_missing_approved_renders('current','chat')
    assert response['count']==0 and response['status']=='nothing_eligible'
    assert batch['read']()==before and batch['read']('older')==other
    assert len(identities(before))==6 and before['candidate_reviews']['1']['receipt']=='existing-success'
    executor.submit.assert_not_called()
    tg._approved_clip_history_from_sheet.assert_not_called()
    tg._active_request_for_video.assert_not_called()


def test_retry_one_transient_source_asset_preserves_six_assets_and_all_approvals(batch,monkeypatch):
    state=batch['read'](); state['candidate_reviews']['0'].update(render_error='Read timed out',render_error_type='ReadTimeout',render_failure_phase='source_access',render_diagnostics={})
    batch['write'](state); before=deepcopy(state)
    executor=Mock(); monkeypatch.setattr(batch['tg'],'RENDER_EXECUTOR',executor)
    item,_=recovery.prepare_queue('current','short',0)
    result=batch['tg']._queue_missing_approved_renders('current','chat',asset_id=item['asset_id'],approved_revision=item['revision'])
    assert result['count']==1 and executor.submit.call_count==1
    assert executor.submit.call_args.kwargs['render_ticket']['asset_id']=='current:short:12'
    assert batch['read']()==before and identities(batch['read']())==identities(before)
    assert not batch['read']()['schedule_master']


def test_wrong_chat_and_wrong_revision_cannot_queue_another_asset(batch,monkeypatch):
    executor=Mock();monkeypatch.setattr(batch['tg'],'RENDER_EXECUTOR',executor)
    assert batch['tg']._queue_missing_approved_renders('current','different-chat')['status']=='held_requires_review'
    assert batch['tg']._queue_missing_approved_renders('current','chat',asset_id='older:short:9',approved_revision='wrong')['count']==0
    executor.submit.assert_not_called()


def test_overwritten_legacy_selection_requires_verified_receipt_not_guessing(batch):
    state=batch['read']();state['recovered_approvals_from_sheet']=True;batch['write'](state)
    with pytest.raises(recovery.RenderReviewRequired,match='receipt restoration'):
        recovery.retry_plan('current','chat')
    assert batch['read']()==state


def test_successful_asset_guard_never_renders_again(batch):
    render=Mock(); guarded=recovery.guarded_render('short')(render)
    assert guarded('current',1,'chat')['status']=='already_rendered'
    render.assert_not_called()
    assert batch['read']()['candidate_reviews']['1']['receipt']=='existing-success'


def test_success_and_attempt_records_are_idempotent_across_duplicate_execution(batch):
    state=batch['read']();state['topic_reviews']['0']={'status':'queued','user_id':'owner'};batch['write'](state)
    calls=[]
    @recovery.guarded_render('highlight')
    def render(rid,index,chat):
        calls.append(index);s=batch['read']()
        s['topic_result']['segments'][index]['completion_check']=batch['proof']
        s['topic_reviews'][str(index)].update(status='rendered',segment_url='https://drive.google.com/file/d/highlight/view')
        batch['write'](s)
    item,_=recovery.prepare_queue('current','highlight',0)
    render('current',0,'chat',render_ticket=item)
    render('current',0,'chat',render_ticket=item)
    assert calls==[0]
    with recovery.evidence.database() as db:
        assert db.execute("SELECT count(*) FROM intelligence_records WHERE kind='render_attempt'").fetchone()[0]==2
        assert db.execute("SELECT count(*) FROM durable_job_leases WHERE state='COMPLETE'").fetchone()[0]==1
    assert len(identities(batch['read']()))==6 and batch['read']()['candidate_reviews']['1']['receipt']=='existing-success'


def test_uncertain_crashed_attempt_is_not_reclaimed_even_after_lease_expiry(batch):
    s=batch['read']();s['topic_reviews']['0']={'status':'queued'};batch['write'](s)
    item,_=recovery.prepare_queue('current','highlight',0)
    from durable_jobs import claim
    assert claim(batch['path'],'current',item['action'],lease_seconds=-1)
    body=Mock();guarded=recovery.guarded_render('highlight')(body)
    assert guarded('current',0,'chat')['status']=='running_or_uncertain_requires_verification'
    body.assert_not_called()


def test_new_editorial_revision_is_required_to_revisit_a_boundary_hold(batch):
    assert recovery.prepare_queue('current','short',0)[1]=='held_requires_review'
    batch['revisions']['current:short:12']={'framing':'owner-approved-framing:2'}
    assert recovery.prepare_queue('current','short',0)[1]=='held_requires_review'
    assert recovery.prepare_queue('current','short',0)[1]=='held_requires_review'
    batch['revisions']['current:short:12']={'editorial':'owner-approved-revision:2'}
    assert recovery.prepare_queue('current','short',0)[1]=='eligible'
    assert batch['read']()['candidate_reviews']['0']['status']=='render_failed'


def test_stale_queued_ticket_cannot_render_changed_asset_revision(batch):
    s=batch['read']();s['topic_reviews']['0']={'status':'queued'};batch['write'](s)
    item,_=recovery.prepare_queue('current','highlight',0)
    s['topic_result']['segments'][0]['end']+=1;batch['write'](s)
    body=Mock();guarded=recovery.guarded_render('highlight')(body)
    assert guarded('current',0,'chat',render_ticket=item)['status']=='held_revision_changed'
    body.assert_not_called()


def test_restore_from_verified_request_receipt_recovers_success_and_preserves_current_copy(batch):
    original=batch['read']()
    recovery.prepare_queue('current','short',1)  # immutable successful-media receipt
    with recovery.evidence.database() as db:
        receipt=recovery._snapshot(db,'current',original)
    corrupted=deepcopy(original)
    corrupted['result']={'segments':[deepcopy(original['result']['segments'][0])]}
    corrupted['candidate_reviews']={'0':original['candidate_reviews']['0']}
    corrupted['recovered_approvals_from_sheet']=True
    corrupted['copy_drafts'][0]['social_caption']='Latest human wording'
    batch['write'](corrupted)
    check=Mock(return_value=True)
    restored=recovery.restore_from_receipt('current',receipt,check)
    assert identities(restored)==identities(original)
    assert restored['candidate_reviews']['1']['receipt']=='existing-success'
    assert restored['copy_drafts'][0]['social_caption']=='Latest human wording'
    assert restored['creator_approval']==original['creator_approval'] and not restored['schedule_master']
    check.assert_called_once()
    # Existing website finished-video controls still expose the recovered successful tile.
    import finished_clip_review
    tiles=finished_clip_review.clips('current',restored)
    assert len(tiles)==6 and sum(c['finished'] for c in tiles)==1
    assert next(c for c in tiles if c['asset_id']=='current:short:13')['media_url'].endswith('/view')


def test_restore_rejects_wrong_request_missing_media_and_scheduled_records(batch):
    with recovery.evidence.database() as db:
        receipt=recovery._snapshot(db,'current',batch['read']())
    with pytest.raises(recovery.RenderReviewRequired,match='this request'):
        recovery.restore_from_receipt('older',receipt,lambda *args:True)
    before=batch['read']()
    with pytest.raises(recovery.RenderReviewRequired,match='independent verification'):
        recovery.restore_from_receipt('current',receipt,lambda *args:False)
    assert batch['read']()==before
    before['schedule_master']={'shorts_status':'sending'};batch['write'](before)
    with pytest.raises(recovery.RenderReviewRequired,match='Submitted selection'):
        recovery.restore_from_receipt('current',receipt,lambda *args:True)


def test_actual_renderer_failure_does_not_overwrite_successful_sibling(batch,monkeypatch):
    tg=batch['tg'];s=batch['read']();s['topic_reviews']['0']={'status':'queued'};batch['write'](s)
    import main
    from clip_completion import CompletionReviewRequired
    monkeypatch.setattr(tg,'_ensure_render_source',lambda *args:Path('isolated-source'))
    monkeypatch.setattr(main,'attach_topic_segment_asset',Mock(side_effect=CompletionReviewRequired('Actual exported speech does not match approved boundary words')))
    for name in ('_refresh_topic_review','_send_short_confirmation','_notify_render_queue_complete'):
        monkeypatch.setattr(tg,name,Mock())
    monkeypatch.setattr(__import__('workflow_test_status'),'journal',Mock())
    before=batch['read']();tg._render_topic_approved('current',0,'chat')
    after=batch['read']()
    assert after['topic_reviews']['0']['status']=='render_failed'
    assert after['candidate_reviews']==before['candidate_reviews']
    assert identities(after)==identities(before)
    tg._render_topic_approved('current',0,'chat')
    assert main.attach_topic_segment_asset.call_count==1


def test_notification_failure_cannot_downgrade_successful_export(batch,monkeypatch):
    tg=batch['tg'];s=batch['read']();s['topic_reviews']['0']={'status':'queued'};batch['write'](s)
    import main
    rendered={**s['topic_result']['segments'][0], 'completion_check':batch['proof'],
              'segment_url':'https://drive.google.com/file/d/completed/view'}
    monkeypatch.setattr(tg,'_ensure_render_source',lambda *a:Path('isolated-source'))
    monkeypatch.setattr(main,'attach_topic_segment_asset',Mock(return_value=rendered))
    # First refresh is progress; the second happens after the durable success write.
    monkeypatch.setattr(tg,'_refresh_topic_review',Mock(side_effect=[None,RuntimeError('notification failed')]))
    monkeypatch.setattr(tg,'_send_short_confirmation',Mock())
    monkeypatch.setattr(__import__('workflow_test_status'),'journal',Mock())
    tg._render_topic_approved('current',0,'chat')
    assert batch['read']()['topic_reviews']['0']['status']=='rendered'
    tg._render_topic_approved('current',0,'chat')
    assert main.attach_topic_segment_asset.call_count==1


def test_concurrent_duplicate_and_selection_change_never_commit_to_a_sibling(batch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    s=batch['read']();s['candidate_reviews']['0']={'status':'queued'};batch['write'](s)
    started,release=Event(),Event();calls=[]
    @recovery.guarded_render('short')
    def render(rid,index,chat):
        calls.append(index);started.set();assert release.wait(5)
        changed=batch['read']()
        recovery.current_index(changed,'short',index)
    with ThreadPoolExecutor(2) as executor:
        first=executor.submit(render,'current',0,'chat');assert started.wait(5)
        assert render('current',0,'chat')['status']=='running_or_uncertain_requires_verification'
        changed=batch['read']();changed['result']['segments'][0]['candidate_id']='new-unapproved-identity'
        batch['write'](changed);release.set()
        with pytest.raises(recovery.RenderReviewRequired,match='changed during rendering'):
            first.result(5)
    assert calls==[0] and batch['read']()==changed
    assert batch['read']()['candidate_reviews']['1']['receipt']=='existing-success'


def test_watchdog_does_not_revisit_deterministic_holds_or_replace_selections(batch,monkeypatch):
    tg=batch['tg'];before=batch['read']()
    monkeypatch.setattr(tg,'_TERMINAL_JOB_STAGES',{'awaiting_review'})
    # Make only this request stale and inspectable; the other request remains terminal.
    with tg._telegram_db() as db:
        db.execute("UPDATE telegram_requests SET status='render_recovery',updated_at='2000-01-01T00:00:00+00:00' WHERE request_id='current'")
    executor=Mock();monkeypatch.setattr(tg,'RENDER_EXECUTOR',executor)
    assert tg._resume_stale_jobs()==0 and tg._resume_stale_jobs()==0
    executor.submit.assert_not_called();assert batch['read']()==before


def legacy_manifest(batch):
    state=batch['read']();entries=[]
    with recovery.evidence.database() as db:
        for lane,(key,reviews) in recovery.LANES.items():
            analysis='cached-'+lane
            db.execute('INSERT INTO source_analyses(analysis_id,status,result) VALUES(?,?,?)',
                       (analysis,'PASSED',json.dumps(state[key])))
            for index,clip in enumerate(state[key]['segments']):
                review={**state[reviews][str(index)],'user_id':'owner','reviewed_at':'original-owner-approval'}
                aid=recovery.asset_id('current',lane,clip,index);fid='feedback:'+aid
                payload=dict(request_id='current',asset_id=aid,source_video_id='same-source',
                    candidate_id=clip['candidate_id'],format='9:16' if lane=='short' else '16:9',
                    bounds={k:clip[k] for k in ('start','end')},review=review,
                    render=clip.get('completion_check'),media_url=review.get('clip_url') or review.get('segment_url'))
                recovery._put(db,'feedback',fid,payload)
                entries.append(dict(asset_id=aid,feedback_id=fid,analysis_id=analysis))
    return entries


def test_legacy_request_scoped_records_restore_exact_six_and_finished_short(batch):
    entries=legacy_manifest(batch);original=batch['read']()
    corrupt=deepcopy(original);corrupt['result']['segments']=[{**original['result']['segments'][0],'candidate_id':'old-other-request','candidate_number':9}]
    corrupt['candidate_reviews']={'0':{'status':'render_failed'}};corrupt['recovered_approvals_from_sheet']=True
    batch['write'](corrupt)
    receipt=recovery.legacy_selection_receipt('current',entries)
    assert batch['read']()==corrupt  # Preparing a plan does not apply it.
    restored=recovery.restore_from_receipt('current',receipt,lambda *a:True)
    assert identities(restored)==identities(original)
    assert restored['candidate_reviews']['1']['receipt']=='existing-success'
    assert restored['copy_drafts']==original['copy_drafts']
    assert restored['creator_approval']==original['creator_approval'] and not restored['schedule_master']


def test_legacy_manifest_rejects_other_request_missing_approval_and_changed_bounds(batch):
    entries=legacy_manifest(batch)
    with pytest.raises(recovery.RenderReviewRequired,match='different request'):
        recovery.legacy_selection_receipt('older',entries)
    original=entries[0]['feedback_id'];entries[0]['feedback_id']='missing'
    with pytest.raises(recovery.RenderReviewRequired,match='different request'):
        recovery.legacy_selection_receipt('current',entries)
    entries[0]['feedback_id']=original
    with recovery.evidence.database() as db:
        payload=recovery._get(db,original,'feedback')
        payload['review']['reviewed_at']=None
        recovery._put(db,'feedback','unapproved',payload)
        payload=deepcopy(payload);payload['review']['reviewed_at']='original';payload['bounds']['end']+=1
        recovery._put(db,'feedback','wrong-bounds',payload)
    entries[0]['feedback_id']='unapproved'
    with pytest.raises(recovery.RenderReviewRequired,match='incomplete'):
        recovery.legacy_selection_receipt('current',entries)
    entries[0]['feedback_id']='wrong-bounds'
    with pytest.raises(recovery.RenderReviewRequired,match='approved bounds differ'):
        recovery.legacy_selection_receipt('current',entries)


def test_legacy_restoration_cannot_substitute_an_earlier_queued_approval_for_a_held_attempt(batch):
    entries=legacy_manifest(batch)
    with recovery.evidence.database() as db:
        payload=recovery._get(db,entries[0]['feedback_id'],'feedback')
        payload['review']={**payload['review'],'status':'queued'}
        recovery._put(db,'feedback','earlier-approval',payload)
    entries[0]['feedback_id']='earlier-approval'
    before=batch['read']()
    with pytest.raises(recovery.RenderReviewRequired,match='incomplete'):
        recovery.legacy_selection_receipt('current',entries)
    assert batch['read']()==before


def test_cache_before_display_numbering_uses_exact_owner_asset_receipt_not_list_position(batch):
    entries=legacy_manifest(batch)
    with recovery.evidence.database() as db:
        cached=json.loads(db.execute("SELECT result FROM source_analyses WHERE analysis_id='cached-short'").fetchone()[0])
        for c in cached['segments']:c.pop('candidate_number')
        db.execute("UPDATE source_analyses SET result=? WHERE analysis_id='cached-short'",(json.dumps(cached),))
    receipt=recovery.legacy_selection_receipt('current',entries)
    with recovery.evidence.database() as db:
        saved=recovery._get(db,receipt,'selected_batch')
    assert [c['candidate_number'] for c in saved['collections']['result']['segments']]==[12,13,14,15]
    assert batch['read']()['candidate_reviews']['1']['receipt']=='existing-success'
    with recovery.evidence.database() as db:
        cached['segments'][0]['candidate_number']=9
        db.execute("UPDATE source_analyses SET result=? WHERE analysis_id='cached-short'",(json.dumps(cached),))
    with pytest.raises(recovery.RenderReviewRequired,match='contradicts'):
        recovery.legacy_selection_receipt('current',entries)


@pytest.mark.parametrize('missing',['proof','status'])
def test_existing_output_without_complete_success_proof_is_held_not_rerendered(batch,missing):
    state=batch['read']()
    if missing=='proof':state['result']['segments'][1]['completion_check']={}
    else:state['candidate_reviews']['1']['status']='render_failed'
    batch['write'](state)
    assert recovery.prepare_queue('current','short',1)[1]=='existing_media_requires_verification'
    body=Mock();recovery.guarded_render('short')(body)('current',1,'chat')
    body.assert_not_called();assert batch['read']()==state


def test_restored_six_asset_review_http_playback_and_approval_controls(batch,monkeypatch):
    from fastapi import FastAPI,Response
    from fastapi.testclient import TestClient
    import caption_review_web as web
    monkeypatch.setenv('RIPPED_SHORTS_PUBLIC_URL','https://review.example.test')
    token=web.ensure_review_session('current')['token']
    monkeypatch.setattr(web,'_preview_response',lambda artifact:Response(b'isolated-completed-video',media_type='video/mp4'))
    app=FastAPI();app.include_router(web.router)
    with TestClient(app) as client:
        page=client.get('/review/'+token)
        assert page.status_code==200 and 'Finished video unavailable' in page.text and 'Complete excerpt transcript' in page.text
        base='/api/ripped-shorts/review/'+token+'/clip/current:short:13'
        assert client.get(base+'/media').content==b'isolated-completed-video'
        assert client.get(base.replace('short:13','short:12')+'/media').status_code==409
        assert client.post(base,json=dict(choice='approve',version=2,playback_verified=True)).status_code==200
        assert batch['read']()['website_clip_reviews']['current:short:13']['choice']=='approve'
        assert not batch['read']()['schedule_master']

