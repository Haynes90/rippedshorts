"""Restore the existing pipeline, never a second renderer or production retry."""
import copy,json,os,time
from concurrent.futures import Future
from datetime import datetime,timezone,timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
import render_recovery as recovery
import intelligence_workflow as workflow
from test_preview_vertical_slice import previews
from test_boundary_specification import passage
from test_completion_cache_expiry import owned


class ImmediateExecutor:
    def submit(self,fn,*args):
        result=Future()
        try:result.set_result(fn(*args))
        except Exception as exc:result.set_exception(exc)
        return result


def website(p):
    state=p['read']();state.update(review_interface='website',video_path=str(p['source']))
    state.pop('preview_first_version',None);state.pop('copy_drafts',None)
    state['source_validation'].update(status='verified',request_id='request')
    p['write'](state);return state


@pytest.mark.parametrize('brand',['TDOG','tcb','tdog'])
def test_saved_website_request_enters_existing_dispatcher_without_any_reanalysis(previews,monkeypatch,brand):
    p=previews;before=website(p);before['show_id']=brand;p['write'](before);tg=p['tg'];seen=[]
    monkeypatch.setattr(tg,'_ensure_storage_headroom',lambda:{'free_bytes':2**40})
    monkeypatch.setattr(tg,'_resume_saved_copy_review',lambda row:False)
    monkeypatch.setattr(tg,'RENDER_EXECUTOR',ImmediateExecutor())
    monkeypatch.setattr(recovery,'preview_lane',lambda rid,lane,source:seen.append((rid,lane,source)))
    forbidden=Mock(side_effect=AssertionError('Completed acquisition/discovery must not repeat'))
    for name in ('download_youtube_resilient','_transcribe','ingest_with_audio_master','_save'):
        monkeypatch.setattr(tg,name,forbidden)
    monkeypatch.setattr(workflow,'start',forbidden)
    monkeypatch.setattr(workflow,'analyze_highlights',forbidden)
    tg._process.__wrapped__('request')
    assert [item[1] for item in seen]==['short','highlight']
    state=p['read']();assert state['preview_first_version']==state['sentence_first_version']==1
    assert state['result']==before['result'] and state['topic_result']==before['topic_result']
    # Reopening after an owner decision cannot dispatch again or erase it.
    state['website_clip_reviews']={'request:short:12':{'choice':'approve','media_revision':'retained'}};p['write'](state)
    tg._process.__wrapped__('request')
    assert len(seen)==2 and p['read']()==state
    forbidden.assert_not_called()


def test_cached_highlights_ignore_only_the_sentence_first_cache_key_change(previews,monkeypatch):
    p=previews;state=website(p);state.pop('topic_result');state['source_intelligence']={'map_id':'verified-map'}
    state['sentence_first_version']=1
    expected={'segments':[{'candidate_id':'saved-highlight','start':30,'end':230,'transcript':'Saved discussion.'}]}
    with recovery.evidence.database() as db:
        db.execute('INSERT INTO source_analyses(analysis_id,status,result) VALUES(?,?,?)',('verified-map:16:9:minimum-180-v2','PASSED',json.dumps(expected)))
    monkeypatch.setattr(workflow,'saved_mapping',lambda state:{'map_id':'verified-map'})
    forbidden=Mock(side_effect=AssertionError('No repeated highlight analysis'))
    monkeypatch.setattr(workflow,'analyze_highlights',forbidden)
    assert workflow.prepared_highlights(state,[],p['source'])==expected
    state['topic_result']=expected
    monkeypatch.setattr(workflow,'saved_mapping',forbidden)
    assert workflow.prepared_highlights(state,[],p['source'])==expected
    forbidden.assert_not_called()


def test_ambiguous_cached_revisions_are_held_without_selecting_one(previews,monkeypatch):
    p=previews;state=website(p);state.pop('topic_result');state['source_intelligence']={'map_id':'map'}
    monkeypatch.setattr(workflow,'saved_mapping',lambda state:{'map_id':'map'})
    with recovery.evidence.database() as db:
        for key in ('old','new'):
            db.execute('INSERT INTO source_analyses(analysis_id,status,result) VALUES(?,?,?)',('map:16:9:'+key,'PASSED',json.dumps({'segments':[]})))
    with pytest.raises(ValueError,match='Multiple saved'):
        workflow.prepared_highlights(state,[],p['source'])


def test_website_uses_verified_rendered_gallery_not_proposal_or_source_players(previews):
    from fastapi.testclient import TestClient
    p=previews;state=p['read']();state['review_interface']='website';p['write'](state)
    state['result']['segments'][0].update(end=16.1,duration=6.1);p['write'](state)
    for lane in ('short','highlight'):recovery.generate_preview('request',lane,0,p['source'])
    with TestClient(p['app']) as client:
        url='/api/ripped-shorts/review/'+p['token'];data=client.get(url).json();page=client.get('/review/'+p['token']).text
        assert data['preview_first'] and data['website_first'] and data['candidate_total']==0
        assert len(data['clips'])==2 and data['preview_review_total']==2
        assert page.count('class="clip-card"')==2 and 'class="candidate-card' not in page
        assert 'data-source-excerpt' not in page and 'data-confirm-candidates' not in page
        assert 'if(data.website_first&&!data.preview_first)' in page
        assert 'setInterval' in page
        before=p['read']()
        assert client.post(url+'/candidate/request:short:12',json={'choice':'select','revision':'old','version':0}).status_code==409
        assert client.post(url+'/candidates/short/confirm').status_code==409
        assert p['read']()==before
        assert client.post(url+'/finish').status_code==409
    assert not p['read']().get('schedule_requested_at')


@pytest.mark.parametrize('lane',['short','highlight'])
def test_twenty_preview_admissions_preserve_excess_historical_proposals(previews,monkeypatch,lane):
    p=previews;state=p['read']();key,_=recovery.LANES[lane]
    template=state[key]['segments'][0]
    state[key]['segments']=[{**template,'candidate_id':'candidate-'+str(i),'candidate_number':i+1,'start':i*250,'end':i*250+200} for i in range(25)]
    state['sentence_first_version']=1;p['write'](state);before=copy.deepcopy(state[key])
    seen=[];monkeypatch.setattr(recovery,'generate_preview',lambda rid,kind,index,source:seen.append(index))
    replacement=Mock(side_effect=AssertionError('No replacement beyond the ceiling'));monkeypatch.setattr(workflow,'replacement_candidates',replacement)
    assert recovery.preview_lane('request',lane,p['source'])==['verified']*20
    assert seen==list(range(20)) and p['read']()[key]==before
    replacement.assert_not_called()


@pytest.mark.parametrize('lane',['short','highlight'])
def test_replacement_admission_cannot_exceed_twenty(previews,monkeypatch,lane):
    p=previews;state=p['read']();key,_=recovery.LANES[lane];template=state[key]['segments'][0]
    state[key]['segments']=[{**template,'candidate_id':'existing-'+str(i),'candidate_number':i+1,'start':i*250,'end':i*250+200} for i in range(19)]
    state['sentence_first_version']=1;p['write'](state);seen=[]
    def render(rid,kind,index,source):
        seen.append(index)
        if index==0:raise recovery.RenderReviewRequired('Individual unresolved speech')
    monkeypatch.setattr(recovery,'generate_preview',render)
    replacement=Mock(return_value={'segments':[{**template,'candidate_id':'replacement-'+str(i),'start':10000+i*250,'end':10200+i*250} for i in range(5)]})
    monkeypatch.setattr(workflow,'replacement_candidates',replacement)
    recovery.preview_lane('request',lane,p['source'])
    assert len(p['read']()[key]['segments'])==20 and seen==list(range(20))
    assert replacement.call_count==1


@pytest.mark.parametrize('emergency',[False,True])
def test_temporary_preview_expiry_is_48_hours_with_approval_exclusions(owned,emergency,monkeypatch):
    p,tg,source,clock=owned;root=p['tmp'];now=clock[0].timestamp()
    cache=root/'review-previews';cache.mkdir()
    recent=cache/'recent.mp4';old=cache/'old.mp4';approved=cache/'approved.mp4'
    for path,age in ((recent,47),(old,49),(approved,168)):
        path.write_bytes(b'cached verified preview');os.utime(path,(now-age*3600,)*2)
    state=p['read']();state.pop('source_cache_closed_at');state.pop('source_cache_closure_actor')
    state['candidate_reviews']['1']['clip_url']='https://drive.google.com/file/d/approved/view';p['write'](state)
    export=source.parent/'previews'/'active-export.mp4';export.parent.mkdir();export.write_bytes(b'needed')
    os.utime(export,(now-7*86400,)*2)
    tg._purge_expired_local_artifacts(retention_hours=12,emergency=emergency)
    assert recent.exists() and not old.exists() and approved.exists() and export.exists() and source.exists()


@pytest.mark.parametrize('emergency',[False,True])
def test_resolved_preview_export_uses_completion_clock_not_old_mtime(owned,emergency):
    p,tg,source,clock=owned;export=source.parent/'previews'/'unused.mp4';export.parent.mkdir();export.write_bytes(b'verified export')
    os.utime(export,(clock[0].timestamp()-7*86400,)*2)
    tg._purge_expired_local_artifacts(retention_hours=12,emergency=emergency)
    clock[0]+=timedelta(hours=47);tg._purge_expired_local_artifacts(retention_hours=12,emergency=emergency)
    assert export.exists()
    clock[0]+=timedelta(hours=2);tg._purge_expired_local_artifacts(retention_hours=12,emergency=emergency)
    assert not export.exists()


@pytest.mark.parametrize('mode',['both','shorts','topics'])
def test_fresh_website_processing_reaches_preview_dispatch_after_cached_selection(previews,monkeypatch,mode):
    p=previews;tg=p['tg'];original=website(p);state=copy.deepcopy(original)
    for key in ('result','topic_result','topic_source_segments'):
        state.pop(key,None)
    state['parsed']={'video_id':'source','source_value':'https://www.youtube.com/watch?v=source'}
    state['source_intelligence']={'map_id':'cached'};p['write'](state)
    with tg._telegram_db() as db:db.execute('UPDATE telegram_requests SET mode=? WHERE request_id=?',(mode,'request'))
    def save(rid,status,state):
        with tg._telegram_db() as db:db.execute('UPDATE telegram_requests SET status=?,state_json=? WHERE request_id=?',(status,json.dumps(state),rid))
    monkeypatch.setattr(tg,'_save',save)
    monkeypatch.setattr(tg,'SOURCE_DIR',p['tmp']/'sources')
    monkeypatch.setattr(tg,'_ensure_storage_headroom',lambda:{'free_bytes':2**40})
    monkeypatch.setattr(tg,'_resume_saved_copy_review',lambda row:False)
    monkeypatch.setattr(tg,'send',Mock())
    monkeypatch.setattr(tg,'reuse_from_drive',lambda *a:{'video_path':p['source'],'segments':original['topic_source_segments']})
    monkeypatch.setattr(tg,'_reusable_youtube_job',lambda *a:None)
    monkeypatch.setattr(__import__('source_ingestion'),'verify_youtube_source',lambda *a,**k:state['source_validation'])
    monkeypatch.setattr(tg,'_approved_clip_history_from_sheet',lambda *a:{'unfinished':[],'rendered':[]})
    monkeypatch.setattr(tg,'_reviewed_short_history_from_sheet',lambda *a:[])
    monkeypatch.setattr(tg,'_configured_short_framework_prompt',lambda:'Existing approved rubric')
    monkeypatch.setattr(tg,'_brand_short_prompt',lambda *a:'Existing brand context')
    monkeypatch.setattr(workflow,'start',lambda *a:(None,{}))
    monkeypatch.setattr(workflow,'shorts_for_review',lambda *a:copy.deepcopy(original['result']))
    monkeypatch.setattr(workflow,'prepared_highlights',lambda *a:copy.deepcopy(original['topic_result']))
    forbidden=Mock(side_effect=AssertionError('Drafts must reach the existing sentence-first constructor'))
    for name in ('download_youtube_resilient','_transcribe','ingest_with_audio_master','_send_candidates','_send_topic_candidates','validate_complete_candidates','_preflight_candidates'):
        monkeypatch.setattr(tg,name,forbidden)
    seen=[];monkeypatch.setattr(tg,'RENDER_EXECUTOR',ImmediateExecutor())
    monkeypatch.setattr(recovery,'preview_lane',lambda rid,lane,source:seen.append(lane))
    tg._process.__wrapped__('request')
    assert seen==({'both':['short','highlight'],'shorts':['short'],'topics':['highlight']}[mode]),p['read']()
    assert p['read']()['topic_source_segments']==original['topic_source_segments']
    forbidden.assert_not_called()


def test_verified_historical_preview_beyond_ceiling_is_reused_without_rendering(previews):
    p=previews;state=p['read']();template=state['result']['segments'][0]
    state['result']['segments']=[{**template,'candidate_number':i+1,'candidate_id':'historical-'+str(i)} for i in range(21)]
    p['write'](state);identity=recovery.asset_id('request','short',state['result']['segments'][20],20)
    ticket=recovery.preview_ticket('request',state,identity)
    saved={'clip_url':'https://drive.google.com/file/d/retained-historical/view','media_receipt_id':'original-receipt'}
    with recovery.evidence.database() as db:
        recovery._put(db,'preview_success','preview:'+identity+':'+ticket['revision']+':success',{'ticket':ticket,'clip':saved})
    assert recovery.generate_preview('request','short',20,p['source'])==saved
    assert p['read']()==state and not p['calls'] and p['files'].create.call_count==0


def test_unrelated_brand_does_not_enter_rendered_preview_dispatch(previews,monkeypatch):
    p=previews;state=website(p);state['show_id']='AGAPE_CHURCH';state['source_intelligence']={'map_id':'existing'}
    monkeypatch.setattr(workflow,'prepared_highlights',lambda *a:state['topic_result'])
    monkeypatch.setattr(p['tg'],'send',Mock());monkeypatch.setattr(p['tg'],'_save',Mock())
    forbidden=Mock(side_effect=AssertionError('Unrelated brand dispatch is forbidden'))
    monkeypatch.setattr(recovery,'start_preview_pipeline',forbidden)
    notification=Mock();monkeypatch.setattr(p['tg'],'_send_topic_candidates',notification)
    p['tg']._process_topics('request',state,'chat','source',p['source'],state['topic_source_segments'],True)
    forbidden.assert_not_called();notification.assert_called_once()


@pytest.mark.parametrize('brand',['tcb','tdog'])
@pytest.mark.parametrize('lane',['short','highlight'])
def test_lowercase_brand_preserves_identity_through_specification_render_and_receipt_reuse(previews,brand,lane):
    p=previews;state=website(p)
    state.update(show_id=brand,preview_first_version=1,sentence_first_version=1)
    state['result']['segments'][0].update(end=16.1,duration=6.1)
    p['write'](state)
    assert workflow.sentence_first(state)
    rendered=recovery.generate_preview('request',lane,0,p['source'])
    assert rendered['boundary_specification_id'] and rendered['media_receipt_id']
    assert p['read']()['show_id']==brand
    calls=p['reviewer'].call_count
    assert recovery.generate_preview('request',lane,0,p['source'])==rendered
    assert p['reviewer'].call_count==calls and len(p['calls'])==1 and p['files'].create.call_count==1
    assert not p['read']().get('schedule_master') and not p['read']().get('website_clip_reviews')
