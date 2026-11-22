import json
import re
import shutil
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import BackgroundTasks
from test_editorial_framing_review import review


def read(r):
    with r['storage'].connect() as db:
        return json.loads(db.execute('SELECT state_json FROM telegram_requests WHERE request_id=?',('existing',)).fetchone()[0])


def write(r,state):
    with r['storage'].connect() as db:
        db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id=?',(json.dumps(state),'existing'))


@pytest.fixture
def candidates(review,monkeypatch):
    r=review
    state={'review_interface':'website','show_id':'TDOG','stage':'awaiting_review',
           'parsed':{'video_id':'source'},'transcript_evidence':{'timed_segments':427},
           'source_validation':{'status':'verified','source_sha256':'retained','transcript_file_id':'transcript'},
           'result':{'segments':[{'candidate_number':i+1,'candidate_id':'short-'+str(i),
               'start':i*60.,'end':i*60.+30.,'duration':30.,'transcript':'How should seeds be stored? Keep them dry and cool.',
               'reason':'A practical question and its complete answer.'} for i in range(5)]},
           'topic_result':{'segments':[{'start':400.,'end':660.,'duration':260.,'candidate_id':'highlight',
               'title':'A focused seed-storage discussion','transcript':'A complete discussion with setup, explanation and conclusion.',
               'reason':'A substantive topic arc.'}]}}
    write(r,state)
    r['submit']=Mock(return_value={'status':'queued'})
    monkeypatch.setattr(r['tg'],'_submit_render',r['submit'])
    monkeypatch.setattr(__import__('intelligence_workflow'),'observe',Mock())
    return r


def base(r): return '/api/ripped-shorts/review/'+r['token']


def choose(r,item,choice='select',**extra):
    return r['client'].post(base(r)+'/candidate/'+item['asset_id'],json={
        'choice':choice,'revision':item['revision'],'version':item['version'],**extra})


def test_both_formats_are_visible_before_render_and_session_is_request_specific(candidates):
    r=candidates;data=r['client'].get(base(r)).json();page=r['client'].get('/review/'+r['token']+'?stage=shorts').text
    assert data['request_id']=='existing' and data['candidate_total']==6
    assert {c['format'] for c in data['candidates']}=={'9:16','16:9'}
    assert page.count('class="candidate-card asset"')==6
    assert '<video' not in page and 'watched this finished clip' not in page
    assert 'Intended transcript and boundaries' in page and 'Full frame on blurred background' in page
    assert 'Status not loaded' not in page and 'Candidate selection remains in Telegram' not in page
    assert 'refreshWorkflow();' in page and 'id="finishButton" disabled' in page
    assert r['client'].post(base(r)+'/finish').status_code==409
    r['submit'].assert_not_called();r['telegram'].assert_not_called()


def test_five_current_style_shorts_remain_visible_without_highlight_materialization(candidates):
    r=candidates;state=read(r);state.pop('topic_result');write(r,state)
    data=r['client'].get(base(r)).json()
    assert data['candidate_total']==5 and len(data['candidates'])==5
    assert data['clips']==[] and data['processing_status']=='Candidate review ready'


def test_choices_persist_and_confirmation_queues_only_selected_format_once(candidates):
    r=candidates;items=r['client'].get(base(r)).json()['candidates']
    original_passage=dict(read(r)['result']['segments'][2])
    with r['storage'].connect() as db:
        other=db.execute('SELECT state_json FROM telegram_requests WHERE request_id="other"').fetchone()[0]
    assert choose(r,items[0],framing='blurred').status_code==200
    assert choose(r,items[1],'reject').status_code==200
    assert choose(r,items[2],'request_adjustment',feedback='Include the question and the complete answer.').status_code==200
    assert choose(r,items[3],'unused').status_code==200
    after=r['client'].get(base(r)).json()
    assert after['candidate_reviewed']==4 and after['candidates'][0]['framing']=='blurred'
    assert after['candidates'][0]['choice']=='select'
    assert read(r)['result']['segments'][2]==original_passage
    assert choose(r,items[0]).status_code==409  # stale revision/version cannot overwrite newer choice
    r['submit'].assert_not_called()
    assert r['client'].post(base(r)+'/candidates/short/confirm').status_code==200
    assert r['client'].post(base(r)+'/candidates/short/confirm').json()['status']=='already_confirmed'
    r['submit'].assert_called_once_with('short','existing',0,'saved-chat')
    assert choose(r,items[5]).status_code==200
    assert r['client'].post(base(r)+'/candidates/highlight/confirm').status_code==200
    assert r['submit'].call_count==2
    state=read(r)
    assert state['candidate_reviews']['0']['status']=='queued' and state['topic_reviews']['0']['status']=='queued'
    assert state['candidate_reviews']['1']['status']=='rejected'
    assert not state.get('schedule_requested_at') and not state.get('schedule_master')
    with r['storage'].connect() as db:
        assert db.execute('SELECT state_json FROM telegram_requests WHERE request_id="other"').fetchone()[0]==other


@pytest.mark.parametrize('action',['approve','reject','topic_approve','topic_reject','shorts_confirm','shorts_page','schedule_now','copy_finish','copy_keep','copy_regenerate','copy_edit_caption','copy_edit_title','copy_edit_description'])
def test_legacy_telegram_buttons_only_navigate_for_website_requests(candidates,action):
    r=candidates;before=read(r)
    update={'callback_query':{'from':{'id':'owner'},'data':'rs:'+action+':existing:0',
            'message':{'chat':{'id':'saved-chat'},'message_id':8}}}
    assert r['tg']._accept_update(update,BackgroundTasks())['status']=='website_review_required'
    assert read(r)==before
    r['submit'].assert_not_called()
    markup=r['telegram'].call_args.args[1]['reply_markup']['inline_keyboard']
    assert len(markup)==1 and 'url' in markup[0][0] and 'callback_data' not in markup[0][0]


def test_both_candidate_notifiers_share_one_receipt_and_preserve_legacy_records(candidates):
    r=candidates;before=read(r)
    r['tg']._send_candidates('saved-chat','existing',before['result'])
    r['tg']._send_topic_candidates('saved-chat','existing',before['topic_result'])
    r['tg']._refresh_short_review_message('saved-chat','existing')
    r['tg']._refresh_topic_review('saved-chat','existing')
    assert r['telegram'].call_count==1 and read(r)==before


@pytest.mark.parametrize('stage,label',[('source_resolution','Source verified'),('selection','discovering candidates'),('error','Blocked')])
def test_empty_processing_and_error_states_are_not_completed_reviews(candidates,stage,label):
    r=candidates;state=read(r);state.pop('result');state.pop('topic_result');state['stage']=stage
    if stage=='source_resolution':state.pop('transcript_evidence')
    write(r,state)
    data=r['client'].get(base(r)).json();page=r['client'].get('/review/'+r['token']).text
    assert data['candidate_total']==0 and label in data['processing_status']
    assert '0/0 reviewed' not in page and 'id="finishButton" disabled' in page
    assert r['client'].post(base(r)+'/finish').status_code==409


def test_page_javascript_parses_without_status_widget_syntax_errors(candidates,tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable for JavaScript parse acceptance')
    page=candidates['client'].get('/review/'+candidates['token']).text
    for index,script in enumerate(re.findall(r'<script>(.*?)</script>',page,re.S)):
        path=tmp_path/(str(index)+'.js');path.write_text(script,encoding='utf-8')
        result=subprocess.run([node,'--check',str(path)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_audited_transition_reuses_cached_highlights_without_processing(candidates):
    r=candidates;state=read(r);state.pop('review_interface');state.pop('topic_result')
    state['source_intelligence']={'map_id':'saved-map'};state['creator_approval']={'approved':True}
    write(r,state)
    import production_intelligence as pi
    with pi.database() as db:
        db.execute('INSERT INTO source_analyses(analysis_id,status,result) VALUES(?,?,?)',
                   ('saved-map:16:9:approved-rubric','PASSED',json.dumps({'segments':[{'start':400.,'end':660.,'transcript':'Complete cached discussion.'}]})))
    result=r['reviews'].enable_website_review('existing','authorized-owner')
    assert result['candidate_count']==6 and result['processing_dispatched'] is False
    current=read(r)
    assert current['result']==state['result'] and current['creator_approval']==state['creator_approval']
    assert current['website_review_transition']['previous_state_sha256']
    assert r['reviews'].enable_website_review('existing','authorized-owner')['status']=='already_enabled'
    r['submit'].assert_not_called()


def test_transition_refuses_legacy_approval_without_changing_it(candidates):
    r=candidates;state=read(r);state.pop('review_interface')
    state['candidate_reviews']={'0':{'status':'rendered','clip_url':'retained-success','approved_at':'original'}}
    write(r,state)
    with pytest.raises(ValueError,match='Existing selections'):
        r['reviews'].enable_website_review('existing','owner')
    assert read(r)==state


def test_copy_notification_has_only_website_navigation(candidates):
    r=candidates;before=read(r)
    r['tg']._refresh_copy_review('saved-chat','existing')
    r['tg']._refresh_copy_review('saved-chat','existing')
    assert r['telegram'].call_count==1 and read(r)==before
    payload=r['telegram'].call_args.args[1]
    assert '/review/' in payload['reply_markup']['inline_keyboard'][0][0]['url']
    assert 'callback_data' not in payload['reply_markup']['inline_keyboard'][0][0]


def test_website_finished_approval_preserves_copy_and_never_schedules(review):
    from test_finished_clip_review import render_fixture,post,read_state,save_state
    r=review;state=render_fixture(r);state['review_interface']='website'
    state['copy_drafts']=[];save_state(r,state)
    response=post(r,choice='approve',version=0,playback_verified=True)
    assert response.status_code==200 and response.json()['copy_review_ready']
    current=read_state(r)
    assert len(current['copy_drafts'])==1 and current['copy_drafts'][0]['copy_status']=='pending'
    assert current['website_clip_reviews']['existing:short:1']['choice']=='approve'
    draft=dict(current['copy_drafts'][0]);draft['social_caption']='Owner text 🌱 #Seeds'
    current['copy_drafts'][0]=draft;save_state(r,current)
    assert post(r,choice='approve',version=1,playback_verified=True).status_code==200
    assert read_state(r)['copy_drafts']==[draft]
    assert not read_state(r).get('schedule_requested_at')
    r['telegram'].assert_not_called()
