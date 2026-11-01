import json
from unittest.mock import Mock
import pytest
from test_editorial_framing_review import review


def render_fixture(r):
    state=r['state']
    state['result']['segments'][0]['completion_check']={k:True for k in ('media_duration','video_stream','audio_stream','format_valid','export_edges')}
    state['candidate_reviews']['0']={'status':'rendered','clip_url':'https://drive.google.com/file/d/verified_finished_asset/view'}
    save_state(r,state)
    return state


def save_state(r,state):
    with r['storage'].connect() as db:
        db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id="existing"',(json.dumps(state),))


def read_state(r):
    with r['storage'].connect() as db:
        return json.loads(db.execute('SELECT state_json FROM telegram_requests WHERE request_id="existing"').fetchone()[0])


def post(r,**data):
    return r['client'].post('/api/ripped-shorts/review/'+r['token']+'/clip/existing:short:1',json=data)


def test_failed_diagnostic_is_not_presented_as_finished_and_comparisons_are_optional(review):
    r=review;r['register']()
    page=r['client'].get('/review/'+r['token'])
    assert page.status_code==200
    assert 'Finished video unavailable' in page.text and 'No source preview is presented as a final export' in page.text
    assert '<details id="editing-workflow" >' in page.text and 'Approve Proposed' in page.text
    assert page.text.index('Complete excerpt transcript')<page.text.index('Finished video unavailable')
    assert post(r,choice='approve',version=0,playback_verified=True).status_code==409
    assert r['client'].get('/api/ripped-shorts/review/'+r['token']+'/clip/existing:short:1/media').status_code==409


def test_finished_clip_approval_durable_and_does_not_authorize_copy_creators_or_scheduling(review):
    r=review;render_fixture(r)
    before=read_state(r)['copy_drafts']
    assert post(r,choice='approve',version=0,playback_verified=False).status_code==409
    response=post(r,choice='approve',version=0,playback_verified=True)
    assert response.status_code==200 and response.json()['collaborator_invitation_authorized'] is False
    state=read_state(r)
    assert state['website_clip_reviews']['existing:short:1']['choice']=='approve'
    assert state['copy_drafts']==before and not state.get('schedule_requested_at')
    assert not state.get('creator_approval')
    r['telegram'].assert_not_called()
    assert post(r,choice='reject',version=0,reason='poor_topic').status_code==409


def test_rejection_reason_is_separate_and_media_fault_does_not_become_topic_dislike(review):
    r=review;render_fixture(r)
    assert post(r,choice='reject',version=0).status_code==409
    assert post(r,choice='reject',version=0,reason='bad_audio',feedback='The topic is good; audio is quiet').status_code==200
    import production_intelligence as intelligence
    context=intelligence.learning_context({'brand_id':'TDOG','show_id':'TDOG'},'9:16')
    item=context['examples'][0]
    assert item['rejection_reason']=='bad_audio' and item['sign']==0
    assert 'audio is quiet' in intelligence.learning_prompt(context)
    from finished_clip_review import blocked_assets
    assert blocked_assets('existing',read_state(r))=={'existing:short:1'}


def test_adjustment_records_specific_fields_and_scheduled_copy_stays_frozen(review):
    r=review;render_fixture(r)
    assert post(r,choice='request_adjustment',version=0,feedback='Include the answer').status_code==409
    assert post(r,choice='request_adjustment',version=0,feedback='Include the answer',changed_fields=['boundaries','topic_context']).status_code==200
    state=read_state(r);assert state['website_clip_reviews']['existing:short:1']['changed_fields']==['boundaries','topic_context']
    state['schedule_master']={'shorts_status':'accepted','shorts_response':{'asset_group_ids':['existing:short:1']}}
    save_state(r,state)
    assert post(r,choice='reject',version=1,reason='poor_topic').status_code==409


def test_render_revision_change_invalidates_approval_without_replacing_it(review):
    r=review;state=render_fixture(r)
    assert post(r,choice='approve',version=0,playback_verified=True).status_code==200
    state=read_state(r);original=dict(state['website_clip_reviews']['existing:short:1'])
    state['candidate_reviews']['0']['clip_url']='https://drive.google.com/file/d/replacement_needs_review/view';save_state(r,state)
    from finished_clip_review import blocked_assets
    assert blocked_assets('existing',state)=={'existing:short:1'}
    assert read_state(r)['website_clip_reviews']['existing:short:1']==original


def test_finish_without_final_renders_does_not_claim_handoff_or_write_request(review):
    r=review;before=read_state(r)
    result=r['client'].post('/api/ripped-shorts/review/'+r['token']+'/finish')
    assert result.status_code==409 and 'nothing was sent' in result.json()['detail']
    assert read_state(r)==before
    page=r['client'].get('/review/'+r['token']).text
    assert "data.status === 'scheduler_accepted'" in page and 'Scheduling is pending' in page


def test_collaborator_section_and_approval_records_remain_unchanged(review):
    r=review;render_fixture(r)
    import creator_approval as creators
    hosts=creators.suggestions('TDOG','source')
    approved=creators.confirm_batch('existing','TDOG','source','channel',[dict(h,intent='invite') for h in hosts],'fixture-owner',True)
    page=r['client'].get('/review/'+r['token']).text
    assert page.count('id="creatorReview"')==1 and 'Add a person' in page and 'Confirm batch creators' in page
    assert page.index('id="creatorReview"')<page.index('id="finished-clips"')
    assert post(r,choice='approve',version=0,playback_verified=True).status_code==200
    assert creators.approved_batch('existing')==approved
    r['telegram'].assert_not_called()


def test_finished_media_endpoint_checks_request_and_proof(review,monkeypatch):
    from fastapi import Response
    r=review;render_fixture(r)
    observed=[]
    monkeypatch.setattr(r['web'],'_preview_response',lambda artifact:(observed.append(artifact) or Response(b'verified-export',media_type='video/mp4')))
    url='/api/ripped-shorts/review/'+r['token']+'/clip/existing:short:1/media'
    assert r['client'].get(url).content==b'verified-export'
    assert observed==[{'drive_file_id':'verified_finished_asset'}]
    other=r['web'].ensure_review_session('other')['token']
    assert r['client'].get(url.replace(r['token'],other)).status_code==409


def test_latest_feedback_ranks_future_candidates_without_erasing_history_or_novelty(review):
    r=review;render_fixture(r)
    assert post(r,choice='reject',version=0,reason='poor_topic',feedback='This example is too generic').status_code==200
    assert post(r,choice='approve',version=1,playback_verified=True,feedback='The complete explanation is useful').status_code==200
    saved=read_state(r)['website_clip_reviews']['existing:short:1']
    assert saved['prior_versions'][0]['choice']=='reject' and saved['prior_versions'][0]['reason']=='poor_topic'
    import production_intelligence as pi
    context=pi.learning_context({'brand_id':'TDOG','show_id':'TDOG','source_video_id':'new-source'},'9:16')
    future={'segments':[{'start':0,'end':20,'category':'education','score':80},{'start':30,'end':55,'category':'humor','score':80}]}
    ranked=pi.rank_with_evidence(future,context)
    assert len(ranked['segments'])==2
    assert ranked['segments'][0]['category']=='education' and ranked['segments'][0]['learning_influence']['score_adjustment']>0
    assert ranked['segments'][1]['category']=='humor' and ranked['segments'][1]['learning_influence']['score_adjustment']==0


def test_optional_framing_reference_does_not_force_routine_comparison_or_auto_approve_it(review):
    r=review;render_fixture(r)
    r['provenance']['optional_reference']=True
    revision=r['register']('framing')
    assert post(r,choice='approve',version=0,playback_verified=True).status_code==200
    assert r['reviews'].reviews('existing')[0]['decision'] is None
    assert not r['reviews'].delivery_blockers('existing',read_state(r))
    # An explicit adjustment remains binding and cannot be ignored.
    r['reviews'].decide('existing',revision,'request_adjustment','Keep the demonstration visible',0)
    assert r['reviews'].delivery_blockers('existing',read_state(r))
