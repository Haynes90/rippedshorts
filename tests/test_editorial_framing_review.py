import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def review(tmp_path, monkeypatch):
    import audio_master_handoff as storage
    import caption_review_web as web
    import clip_review_state as reviews
    import telegram_intake as tg
    import review_navigation as navigation
    monkeypatch.setattr(storage, 'DB_PATH', tmp_path/'isolated.db')
    monkeypatch.setattr(storage, 'DATA_DIR', tmp_path)
    monkeypatch.setenv('RIPPED_SHORTS_PUBLIC_URL', 'https://review.example.test')
    monkeypatch.setattr(tg, '_authorized', lambda chat, user: chat == 'saved-chat' and user == 'owner')
    telegram = Mock(return_value={'ok': True, 'result': {'message_id': 7}})
    monkeypatch.setattr(tg, 'telegram', telegram)
    state = {'show_id': 'TDOG', 'parsed': {'video_id': 'source'},
             'result': {'segments': [{'start': 10., 'end': 30., 'duration': 20., 'transcript': 'Original complete thought.', 'category': 'education', 'candidate_id': 'candidate'}]},
             'candidate_reviews': {'0': {'status': 'render_failed', 'render_error': 'Retained diagnostic'}},
             'copy_drafts': [{'asset_id': 'existing:short:1', 'asset_type': '9:16_SHORT', 'social_caption': 'Exact approved words', 'copy_status': 'edited'}]}
    with tg._telegram_db() as db:
        db.execute('INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)', ('existing','key','saved-chat','owner','awaiting_render_retry','both','youtube','source',json.dumps(state),'now','now'))
        db.execute('INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)', ('other','key2','saved-chat','owner','review','both','youtube','source',json.dumps(state),'now','now'))
    media = tmp_path/'preview.mp4'
    media.write_bytes(b'isolated-video-bytes-not-real-media')
    artifact = {'local_path': str(media), 'size': media.stat().st_size, 'sha256': hashlib.sha256(media.read_bytes()).hexdigest()}
    provenance = {'source_video_id': 'source', 'source_duration': 100, 'verified_media': True,
                  'proposed_transcript': 'The proposed complete thought.', 'timing_provenance': 'provider_segment_timing'}
    def register(kind='editorial', **kwargs):
        return reviews.register('existing','existing:short:1',kind,[10.,30.],[10.,35.],
            {'original': artifact, 'proposed': artifact} if kind == 'editorial' else {'blurred': artifact}, provenance, notify=False, **kwargs)
    token = web.ensure_review_session('existing')['token']
    app = FastAPI(); app.include_router(web.router)
    with TestClient(app) as client:
        yield dict(storage=storage, web=web, reviews=reviews, tg=tg, navigation=navigation, telegram=telegram,
                   state=state, register=register, token=token, client=client, artifact=artifact, provenance=provenance)


def endpoint(r, revision):
    return f'/api/ripped-shorts/review/{r["token"]}/comparison/{revision}'


def test_editorial_and_framing_decisions_are_separate_resumable_and_do_not_mutate_request(review):
    r=review; editorial=r['register'](); framing=r['register']('framing')
    with r['storage'].connect() as db:
        before=dict(db.execute('SELECT * FROM telegram_requests WHERE request_id="existing"').fetchone())
    assert r['client'].post(endpoint(r,editorial),json={'choice':'proposed','version':0}).json()['rendering_started'] is False
    assert len(r['reviews'].pending('existing'))==1
    assert r['client'].post(endpoint(r,framing),json={'choice':'blurred','version':0}).json()['scheduling_authorized'] is False
    page=r['client'].get('/review/'+r['token'])
    assert page.status_code==200 and 'Approve Proposed' in page.text and 'Choose Blurred Full Frame' in page.text
    assert 'Saved decision: proposed' in page.text and '<video controls playsinline' in page.text
    assert 'max-width:600px' in page.text and 'data-index' in page.text
    with r['storage'].connect() as db:
        assert dict(db.execute('SELECT * FROM telegram_requests WHERE request_id="existing"').fetchone())==before
    assert r['reviews'].delivery_blockers('existing',r['state'])
    assert r['client'].post('/api/ripped-shorts/review/'+r['token']+'/finish').status_code==409
    r['telegram'].assert_not_called()


def test_feedback_required_optimistic_version_and_frozen_copy(review):
    r=review; rev=r['register']()
    assert r['client'].post(endpoint(r,rev),json={'choice':'request_edit'}).status_code==409
    assert r['client'].post(endpoint(r,rev),json={'choice':'request_edit','feedback':'Include the answer','version':0}).status_code==200
    assert r['client'].post(endpoint(r,rev),json={'choice':'original','version':0}).status_code==409
    assert r['reviews'].pending('existing')[0]['feedback']=='Include the answer'
    state={**r['state'],'schedule_master':{'shorts_status':'accepted'}}
    with r['storage'].connect() as db:
        db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id="existing"',(json.dumps(state),))
    assert r['client'].post(endpoint(r,rev),json={'choice':'original','version':1}).status_code==409


def test_authenticated_video_range_expiry_and_cross_request_isolation(review):
    r=review; rev=r['register'](); url=endpoint(r,rev)+'/media/original'
    response=r['client'].get(url,headers={'Range':'bytes=0-7'})
    assert response.status_code==206 and response.content==b'isolated'
    other=r['web'].ensure_review_session('other')['token']
    assert r['client'].get(url.replace(r['token'],other)).status_code==404
    assert r['client'].post(endpoint(r,rev).replace(r['token'],other),json={'choice':'original','version':0}).status_code==409
    with r['storage'].connect() as db:
        db.execute('UPDATE copy_review_sessions SET expires_at="2020-01-01T00:00:00+00:00" WHERE token=?',(r['token'],))
    assert r['client'].get(url).status_code==410
    fresh=r['web'].ensure_review_session('existing')['token']
    assert fresh!=r['token'] and len(r['reviews'].reviews('existing'))==1


def test_notification_receipts_dedupe_renew_and_uncertain_no_resend(review):
    r=review
    assert r['navigation'].notify('existing','editorial','revision')['status']=='delivered'
    assert r['navigation'].notify('existing','editorial','revision')['status']=='already_delivered'
    assert r['telegram'].call_count==1
    payload=r['telegram'].call_args.args[1]
    assert payload['chat_id']=='saved-chat'
    button=payload['reply_markup']['inline_keyboard'][0][0]
    assert button['text']=='🌐 Open Clip Review' and '?stage=editorial#editorial' in button['url']
    with r['storage'].connect() as db:
        db.execute('UPDATE copy_review_sessions SET expires_at="2020-01-01T00:00:00+00:00"')
    assert r['navigation'].notify('existing','editorial','revision')['status']=='delivered'
    assert r['telegram'].call_args.args[0]=='editMessageText'
    r['telegram'].side_effect=TimeoutError('uncertain')
    assert r['navigation'].notify('existing','framing','revision')['status']=='verification_required'
    calls=r['telegram'].call_count
    assert r['navigation'].notify('existing','framing','revision')['status']=='verification_required'
    assert r['telegram'].call_count==calls


def test_review_integrity_and_old_render_cannot_satisfy_new_revision(review):
    r=review; rev=r['register']()
    assert r['register']()==rev
    r['reviews'].decide('existing',rev,'proposed','Include the full explanation',0)
    prepared,revisions=r['reviews'].prepare_segment('existing','existing:short:1',r['state']['result']['segments'][0])
    assert prepared['end']==35 and prepared['transcript']=='The proposed complete thought.'
    assert r['state']['result']['segments'][0]['end']==30
    r['state']['candidate_reviews']['0']['rendered_review_revisions']=revisions
    assert not r['reviews'].delivery_blockers('existing',r['state'])
    r['reviews'].decide('existing',rev,'original','',1)
    assert r['reviews'].delivery_blockers('existing',r['state'])
    r['provenance']['source_video_id']='wrong'
    with pytest.raises(ValueError,match='source binding'): r['register']()


def test_review_feedback_reaches_existing_learning_without_claiming_explicit_preference(review):
    r=review; rev=r['register']()
    r['reviews'].decide('existing',rev,'proposed','Preserve the answer and payoff',0)
    import production_intelligence as intelligence
    context=intelligence.learning_context({'brand_id':'TDOG','show_id':'TDOG','source_video_id':'later'},'9:16')
    assert context['examples'][0]['boundary_changes']=={'before':[10.,30.],'after':[10.,35.]}
    prompt=intelligence.learning_prompt(context)
    assert 'Preserve the answer' in prompt and 'not owner instructions' in context['interpretation']


def test_preview_tampering_is_detected(review):
    r=review; rev=r['register']()
    Path(r['artifact']['local_path']).write_bytes(b'tampered')
    assert r['client'].get(endpoint(r,rev)+'/media/original').status_code==409


def test_new_revision_preserves_history_and_rejects_superseded_controls(review):
    r=review; original=r['register']()
    r['reviews'].decide('existing',original,'request_edit','Need the response',0)
    r['provenance']['explanation']='New comparison after explicit edit request'
    revised=r['register']()
    assert revised!=original and r['reviews'].reviews('existing')[0]['version']==0
    assert r['client'].post(endpoint(r,original),json={'choice':'original','version':1}).status_code==409
    with r['storage'].connect() as db:
        assert db.execute('SELECT feedback FROM clip_review_decisions WHERE revision_id=?',(original,)).fetchone()[0]=='Need the response'


def test_independent_framing_and_boundary_evidence_both_influence_future_prompt(review):
    r=review; editorial=r['register'](); framing=r['register']('framing')
    r['reviews'].decide('existing',editorial,'proposed','',0)
    r['reviews'].decide('existing',framing,'blurred','',0)
    import production_intelligence as intelligence
    context=intelligence.learning_context({'brand_id':'TDOG','show_id':'TDOG'},'9:16')
    assert {e['review_kind'] for e in context['examples']}=={'editorial','framing'}
    assert any(e['boundary_changes'] for e in context['examples'])
    assert any(e['framing_choice']=='blurred' for e in context['examples'])


def test_review_hold_is_asset_specific_and_smart_cannot_silently_fall_back(review):
    r=review; r['register']()
    state=r['state']
    state['result']['segments'].append({'start':40,'end':60,'candidate_number':2})
    state['candidate_reviews']['1']={'status':'rendered','clip_url':'private-sibling'}
    blockers=r['reviews'].delivery_blockers('existing',state)
    assert blockers and all('existing:short:1:' in b for b in blockers)
    with pytest.raises(ValueError,match='reproduced safely'):
        r['reviews'].verify_render_choice({'framing_mode':'smart'},{'framing_plan':{'layout':'blurred_full_frame'}})


def test_invalid_destination_and_invalid_duration_cannot_create_review_delivery(review,monkeypatch):
    r=review
    monkeypatch.setattr(r['tg'],'_authorized',lambda *args:False)
    with pytest.raises(ValueError,match='authorized existing request'):r['navigation'].notify('existing','editorial')
    r['telegram'].assert_not_called()
    r['provenance']['source_duration']=float('nan')
    with pytest.raises(ValueError,match='duration'):r['register']()
