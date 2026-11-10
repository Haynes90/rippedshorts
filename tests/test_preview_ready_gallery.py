"""Eligibility projection only: preserve all candidates and receipts, no live choices."""
import copy
import pytest
from fastapi.testclient import TestClient
from test_boundary_specification import passage
from test_preview_vertical_slice import previews
import render_recovery as recovery
from boundary_specification import BoundaryReviewRequired


def test_three_ready_two_held_one_deferred_gallery_and_counts(previews):
    p=previews;s=p['read']();s['result']['segments']=s['result']['segments'][:1]
    template=s['topic_result']['segments'][0]
    s['topic_result']['segments']=[{**copy.deepcopy(template),'candidate_id':'highlight-'+str(i),'title':'Highlight '+str(i)} for i in range(5)]
    for i in (2,3):s['topic_result']['segments'][i].update(start=230,end=30)
    s['copy_drafts']=[];s['acceptance_demo']=True;p['write'](s)
    for lane,index in [('short',0),('highlight',0),('highlight',1)]:
        recovery.generate_preview('request',lane,index,p['source'])
    for i in (2,3):
        with pytest.raises(BoundaryReviewRequired):recovery.generate_preview('request','highlight',i,p['source'])
    before=p['read']()
    base='/api/ripped-shorts/review/'+p['token']
    with TestClient(p['app']) as client:
        data=client.get(base).json();page=client.get('/review/'+p['token']).text
        assert len(data['clips'])==3 and all(c['finished'] for c in data['clips'])
        assert data['preview_review_total']==3 and data['preview_review_completed']==0
        assert page.count('class="clip-card"')==3 and page.count('<video controls playsinline')==3
        assert page.count('I watched this finished clip')==3
        assert 'BoundaryReviewRequired' not in page and 'Finished video unavailable' not in page
        assert 'Selection evidence' not in page
        assert 'id="finishButton" disabled' in page and 'not yet enabled for scheduling' in page
        assert {c['format'] for c in data['clips']}=={'9:16','16:9'}
        for item in data['clips']:
            assert client.get(base+'/clip/'+item['asset_id']+'/media').status_code==200
        rejected=data['clips'][0]
        response=client.post(base+'/clip/'+rejected['asset_id'],json={'choice':'reject','version':0,'reason':'weak_hook'})
        assert response.status_code==200 and not response.json()['scheduling_started']
        assert client.get(base).json()['preview_review_completed']==1
        assert client.get(base).json()['preview_review_total']==3
        assert client.post(base+'/clip/'+rejected['asset_id'],json={'choice':'reject','version':0,'reason':'weak_hook'}).status_code==409
        for item in recovery.preview_items('request',before):
            if item[1]=='highlight' and item[0].endswith((':3',':4',':5')):
                assert client.post(base+'/clip/'+item[0],json={'choice':'approve','version':0,'playback_verified':True,'framing_verified':True}).status_code==409
        assert client.post(base+'/finish').status_code==409
    after=p['read']()
    assert after['result']==before['result'] and after['topic_result']==before['topic_result']
    assert after['topic_reviews']==before['topic_reviews'] and not after.get('schedule_requested_at')
    assert len(p['calls'])==3 and p['files'].create.call_count==3 and not p['files'].delete.called
    assert len(list(recovery.preview_items('request',after)))==6


def test_unresolved_sibling_does_not_block_selected_finished_copy(previews,monkeypatch):
    p=previews;clip=recovery.generate_preview('request','short',0,p['source'])
    base='/api/ripped-shorts/review/'+p['token'];asset=clip['boundary_asset_id']
    with TestClient(p['app']) as client:
        assert client.post(base+'/clip/'+asset,json={'choice':'approve','version':0,'playback_verified':True,'framing_verified':True}).status_code==200
    s=p['read']();s['copy_drafts']=[dict(asset_id=asset,copy_status='approved',social_caption='Owner-approved copy',video_title='Approved title')];p['write'](s)
    import caption_review_web as web
    monkeypatch.setattr(__import__('creator_approval'),'approved_batch',lambda _:dict(approval_revision=1))
    payload=web._payload_for_session(p['token'])
    assert payload['schedule_blockers']==[] and payload['preview_review_total']==1
    assert payload['preview_review_completed']==1
    # This is eligibility evidence, not an owner final approval or scheduler transaction.
    assert not p['read']().get('schedule_requested_at') and not p['read']().get('schedule_master')


def test_missing_receipt_removes_player_and_copy_card_without_changing_records(previews):
    p=previews;clip=recovery.generate_preview('request','short',0,p['source']);s=p['read']()
    s['copy_drafts']=[dict(asset_id=clip['boundary_asset_id'],copy_status='approved')];p['write'](s)
    import production_intelligence as intelligence
    with intelligence.database() as db:db.execute('DELETE FROM intelligence_records WHERE record_id=?',(clip['media_receipt_id'],))
    before=p['read']()
    with TestClient(p['app']) as client:
        payload=client.get('/api/ripped-shorts/review/'+p['token']).json()
        assert payload['clips']==[] and payload['items']==[] and payload['preview_review_total']==0
        assert payload['schedule_blockers']
        page=client.get('/review/'+p['token']).text
        assert 'I watched this finished clip' not in page and '<video controls' not in page
    assert p['read']()==before and len(p['calls'])==1
