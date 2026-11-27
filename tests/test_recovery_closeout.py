"""Exact review/receipt accounting and real disposable cache cleanup; no dispatch."""
import copy
import json
from unittest.mock import Mock

import pytest
import batch_progress as progress
from workflow_test_status import project
from test_workflow_test_status import complete
from test_boundary_specification import passage
from test_completion_cache_expiry import owned
from test_finished_clip_review import review, render_fixture, read_state, post


def reviewed(complete):
    state,creator,scheduler=copy.deepcopy(complete)
    state['clip_review_ui_version']=2
    from finished_clip_review import clips
    clip=clips('request',state)[0]
    state['website_clip_reviews']={clip['asset_id']:{'choice':'approve','version':1,
        'media_revision':clip['media_revision'],'actor':'owner','decided_at':'approved'}}
    scheduler['delivery_manifest']=[{'asset_group_id':clip['asset_id'],'delivery_id':'delivery'}]
    scheduler['delivery_ids']=['delivery']
    scheduler['delivery_validation']['deliveries']=[dict(asset_group_id=clip['asset_id'],
        delivery_id='delivery',scheduled_at='2030-01-01T12:00:00Z',destination_id='approved-account',schedule_record_valid=True)]
    initial=copy.deepcopy(scheduler)
    initial['delivery_validation']['deliveries'][0].update(scheduled_at='',schedule_record_valid=False)
    state['schedule_master']={'shorts_status':'accepted','shorts_response':initial}
    return state,scheduler


def test_validated_playback_approval_is_persisted_with_its_exact_media(review):
    render_fixture(review)
    assert post(review,choice='approve',version=0,playback_verified=True).status_code==200
    state=read_state(review);asset='existing:short:1'
    assert state['website_playback_checks'][asset]['verified'] is True
    assert state['website_playback_checks'][asset]['media_revision']==state['website_clip_reviews'][asset]['media_revision']


def test_legacy_validated_approval_counts_but_changed_media_does_not(complete):
    state,scheduler=reviewed(complete)
    before=copy.deepcopy(state)
    assert progress.summarize('request',state,scheduler)['completed_clips']==1
    assert state==before  # No fabricated decision or migration.
    state['result']['segments'][0]['rendered_bounds']=[1,5]
    assert progress.summarize('request',state,scheduler)['formats']['short']['playable']==0


def test_unused_copy_does_not_change_the_selected_batch_digest(complete):
    state,scheduler=reviewed(complete)
    state['copy_drafts'].append(dict(asset_id='request:highlight:4',asset_type='16:9_HIGHLIGHT',
        video_title='Unused owner draft',video_description='Keep these words',copy_status='pending'))
    before=copy.deepcopy(state)
    result=project('request',state,scheduler=scheduler)
    assert result['checkpoints']['SCHEDULER_ACCEPTED']['status']=='PASSED'
    assert state==before
    state['copy_drafts'][0]['social_caption']='A different approved caption'
    assert project('request',state,scheduler=scheduler)['checkpoints']['SCHEDULER_ACCEPTED']['status']=='BLOCKED'


@pytest.mark.parametrize('fault',['missing_delivery','wrong_request','wrong_revision','wrong_delivery','unavailable','simulated','simulated_validation'])
def test_incomplete_or_mismatched_readback_never_releases_source(owned,complete,monkeypatch,fault):
    p,tg,source,clock=owned
    state,scheduler=reviewed(complete);state['video_path']=str(source);p['write'](state)
    if fault=='missing_delivery':scheduler['delivery_validation']['deliveries']=[]
    if fault=='wrong_request':scheduler['request_id']='another-request'
    if fault=='wrong_revision':scheduler['approved_caption_revision']='changed'
    if fault=='wrong_delivery':scheduler['delivery_ids']=['unrelated-delivery']
    if fault=='unavailable':scheduler=None
    if fault=='simulated':scheduler['evidence_mode']='SIMULATED'
    if fault=='simulated_validation':scheduler['delivery_validation']['evidence_mode']='SIMULATED'
    monkeypatch.setattr(__import__('caption_review_web'),'_scheduler_readback',Mock(return_value=scheduler))
    assert tg._source_cache_eligibility(set())[source.parent.resolve()] is None
    tg._purge_expired_local_artifacts(emergency=True)
    assert source.exists()


def test_live_scheduling_readback_releases_only_eligible_source_and_is_reused(owned,complete,monkeypatch):
    p,tg,source,clock=owned
    state,scheduler=reviewed(complete);state['video_path']=str(source);p['write'](state)
    before=copy.deepcopy(state)
    preview=source.parent/'previews'/'finished.mp4';preview.parent.mkdir();preview.write_bytes(b'retained preview')
    readback=Mock(return_value=scheduler)
    monkeypatch.setattr(__import__('caption_review_web'),'_scheduler_readback',readback)
    assert tg._source_cache_eligibility(set())[source.parent.resolve()] is not None
    assert progress.summarize('request',p['read']())['completed_clips']==1
    unrelated=p['tmp']/'unrelated-cache.mp4';unrelated.write_bytes(b'preserve unrelated file')
    import os,time
    os.utime(unrelated,(time.time()-3*86400,)*2)
    tg._purge_expired_local_artifacts(paths={source})
    assert not source.exists() and preview.exists()
    assert unrelated.exists()
    assert p['read']()==before
    assert readback.call_count==1
    # A material manual/media revision cannot reuse the completion evidence.
    changed=copy.deepcopy(state);changed['result']['segments'][0]['rendered_bounds']=[1,5];p['write'](changed)
    assert progress._completion_receipt('request',changed) is None


def test_shared_unresolved_owner_still_protects_the_source(owned,complete,monkeypatch):
    p,tg,source,clock=owned
    state,scheduler=reviewed(complete);state['video_path']=str(source);p['write'](state)
    other={'video_path':str(source)}
    with tg._telegram_db() as db:
        db.execute('INSERT INTO telegram_requests VALUES(?,?,?,?,?,?,?,?,?,?,?)',
            ('other','other','chat','owner','awaiting_review','both','youtube','source',json.dumps(other),'now','now'))
    monkeypatch.setattr(__import__('caption_review_web'),'_scheduler_readback',Mock(return_value=scheduler))
    assert tg._source_cache_eligibility(set())[source.parent.resolve()] is None
    tg._purge_expired_local_artifacts()
    assert source.exists()
