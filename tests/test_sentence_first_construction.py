import copy
import os
import pytest
from test_preview_vertical_slice import previews
from test_boundary_specification import passage
import boundary_specification as boundary
import render_recovery as recovery


def enable(p):
    s=p['read']();s['sentence_first_version']=1;p['write'](s)


def test_early_ending_is_constructed_at_measured_transition_before_freezing(previews,monkeypatch):
    p=previews;enable(p)
    s=p['read']();s['result']['segments'][0].update(end=15.7,duration=5.7);p['write'](s)
    def edge(source,t,side,duration):
        intervals=[[9.8,10.4]] if side=='start' else [[15.85,16.2]]
        return dict(edge=side,cut=t,quiet_intervals=intervals,measured_clear=any(a+.02<=t<=b-.02 for a,b in intervals))
    monkeypatch.setattr(boundary,'acoustic_edge',edge)
    clip=recovery.generate_preview('request','short',0,p['source'])
    assert clip['end']==pytest.approx(15.87)
    assert clip['boundary_specification_id']
    assert p['calls']==[('9:16',10.,15.87)]
    assert not p['read']()['candidate_reviews']
    assert recovery.generate_preview('request','short',0,p['source'])==clip
    assert len(p['calls'])==1 and p['files'].create.call_count==1


def test_missing_question_expands_using_verified_context_and_rechecks(previews,monkeypatch):
    p=previews;enable(p)
    original=p['reviewer'].side_effect
    def review(units,start,end,aspect,*,context):
        result=original(units,start,end,aspect,context=context)
        if start>0:result.update(first_word=0,setup_units=[0])
        return result
    p['reviewer'].side_effect=review
    clip=recovery.generate_preview('request','short',0,p['source'])
    assert clip['start']==0 and 'best way to store' in clip['transcript']
    assert p['reviewer'].call_count==2 and len(p['calls'])==1


def test_word_timed_complete_speech_does_not_require_silence(previews,monkeypatch):
    p=previews;enable(p);s=p['read']()
    s['topic_source_segments'][1]['words']=[{'word':'Complete answer.','start':10.2,'end':15.8}]
    p['write'](s)
    monkeypatch.setattr(boundary,'acoustic_edge',lambda *a:dict(quiet_intervals=[],measured_clear=False))
    assert recovery.generate_preview('request','short',0,p['source'])['boundary_specification_id']


def test_unresolvable_is_held_without_repeating_identical_model_input(previews,monkeypatch):
    p=previews;enable(p)
    monkeypatch.setattr(boundary,'acoustic_edge',lambda *a:dict(quiet_intervals=[],measured_clear=False))
    with pytest.raises(boundary.BoundaryReviewRequired):recovery.generate_preview('request','short',0,p['source'])
    count=p['reviewer'].call_count
    with pytest.raises(recovery.RenderReviewRequired):recovery.generate_preview('request','short',0,p['source'])
    assert p['reviewer'].call_count==count and not p['calls']


def test_selected_assets_and_legacy_requests_cannot_be_reconstructed(previews):
    p=previews;enable(p);s=p['read']();s['candidate_reviews']['0']={'status':'render_failed'};p['write'](s)
    before=copy.deepcopy(s)
    with pytest.raises(boundary.BoundaryReviewRequired,match='DRAFT_AUTHORITY_REQUIRED'):
        boundary.construct_preview_candidate('request','short',0,p['source'])
    assert p['read']()==before


def test_failed_candidate_gets_distinct_replacement_and_siblings_survive(previews,monkeypatch):
    p=previews;enable(p)
    import intelligence_workflow as workflow
    def review(units,start,end,aspect,*,context):
        a,b=context['included_unit_indices']
        return dict(complete_start=start>=20,complete_end=True,complete_thought=start>=20,
            context_sufficient=True,first_word=a,last_word=b,setup_units=[a],payoff_units=[b])
    p['reviewer'].side_effect=review
    proposed=dict(candidate_id='independent-replacement',candidate_number=99,start=30,end=45,duration=15,
                  transcript='Here is a complete, substantial seed-storage explanation.',reason='A useful complete explanation.')
    # Supply an actual bounded passage for this independent candidate.
    s=p['read']();s['topic_source_segments'][3]['end']=44.8;p['write'](s)
    calls=[]
    def discover(state,segments,source,lane,attempted,n):
        calls.append(n)
        return {'segments':[proposed]} if n==1 else {'segments':[]}
    monkeypatch.setattr(workflow,'replacement_candidates',discover)
    monkeypatch.setattr(__import__('review_navigation'),'notify_safely',lambda *a:None)
    result=recovery.preview_lane('request','short',p['source'])
    assert result.count('held')==1 and result.count('verified')==2
    assert calls==[1,2]
    assert p['read']()['result']['segments'][0]['candidate_id']=='selected-short'
    assert p['read']()['result']['segments'][-1]['candidate_id']=='independent-replacement'
    assert not p['read']()['candidate_reviews'] and not p['read']().get('schedule_master')


