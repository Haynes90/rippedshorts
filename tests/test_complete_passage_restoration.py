"""Replay actual retained transcript/acoustic evidence in isolated storage.

Semantic responses and Drive are explicit doubles. These tests prove resolver
behavior, not new live editorial approval, source listening or production renders.
"""
import copy,json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import boundary_specification as boundary
import render_recovery as recovery
from test_boundary_specification import passage
from test_preview_vertical_slice import previews

EVIDENCE=json.loads((Path(__file__).parent/'fixtures/tdog_boundary_replay.json').read_text(encoding='utf-8'))


def actual_candidate(p,asset,monkeypatch):
    original=next(c for c in EVIDENCE['original_candidates'] if c['asset_id'].endswith(':'+asset))
    lane,number=asset.split(':');key='result' if lane=='short' else 'topic_result'
    candidate={k:v for k,v in original.items() if k not in ('asset_id','format')}
    candidate.update(candidate_id='fixture-'+asset,candidate_number=int(number),duration=candidate['end']-candidate['start'],score=85)
    s=p['read']();s.update(sentence_first_version=1,topic_source_segments=copy.deepcopy(EVIDENCE['segments']))
    s[key]['segments']=[candidate];p['write'](s)
    monkeypatch.setattr(boundary.speech,'media_duration',lambda _:2600.124082)
    return lane,candidate


@pytest.mark.parametrize('asset',['short:2','short:3','short:5','highlight:1','highlight:2'])
def test_real_held_passages_resolve_recorded_acoustic_margins_without_losing_speech(previews,monkeypatch,asset):
    p=previews;lane,original=actual_candidate(p,asset,monkeypatch)
    assessment=next(a for a in EVIDENCE['assessments'] if a['asset']==asset)
    assert assessment['reasons']==['ACOUSTIC_EDGE_UNRESOLVED']
    assert all(assessment['semantic'][k] is True for k in ('complete_start','complete_end','complete_thought','context_sufficient'))
    def measured(source,cut,edge,duration):
        saved=assessment['acoustic'][edge]
        assert saved['scan'][0]<=cut<=saved['scan'][1]
        intervals=saved['quiet_intervals']
        return dict(edge=edge,cut=cut,scan=saved['scan'],quiet_intervals=intervals,
            measured_clear=any(a+.02<=cut<=b-.02 for a,b in intervals),resolution_uncertainty_seconds=.02)
    monkeypatch.setattr(boundary,'acoustic_edge',measured)
    review=p['reviewer'].side_effect
    def semantic(units,start,end,aspect,*,context):
        value=review(units,start,end,aspect,context=context)
        assert context['editorial_contract']['original']['reason']==original['reason']
        return value
    p['reviewer'].side_effect=semantic
    rendered=recovery.generate_preview('request',lane,0,p['source'])
    spec=boundary.consume(p['source'],rendered,'9:16' if lane=='short' else '16:9')
    assert spec['status']=='VALIDATED' and all(e['measured_clear'] for e in spec['acoustic'].values())
    captured=spec['speech_units']
    assert rendered['start']<=min(u['start'] for u in captured) and rendered['end']>=max(u['end'] for u in captured)
    assert rendered['transcript']==' '.join(u['word'] for u in captured)
    assert len(rendered['transcript'].split())>30
    assert rendered['end']-rendered['start']<=90 if lane=='short' else rendered['end']-rendered['start']>=180
    assert p['reviewer'].call_count==1 and len(p['calls'])==1
    assert recovery.generate_preview('request',lane,0,p['source'])==rendered
    assert p['reviewer'].call_count==1 and len(p['calls'])==1 and p['files'].create.call_count==1
    assert not p['read']().get('website_clip_reviews') and not p['read']().get('schedule_master')
    tampered=copy.deepcopy(rendered);tampered['construction_provenance']['intended_unit_range'][0]+=1
    with pytest.raises(boundary.BoundaryReviewRequired,match='INTENDED_SPEECH_REVISION_CHANGED'):
        boundary.consume(p['source'],tampered,'9:16' if lane=='short' else '16:9')


def test_real_short1_original_question_and_recipe_are_available_before_any_contraction(previews,monkeypatch):
    p=previews;lane,original=actual_candidate(p,'short:1',monkeypatch);before=p['read']()
    model=p['reviewer'].side_effect
    def old_tail_choice(units,start,end,aspect,*,context):
        assert context['editorial_contract']['original']['reason']==original['reason']
        combined=' '.join(u['word'] for u in units)
        assert 'Why did I get molasses in general' in combined and 'one part molasses to two parts water' in combined
        value=model(units,start,end,aspect,context=context)
        a=next(i for i,u in enumerate(units) if u['start']==241.17999267578125)
        b=max(i for i,u in enumerate(units) if u['end']<=247.33999633789062)
        value.update(first_word=a,last_word=b,setup_units=[a],payoff_units=[b])
        value['editorial'].update(eligible=False,original_message_preserved=False,hook_fulfilled=False,payoff_preserved=False)
        return value
    p['reviewer'].side_effect=old_tail_choice
    with pytest.raises(boundary.BoundaryReviewRequired) as held:
        recovery.generate_preview('request',lane,0,p['source'])
    assert 'EDITORIAL_VALUE_NOT_PRESERVED' in held.value.diagnostics['reasons']
    assert p['reviewer'].call_count==1 and not p['calls'] and p['files'].create.call_count==0
    assert p['read']()['result']==before['result']
    with pytest.raises(recovery.RenderReviewRequired):recovery.generate_preview('request',lane,0,p['source'])
    assert p['reviewer'].call_count==1


def test_valid_setup_reference_outside_proposed_span_expands_instead_of_invalid_reference(previews):
    p=previews;s=p['read']();s['sentence_first_version']=1;p['write'](s)
    review=p['reviewer'].side_effect
    def missing_setup(units,start,end,aspect,*,context):
        result=review(units,start,end,aspect,context=context)
        if start>0:result['setup_units']=[0]  # Full question preceding the selected answer.
        return result
    p['reviewer'].side_effect=missing_setup
    clip=recovery.generate_preview('request','short',0,p['source'])
    assert clip['start']==0 and 'best way to store' in clip['transcript']
    assert p['reviewer'].call_count==2 and len(p['calls'])==1


def test_quiet_inside_intended_speech_cannot_be_used_to_cut_it_away(previews,monkeypatch):
    p=previews;s=p['read']();s['sentence_first_version']=1;p['write'](s)
    monkeypatch.setattr(boundary,'acoustic_edge',lambda source,cut,edge,duration:
        dict(edge=edge,cut=cut,quiet_intervals=[[10.3,10.6]] if edge=='start' else [[15.2,15.5]],measured_clear=False))
    before=p['read']()
    with pytest.raises(boundary.BoundaryReviewRequired):recovery.generate_preview('request','short',0,p['source'])
    assert p['read']()['result']==before['result'] and not p['calls'] and p['files'].create.call_count==0


def test_gallery_accounts_for_seven_without_admitting_five_holds_or_blocking_ready_reviews(previews):
    p=previews;s=p['read']();s['result']['segments'][0].update(end=16.1,duration=6.1);p['write'](s)
    for i in (0,1):recovery.generate_preview('request','short',i,p['source'])
    s=p['read']()
    for lane,key,count in [('short','result',3),('highlight','topic_result',1)]:
        for n in range(count):
            c=copy.deepcopy(s[key]['segments'][0]);c.update(candidate_id=f'{lane}-held-{n}',candidate_number=30+n)
            s[key]['segments'].append(c)
    p['write'](s)
    for lane,key in [('short','result'),('highlight','topic_result')]:
        for i,c in enumerate(s[key]['segments']):
            if lane=='short' and i<2:continue
            identity=recovery.asset_id('request',lane,c,i);t=recovery.preview_ticket('request',s,identity)
            with boundary.evidence.database() as db:
                recovery._put(db,'preview_hold','preview:'+identity+':'+t['revision']+':hold',dict(category='BoundaryReviewRequired',diagnostics={'category':'NO_EVIDENCE_SUPPORTED_PROGRESS'}))
    before=p['read']()
    with TestClient(p['app']) as client:
        d=client.get('/api/ripped-shorts/review/'+p['token']).json()
        assert len(d['clips'])==2 and len(d['preview_statuses'])==7
        assert (d['preview_review_completed'],d['preview_review_total'])==(0,2)
        html=client.get('/review/'+p['token']).text
        assert html.count('class="clip-card"')==2 and html.count('class="preview-status"')==5
        assert '2/5 rendered previews available' in html and '0/2 rendered previews available' in html
        assert 'awaiting boundary resolution' in html and 'NO_EVIDENCE_SUPPORTED_PROGRESS' not in html
        assert html.count('I watched this finished clip')==2
    assert p['read']()==before and len(p['calls'])==2
