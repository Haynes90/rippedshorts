"""Actual production boundary values, isolated source/semantic/Drive fixtures."""
import copy,json
import pytest
import boundary_specification as boundary
import render_recovery as recovery
from test_preview_vertical_slice import previews
from test_boundary_specification import passage


TAIL="You want to increase your plant's uptake of nutrients Uptake of nutrients increased means your plants will grow better"
RECIPE='Clear hook about feeding inoculants, gives the why and a practical dilution recipe — strong standalone value for growers.'


def contraction(p,eligible):
    s=p['read']();s['sentence_first_version']=1
    tail=('For a quick seed viability check, sprout ten seeds on a damp towel and count how many germinate.' if eligible else TAIL)
    reason='A self-contained practical seed-viability check.' if eligible else RECIPE
    setup='Long surrounding discussion before the seed-viability method.' if eligible else 'Original setup and method for a practical dilution recipe.'
    s['topic_source_segments']=[dict(start=168.27999877929688,end=240.9,text=setup),
        dict(start=241.17999267578125,end=247.33999633789062,text=tail)]
    s['result']['segments'][0].update(start=168.27999877929688,end=247.33999633789062,duration=79.05999755859375,
        transcript=setup+' '+tail,score=85,reason=reason,
        hook='A seed-viability check' if eligible else 'A practical dilution recipe',payoff='The useful method')
    p['write'](s);original=p['reviewer'].side_effect
    def review(units,start,end,aspect,*,context):
        value=original(units,start,end,aspect,context=context)
        if start<241:
            value.update(complete_start=False,first_word=1,last_word=1,setup_units=[1],payoff_units=[1])
        else:
            assert context['editorial_contract']['original']['score']==85
            assert context['editorial_contract']['original']['reason']==reason
            value['editorial'].update(eligible=eligible,original_message_preserved=eligible,
                hook_fulfilled=eligible,payoff_preserved=eligible,score=83 if eligible else 12,
                reason='A genuinely useful standalone final passage.' if eligible else 'The generic sentence no longer delivers the promised recipe.')
        return value
    p['reviewer'].side_effect=review;return s


def test_real_short1_contraction_cannot_inherit_recipe_score_or_reach_renderer(previews):
    p=previews;before=contraction(p,False)
    with pytest.raises(boundary.BoundaryReviewRequired) as caught:recovery.generate_preview('request','short',0,p['source'])
    assert 'EDITORIAL_VALUE_NOT_PRESERVED' in caught.value.diagnostics['reasons']
    assert p['read']()['result']==before['result'] and not p['calls'] and p['files'].create.call_count==0
    calls=p['reviewer'].call_count;assert calls==2
    with pytest.raises(recovery.RenderReviewRequired):recovery.generate_preview('request','short',0,p['source'])
    assert p['reviewer'].call_count==calls


def test_valuable_short_passage_uses_new_editorial_evidence_and_reuses_exact_media(previews):
    p=previews;contraction(p,True)
    clip=recovery.generate_preview('request','short',0,p['source'])
    assert clip['end']-clip['start']<10 and clip['score']==83 and clip['reason']!=RECIPE
    assert not any(k in clip for k in ('score_total','hook','payoff','why_it_will_work'))
    assert clip['construction_provenance']['editorial_origin']['score']==85
    spec=boundary.consume(p['source'],clip,'9:16');assert spec['semantic']['editorial']['eligible']
    changed=copy.deepcopy(clip);changed['reason']=RECIPE
    with pytest.raises(boundary.BoundaryReviewRequired,match='EDITORIAL_REVISION_CHANGED'):boundary.consume(p['source'],changed,'9:16')
    calls=p['reviewer'].call_count
    assert recovery.generate_preview('request','short',0,p['source'])==clip
    assert p['reviewer'].call_count==calls and len(p['calls'])==1 and p['files'].create.call_count==1


def test_existing_verified_media_does_not_get_reassessed_or_replaced(previews):
    p=previews;s=p['read']();s['result']['segments'][0].update(end=16.1,duration=6.1);p['write'](s)
    clip=recovery.generate_preview('request','short',0,p['source']);before=p['read']();calls=p['reviewer'].call_count
    assert recovery.generate_preview('request','short',0,p['source'])==clip
    assert p['read']()==before and p['reviewer'].call_count==calls and len(p['calls'])==1


def test_legacy_changed_excerpt_does_not_advertise_obsolete_recipe_or_change_media(previews):
    p=previews;original=p['reviewer'].side_effect;s=contraction(p,False);s.pop('sentence_first_version')
    p['reviewer'].side_effect=original
    c=s['result']['segments'][0]
    c.update(start=241.17999267578125,end=247.33999633789062,duration=6.160003662109375,transcript=TAIL,
        construction_provenance={'original_proposal':[168.27999877929688,247.33999633789062]})
    p['write'](s);clip=recovery.generate_preview('request','short',0,p['source']);before=p['read']()
    from finished_clip_review import clips
    item=clips('request',before)[0]
    assert item['finished'] and item['transcript']==TAIL and item['media_url']==clip['clip_url']
    assert 'not verified for this passage' in item['reason'] and 'practical dilution recipe' not in item['reason']
    assert p['read']()==before and len(p['calls'])==1 and p['files'].create.call_count==1


def test_changed_passage_editorial_is_part_of_the_existing_semantic_call(monkeypatch):
    import clip_completion as speech
    from types import SimpleNamespace
    calls=[]
    def post(url,**kwargs):
        calls.append((url,kwargs['json']))
        return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'choices':[{'message':{'content':'{"complete_thought":true}'}}]})
    monkeypatch.setattr(speech.requests,'post',post)
    context={'editorial_contract':{'original':{'reason':RECIPE},'playbook':'Existing quality-first rubric; reject generic advice.'}}
    result=speech.review_thought([{'word':TAIL,'start':241.18,'end':247.34}],241.18,247.34,'9:16',context=context)
    assert len(calls)==1 and calls[0][0]=='https://api.openai.com/v1/chat/completions'
    prompt=calls[0][1]['messages'][1]['content']
    assert json.dumps(context) in prompt and 'Existing quality-first rubric' in prompt and 'Do not impose a new minimum duration' in prompt
    assert result['provenance']['prompt_version']=='speech-context-editorial-v4'
