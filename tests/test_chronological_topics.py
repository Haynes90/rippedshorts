"""Isolated conversational contracts, not claims of live provider/media success."""
import copy
import json
from unittest.mock import Mock

import pytest
import hierarchical_intelligence as hi
import production_intelligence as pi
from test_hierarchical_intelligence import SOURCE, TIMELINE, generate, isolated, topic


def returning_topics():
    items=[topic('seed-storage',0),topic('viewer-question',3),
           topic('funny-exchange',5),topic('return-to-storage',7)]
    items[0]['related_keys']=['return-to-storage']  # verified forward reference
    items[-1]['related_keys']=['seed-storage']     # returning subject, not child
    return {'range_contract':pi.TOPIC_RECIPE,'topics':items,
            'provenance':{'prompt_version':'chronological-topics-v2:topics','prompt_sha256':'fixture-sha'}}


def test_chronology_then_thematic_links_covers_whole_conversation_without_mutation():
    value=returning_topics(); original=copy.deepcopy(value)
    result=hi.chronological_topics(value,12,(0,11))
    assert value==original
    assert [(t['first_segment'],t['last_segment']) for t in result['topics']]==[(0,2),(3,4),(5,6),(7,11)]
    assert result['topics'][0]['related_keys']==['return-to-storage']
    assert result['topics'][-1]['related_keys']==['seed-storage']
    assert all(t['parent_key'] is None for t in result['topics'])


@pytest.mark.parametrize('kind,mutate',[
    ('TEMPORAL_PARENT_NOT_ALLOWED',lambda p:p['topics'][-1].update(parent_key='seed-storage')),
    ('TEMPORAL_PARENT_NOT_ALLOWED',lambda p:p['topics'][-1].update(parent_key=42)),
    ('UNKNOWN_TARGET',lambda p:p['topics'][0].update(related_keys=['private unknown source words'])),
    ('REFERENCE_TYPE',lambda p:p['topics'][0].update(related_keys=[False])),
    ('SELF_REFERENCE',lambda p:p['topics'][0].update(related_keys=['seed-storage'])),
    ('DUPLICATE_LINK',lambda p:p['topics'][0].update(related_keys=['return-to-storage']*2)),
    ('LINK_COLLECTION_TYPE',lambda p:p['topics'][0].update(related_keys=None)),
])
def test_invalid_links_parent_contract_and_safe_durable_hold(isolated,kind,mutate):
    value=returning_topics();mutate(value);original=copy.deepcopy(value)
    calls=Mock(side_effect=lambda:hi.chronological_topics(value,12,(0,11)))
    with pytest.raises(pi.TopicValidationError) as caught:pi.cached_analysis('topics:new',calls)
    d=caught.value.diagnostics
    assert d['failure_kind']==kind and d['prompt_sha256']=='fixture-sha'
    assert d['topic_key']['sha256'] and d['rejected_reference']['type']
    assert 'private unknown source words' not in json.dumps(d)
    with pi.database() as db:
        saved=json.loads(db.execute('select result from source_analyses').fetchone()[0])
    assert saved['validation_failure']==d and value==original
    with pytest.raises(ValueError) as held:pi.cached_analysis('topics:new',calls)
    state={};assert pi.hold_analysis_failure(state,held.value)
    assert pi.analysis_recovery_blocked(state) and calls.call_count==1


@pytest.mark.parametrize('mutate',[
    lambda p:p['topics'][0].update(first_segment=1),
    lambda p:p['topics'][2].update(first_segment=2),
    lambda p:p['topics'][2].update(first_segment=True),
    lambda p:p['topics'][2].update(last_segment=4),
    lambda p:p['topics'][2].update(key='seed-storage'),
    lambda p:p['topics'][2].update(first_segment=12),
])
def test_invalid_starts_and_provider_ends_never_repaired(mutate):
    value=returning_topics();mutate(value);original=copy.deepcopy(value)
    with pytest.raises(pi.TopicValidationError):hi.chronological_topics(value,12,(0,11))
    assert value==original


def test_schema_removes_ambiguous_topic_parent_without_adding_model_stage():
    item=hi.schema('topics',1189,(130,350))['properties']['topics']['items']['properties']
    assert item['parent_key']=={'type':'null'}
    assert item['first_segment']=={'type':'integer','minimum':130,'maximum':350}
    prompt=hi.instructions('topics',(130,350))
    assert 'CHRONOLOGICALLY first' in prompt and 'complete question and answer' in prompt
    assert 'After identifying ALL topic keys' in prompt and 'Returning subjects' in prompt


def test_recovery_reuses_exact_legacy_passes_without_reopening_legacy_failure(isolated,monkeypatch):
    old_versions={k:v for k,v in hi.STAGE_VERSIONS.items() if k!='topics'}
    def failed_old(stage,transcript,context,bounds,count):
        value=generate(stage,transcript,context,bounds,count)
        if stage=='topics' and bounds[0]==8:
            raise pi.TopicValidationError('Semantic hierarchy must follow depth-first parent order',
                                          topic_position=10)
        return value
    with monkeypatch.context() as m:
        m.setattr(hi,'STAGE_VERSIONS',old_versions)
        with pytest.raises(pi.TopicValidationError):hi.analyze(TIMELINE,SOURCE,failed_old)
    with pi.database() as db:
        before={r['analysis_id']:dict(r) for r in db.execute('select * from source_analyses')}
    calls=Mock(side_effect=generate)
    result=hi.analyze(TIMELINE,SOURCE,calls)
    assert [c.args[0] for c in calls.call_args_list]==['topics','moments']
    assert calls.call_args_list[0].args[3]==(8,11)
    assert len(result['stage_evidence'])==7  # legacy accepted topics need no new link checkpoint
    assert ':hierarchical-source-v1:topics:' in result['stage_evidence'][3]['analysis_id']
    assert ':chronological-spans-v3:topics:' in result['stage_evidence'][4]['analysis_id']
    with pi.database() as db:
        after={r['analysis_id']:dict(r) for r in db.execute('select * from source_analyses')}
    assert all(after[key]==row for key,row in before.items())
    assert hi.analyze(TIMELINE,SOURCE,Mock(side_effect=AssertionError('repeat')))==result


def test_complete_conversational_path_reaches_both_real_selectors_and_learning(isolated,monkeypatch):
    """Model and media I/O mocked; real map, selectors, timing checks and ledger."""
    import main
    import telegram_intake as tg
    import intelligence_workflow as workflow
    import selection_specs
    texts=[
        'Today we are discussing how to save tomato seeds.',
        'Which tomatoes should gardeners choose for seed saving?',
        'Choose ripe open-pollinated fruit from healthy plants.',
        'How do you remove the gel from tomato seeds?',
        'Ferment briefly in water, then rinse and dry thoroughly.',
        'I labelled my seeds dinner, so nobody planted them.',
        'That is one way to keep the family from eating your seed collection!',
        'A viewer asks whether damp seeds can go into a sealed jar.',
        'No, moisture can cause mold and spoil the entire collection.',
        'Returning to storage, use fully dry seeds in a cool dark place.',
        'Test a small sample for germination before planting the whole packet.',
        'That completes our seed storage discussion. Thank you for watching.',
    ]
    segments=[{'start':i*30.,'end':(i+1)*30.,'duration':30.,'text':text} for i,text in enumerate(texts)]
    state={'show_id':'TDOG','parsed':{'video_id':'conversational-fixture'},'vid_title':'Seed saving Q&A'}
    def provider(stage,transcript,context,bounds,count):
        if stage=='episode':return generate(stage,transcript,context,bounds,count)
        if stage=='relationships':
            a,b=context['verified_conversations']
            return {'relationships':[{'conversation_id':a['conversation_id'],'related_ids':[b['conversation_id']]}]}
        if stage=='moments':
            return {'moments':[{'first_segment':5,'last_segment':6,'title':'Seed dinner joke',
                               'summary':'Complete humorous exchange across the conversation transition','category':'humor'}]}
        starts=(0,6) if stage=='conversations' else (bounds[0],bounds[0]+3)
        nodes=[topic('seed-saving-question',starts[0]),topic('returning-storage',starts[1])]
        if stage=='conversations':
            for node in nodes:node.pop('related_keys')
        else:
            nodes[0]['related_keys']=[nodes[1]['key']]
            nodes[1]['related_keys']=[nodes[0]['key']]
        return {'range_contract':pi.TOPIC_RECIPE,'topics':nodes}
    monkeypatch.setattr(hi,'provider',provider)
    monkeypatch.setattr(selection_specs,'highlight_framework',lambda:'Approved minimum three-minute complete explanation contract.')
    # No retained video is accessed in this isolated contract test.
    monkeypatch.setattr(tg,'_preflight_candidates',lambda video,clips:clips)
    for lane in ('9:16','16:9'):
        pi.record('feedback','past-review-'+lane,{'candidate_id':'past-'+lane,'brand_id':'TDOG',
            'show_id':'TDOG','format':lane,'decision':'approved','category':'humor' if lane=='9:16' else 'education',
            'feedback':'Keep the complete useful answer and payoff.'})
    with pi.database() as db:db.execute("UPDATE intelligence_records SET synced_digest=digest WHERE kind='feedback'")
    captured={}
    def short(segments,prompt):
        captured['9:16']=prompt
        return {'segments':[{'start':150.,'end':210.,'transcript':' '.join(texts[5:7]),
            'category':'humor','score':85,'reason':'A complete humorous exchange, not a chapter excerpt.'}]}
    def highlight(segments,**kw):
        captured['16:9']=kw['intelligence_context']
        return [{'start':0.,'end':180.,'title':'Saving tomato seeds','score':90,
                 'highlight_type':'education','reason':'A complete explanation with questions and responses.'}]
    monkeypatch.setattr(main,'call_openai_for_clips',short)
    monkeypatch.setattr(tg,'_topic_break_suggestions',highlight)
    future,mapping=workflow.start(state,segments,'isolated-no-media.mp4','Approved strong hook and complete-thought rubric.')
    shorts=future.result(timeout=5)
    highlights=workflow.prepared_highlights(state,segments,'isolated-no-media.mp4')
    assert len(shorts['segments'])==len(highlights['segments'])==1
    assert shorts['segments'][0]['start']==150 and shorts['segments'][0]['end']==210
    assert highlights['segments'][0]['duration']>=180
    assert not shorts['validation_rejections']
    assert mapping['episode_moments'][0]['first_segment']==5
    assert len(mapping['stage_evidence'])==8
    for lane,result in [('9:16',shorts),('16:9',highlights)]:
        assert 'hierarchical-source-v4' in captured[lane]
        assert 'returning-storage' in captured[lane] and 'Seed dinner joke' in captured[lane]
        assert 'Keep the complete useful answer and payoff.' in captured[lane]
        candidate=result['segments'][0]
        assert candidate['learning_influence']['score_adjustment']>0
        assert candidate['selection_provenance']['learning_evidence_ids']==['past-review-'+lane]
        assert len(candidate['selection_provenance']['hierarchy_stage_ids'])==8
        assert candidate['candidate_id'] and candidate['topic_ids']
    assert 'SENTENCE-LEVEL CLIP FARMING' in captured['9:16']
    assert 'focused topic arcs' in captured['16:9']
    assert 'candidate_reviews' not in state and 'topic_reviews' not in state
