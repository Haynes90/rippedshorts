import copy
import json
from unittest.mock import Mock

import pytest
import hierarchical_intelligence as hi
import production_intelligence as pi


@pytest.fixture
def isolated(tmp_path,monkeypatch):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff,'DB_PATH',tmp_path/'isolated.db')


SOURCE={'brand_id':'TDOG','show_id':'TDOG','source_video_id':'fixture'}
TIMELINE=[{'index':i,'start':i*60,'end':(i+1)*60,'text':'Private transcript.'} for i in range(12)]


def topic(key,start,parent=None):
    return {'key':key,'title':key,'summary':'Meaningful discussion','first_segment':start,
            'parent_key':parent,'related_keys':[],'tags':['gardening']}


def generate(stage,transcript,context,bounds,count):
    provenance={'prompt_version':hi.stage_version(stage)+':'+stage,'prompt_sha256':'tested-prompt-hash'}
    if stage=='episode':
        return {**{k:'Seed saving, advice and audience questions' for k in
                ('title','summary','beginning','middle','ending','discussion_flow')},'themes':['seed saving'],'provenance':provenance}
    if stage=='relationships':return {'relationships':[],'provenance':provenance}
    if stage=='conversations':
        topics=[topic('seed discussion',0),topic('audience questions',8)]
        for item in topics:item.pop('related_keys')
    elif stage=='topics':topics=[topic('introduction',bounds[0]),topic('practical advice',bounds[0]+2),topic('supporting context',bounds[0]+3)]
    else:return {'moments':[{'title':'Humorous transition','summary':'Complete exchange across chapters',
                    'category':'humor','first_segment':7,'last_segment':9}],'provenance':provenance}
    return {'range_contract':pi.TOPIC_RECIPE,'topics':topics,'provenance':provenance}


def test_full_top_down_understanding_hierarchy_and_transition_moments(isolated):
    calls=Mock(side_effect=generate)
    result=hi.analyze(TIMELINE,SOURCE,calls)
    assert [c.args[0] for c in calls.call_args_list]==['episode','conversations','relationships','topics','topics','moments']
    assert len(calls.call_args_list[0].args[1])==len(calls.call_args_list[1].args[1])==len(calls.call_args_list[-1].args[1])==12
    children=[c for c in calls.call_args_list if c.args[0]=='topics']
    assert [s['index'] for s in children[1].args[1]]==[8,9,10,11]  # GLOBAL indices retained
    roots=[t for t in result['topics'] if t['parent_key'] is None]
    assert [(t['first_segment'],t['last_segment']) for t in roots]==[(0,7),(8,11)]
    moment=result['episode_moments'][0]
    assert moment['first_segment']==7 and moment['last_segment']==9
    assert 'conversation:seed discussion' in moment['topic_keys'] and 'conversation:audience questions' in moment['topic_keys']
    assert result['episode']['themes']==['seed saving']
    assert len(result['stage_evidence'])==8
    assert hi.analyze(TIMELINE,SOURCE,Mock(side_effect=AssertionError('duplicate provider call')))==result


@pytest.mark.parametrize('bad_stage',['episode','conversations','topics','moments'])
def test_failure_is_durable_sanitized_and_completed_stages_never_repeat(isolated,bad_stage):
    def bad(stage,*args):
        value=generate(stage,*args)
        if stage==bad_stage:
            if stage=='episode':value['summary']=''
            elif stage=='moments':value['moments'][0].update(first_segment=9,last_segment=7)
            else:value['topics'][0]['first_segment']='private invalid reference'
        return value
    calls=Mock(side_effect=bad)
    with pytest.raises(pi.TopicValidationError):hi.analyze(TIMELINE,SOURCE,calls)
    count=calls.call_count
    with pytest.raises(ValueError):hi.analyze(TIMELINE,SOURCE,calls)
    assert calls.call_count==count
    with pi.database() as db:
        rows=db.execute('select * from source_analyses').fetchall()
        failed=[r for r in rows if r['status']=='FAILED']
        diagnostic=json.loads(failed[0]['result'])['validation_failure']
        assert diagnostic['stage']==bad_stage and diagnostic['stage_analysis_id']
        assert 'private invalid reference' not in failed[0]['result'] and 'Private transcript' not in failed[0]['result']
        assert len([r for r in rows if r['status']=='PASSED'])==count-1+(2 if bad_stage=='moments' else 0)


def test_child_outside_parent_and_missing_conversation_beginning_fail(isolated):
    def bad(stage,*args):
        result=generate(stage,*args)
        if stage=='topics':result['topics'][1]['first_segment']=11
        return result
    with pytest.raises(pi.TopicValidationError):hi.analyze(TIMELINE,SOURCE,bad)


def test_successful_children_survive_one_explicit_failed_child_reopen(isolated):
    def bad(stage,transcript,context,bounds,count):
        value=generate(stage,transcript,context,bounds,count)
        if stage=='topics' and bounds[0]==8:value['topics'][0]['first_segment']=0
        return value
    with pytest.raises(pi.TopicValidationError):hi.analyze(TIMELINE,SOURCE,bad)
    with pi.database() as db:
        failed=db.execute("select analysis_id from source_analyses where status='FAILED'").fetchone()[0]
    pi.retry_failed(failed)  # single explicit isolated authorization; not a broad reset
    calls=Mock(side_effect=generate)
    result=hi.analyze(TIMELINE,SOURCE,calls)
    assert [c.args[0] for c in calls.call_args_list]==['topics','moments']
    assert result['topics'] and len(result['stage_evidence'])==8


def test_outer_map_retains_optional_failure_but_can_continue_without_moments(isolated,monkeypatch):
    def bad(stage,*args):
        value=generate(stage,*args)
        if stage=='moments':value['moments'][0]['last_segment']=99
        return value
    monkeypatch.setattr(hi,'provider',bad)
    segments=[{k:v for k,v in s.items() if k!='index'} for s in TIMELINE]
    first=hi.mapping(SOURCE,segments)
    assert first['analysis_path']=='topics-only' and first['episode_moments']==[]
    assert first['metadata_failures'][0]['diagnostics']['stage']=='moments'
    assert hi.mapping(SOURCE,segments)==first
    with pi.database() as db:
        saved=json.loads(db.execute("select result from source_analyses where status='FAILED'").fetchone()[0])
    assert saved['validation_failure']['stage']=='moments'
    assert pi.analysis_recovery_blocked({'analysis_validation_hold':saved['validation_failure']})


def test_compiled_map_and_moments_reach_existing_candidate_provenance(isolated,monkeypatch):
    monkeypatch.setattr(hi,'provider',generate)
    result=hi.mapping(SOURCE,TIMELINE)
    assert result['analysis_version']==hi.VERSION
    assert result['episode_moments'] and result['topics'][2]['parent_topic_id']
    for lane in ('9:16','16:9'):
        clips={'segments':[{'start':420,'end':600,'category':'humor','score':90}]}
        context={'examples':[{'evidence_id':'scoped-owner-feedback'}],'confidence':0.5}
        pi.candidate_records(result,lane,clips,context)
        evidence=clips['segments'][0]['selection_provenance']
        assert len(evidence['hierarchy_stage_ids'])==8
        assert evidence['learning_scope']=={'brand_id':'TDOG','show_id':'TDOG','format':lane}
        assert evidence['learning_evidence_ids']==['scoped-owner-feedback']


def test_provider_schema_constrains_global_child_indices_and_does_not_request_ends():
    contract=hi.schema('topics',1189,(1120,1188))
    fields=contract['properties']['topics']['items']['properties']
    assert fields['first_segment']=={'type':'integer','minimum':1120,'maximum':1188}
    assert 'last_segment' not in fields and 'memorable_moments' not in fields
    assert 'ACROSS conversation transitions' in hi.instructions('moments',(0,1188))
