"""Isolated semantic-boundary and actual watchdog-function regression tests."""
import ast
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import threading
from unittest.mock import Mock

import pytest
import production_intelligence as pi


@pytest.fixture
def db(tmp_path, monkeypatch):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'isolated.db')


def boundary(key, first, parent=None, **extra):
    return dict(key=key, title='Semantic '+key, first_segment=first, parent_key=parent,
                related_keys=[], subtopics=[], questions_responses=[], memorable_moments=[], **extra)


def response(*topics):
    return {'range_contract':pi.TOPIC_RECIPE, 'topics':list(topics),
            'provenance':{'prompt_version':pi.TOPIC_RECIPE, 'prompt_sha256':'actual-test-prompt'}}


SOURCE = {'brand_id':'TDOG', 'source_video_id':'retained-test-source', 'show_id':'TDOG'}
TIMELINE = [{'start':i*10, 'end':(i+1)*10, 'text':'Private timed speech.'} for i in range(12)]


def test_semantic_transitions_derive_unequal_ranges_and_preserve_hierarchy(db):
    original=response(boundary('seed-saving',0), boundary('drying',2,'seed-saving'),
                      boundary('storage',5,'seed-saving'), boundary('audience-questions',9))
    original['topics'][0]['memorable_moments']=[{'first_segment':7,'last_segment':8,'title':'Unexpected lesson'}]
    mapping=pi.topic_map(SOURCE,TIMELINE,lambda _:original)
    topics=mapping['topics']
    assert [(t['first_segment'],t['last_segment']) for t in topics]==[(0,8),(2,4),(5,8),(9,11)]
    assert topics[1]['parent_topic_id']==topics[0]['topic_id']
    assert topics[2]['transcript_ref']['last_segment']==8
    assert topics[-1]['end']==120
    assert all('last_segment' not in t for t in original['topics'])  # no mutation of provider evidence
    assert topics[0]['memorable_moments']==original['topics'][0]['memorable_moments']


@pytest.mark.parametrize('topics',[
    [], [boundary('a',1)], [boundary('a',False)], [boundary('a','0')],
    [boundary('a',-1)], [boundary('a',12)],
    [boundary('a',0),boundary('b',0)], [boundary('a',0),boundary('b',8),boundary('c',6)],
    [boundary('a',0),boundary('a',5)], [boundary('a',0,'missing')],
    [boundary('a',0),boundary('b',5),boundary('c',6,'a')],  # closed parent
    [boundary('a',0),boundary('b',9,'a'),boundary('c',8)],  # child beyond parent's derived end
    [boundary('a',0),boundary('b',2,'a'),boundary('c',1,'b')],
    [boundary('a',0,last_segment=11)],  # never silently ignore legacy endpoints
    [boundary('a',0),boundary('b',5,last_segment=3)],  # actual reversed interval shape
])
def test_invalid_semantic_sequences_are_held_without_repairs(db, topics):
    analyze=Mock(return_value=response(*topics))
    with pytest.raises(pi.TopicValidationError):
        pi.topic_map(SOURCE,TIMELINE,analyze)
    with pi.database() as conn:
        row=conn.execute('select * from source_analyses').fetchone()
        assert row['status']=='FAILED'
        diagnostic=json.loads(row['result'])['validation_failure']
        assert diagnostic['prompt_sha256']=='actual-test-prompt'
        assert 'Private timed speech' not in row['result'] and 'Semantic a' not in row['result']
        assert not conn.execute("select 1 from intelligence_records").fetchone()
    with pytest.raises(ValueError,match='Saved analysis failed') as saved:
        pi.topic_map(SOURCE,TIMELINE,analyze)
    assert saved.value.validation_held
    analyze.assert_called_once()


@pytest.mark.parametrize('field',['subtopics','questions_responses','memorable_moments'])
def test_nested_refs_checked_against_derived_parent_without_adjustment(db,field):
    data=response(boundary('a',0),boundary('b',6))
    data['topics'][0][field]=[{'first_segment':5,'last_segment':6}]
    with pytest.raises(pi.NestedReferenceError) as error:
        pi.topic_map(SOURCE,TIMELINE,lambda _:data)
    assert error.value.diagnostics['parent_last_segment']==5
    assert error.value.diagnostics['last_segment']['value']==6


def test_full_1189_segment_coverage_uses_model_selected_starts_not_time_slices(db):
    timeline=[{'start':i*2,'end':i*2+2,'text':'Verified fixture speech.'} for i in range(1189)]
    starts=[0,61,143,410,877,1120,1161]
    mapping=pi.topic_map(SOURCE,timeline,lambda _:response(*(boundary(str(i),n) for i,n in enumerate(starts))))
    assert [t['first_segment'] for t in mapping['topics']]==starts
    assert mapping['topics'][-2]['last_segment']==1160
    assert mapping['topics'][-1]['last_segment']==1188
    assert set().union(*(set(range(t['first_segment'],t['last_segment']+1)) for t in mapping['topics']))==set(range(1189))


def watchdog(rows):
    """Execute the production watchdog function with isolated rows and no workers."""
    source=(Path(__file__).resolve().parents[1]/'telegram_intake.py').read_text(encoding='utf-8')
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_resume_stale_jobs')
    database=Mock()
    database.execute.return_value.fetchall.return_value=rows
    @contextmanager
    def isolated_db():
        yield database
    save,executor=Mock(),Mock()
    env={'_ensure_storage_headroom':lambda:{'free_bytes':10**15},'os':os,
         '_LOCK':threading.RLock(),'_telegram_db':isolated_db,'datetime':datetime,'timezone':timezone,
         'timedelta':timedelta,'json':json,'logger':Mock(),'_TERMINAL_JOB_STAGES':{'complete','superseded'},
         '_save':save,'RENDER_EXECUTOR':executor,'now':lambda:'isolated-time',
         '_process':Mock(),'_handoff_shorts_to_schedule_master':Mock(),'_queue_missing_approved_renders':Mock()}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-watchdog>','exec'),env)
    return env['_resume_stale_jobs'],save,executor


def job(state):
    return dict(request_id='same-request',status='error',state_json=json.dumps(state),
                updated_at=(datetime.now(timezone.utc)-timedelta(hours=2)).isoformat(),chat_id='isolated')


def test_legacy_failed_map_not_revisited_even_with_machine_stage(db):
    try:
        pi.cached_analysis('old-map',lambda:(_ for _ in ()).throw(pi.TopicReferenceError(12,1161,1120,1189,{})))
    except pi.TopicReferenceError:
        pass
    state={'stage':'error','source_intelligence':{'map_id':'old-map'},'watchdog_resume_count':0}
    run,save,executor=watchdog([job(state)])
    for _ in range(5):
        assert run()==0
    save.assert_not_called();executor.submit.assert_not_called()


def test_one_explicit_recovery_failure_cannot_schedule_more_attempts(db):
    analyze=Mock(side_effect=pi.TopicValidationError('Invalid semantic boundary contract'))
    for authorized_attempt in range(2):
        if authorized_attempt:
            pi.retry_failed('same-map')  # exactly one explicit reopening in this isolated test
        with pytest.raises(pi.TopicValidationError) as failure:
            pi.cached_analysis('same-map',analyze)
        state={'source_intelligence':{'map_id':'same-map'}}
        assert pi.hold_analysis_failure(state,failure.value)
        run,save,executor=watchdog([job(state)])
        for _ in range(4):assert run()==0
        save.assert_not_called();executor.submit.assert_not_called()
    assert analyze.call_count==2  # original attempt + one explicitly authorized recovery only
    with pi.database() as conn:
        assert conn.execute('select count(*) from analysis_failures').fetchone()[0]==2


def test_transient_error_is_not_content_hold_and_watchdog_still_bounds_recovery(db):
    error=TimeoutError('Transient operation')
    assert not pi.hold_analysis_failure({},error)
    state={'stage':'processing','watchdog_resume_count':0}
    run,save,executor=watchdog([job(state)])
    assert run()==1 and executor.submit.call_count==1
    state['watchdog_resume_count']=3
    run,save,executor=watchdog([job(state)])
    assert run()==0
    executor.submit.assert_not_called()


def test_validation_hold_precedes_even_render_recovery(db):
    state={'stage':'render_recovery','analysis_validation_hold':{'analysis_id':'held'},
           'candidate_reviews':{'0':{'status':'render_failed'}}}
    run,save,executor=watchdog([job(state)])
    assert run()==0
    save.assert_not_called();executor.submit.assert_not_called()


def test_bounded_recovery_in_flight_and_failed_never_auto_resubmits(db):
    for stage in ('processing','error','render_recovery'):
        state={'stage':stage,'analysis_recovery':{'mode':'owner_bounded','attempt':1}}
        run,save,executor=watchdog([job(state)])
        assert run()==0
        save.assert_not_called();executor.submit.assert_not_called()


def test_held_newest_source_cannot_revive_an_older_duplicate(db):
    held=job({'stage':'error','parsed':{'video_id':'same-source'},'analysis_validation_hold':{'analysis_id':'held'}})
    older=job({'stage':'processing','parsed':{'video_id':'same-source'}})
    older['request_id']='older-protected-request'
    run,save,executor=watchdog([held,older])
    assert run()==0
    save.assert_not_called();executor.submit.assert_not_called()


def test_new_recipe_preserves_failed_evidence_and_successful_legacy_map(db):
    legacy=lambda _: {'topics':[boundary('a',0,last_segment=11)]}
    original=pi.topic_map(SOURCE,TIMELINE,legacy)
    def upgraded(_):raise AssertionError('Do not replace completed source analysis')
    upgraded.analysis_recipe=pi.TOPIC_RECIPE
    assert pi.topic_map(SOURCE,TIMELINE,upgraded)==original


def test_new_recipe_is_separate_not_an_implicit_reset_of_exhausted_history(db):
    failed=Mock(side_effect=pi.TopicReferenceError(12,1161,1120,1189,{}))
    for attempt in range(3):
        if attempt:pi.retry_failed(pi.identity(pi.VERSION,SOURCE['brand_id'],SOURCE['source_video_id'],pi.identity(pi.normalize_transcript(TIMELINE))))
        with pytest.raises(pi.TopicReferenceError):pi.topic_map(SOURCE,TIMELINE,failed)
    def reviewed_recipe(_):return response(boundary('a',0))
    reviewed_recipe.analysis_recipe=pi.TOPIC_RECIPE
    mapping=pi.topic_map(SOURCE,TIMELINE,reviewed_recipe)
    with pi.database() as conn:
        assert conn.execute('select count(*) from analysis_failures').fetchone()[0]==3
        assert conn.execute('select status from source_analyses where analysis_id=?',(mapping['map_id'],)).fetchone()[0]=='FAILED'
    assert pi.analysis_recovery_blocked({'source_intelligence':{'map_id':mapping['map_id']}})
    assert not pi.analysis_recovery_blocked({'source_intelligence':{'map_id':mapping['map_id'],
                  'analysis_id':mapping['map_id']+':'+pi.TOPIC_RECIPE}})


def test_both_candidate_lanes_and_learning_accept_derived_map(db):
    mapping=pi.topic_map(SOURCE,TIMELINE,lambda _:response(boundary('seeds',0),boundary('questions',8)))
    for lane in ('9:16','16:9'):
        result={'segments':[{'start':20,'end':50,'category':'education','score':80}]}
        context={'examples':[{'evidence_id':'saved-owner-decision'}],'confidence':0.5}
        pi.candidate_records(mapping,lane,result,context)
        candidate=result['segments'][0]
        assert candidate['topic_ids']==[mapping['topics'][0]['topic_id']]
        assert candidate['selection_provenance']['learning_evidence_ids']==['saved-owner-decision']


@pytest.mark.parametrize('content',['not JSON', '[]', '{"private response":'])
def test_malformed_provider_output_is_deterministic_and_sanitized(db,monkeypatch,content):
    import requests
    provider=Mock()
    provider.json.return_value={'choices':[{'message':{'content':content}}]}
    post=Mock(return_value=provider)
    monkeypatch.setattr(requests,'post',post)
    with pytest.raises(pi.TopicValidationError) as failure:
        pi.topic_map(SOURCE,TIMELINE,pi.model_topic_map)
    assert failure.value.diagnostics['prompt_sha256']
    assert content not in json.dumps(failure.value.diagnostics)
    assert pi.hold_analysis_failure({},failure.value)
    with pytest.raises(ValueError):pi.topic_map(SOURCE,TIMELINE,pi.model_topic_map)
    post.assert_called_once()


def test_provider_timeout_is_operational_not_semantic_validation(db,monkeypatch):
    import requests
    post=Mock(side_effect=requests.Timeout('Sanitized test timeout'))
    monkeypatch.setattr(requests,'post',post)
    with pytest.raises(requests.Timeout) as failure:
        pi.topic_map(SOURCE,TIMELINE,pi.model_topic_map)
    assert not pi.hold_analysis_failure({},failure.value)
    assert not pi.analysis_recovery_blocked({'source_intelligence':{'map_id':failure.value.analysis_id.split(':')[0],
               'analysis_id':failure.value.analysis_id}})
    with pi.database() as conn:
        row=conn.execute('select * from source_analyses').fetchone()
    assert row['error']=='Timeout'
    saved=json.loads(row['result'])
    assert saved['operational_failure']['category']=='PROVIDER_UNAVAILABLE'
    assert 'validation_failure' not in saved  # timeout is operational, never semantic evidence


def test_provider_cannot_omit_contract_and_fall_back_to_independent_ranges(db,monkeypatch):
    import requests
    provider=Mock()
    provider.json.return_value={'choices':[{'message':{'content':json.dumps({'topics':[boundary('a',0,last_segment=11)]})}}]}
    monkeypatch.setattr(requests,'post',Mock(return_value=provider))
    with pytest.raises(pi.TopicValidationError,match='boundary contract'):
        pi.topic_map(SOURCE,TIMELINE,pi.model_topic_map)


def test_learning_latest_decision_wins_when_database_timestamps_tie(db):
    for decision in ('approved','rejected'):
        pi.record('feedback',decision,{'candidate_id':'same-candidate','brand_id':'TDOG','show_id':'TDOG',
                 'format':'9:16','decision':decision})
    with pi.database() as conn:
        conn.execute("update intelligence_records set updated_at='2026-11-01T00:00:00+00:00'")
    context=pi.learning_context(SOURCE,'9:16')
    assert len(context['examples'])==1
    assert context['examples'][0]['decision']=='rejected' and context['examples'][0]['sign']==-1
