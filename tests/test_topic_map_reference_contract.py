import json
from unittest.mock import Mock

import pytest
import production_intelligence as pi


@pytest.mark.parametrize('first,last', [(-1, 1), (0, 2), ('0', 1), (False, 1), (0.0, 1), (1, 0), (10**400, 10**400)])
def test_invalid_reference_is_held_with_durable_sanitized_shape(tmp_path, monkeypatch, first, last):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'existing.db')
    timeline = [{'start': 0, 'end': 4, 'text': 'First sentence.'},
                {'start': 4, 'end': 8, 'text': 'Second sentence.'}]
    response = {'topics': [{'key': 'a', 'title': 'Private provider text', 'first_segment': first, 'last_segment': last}],
                'provenance': {'prompt_version': 'tested', 'prompt_sha256': 'hash'}}
    analyze = Mock(return_value=response)
    source = {'brand_id': 'TDOG', 'source_video_id': 'existing-source'}
    with pytest.raises(pi.TopicReferenceError):
        pi.topic_map(source, timeline, analyze)
    with pi.database() as db:
        row = db.execute('select * from source_analyses').fetchone()
        failure = json.loads(row['result'])['validation_failure']
        assert row['status'] == 'FAILED' and failure['segment_count'] == 2
        assert failure['first_segment']['type'] == type(first).__name__
        assert failure['allowed_max'] == 1 and failure['prompt_sha256'] == 'hash'
        assert 'Private provider text' not in row['result'] and 'sentence' not in row['result']
        assert db.execute("select count(*) from intelligence_records where kind='topic'").fetchone()[0] == 0
    with pytest.raises(ValueError, match='Saved analysis failed'):
        pi.topic_map(source, timeline, analyze)
    analyze.assert_called_once()  # deterministic validation failure cannot repeat a provider request


def test_provider_contract_uses_actual_zero_based_inclusive_range(monkeypatch):
    import requests
    timeline = [{'index': i, 'start': i, 'end': i+1, 'text': 'Timed source.'} for i in range(1189)]
    response = Mock()
    response.json.return_value = {'choices': [{'message': {'content': json.dumps({'topics': []})}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, 'post', post)
    monkeypatch.setenv('OPENAI_API_KEY', 'isolated-never-sent')
    monkeypatch.setenv('OPENAI_MODEL', 'gpt-5-mini')
    result = pi.model_topic_map(timeline)
    payload = post.call_args.kwargs['json']
    contract = payload['response_format']['json_schema']
    topic = contract['schema']['properties']['topics']['items']
    assert contract['strict'] is True and contract['schema']['additionalProperties'] is False
    assert topic['properties']['first_segment'] == {'type': 'integer', 'minimum': 0, 'maximum': 1188}
    nested = topic['properties']['subtopics']['items']['properties']['last_segment']
    assert nested['maximum'] == 1188 and nested['type'] == 'integer'
    assert 'never seconds, strings or one-based positions' in payload['messages'][1]['content']
    assert payload['model'] == 'gpt-5-mini'
    assert 'GLOBAL inclusive transcript indices' in topic['properties']['subtopics']['items']['description']
    assert 'topic.first_segment <= item.first_segment <= item.last_segment <= topic.last_segment' in payload['messages'][1]['content']
    assert result['provenance']['prompt_version'] == 'semantic-source-map-v3-nested-contract'


def test_strict_schema_does_not_replace_full_source_coverage_check(tmp_path, monkeypatch):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'existing.db')
    with pytest.raises(ValueError, match='omitted part'):
        pi.topic_map({'brand_id':'TDOG', 'source_video_id':'source'},
                     [{'start':0,'end':1,'text':'First.'}, {'start':1,'end':2,'text':'Last.'}],
                     lambda _: {'topics':[{'key':'a','title':'Partial','first_segment':0,'last_segment':0}]})


NESTED_FIELDS = ['subtopics', 'questions_responses', 'memorable_moments']


def nested_fixture(field, first=1, last=2):
    timeline = [{'start':i*4, 'end':(i+1)*4, 'text':'Private transcript sentence.'} for i in range(4)]
    response = {'topics':[
        {'key':'intro', 'title':'Private opening', 'first_segment':0, 'last_segment':0},
        {'key':'main', 'title':'Private explanation', 'first_segment':1, 'last_segment':2,
         field:[{'first_segment':1, 'last_segment':1}, {'first_segment':first, 'last_segment':last}]},
        {'key':'outro', 'title':'Private closing', 'first_segment':3, 'last_segment':3}],
        'provenance':{'prompt_sha256':'actual-prompt-hash', 'prompt_version':'nested-v3', 'provider':'openai','model':'gpt-5-mini'}}
    return timeline, response


@pytest.mark.parametrize('field', NESTED_FIELDS)
@pytest.mark.parametrize('first,last', [(0,2),(1,3),(2,1),('1',2),(True,2),(1.0,2),(None,2),('Private speech or secret',2)])
def test_nested_failure_retains_actual_scope_types_and_safe_values(tmp_path, monkeypatch, field, first, last):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'existing.db')
    timeline, response = nested_fixture(field, first, last)
    analyze = Mock(return_value=response)
    source = {'brand_id':'TDOG', 'source_video_id':'same-source'}
    with pytest.raises(pi.NestedReferenceError, match='within its topic'):
        pi.topic_map(source, timeline, analyze)
    # Reopen SQLite after the exception: diagnostics must survive transaction closure.
    with pi.database() as db:
        row=db.execute('select status,result from source_analyses').fetchone()
        failure=json.loads(row['result'])['validation_failure']
        assert row['status']=='FAILED'
        assert failure['category']=='NESTED_TOPIC_REFERENCE_INVALID'
        assert (failure['topic_position'],failure['nested_field'],failure['item_position'])==(1,field,1)
        assert (failure['parent_first_segment'],failure['parent_last_segment'])==(1,2)
        assert (failure['segment_count'],failure['allowed_min'],failure['allowed_max'])==(4,0,3)
        assert failure['first_segment']['type']==type(first).__name__
        assert failure['last_segment']=={'type':type(last).__name__,'value':last}
        assert failure['prompt_sha256']=='actual-prompt-hash' and failure['prompt_version']=='nested-v3'
        assert failure['model']=='gpt-5-mini' and failure['provider']=='openai'
        if first=='Private speech or secret':
            assert failure['first_segment']=={'type':'str','value':None,'redacted':True}
        else:
            assert failure['first_segment']['value']==first
        assert 'Private' not in row['result'] and 'transcript sentence' not in row['result']
        assert not db.execute("select 1 from intelligence_records where kind in ('source','topic')").fetchone()
    with pytest.raises(ValueError, match='Saved analysis failed'):
        pi.topic_map(source, timeline, analyze)
    analyze.assert_called_once()


@pytest.mark.parametrize('field',NESTED_FIELDS)
def test_valid_nested_endpoints_preserve_complete_map_without_adjustment(tmp_path, monkeypatch, field):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'existing.db')
    timeline,response=nested_fixture(field)
    mapping=pi.topic_map({'brand_id':'TDOG','source_video_id':'source'},timeline,lambda _:response)
    assert mapping['topics'][1][field]==response['topics'][1][field]
    assert (mapping['topics'][1]['start'],mapping['topics'][1]['end'])==(4,12)


@pytest.mark.parametrize('malformed', [None, False, {'first_segment':1}, 'Private text'])
def test_malformed_nested_collection_is_diagnostic_not_silently_empty(tmp_path, monkeypatch, malformed):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'existing.db')
    timeline,response=nested_fixture('subtopics');response['topics'][1]['subtopics']=malformed
    with pytest.raises(pi.NestedReferenceError) as error:
        pi.topic_map({'brand_id':'TDOG','source_video_id':'source'},timeline,lambda _:response)
    assert error.value.diagnostics['nested_collection_type']==type(malformed).__name__
    assert error.value.diagnostics['item_position'] is None
    assert 'Private' not in json.dumps(error.value.diagnostics)


def test_non_object_moment_is_preserved_without_private_value(tmp_path, monkeypatch):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'existing.db')
    timeline,response=nested_fixture('questions_responses');response['topics'][1]['questions_responses'][1]='Private speech'
    with pytest.raises(pi.NestedReferenceError) as error:
        pi.topic_map({'brand_id':'TDOG','source_video_id':'source'},timeline,lambda _:response)
    assert error.value.diagnostics['nested_item_type']=='str'
    assert error.value.diagnostics['item_position']==1
    assert 'Private' not in json.dumps(error.value.diagnostics)
