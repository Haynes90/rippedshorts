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
    assert result['provenance']['prompt_version'] == 'semantic-source-map-v2-index-contract'


def test_strict_schema_does_not_replace_full_source_coverage_check(tmp_path, monkeypatch):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff, 'DB_PATH', tmp_path/'existing.db')
    with pytest.raises(ValueError, match='omitted part'):
        pi.topic_map({'brand_id':'TDOG', 'source_video_id':'source'},
                     [{'start':0,'end':1,'text':'First.'}, {'start':1,'end':2,'text':'Last.'}],
                     lambda _: {'topics':[{'key':'a','title':'Partial','first_segment':0,'last_segment':0}]})
