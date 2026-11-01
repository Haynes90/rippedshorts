from unittest.mock import Mock
import pytest
import selection_specs as specs


def test_approved_doc_candidate_contract_does_not_silently_drop_clips_or_scores():
    data={'analysis':{'audience':'gardeners'},'clips':[{'start':0,'end':20,'transcript':'Complete thought.','score_total':92,'why_it_will_work':'A clear payoff.','hook':'Truthful hook'}]}
    result=specs.selector_result(data)
    assert result['segments'][0]['score']==92 and result['segments'][0]['reason']=='A clear payoff.'
    assert result['segments'][0]['hook']=='Truthful hook' and result['analysis']==data['analysis']
    assert result['clips']==data['clips']


def test_missing_candidate_array_is_a_validation_failure_not_no_qualifying_material():
    with pytest.raises(ValueError,match='omitted'):specs.selector_result({'analysis':{}})
    assert specs.selector_result({'clips':[]})['segments']==[]


def test_missing_score_is_explicit_and_nonfinite_score_is_rejected():
    item=specs.selector_result({'segments':[{}]})['segments'][0]
    assert not item['ai_score_available']
    with pytest.raises(ValueError,match='finite'):
        specs.selector_result({'segments':[{'score':float('nan')}]})


def test_editable_highlight_doc_is_reused_and_permission_failure_uses_packaged_spec(monkeypatch):
    import google_drive
    monkeypatch.setattr(specs,'_cache',None);monkeypatch.setattr(specs,'_disabled',False)
    reader=Mock(side_effect=RuntimeError('HTTP 403 permission denied'))
    monkeypatch.setattr(google_drive,'read_google_doc_text',reader)
    first=specs.highlight_framework()
    assert 'Every highlight must be at least 3 minutes' in first
    assert specs.highlight_framework()==first and reader.call_count==1
    assert reader.call_args.args[0]=='1QyDKGKJscc107U38QsWtLP6fhqv8VPZhdoSwIC-bQdM'
