import hashlib
import json
from unittest.mock import Mock
import pytest
from test_editorial_framing_review import review

def test_actual_highlight_payload_hash_survives_boundary_normalization(monkeypatch):
    import telegram_intake as tg
    monkeypatch.setenv('OPENAI_API_KEY','isolated-not-a-real-key')
    response=Mock(status_code=200)
    response.json.return_value={'choices':[{'message':{'content':json.dumps({'segments':[{'start':0,'end':210,'reason':'Question plus complete answer','score':80}]})}}]}
    post=Mock(return_value=response);monkeypatch.setattr(tg.requests,'post',post)
    timeline=[{'start':0,'end':210,'duration':210,'text':'A complete substantial answer.'}]
    suggestions=tg._topic_break_suggestions(timeline,intelligence_context='approved rubric and full source map')
    actual=post.call_args.kwargs['json']['messages'][1]['content']
    normalized=tg._build_contiguous_topic_segments(timeline,suggestions)
    assert normalized[0]['selection_provenance']['prompt_sha256']==hashlib.sha256(actual.encode()).hexdigest()
    assert normalized[0]['reason']=='Question plus complete answer'


def test_candidate_provenance_has_exact_map_and_feedback_references(review):
    import production_intelligence as pi
    mapping={'map_id':'map','analysis_version':'source-v1','transcript_revision':'timed-r1',
             'source':{'brand_id':'TDOG','show_id':'TDOG','source_video_id':'source'},
             'topics':[{'topic_id':'topic','start':0,'end':40}]}
    pi.record('topic','topic',{'topic_id':'topic','candidate_ids':[]})
    evidence=pi.prompt_evidence('isolated-model','actual supplied prompt','9:16')
    result=pi.candidate_records(mapping,'9:16',{'selection_provenance':evidence,'segments':[{'start':0,'end':30,'reason':'The answer reaches its payoff.'}]},
        {'examples':[{'evidence_id':'owner-feedback-1'}],'confidence':.4})
    proof=result['segments'][0]['selection_provenance']
    assert proof['map_id']=='map' and proof['transcript_revision']=='timed-r1'
    assert proof['learning_evidence_ids']==['owner-feedback-1'] and proof['topic_map_revision']
    assert proof['prompt_sha256']==hashlib.sha256(b'actual supplied prompt').hexdigest()


