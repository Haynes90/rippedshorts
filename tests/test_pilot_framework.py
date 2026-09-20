import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import clip_completion as c
import pilot_download as d
import pilot_selection as s
from pilot_render import render_approved
from test_review_hub import studio


def section(tmp_path):
    path = tmp_path / 'source.mp4'
    path.write_bytes(b'source')
    timed = [dict(start=i*12, end=(i+1)*12-1, duration=11, text=text,
                  words=[dict(start=i*12, end=(i+1)*12-1, word=text)])
             for i, text in enumerate(['Welcome everybody.', 'Water plants deeply.', 'Roots need oxygen.', 'Next topic.'])]
    return dict(path=path, offset=900, duration=48, timed=timed)


def selection(monkeypatch, tmp_path, decision=None):
    calls = []
    review = dict(candidate_id=0, first_sentence=1, last_sentence=2, complete_start=True,
                  complete_end=True, complete_thought=True, distinct=True)
    review.update(decision or {})
    def ask(stage, instruction, data, work):
        calls.append(stage)
        return {'topic-map': {'classification': {'domain':'gardening'}, 'topics':[{'first_sentence':0,'last_sentence':3}]},
                'proposals': {'shorts':[{'first_sentence':0,'last_sentence':3,'title':'Watering'}], 'highlights':[]},
                'semantic-review': {'reviews':[review]}}[stage]
    monkeypatch.setattr(s, 'ask', ask)
    source = section(tmp_path)
    result = s.select([source], 'both', [], tmp_path)
    return source, result, calls


def test_semantic_trimming_happens_before_human_approval(monkeypatch, tmp_path):
    _, result, calls = selection(monkeypatch, tmp_path)
    assert calls == ['topic-map', 'proposals', 'semantic-review']
    clip = result['shorts'][0]
    assert (clip['start'],clip['end']) == (12,35)
    assert clip['transcript'] == 'Water plants deeply. Roots need oxygen.'
    assert clip['ai_proposal']['first_sentence'] == 0
    assert clip['source_offset'] == 900


def test_punctuation_does_not_allow_incomplete_thought(monkeypatch, tmp_path):
    _, result, _ = selection(monkeypatch, tmp_path, {'complete_thought':False})
    assert result['shorts'] == []


def test_selection_cannot_cross_section_gap(tmp_path):
    one = section(tmp_path)['timed']
    sentences = [{**one[0],'section':0},{**one[1],'section':1}]
    with pytest.raises(ValueError, match='gaps'):
        s.sentence_span({'first_sentence':0,'last_sentence':1},sentences)


def test_render_keeps_preapproved_thought_without_second_model_selection(monkeypatch, tmp_path):
    source, result, _ = selection(monkeypatch, tmp_path)
    clip = result['shorts'][0]
    monkeypatch.setattr(c,'media_duration',lambda _:48)
    monkeypatch.setattr(c,'pause_end',lambda *args:35.2)
    monkeypatch.setattr(c,'verify_render',Mock())
    monkeypatch.setattr(c,'review_thought',Mock(side_effect=AssertionError('Must not reselect after approval')))
    renderer = Mock()
    rendered = render_approved(source['path'],clip,tmp_path/'out.mp4',renderer,'9:16')
    assert rendered['completion_check']['semantic_before_approval']
    assert renderer.call_args.args[1] == 11.85
    c.verify_render.assert_called_once()


def test_changed_source_or_transcript_cannot_use_saved_proof(monkeypatch, tmp_path):
    source, result, _ = selection(monkeypatch, tmp_path)
    clip = result['shorts'][0]
    source['path'].write_bytes(b'replaced source')
    with pytest.raises(c.CompletionReviewRequired,match='source changed'):
        render_approved(source['path'],clip,tmp_path/'out.mp4',Mock(),'9:16')


def test_rapidapi_first_never_sends_key_to_media(monkeypatch, tmp_path):
    monkeypatch.setenv('RAPIDAPI_KEY','unit-test-secret')
    calls = []
    class Media:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def raise_for_status(self): pass
        def iter_content(self,*a): return [b'video']
    def get(url, **kwargs):
        calls.append((url,kwargs))
        if len(calls)==1:
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:{'downloadUrl':'https://media.example/video.mp4'})
        return Media()
    monkeypatch.setattr(d.requests,'get',get)
    monkeypatch.setattr(d,'media_ok',lambda p:p.exists())
    output=d.download_youtube('abcdefghijk','https://youtu.be/abcdefghijk',tmp_path)
    assert output.read_bytes()==b'video'
    assert 'rapidapi.com/download_video/abcdefghijk' in calls[0][0]
    assert calls[0][1]['headers']['x-rapidapi-key']=='unit-test-secret'
    assert 'headers' not in calls[1][1]
    assert d.download_youtube('abcdefghijk','unused',tmp_path)==output
    assert len(calls)==2


def test_missing_rapidapi_key_fails_before_bot_loop(monkeypatch, tmp_path):
    monkeypatch.delenv('RAPIDAPI_KEY',raising=False)
    monkeypatch.setattr(d.requests,'get',Mock(side_effect=AssertionError('No HTTP expected')))
    with pytest.raises(RuntimeError,match='RAPIDAPI_KEY'):
        d.download_youtube('abcdefghijk','url',tmp_path)


def test_rebuild_archives_decisions_and_blocks_duplicate_requests(studio):
    client, fake, seed=studio
    seed(state={'basic_pilot':True,'candidate_reviews':{'0':{'status':'rejected'}}})
    with fake._telegram_db() as db:
        db.execute('CREATE TABLE durable_job_leases (job_id TEXT, action TEXT)')
    assert client.post('/review/api/projects/one/rebuild').status_code==202
    assert client.post('/review/api/projects/one/rebuild').status_code==409
    with fake._telegram_db() as db:
        state=json.loads(db.execute('SELECT state_json FROM telegram_requests').fetchone()[0])
    assert state['selection_history'][0]['candidate_reviews']['0']['status']=='rejected'
    assert state['result']=={}
    fake.RENDER_EXECUTOR.submit.assert_called_once()


def test_rebuild_never_replaces_rendered_clips(studio):
    client, fake, seed=studio
    seed(state={'basic_pilot':True,'candidate_reviews':{'0':{'status':'rendered'}}})
    assert client.post('/review/api/projects/one/rebuild').status_code==409
    fake.RENDER_EXECUTOR.submit.assert_not_called()
