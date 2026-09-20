import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from review_basic import normalize_ranges, learning_examples
from test_review_hub import studio


def test_ranges_merge_overlap_and_preserve_gaps():
    assert normalize_ranges([{"start":100,"end":200},{"start":0,"end":120},{"start":300,"end":400}]) == [{"start":0,"end":200},{"start":300,"end":400}]


@pytest.mark.parametrize("ranges", [[],[{"start":0,"end":3601}],[{"start":3,"end":2}],[{"start":0,"end":float('inf')}], [{"start":0,"end":10}]*5])
def test_invalid_or_overbudget_ranges(ranges):
    with pytest.raises(ValueError): normalize_ranges(ranges)


def test_transcript_is_owner_scoped(studio):
    client, _, seed=studio
    seed(state={"transcript_text":"[00:10:00] Selected content."})
    seed("other",chat="different",state={"transcript_text":"private"})
    assert client.get('/review/api/projects/one/transcript.txt').text == '[00:10:00] Selected content.'
    assert client.get('/review/api/projects/other/transcript.txt').status_code == 404


def test_render_limit_reserves_inflight_clips(studio):
    client, fake, seed=studio
    seed(state={"basic_pilot":True,"plan":"trial","candidate_reviews":{"1":{"status":"queued"},"2":{"status":"rendered"}}})
    assert client.post('/review/api/projects/one/shorts/0/decision',json={"decision":"approve"}).status_code == 409
    fake.RENDER_EXECUTOR.submit.assert_not_called()


def test_learning_never_crosses_customer_accounts(studio):
    _, fake, seed=studio
    seed(state={"account_id":"pilot:alice","candidate_reviews":{"0":{"status":"rendered"}}})
    seed("two",state={"account_id":"pilot:bob","candidate_reviews":{"0":{"status":"rejected"}}})
    with fake._telegram_db() as db:
        examples=learning_examples(db,'pilot:alice')
    assert len(examples)==1
    assert examples[0]['choice']=='kept'


def test_processing_transcribes_only_trimmed_sections(studio, tmp_path, monkeypatch):
    _, fake, seed=studio
    import main
    import pilot_download
    import pilot_selection
    import telegram_intake as engine
    import clip_completion
    import durable_jobs
    import review_basic
    seed(state={"account_id":"pilot:test","customer_reference":"test","basic_pilot":True,
                "parsed":{"source_kind":"youtube","video_id":"abcdefghijk","source_value":"https://youtu.be/abcdefghijk"},
                "selected_ranges":[{"start":100,"end":150},{"start":300,"end":350}]})
    monkeypatch.setattr(engine,'_telegram_db',fake._telegram_db)
    monkeypatch.setattr(engine,'SOURCE_DIR',tmp_path)
    monkeypatch.setattr(engine,'_LOCK',fake._LOCK)
    saved=[]
    monkeypatch.setattr(engine,'_save',lambda pid,status,state:saved.append((status,dict(state))))
    monkeypatch.setattr(durable_jobs,'claim',lambda *a,**k:True)
    monkeypatch.setattr(durable_jobs,'finish',Mock())
    monkeypatch.setattr(pilot_download,'download_youtube',lambda *a:tmp_path/'original.mp4')
    monkeypatch.setattr(clip_completion,'media_duration',lambda _:1000)
    trimmed=[]
    def cut(source,output,start,end):
        trimmed.append((start,end)); output.touch()
    monkeypatch.setattr(review_basic,'cut_section',cut)
    transcribed=[]
    def transcribe(path):
        transcribed.append(path.name)
        return [{"start":0,"end":30,"duration":30,"text":"Complete thought."}]
    monkeypatch.setattr(clip_completion,'transcribe_source',transcribe)
    monkeypatch.setattr(main,'call_openai_for_clips',lambda *a:{"segments":[{"start":0,"duration":30,"transcript":"Complete thought."}]})
    monkeypatch.setattr(engine,'validate_complete_candidates',lambda result,*_:result)
    def selection(sections, *args):
        return {'shorts': [{'start':0,'duration':30,'transcript':'Complete thought.', 'source_offset':section['offset']} for section in sections],
                'highlights': [], 'understanding': {}, 'proposals': {}, 'semantic_reviews': {}}
    monkeypatch.setattr(pilot_selection,'select',selection)
    monkeypatch.setattr(engine,'_topic_break_suggestions',lambda *a,**k:[])
    monkeypatch.setattr(engine,'_build_contiguous_topic_segments',lambda *a:[])
    review_basic.process('one')
    assert trimmed==[(100,150),(300,350)]
    assert transcribed==['section-0.mp4','section-1.mp4']
    assert saved[-1][0]=='awaiting_review'
    assert '[00:01:40]' in saved[-1][1]['transcript_text']
    assert '[00:05:00]' in saved[-1][1]['transcript_text']
    assert saved[-1][1]['result']['segments'][1]['source_offset']==300
