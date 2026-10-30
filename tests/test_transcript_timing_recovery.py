import math
import pytest
from transcript_timing import normalize_response, TranscriptTimingError, cached_response, saved_segments

def response():
    return dict(duration=600, words=[
        dict(word='Start.',start=0,end=1),
        dict(word='',start=228.1999969482422,end=228.1999969482422),
        dict(word='End.',start=229,end=230)],
        segments=[dict(text='Start.',start=0,end=1),dict(text='',start=213.8000030517578,end=213.8000030517578),dict(text='End.',start=229,end=230)])

def test_actual_cached_tdog_blank_zero_duration_token_is_not_speech():
    result=normalize_response(response(),offset=600)
    assert [w['word'] for w in result['words']]==['Start.','End.']
    assert [w['start'] for w in result['words']]==[600,829]
    assert result['timing_diagnostics']['discarded_blank_zero_duration_tokens']==1
    assert result['segments'][1]['end']==830
    assert result['timing_diagnostics']['discarded_blank_zero_duration_segments']==1
    assert result['timing_basis']=='provider_word_timing'

@pytest.mark.parametrize('words',[None,[],[dict(word='x',start=2,end=1)],[dict(word='x',start=float('nan'),end=1)]])
def test_missing_or_invalid_words_use_only_honest_segment_timing(words):
    data=response();data['words']=words
    result=normalize_response(data)
    assert result['timing_basis']=='provider_segment_timing'
    assert result['words']==[]
    assert result['requires_acoustic_boundary_validation']
    assert all('word' not in s for s in result['segments'])

@pytest.mark.parametrize('segment',[dict(text='x',start=-1,end=2),dict(text='x',start=3,end=2),
    dict(text='x',start=0,end=math.inf),dict(text='x',start=0,end=601),
    dict(text='x',start='bad',end=2)])
def test_malformed_segment_timing_still_blocks(segment):
    with pytest.raises(TranscriptTimingError):
        normalize_response(dict(segments=[segment]),duration=600)

def test_deterministic_bad_response_is_cached_not_requested_again(tmp_path):
    audio=tmp_path/'chunk.mp3';audio.write_bytes(b'cached audio')
    calls=[]
    def fetch():
        calls.append(True);return dict(words=[],segments=[])
    for _ in range(2):
        with pytest.raises(TranscriptTimingError):
            normalize_response(cached_response(audio,'whisper-1',fetch))
    assert len(calls)==1

def test_completed_timed_txt_is_reusable_without_invented_word_precision():
    text='Title\nYouTube ID: video\n\nTranscript\n[00:00:01.000 - 00:00:03.200] A complete thought.\n'
    result=saved_segments(text,'video',5)
    assert result['words']==[]
    assert result['segments'][0]['end']==3.2
    assert result['timing_basis']=='provider_segment_timing'
    with pytest.raises(TranscriptTimingError):saved_segments(text,'other',5)

def test_valid_words_without_segments_remain_available():
    result=normalize_response(dict(words=[dict(word='Done.',start=0,end=1)]))
    assert result['timing_basis']=='provider_word_timing'


def window_fixture(monkeypatch,tmp_path,fetch):
    import clip_completion as clip
    from pathlib import Path
    monkeypatch.setenv('OPENAI_API_KEY','test-only')
    monkeypatch.setattr(clip,'_run',lambda args:Path(args[-1]).write_bytes(b'isolated audio'))
    monkeypatch.setattr(clip.time,'sleep',lambda _:None)
    monkeypatch.setattr(clip.requests,'post',fetch)
    return clip, tmp_path/'video.mp4'


def test_valid_words_and_offsets_survive_actual_window_adapter(monkeypatch,tmp_path):
    from types import SimpleNamespace
    def fetch(*args,**kw):
        assert ('timestamp_granularities[]','segment') in kw['data']
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=response)
    clip,video=window_fixture(monkeypatch,tmp_path,fetch)
    units=clip.transcribe_window(video,600,1200)
    assert len(units)==2 and units[-1]['end']==830


def test_segment_fallback_contains_no_fake_words(monkeypatch,tmp_path):
    from types import SimpleNamespace
    data=dict(segments=[dict(text='A complete thought.',start=0,end=2)])
    clip,video=window_fixture(monkeypatch,tmp_path,lambda *a,**k:SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=lambda:data))
    units=clip.transcribe_window(video,100,110)
    assert units[0]['start']==100 and units[0]['end']==102
    assert 'word' not in units[0]
    assert units[0]['timing_basis']=='provider_segment_timing'
    assert clip._tokens(units)==['a','complete','thought']


def test_transient_provider_timeout_retries_but_bad_timing_does_not(monkeypatch,tmp_path):
    import requests
    from types import SimpleNamespace
    calls=[]
    def fetch(*a,**k):
        calls.append(True)
        if len(calls)==1:raise requests.Timeout()
        return SimpleNamespace(status_code=200,raise_for_status=lambda:None,json=response)
    clip,video=window_fixture(monkeypatch,tmp_path,fetch)
    assert clip.transcribe_window(video,0,600)
    assert len(calls)==2


def test_segment_edges_require_actual_acoustic_evidence(monkeypatch):
    import clip_completion as clip
    from types import SimpleNamespace
    monkeypatch.setattr(clip,'_run',lambda _:SimpleNamespace(stderr=''))
    with pytest.raises(clip.CompletionReviewRequired,match='acoustic'):
        clip.segment_edge_guard('video',0,10,True)
    monkeypatch.setattr(clip,'_run',lambda _:SimpleNamespace(stderr='silence_start: 0.1\nsilence_end: 0.5'))
    clip.segment_edge_guard('video',0,10,True)
    with pytest.raises(clip.CompletionReviewRequired):clip.segment_edge_guard('video',0,10,False)


def test_upstream_media_checkpoint_survives_transcription_failure():
    from workflow_test_status import project
    report=project('request',dict(show_id='TDOG',parsed={'video_id':'video'},error='timing failed',
        audio_master_ingest={'checkpoints':{'MEDIA_READY':{'status':'PASSED','evidence':{'duration':600,'audio':True,'video':True}}}}))
    assert report['checkpoints']['MEDIA_READY']['status']=='PASSED'
    assert report['checkpoints']['TRANSCRIPT_READY']['status']=='FAILED'
    assert report['blocking_stage']=='TRANSCRIPT_READY'

