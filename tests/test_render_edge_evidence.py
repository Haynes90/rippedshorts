from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import clip_completion as clip


@pytest.fixture
def render(monkeypatch,tmp_path):
    output=tmp_path/'export.mp4'
    monkeypatch.setattr(clip,'media_duration',lambda path:100 if str(path)=='source' else 5)
    monkeypatch.setattr(clip,'review_thought',lambda *a:{'complete_start':True,'complete_end':True,'complete_thought':True})
    monkeypatch.setattr(clip,'_run',lambda args:SimpleNamespace(stdout=json.dumps({'streams':[{'codec_type':'video','width':1080,'height':1920},{'codec_type':'audio'}]})))
    segment={'start':10,'end':15,'transcript':'This is a complete thought.'}
    renderer=lambda *a:output.write_bytes(b'fixture output')
    return segment,output,renderer


def test_segment_end_is_not_mistaken_for_a_measured_word_endpoint(monkeypatch,render):
    segment,output,renderer=render
    # Actual failing shape: provider emits a segment ending at the sample boundary.
    # Only independent measured silence may establish the acoustic edge.
    monkeypatch.setattr(clip,'transcribe_window',lambda *a:[{'text':segment['transcript'],'start':0,'end':5,'timing_basis':'provider_segment_timing'}])
    guards=[]
    monkeypatch.setattr(clip,'segment_edge_guard',lambda *a:guards.append(a[-1]))
    result=clip.render_complete_clip(Path('source'),segment,output,renderer,'9:16')
    assert guards==[False,True] and result['completion_check']['export_edges']


def test_segment_timing_still_holds_if_acoustic_edge_is_not_safe(monkeypatch,render):
    segment,output,renderer=render
    monkeypatch.setattr(clip,'transcribe_window',lambda *a:[{'text':segment['transcript'],'start':0,'end':5,'timing_basis':'provider_segment_timing'}])
    def unsafe(*a):raise clip.CompletionReviewRequired('unsafe acoustic edge')
    monkeypatch.setattr(clip,'segment_edge_guard',unsafe)
    with pytest.raises(clip.CompletionReviewRequired,match='acoustic'):
        clip.render_complete_clip(Path('source'),segment,output,renderer,'9:16')
    assert not output.exists()


def test_word_timing_keeps_real_final_word_safety_margin(monkeypatch,render):
    segment,output,renderer=render
    monkeypatch.setattr(clip,'transcribe_window',lambda *a:[{'word':w,'start':i,'end':i+1} for i,w in enumerate(segment['transcript'].split())])
    with pytest.raises(clip.CompletionReviewRequired,match='close to speech'):
        clip.render_complete_clip(Path('source'),segment,output,renderer,'9:16')


def test_mismatched_boundary_does_not_publish_or_silently_trim_and_keeps_diagnostics(monkeypatch,render):
    segment,output,renderer=render
    monkeypatch.setattr(clip,'transcribe_window',lambda *a:[{'word':'Unrelated speech.', 'start':0,'end':1}])
    with pytest.raises(clip.CompletionReviewRequired) as caught:
        clip.render_complete_clip(Path('source'),segment,output,renderer,'9:16')
    assert caught.value.diagnostics['stage']=='export_boundary_alignment'
    assert caught.value.diagnostics['expected_tokens']!=caught.value.diagnostics['heard_tokens']
    assert not output.exists()
