import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import clip_completion as c


def words(text='This is a complete thought.', start=10, step=1):
    return [{'word': w, 'start': start+i*step, 'end': start+i*step+.6} for i,w in enumerate(text.split())]


def review(ws):
    return dict(complete_start=True, complete_end=True, complete_thought=True,
                first_word=0, last_word=len(ws)-1, reason='Setup and payoff complete')


@pytest.mark.parametrize('aspect', ['16:9','9:16'])
def test_incomplete_thought_is_rejected_even_with_period(aspect):
    ws=words(); r=review(ws); r['complete_thought']=False
    with pytest.raises(c.CompletionReviewRequired): c.checked_selection(ws,r,10,14.6,aspect)


def test_short_cannot_cut_to_fit_limit():
    ws=words(start=0,step=30)
    with pytest.raises(c.CompletionReviewRequired,match='90-second'): c.checked_selection(ws,review(ws),0,120.6,'9:16')
    assert c.checked_selection(ws,review(ws),0,120.6,'16:9') == (0,4)


@pytest.mark.parametrize('last', ['unfinished', 'unfinished,', 'because'])
def test_sentence_must_finish(last):
    ws=words('A thought '+last)
    with pytest.raises(c.CompletionReviewRequired,match='sentence'): c.checked_selection(ws,review(ws),10,12.6,'16:9')


def test_invalid_or_unrelated_boundaries_rejected():
    ws=words(); r=review(ws); r['first_word']=1
    with pytest.raises(c.CompletionReviewRequired): c.checked_selection(ws,r,9,14.6,'16:9')
    r['first_word']=True
    with pytest.raises(c.CompletionReviewRequired): c.checked_selection(ws,r,10,14.6,'16:9')


@pytest.mark.parametrize('data', [{}, {'words': []}, {'words': [{'word':'x','start':float('nan'),'end':1}]}, {'words':[{'word':'x','start':0,'end':99}]}])
def test_missing_invalid_word_transcripts_fail(data):
    with pytest.raises(c.CompletionReviewRequired): c.validate_words(data,10)


def test_transcript_retries_empty_response_and_reextracts(monkeypatch):
    calls=[]
    def extract(args):
        Path(args[-1]).write_bytes(b'audio'); calls.append(args)
    payloads=iter([{'words':[]},{'words':words(start=0)}])
    monkeypatch.setenv('OPENAI_API_KEY','test')
    monkeypatch.setattr(c,'_run',extract)
    monkeypatch.setattr(c.time,'sleep',lambda _:None)
    monkeypatch.setattr(c.requests,'post',lambda *a,**k:SimpleNamespace(raise_for_status=lambda:None,json=lambda:next(payloads)))
    result=c.transcribe_window(Path('source'),100,110)
    assert len(calls)==2 and result[0]['start']==100


def test_overlap_recombines_sentence_crossing_chunk_boundary(monkeypatch):
    monkeypatch.setattr(c,'media_duration',lambda _:240)
    monkeypatch.setenv('RIPPED_TRANSCRIPTION_CHUNK_SECONDS','120')
    calls=[]
    def transcribe(_,start,end):
        calls.append((start,end))
        return [{'word':'First','start':118,'end':119}, {'word':'complete.','start':121,'end':122}]
    monkeypatch.setattr(c,'transcribe_window',transcribe)
    result=c.transcribe_source(Path('source'))
    assert sorted(calls)==[(0,124.0),(116.0,240)]
    assert len(result)==1 and result[0]['text']=='First complete.'


def test_missing_chunk_does_not_silently_succeed(monkeypatch):
    monkeypatch.setattr(c,'media_duration',lambda _:240)
    monkeypatch.setenv('RIPPED_TRANSCRIPTION_CHUNK_SECONDS','120')
    monkeypatch.setattr(c,'transcribe_window',lambda *args: words(start=1))
    with pytest.raises(c.CompletionReviewRequired): c.transcribe_source(Path('source'))


def test_audio_pause_is_before_next_word(monkeypatch):
    monkeypatch.setattr(c,'_run',lambda _:SimpleNamespace(stderr='silence_start: 0.35\nsilence_end: 0.9'))
    cut=c.pause_end(Path('source'),10,10.7)
    assert 10.05 < cut < 10.7
    monkeypatch.setattr(c,'_run',lambda _:SimpleNamespace(stderr=''))
    with pytest.raises(c.CompletionReviewRequired): c.pause_end(Path('source'),10,10.7)


def test_real_audio_pause_detection(monkeypatch,tmp_path):
    ffmpeg=pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    monkeypatch.setenv('FFMPEG_BINARY',ffmpeg)
    audio=tmp_path/'pause.wav'
    c._run([ffmpeg,'-v','error','-y','-f','lavfi','-i','sine=frequency=440:duration=1',
            '-af','apad=pad_dur=1',str(audio)])
    assert 1 < c.pause_end(audio,1,1.8) < 1.8


@pytest.mark.parametrize('aspect',['16:9','9:16'])
def test_failed_export_rebuilds_and_rechecks_before_success(monkeypatch,tmp_path,aspect):
    ws=words(); calls=[]; renders=[]; verifications=[]
    monkeypatch.setattr(c,'media_duration',lambda _:200)
    monkeypatch.setattr(c,'transcribe_window',lambda *args:calls.append(args) or ws)
    monkeypatch.setattr(c,'review_thought',lambda *args:review(ws))
    monkeypatch.setattr(c,'pause_end',lambda *args:14.9)
    def verify(*args):
        verifications.append(args)
        if len(verifications)==1: raise c.CompletionReviewRequired('last word missing')
    monkeypatch.setattr(c,'verify_render',verify)
    result=c.render_complete_clip(Path('source'),{'start':10,'end':14.6},tmp_path/'clip.mp4',lambda *a:renders.append(a),aspect)
    assert len(calls)==len(renders)==len(verifications)==2
    assert calls[1][2]>calls[0][2]
    assert result['completion_check']['status']=='verified'
    assert result['completion_check']['attempt']==2


def test_persistent_failure_stops_after_two_attempts(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(c,'media_duration',lambda _:200)
    def fail(*args):
        calls.append(args); raise c.CompletionReviewRequired('Missing transcript')
    monkeypatch.setattr(c,'transcribe_window',fail)
    with pytest.raises(c.CompletionReviewRequired,match='two attempts'):
        c.render_complete_clip(Path('source'),{'start':10,'end':20},tmp_path/'clip',None,'9:16')
    assert len(calls)==2


@pytest.mark.parametrize('text', ['This is a complete', 'This is a complete thought. Next'])
def test_export_cannot_omit_last_word_or_add_next_sentence(monkeypatch,text):
    ws=words(start=0)
    monkeypatch.setattr(c,'media_duration',lambda _:5)
    monkeypatch.setattr(c,'transcribe_window',lambda *a:words(text,start=0))
    with pytest.raises(c.CompletionReviewRequired): c.verify_render(Path('clip'),ws,0,5)


def test_export_edges_and_tail_padding_pass(monkeypatch):
    ws=words(start=0)
    monkeypatch.setattr(c,'media_duration',lambda _:5)
    monkeypatch.setattr(c,'transcribe_window',lambda *a:ws)
    c.verify_render(Path('clip'),ws,0,5)


@pytest.mark.parametrize('name',['attach_clip_assets','attach_topic_segment_asset'])
def test_both_upload_paths_require_verified_render(name):
    tree=ast.parse(Path('main.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    calls=sorted((n.lineno,n.func.id) for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name))
    gate=next(line for line,call in calls if call=='render_complete_clip')
    upload=next(line for line,call in calls if call=='upload_clip_to_drive')
    assert gate < upload
# Additional integration checks run the real upload functions with service dependencies injected.
@pytest.mark.parametrize('name,aspect',[('attach_clip_assets','9:16'),('attach_topic_segment_asset','16:9')])
def test_upload_is_blocked_on_verification_failure(name,aspect,tmp_path):
    import uuid
    tree=ast.parse(Path('main.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    calls=[]
    def reject(*args):
        assert args[-1]==aspect
        raise c.CompletionReviewRequired('Unfinished sentence')
    env={'Path':lambda value: tmp_path if value=='/tmp' else Path(value),'Optional':object,'render_complete_clip':reject,'uuid':uuid,
         'create_clip_file':lambda *a:None,'create_topic_segment_file':lambda *a:None,
         'cleanup_old_temp_downloads':lambda **k:None,
         'upload_clip_to_drive':lambda *a,**k:calls.append(a)}
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'upload_integration','exec'),env)
    segment={'start':10,'duration':5,'end':15}
    with pytest.raises(c.CompletionReviewRequired):
        if aspect=='16:9': env[name](segment,'test',tmp_path/'source',1,vid_title='test')
        else: env[name]({'segments':[segment]},'test',None,video_path_override=tmp_path/'source',vid_title='test')
    assert calls==[]

def test_restore_asr_punctuation_without_changing_words():
    ws=[{'word':'Hello','start':0,'end':1},{'word':'world','start':1,'end':2},{'word':'Next','start':2,'end':3}]
    restored=c.restore_punctuation(ws,'Hello world. Next!')
    assert [w['word'] for w in restored]==['Hello','world.','Next!']
    assert [w['end'] for w in restored]==[1,2,3]
    assert c.restore_punctuation(ws,'Different text.')==ws
