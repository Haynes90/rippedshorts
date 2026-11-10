"""Actual persistence/HTTP/selection, plus FFmpeg in both formats on CI.

Drive is an isolated API double, semantic provider responses are injected. Never
claim a real destination upload or human audible-speech approval from this fixture.
"""
import copy
import os
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import math
import wave
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import render_recovery as recovery
import boundary_specification as boundary
import clip_completion as speech
from test_boundary_specification import passage, REAL_DURATION, REAL_EDGE

REAL_RENDER=speech.render_complete_clip


@pytest.fixture
def previews(passage,monkeypatch):
    import main, audio_master_handoff as storage, caption_review_web as web
    p=passage;state=p['read']();state.update(preview_first_version=1,candidate_reviews={},topic_reviews={})
    p['write'](state);objects={};blobs={};calls=[]
    def create(**kw):
        body=kw['body'];data=Path(kw['media_body']._filename).read_bytes();fid='fixture-media-'+str(len(objects))
        objects[fid]=dict(id=fid,size=str(len(data)),mimeType='video/mp4',md5Checksum=hashlib.md5(data).hexdigest(),
            parents=body['parents'],appProperties=body['appProperties'],trashed=False)
        blobs[fid]=data
        return SimpleNamespace(execute=lambda:{'id':fid})
    files=Mock();files.create.side_effect=create
    def listed(**kw):
        matches=[o for o in objects.values() if o['appProperties']['ripped_media_receipt'] in kw['q']]
        return SimpleNamespace(execute=lambda:{'files':copy.deepcopy(matches)})
    files.list.side_effect=listed
    files.get.side_effect=lambda **kw:SimpleNamespace(execute=lambda:copy.deepcopy(objects[kw['fileId']]))
    service=Mock();service.files.return_value=files
    monkeypatch.setattr(main,'get_google_services',lambda:(service,None,None))
    monkeypatch.setattr(main,'_drive_title_folder',lambda title:'library')
    monkeypatch.setattr(storage,'drive_metadata',lambda fid:copy.deepcopy(objects[fid]))
    monkeypatch.setattr(storage,'download_drive',lambda fid,path,metadata:Path(path).write_bytes(blobs[fid]))
    monkeypatch.setattr(__import__('review_navigation'),'notify_safely',Mock())
    monkeypatch.setattr(__import__('intelligence_workflow'),'observe',Mock())
    def render(source,clip,output,renderer,aspect):
        assert boundary.consume(source,clip,aspect)
        calls.append((aspect,clip['start'],clip['end']));output.write_bytes(b'isolated output '+aspect.encode()*100)
        return {**clip,'duration':float(clip['end'])-float(clip['start']),'completion_check':{k:True for k in recovery.PROOF},'boundary_export_evidence':{'full_decode':True}}
    monkeypatch.setattr(speech,'render_complete_clip',render)
    app=FastAPI();app.include_router(web.router)
    token=web.ensure_review_session('request')['token']
    return dict(**p,objects=objects,blobs=blobs,calls=calls,files=files,token=token,app=app,main=main)


def test_playable_before_selection_then_reuse_after_caption_edit_and_new_connection(previews):
    p=previews
    state=p['read']();state['result']['segments'][0].update(end=16.1,duration=6.1);p['write'](state)
    for lane in ('short','highlight'):recovery.generate_preview('request',lane,0,p['source'])
    assert not p['read']()['candidate_reviews'] and not p['read']()['topic_reviews']
    base='/api/ripped-shorts/review/'+p['token']
    with TestClient(p['app']) as client:
        data=client.get(base).json();ready=[c for c in data['clips'] if c['finished']]
        assert len(ready)==2 and {c['format'] for c in ready}=={'9:16','16:9'}
        html=client.get('/review/'+p['token']).text
        assert '<video controls' in html and 'Leave unused' in html and 'Watch previews' in html
        for item in ready:
            media=client.get(base+'/clip/'+item['asset_id']+'/media')
            assert media.status_code==200 and media.content==p['blobs'][item['media_url'].split('/d/')[1].split('/')[0]]
        item=ready[0]
        assert client.post(base+'/clip/'+item['asset_id'],json={'choice':'approve','version':0}).status_code==409
        selected=client.post(base+'/clip/'+item['asset_id'],json={'choice':'approve','version':0,'playback_verified':True,'framing_verified':True})
        assert selected.status_code==200 and not selected.json()['scheduling_started']
    state=p['read']();state['copy_drafts']=[{'asset_id':item['asset_id'],'social_caption':'Owner copy change'}];p['write'](state)
    for lane in ('short','highlight'):recovery.generate_preview('request',lane,0,p['source'])
    with TestClient(p['app']) as restarted:
        assert restarted.get(base).json()['clips'][0]['approval_current']
    assert len(p['calls'])==2 and p['files'].create.call_count==2
    assert len(__import__('batch_progress').selected_assets('request',p['read']()))==1
    assert not p['read']().get('schedule_master') and not p['read']().get('schedule_requested_at')
    assert p['tg']._render_approved('request',0,'chat')['status']=='already_rendered'


@pytest.mark.parametrize('choice',['reject','unused','request_adjustment'])
def test_nonselection_never_deletes_preview_or_selects_or_schedules(previews,choice):
    p=previews;recovery.generate_preview('request','short',0,p['source'])
    asset='request:short:12';body=dict(choice=choice,version=0,reason='weak_hook',
        feedback='Needs more setup',changed_fields=['boundaries'])
    with TestClient(p['app']) as client:
        base='/api/ripped-shorts/review/'+p['token']
        assert client.post(base+'/clip/'+asset,json=body).status_code==200
        assert client.get(base+'/clip/'+asset+'/media').status_code==200
    assert not __import__('batch_progress').selected_assets('request',p['read']())
    assert p['files'].delete.call_count==0 and len(p['objects'])==1


def test_individual_failure_preserves_sibling_and_does_not_repeat_provider(previews):
    p=previews;review=p['reviewer'].side_effect
    def semantic(units,start,end,aspect,**kw):
        value=review(units,start,end,aspect,**kw)
        if aspect=='9:16':value['complete_thought']=False
        return value
    p['reviewer'].side_effect=semantic
    assert recovery.preview_lane('request','short',p['source'])==['held','held']
    assert recovery.preview_lane('request','highlight',p['source'])==['verified']
    before=p['reviewer'].call_count
    assert recovery.preview_lane('request','short',p['source'])==['held','held']
    assert p['reviewer'].call_count==before and p['files'].create.call_count==1


def test_legacy_selected_request_cannot_be_automatically_migrated(previews):
    p=previews;state=p['read']();state.pop('preview_first_version');state['candidate_reviews']={'0':{'status':'render_failed'}};p['write'](state)
    with pytest.raises(recovery.RenderReviewRequired):recovery.start_preview_pipeline('request',p['source'])
    assert p['read']()==state and not p['calls']


def test_word_timing_can_support_sound_without_compulsory_silence(previews):
    p=previews;state=p['read']();state['topic_source_segments'][1]['words']=[dict(word='Complete.',start=10.3,end=15.7)];p['write'](state)
    p['edge'].side_effect=lambda *args,**kw:dict(measured_clear=False,quiet_intervals=[],cut=args[1])
    assert recovery.generate_preview('request','short',0,p['source'])['completion_check']['export_edges']


@pytest.mark.parametrize('lane',['short','highlight'])
def test_actual_ffmpeg_to_retention_authenticated_playback_and_selection(previews,monkeypatch,lane):
    if not shutil.which('ffprobe'):pytest.skip('Real FFprobe/FFmpeg tools installed on CI')
    p=previews;import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();monkeypatch.setenv('FFMPEG_BINARY',ffmpeg)
    monkeypatch.setattr(speech,'render_complete_clip',REAL_RENDER)
    monkeypatch.setattr(speech,'media_duration',REAL_DURATION);monkeypatch.setattr(boundary,'acoustic_edge',REAL_EDGE)
    source=p['tmp']/'real-fixture.mp4';wav=p['tmp']/'real-fixture.wav'
    from array import array
    samples=array('h')
    for i in range(183*16000):
        t=i/16000;gate=max(0.,min(1.,(t-1.2)/.04,(181.2-t)/.04))
        if 3.8<t<4.4:gate=0.
        samples.append(int(32767*gate*(.25*math.sin(2*math.pi*(220*t+20*math.sin(t*.3)))+.1*math.sin(2*math.pi*(370*t+7*math.sin(t*.7))))))
    with wave.open(str(wav),'wb') as w:w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(samples.tobytes())
    subprocess.run([ffmpeg,'-v','error','-y','-f','lavfi','-i','color=c=blue:s=320x180:r=5:d=183','-i',str(wav),'-c:v','libx264','-c:a','aac','-b:a','192k','-shortest',str(source)],check=True)
    state=p['read']();state['source_validation']['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    state['topic_source_segments']=[dict(start=1.2,end=3.8,text='Complete fixture statement.'),dict(start=4.4,end=181.2,text='Complete extended fixture explanation.')]
    key='result' if lane=='short' else 'topic_result'
    state[key]['segments'][0].update(start=.9,end=4.1 if lane=='short' else 181.5,duration=3.2 if lane=='short' else 180.6,framing_mode='blurred' if lane=='short' else None)
    state[key]['segments'][0]['transcript']='Complete fixture statement.' if lane=='short' else 'Complete fixture statement. Complete extended fixture explanation.'
    p['write'](state)
    render=Mock(wraps=p['main'].create_clip_file if lane=='short' else p['main'].create_topic_segment_file)
    monkeypatch.setattr(p['main'],'create_clip_file' if lane=='short' else 'create_topic_segment_file',render)
    clip=recovery.generate_preview('request',lane,0,source)
    assert clip['boundary_export_evidence']['full_decode']
    assert not __import__('batch_progress').selected_assets('request',p['read']())
    base='/api/ripped-shorts/review/'+p['token'];asset=clip['boundary_asset_id']
    with TestClient(p['app']) as client:
        data=client.get(base).json();assert next(c for c in data['clips'] if c['asset_id']==asset)['finished']
        media=client.get(base+'/clip/'+asset+'/media',headers={'Range':'bytes=0-127'})
        assert media.status_code==206 and len(media.content)==128
        assert client.get('/api/ripped-shorts/review/invalid/clip/'+asset+'/media').status_code==404
        assert client.post(base+'/clip/'+asset,json={'choice':'approve','version':0,'playback_verified':True,'framing_verified':True}).status_code==200
    assert recovery.generate_preview('request',lane,0,source)==clip
    assert render.call_count==1 and p['files'].create.call_count==1 and not p['read']().get('schedule_master')
    if os.getenv('PREVIEW_ACCEPTANCE_EXPORT_DIR'):
        # Synthetic acceptance evidence only, for actual browser playback QA.
        # Never export production requests, credentials or retained source media.
        import sqlite3,audio_master_handoff as storage
        root=Path(os.environ['PREVIEW_ACCEPTANCE_EXPORT_DIR'])/lane
        root.mkdir(parents=True,exist_ok=True)
        state=p['read']();state['acceptance_demo']=True;p['write'](state)
        with sqlite3.connect(storage.DB_PATH) as original,sqlite3.connect(root/'ripped_shorts.db') as target:original.backup(target)
        shutil.copytree(p['tmp']/'review-previews',root/'review-previews',dirs_exist_ok=True)
        (root/'review.json').write_text(json.dumps({'token':p['token'],'format':lane,'evidence':'synthetic_ffmpeg_not_production'}))


def test_preview_receipts_survive_an_actual_fresh_python_process(previews):
    p=previews;recovery.generate_preview('request','short',0,p['source'])
    import sys,audio_master_handoff as storage
    code="import sys,json,pathlib; import audio_master_handoff as s; s.DB_PATH=pathlib.Path(sys.argv[1]); import production_intelligence as e,render_recovery as r; db=e.database(); c=db.__enter__(); state=json.loads(c.execute('select state_json from telegram_requests').fetchone()[0]); db.__exit__(None,None,None); print(sum(bool(x[3].get('media_receipt_id')) for x in r.preview_items('request',state)))"
    result=subprocess.run([sys.executable,'-c',code,str(storage.DB_PATH)],capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='1' and len(p['calls'])==1


def test_discovery_dispatches_both_lanes_on_existing_bounded_executor_once(previews,monkeypatch):
    p=previews;state=p['read']();state.pop('preview_first_version');state.pop('copy_drafts',None);p['write'](state)
    import concurrent.futures,threading,intelligence_workflow as workflow
    gate=threading.Barrier(2);seen=[]
    def lane(rid,lane,source):gate.wait(timeout=5);seen.append(lane);return ['verified']
    monkeypatch.setattr(recovery,'preview_lane',lane)
    monkeypatch.setattr(workflow,'prepared_highlights',lambda *args:state['topic_result'])
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        monkeypatch.setattr(p['tg'],'RENDER_EXECUTOR',pool)
        results=recovery.start_preview_pipeline('request',p['source'])
        assert all(f.result(timeout=10)==['verified'] for f in results.values())
        assert recovery.start_preview_pipeline('request',p['source'])=={}
    assert sorted(seen)==['highlight','short'] and not p['read']()['candidate_reviews']
