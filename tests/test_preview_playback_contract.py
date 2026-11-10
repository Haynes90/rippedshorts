"""Isolated ledger/HTTP checks. Live owner decisions are never submitted by tests."""
import copy
import json
from unittest.mock import Mock
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from test_boundary_specification import passage
from test_preview_vertical_slice import previews
import boundary_specification as boundary
import render_recovery as recovery


def test_caption_and_primary_cards_resolve_same_verified_media(previews):
    p=previews;clip=recovery.generate_preview('request','short',0,p['source'])
    asset=clip['boundary_asset_id'];s=p['read']()
    s['candidate_reviews']['0']={'status':'rendered','clip_url':'https://drive.google.com/file/d/stale-legacy-id/view'}
    s['copy_drafts']=[dict(asset_id=asset,asset_type='9:16_SHORT',candidate_number=12,social_caption='Owner caption')]
    p['write'](s)
    import caption_review_web as web
    payload=web._payload_for_session(p['token'])
    assert payload['items'][0]['drive_url']==clip['clip_url']
    assert '/clip/'+__import__('urllib.parse',fromlist=['quote']).quote(asset,safe='')+'/media' in payload['items'][0]['preview_url']
    assert payload['clips'][0]['provenance']
    with TestClient(p['app']) as client:
        page=client.get('/review/'+p['token'])
        assert page.status_code==200 and page.headers['cache-control']=='private, no-store'
        assert page.text.count('<video controls playsinline')>=2
        assert 'Selection evidence' not in page.text and 'Why this clip?' in page.text
        media=payload['items'][0]['preview_url']
        assert client.get(media,headers={'Range':'bytes=0-127'}).status_code==206
        assert client.get(media,headers={'Range':'bytes=-128'}).status_code==206
        assert client.get(media.replace(p['token'],'invalid')).status_code==404
    assert len(p['calls'])==1 and p['files'].create.call_count==1


def test_missing_retention_receipt_is_not_selectable_and_never_rerenders(previews):
    p=previews;clip=recovery.generate_preview('request','short',0,p['source'])
    import production_intelligence as intelligence
    with intelligence.database() as db:db.execute('DELETE FROM intelligence_records WHERE record_id=?',(clip['media_receipt_id'],))
    import caption_review_web as web
    assert not web._media_for_state(p['read'](),'request')
    items=__import__('finished_clip_review').clips('request',p['read']())
    assert items[0]['finished'] is False and 'receipt unavailable' in items[0]['error']
    with TestClient(p['app']) as client:
        base='/api/ripped-shorts/review/'+p['token']+'/clip/'+clip['boundary_asset_id']
        assert client.get(base+'/media').status_code==409
        assert client.post(base,json={'choice':'approve','version':0,'playback_verified':True,'framing_verified':True}).status_code==409
    assert len(p['calls'])==1 and p['files'].create.call_count==1


@pytest.mark.parametrize('failure', ['missing','silent'])
def test_source_audio_failure_precedes_semantic_and_video_work(previews,monkeypatch,failure):
    p=previews
    import main
    reason='SOURCE_AUDIO_UNAVAILABLE' if failure=='missing' else 'PASSAGE_AUDIO_SILENT'
    check=Mock(side_effect=main.SourceAudioReviewRequired('Unusable test passage',reason))
    monkeypatch.setattr(main,'_assert_source_audio',check)
    with pytest.raises(boundary.BoundaryReviewRequired,match=reason):
        recovery.generate_preview('request','short',0,p['source'])
    assert not p['calls'] and not p['reviewer'].called and not p['files'].create.called
    with pytest.raises(recovery.RenderReviewRequired):recovery.generate_preview('request','short',0,p['source'])
    assert check.call_count==1


def test_validated_source_audio_cached_without_repeating_semantics(passage,monkeypatch):
    p=passage;check=Mock(return_value=-2.)
    monkeypatch.setattr(__import__('main'),'_assert_source_audio',check)
    first=p['prepare']();second=p['prepare']()
    assert first==second and check.call_count==1 and p['reviewer'].call_count==1
    spec=boundary.consume(p['source'],first,'9:16')
    assert spec['source_audio']['status']=='VERIFIED' and spec['source_audio']['max_volume_db']==-2.


def test_partial_corrupt_audio_decode_cannot_pass_peak_measurement(monkeypatch):
    import main
    run=Mock(side_effect=[SimpleNamespace(returncode=0,stdout='audio',stderr=''),
        SimpleNamespace(returncode=1,stdout='',stderr='max_volume: -1.0 dB')])
    monkeypatch.setattr(main.subprocess,'run',run)
    assert main._audio_max_volume_db('corrupt-fixture') is None



def test_actual_silent_source_is_held_before_semantics_or_video_render(previews,monkeypatch):
    import shutil,subprocess,hashlib
    if not shutil.which('ffprobe'):pytest.skip('FFmpeg tools are installed on CI')
    import imageio_ffmpeg
    from test_preview_vertical_slice import REAL_SOURCE_AUDIO,REAL_DURATION
    p=previews;source=p['tmp']/'silent.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-f','lavfi','-i','color=s=320x180:d=3','-f','lavfi','-i','anullsrc=r=16000:cl=mono','-t','3','-c:v','libx264','-c:a','aac',str(source)],check=True)
    s=p['read']();s['source_validation']['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    s['topic_source_segments']=[dict(start=.2,end=2.8,text='Verified test words.')]
    s['result']['segments'][0].update(start=.1,end=2.9,duration=2.8,transcript='Verified test words.')
    p['write'](s)
    monkeypatch.setattr(p['main'],'_assert_source_audio',REAL_SOURCE_AUDIO)
    monkeypatch.setattr(__import__('clip_completion'),'media_duration',REAL_DURATION)
    with pytest.raises(boundary.BoundaryReviewRequired,match='PASSAGE_AUDIO_SILENT'):
        recovery.generate_preview('request','short',0,source)
    assert not p['reviewer'].called and not p['calls'] and not p['files'].create.called



def test_silent_passage_does_not_block_healthy_sibling(previews,monkeypatch):
    import main
    p=previews
    def source_audio(path,*,start,duration,aspect):
        if start<15:raise main.SourceAudioReviewRequired('Silent test passage','PASSAGE_AUDIO_SILENT')
        return -2.
    monkeypatch.setattr(main,'_assert_source_audio',source_audio)
    assert recovery.preview_lane('request','short',p['source'])==['held','verified']
    assert len(p['calls'])==1 and p['calls'][0][1]==20.
    assert not p['read']()['candidate_reviews']


def test_player_gate_requires_decoded_media_and_preserves_frozen_approvals(previews,tmp_path):
    import shutil,subprocess
    if not shutil.which('node'):pytest.skip('Existing Node runtime required for browser gate test')
    p=previews;clip=recovery.generate_preview('request','short',0,p['source'])
    cards=__import__('clip_review_ui').finished_cards(__import__('finished_clip_review').clips('request',p['read']()),p['token'])
    script=cards.split('<script>',1)[1].split('async function clipDecision',1)[0]
    harness="""const assert=require('node:assert/strict');
const listeners={};const video={readyState:1,error:null,addEventListener:(e,f)=>listeners[e]=f};
const approve={disabled:false},status={textContent:''};
const card={dataset:{frozen:'0'},querySelector:s=>s==='video'?video:s==='[data-approve-clip]'?approve:status};
global.document={querySelectorAll:()=>[card]};
"""+script+"""
assert.equal(approve.disabled,true);
video.readyState=3;listeners.loadeddata();assert.equal(approve.disabled,false);
video.error={code:4};listeners.error();assert.equal(approve.disabled,true);assert.match(status.textContent,/could not load/);
video.error=null;card.dataset.frozen='1';listeners.loadeddata();assert.equal(approve.disabled,true);
"""
    file=tmp_path/'player-gate.cjs';file.write_text(harness,encoding='utf-8')
    subprocess.run(['node',str(file)],check=True,capture_output=True,text=True)



def test_render_scope_reuses_exact_source_measurement_and_always_resets(passage,monkeypatch):
    import main,clip_completion as speech
    from test_preview_vertical_slice import REAL_SOURCE_AUDIO
    p=passage;prepared=p['prepare']()
    measure=Mock(return_value=-2.)
    monkeypatch.setattr(main,'_audio_max_volume_db',measure)
    monkeypatch.setattr(main,'_assert_source_audio',REAL_SOURCE_AUDIO)
    def renderer(source,start,duration,out):
        assert main._assert_source_audio(source,start=start,duration=duration,aspect='9:16')==-1.
        assert not measure.called
        raise RuntimeError('isolated renderer stopped')
    with pytest.raises(RuntimeError,match='isolated renderer stopped'):
        speech.render_complete_clip(p['source'],prepared,p['tmp']/'out.mp4',renderer,'9:16')
    assert speech._VALIDATED_SOURCE_AUDIO.get() is None
    assert main._assert_source_audio(p['source'],start=10.,duration=6.,aspect='9:16')==-2.
    assert measure.call_count==1
