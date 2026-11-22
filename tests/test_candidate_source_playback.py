import hashlib
import json
import shutil
import subprocess
from urllib.parse import quote

import pytest
from test_website_candidate_review import candidates,read,write,base
from test_editorial_framing_review import review


@pytest.fixture
def source(candidates,tmp_path):
    r=candidates;path=tmp_path/'retained.mp4';path.write_bytes(b'verified-private-source-media')
    state=read(r);state['topic_result']['segments'].append(dict(state['topic_result']['segments'][0],candidate_id='highlight-two'))
    state.update(video_path=str(path),source_validation={'status':'verified','request_id':'existing','source_id':'source','source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'size':path.stat().st_size,'duration':800})
    write(r,state);r['web']._candidate_source_fingerprint.cache_clear()
    return r,path


def url(r,item):
    return base(r)+'/candidate/'+quote(item['asset_id'],safe='')+'/source?revision='+item['revision']


def test_all_seven_players_share_retained_source_and_never_dispatch(source):
    r,path=source;before=read(r);data=r['client'].get(base(r)).json()
    assert len(data['candidates'])==7
    for item in data['candidates']:
        response=r['client'].get(url(r,item),headers={'Range':'bytes=0-7'})
        assert response.status_code==206 and response.content==path.read_bytes()[:8]
        assert response.headers['accept-ranges']=='bytes'
        assert response.headers['cache-control']=='private, no-store'
    assert r['web']._candidate_source_fingerprint.cache_info().misses==1
    page=r['client'].get('/review/'+r['token']).text
    assert page.count('data-source-excerpt data-start=')==7
    assert page.count('Source excerpt — not a finished render.')==7
    assert 'preload="none"' in page and 'data-excerpt-seek' in page
    assert read(r)==before
    r['submit'].assert_not_called();r['telegram'].assert_not_called()


@pytest.mark.parametrize('failure',['stale','wrong_request','source_identity','missing','corrupt','outside','bounds','transcript'])
def test_source_proof_and_session_guards_fail_closed(source,tmp_path,failure):
    r,path=source;item=r['client'].get(base(r)).json()['candidates'][0];target=url(r,item)
    state=read(r)
    if failure=='stale':target+='changed'
    elif failure=='wrong_request':target=target.replace(r['token'],r['web'].ensure_review_session('other')['token'])
    elif failure=='source_identity':state['source_validation']['source_id']='different'
    elif failure=='missing':path.unlink()
    elif failure=='corrupt':path.write_bytes(b'x'*path.stat().st_size)
    elif failure=='outside':state['video_path']=str(tmp_path.parent/'outside.mp4')
    elif failure=='bounds':state['source_validation']['duration']=1
    elif failure=='transcript':state['transcript_evidence']={}
    write(r,state)
    assert r['client'].get(target).status_code==409
    r['submit'].assert_not_called()


def test_real_mp4_source_has_decodable_video_audio_and_range_playback(source,tmp_path):
    ffmpeg=shutil.which('ffmpeg');probe=shutil.which('ffprobe')
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:pytest.skip('Existing FFmpeg tools unavailable')
    r,path=source;path.unlink()
    subprocess.run([ffmpeg,'-v','error','-f','lavfi','-i','testsrc2=size=160x90:rate=10','-f','lavfi','-i','sine=frequency=440:sample_rate=44100','-t','2','-c:v','libx264','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(path)],check=True)
    state=read(r);state['source_validation'].update(size=path.stat().st_size,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),duration=2)
    state['result']['segments'][0].update(start=.2,end=1.6,duration=1.4)
    write(r,state);item=r['client'].get(base(r)).json()['candidates'][0]
    response=r['client'].get(url(r,item));assert response.status_code==200
    output=tmp_path/'served.mp4';output.write_bytes(response.content)
    if probe:
        streams=json.loads(subprocess.check_output([probe,'-v','error','-show_streams','-of','json',str(output)]))['streams']
        assert {'video','audio'}<={s['codec_type'] for s in streams}
    decoded=subprocess.run([ffmpeg,'-xerror','-i',str(output),'-af','volumedetect','-f','null','-'],capture_output=True,text=True,check=True)
    assert 'Video:' in decoded.stderr and 'Audio:' in decoded.stderr
    assert 'mean_volume:' in decoded.stderr and 'mean_volume: -inf' not in decoded.stderr


def test_excerpt_controls_clamp_seek_stop_replay_and_pause_siblings(source,tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node unavailable')
    r,_=source;page=r['client'].get('/review/'+r['token']).text
    script=page[page.index("document.querySelectorAll('[data-source-excerpt]')"):page.index("document.querySelectorAll('[data-candidate-choice]')")]
    harness="""
const assert=require('node:assert/strict');let timers=[];
global.setTimeout=(fn,ms)=>{timers.push({fn,ms});return timers.length;};global.clearTimeout=()=>{};
function player(start,end){const handlers={},seek={value:0,addEventListener(n,fn){this[n]=fn;}},status={};return {dataset:{start,end},currentTime:0,playbackRate:1,paused:true,handlers,seek,status,addEventListener(n,fn){handlers[n]=fn;},pause(){this.paused=true;handlers.pause?.();},closest(){return {querySelector:s=>s==='[data-excerpt-seek]'?seek:status};}};}
const a=player(10,30),b=player(168.27999877929688,247.33999633789062);
let clock=0,seeks=0;Object.defineProperty(b,'currentTime',{get(){return clock;},set(value){clock=Math.floor(value*1e6)/1e6;if(++seeks>10)throw Error('Repeated quantized seek');b.handlers.seeking?.();}});
global.document={querySelectorAll:()=>[a,b]};
"""+script+"""
b.handlers.loadedmetadata();assert(Math.abs(b.currentTime-Number(b.dataset.start))<.001);assert.equal(seeks,1);
b.currentTime=Number(b.dataset.end);b.paused=false;b.handlers.play();assert(Math.abs(b.currentTime-Number(b.dataset.start))<.001);
a.handlers.loadedmetadata();assert.equal(a.currentTime,10);
b.paused=false;a.paused=false;a.handlers.play();assert(b.paused);assert.equal(timers.at(-1).ms,20000);
timers.at(-1).fn();assert.equal(a.currentTime,10);assert(!a.paused); // buffering cannot skip speech
a.handlers.waiting();a.handlers.playing();
a.currentTime=5;a.handlers.seeking();assert.equal(a.currentTime,10);
a.seek.value=7;a.seek.input();assert.equal(a.currentTime,17);
a.currentTime=31;a.handlers.timeupdate();assert(a.paused);assert.equal(a.currentTime,30);
a.paused=false;a.handlers.play();assert.equal(a.currentTime,10);
a.currentTime=31;a.handlers.seeking();assert(a.paused);assert.equal(a.currentTime,30);
a.handlers.error();assert(a.status.textContent.includes('decisions are preserved'));
"""
    path=tmp_path/'controls.cjs';path.write_text(harness,encoding='utf-8')
    result=subprocess.run([node,str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
