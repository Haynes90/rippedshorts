"""Existing HTTP persistence and the actual embedded review JavaScript."""
import copy,json,re,shutil,subprocess
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from test_preview_vertical_slice import previews
from test_boundary_specification import passage


def test_two_decisions_persist_count_and_do_not_duplicate_or_touch_another_request(previews):
    p=previews;import render_recovery as recovery
    s=p['read']();s.update(review_interface='website')
    s['result']['segments'][0].update(candidate_number=1,end=16.1,duration=6.1)
    s['result']['segments'][1].update(candidate_number=4);p['write'](s)
    for index in (0,1):recovery.generate_preview('request','short',index,p['source'])
    with p['tg']._telegram_db() as db:
        db.execute("INSERT INTO telegram_requests SELECT 'unrelated','other-key',chat_id,user_id,status,mode,source_kind,source_value,state_json,created_at,updated_at FROM telegram_requests WHERE request_id='request'")
        other=dict(db.execute("SELECT * FROM telegram_requests WHERE request_id='unrelated'").fetchone())
    base='/api/ripped-shorts/review/'+p['token']
    with TestClient(p['app']) as client:
        before=client.get(base).json();one,four=before['clips']
        approve=dict(choice='approve',version=0,media_revision=one['media_revision'],playback_verified=True,framing_verified=True)
        reject=dict(choice='reject',version=0,media_revision=four['media_revision'],reason='weak_hook',feedback='Owner fixture wording 🌱\nKeep this feedback.')
        for item,body,missing in [(one,approve,{'framing_verified':False}),(four,reject,{'reason':''})]:
            failed=client.post(base+'/clip/'+item['asset_id'],json={**body,**missing})
            assert failed.status_code==409 and failed.json()['detail']
        assert client.get(base).json()['preview_review_completed']==0
        for count,(item,body) in enumerate(((one,approve),(four,reject)),1):
            result=client.post(base+'/clip/'+item['asset_id'],json=body)
            assert result.status_code==200 and result.json()['status']=='saved'
            assert result.json()['decision']['media_revision']==item['media_revision']
            assert client.get(base).json()['preview_review_completed']==count
            assert client.post(base+'/clip/'+item['asset_id'],json=body).status_code==409
            repeated=client.post(base+'/clip/'+item['asset_id'],json={**body,'version':1})
            assert repeated.status_code==200 and repeated.json()['unchanged'] and repeated.json()['decision']['version']==1
        # The exact served media revision is checked independently of the version.
        assert client.post(base+'/clip/'+one['asset_id'],json={**approve,'version':1,'media_revision':'different'}).status_code==409
    with TestClient(p['app']) as reopened:
        d=reopened.get(base).json();assert (d['preview_review_completed'],d['preview_review_total'])==(2,2)
        decisions=[c['decision'] for c in d['clips']]
        assert [x['choice'] for x in decisions]==['approve','reject']
        assert all(x['version']==1 and not x['prior_versions'] for x in decisions)
        assert decisions[1]['feedback']==reject['feedback'] and not d['schedule_requested']
    with p['tg']._telegram_db() as db:assert dict(db.execute("SELECT * FROM telegram_requests WHERE request_id='unrelated'").fetchone())==other
    assert len(p['calls'])==2 and p['files'].create.call_count==2


def test_embedded_review_js_validates_preserves_reconciles_and_counts(tmp_path):
    node=shutil.which('node')
    if not node:pytest.skip('Node is provided on CI')
    from clip_review_ui import FINISHED_REVIEW_SCRIPT
    script=FINISHED_REVIEW_SCRIPT.removeprefix('<script>').removesuffix('</script>')
    harness=r'''
const assert=require('node:assert/strict');
const TOKEN='isolated',INITIAL={request_id:'request',preview_first:true};let state=INITIAL;
let counter=[],posts=0,mode='ok',release,gets=0,reloads=0;
const storage=new Map(),localStorage={getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)};
const location={reload:()=>reloads++};
const ids=new Map(),cards=[],saved={};
function element(value=''){return {value,checked:false,textContent:'',handlers:{},setAttribute(k,v){this[k]=v;}};}
for(const asset of ['request:short:1','request:short:4']){
 const key=asset.replaceAll(':','-'),video={readyState:4,error:null,handlers:{},addEventListener(n,f){this.handlers[n]=f;}};
 const controls={};for(const action of ['approve','reject','request_adjustment','unused'])controls[action]={dataset:{clipAction:action},disabled:false};
 const card={dataset:{asset,version:'0',mediaRevision:'media-'+asset,preview:'1',frozen:'0'},handlers:{},
  querySelector(s){return s==='video'?video:s==='[data-player-status]'?this.player:this.requirements;},
  querySelectorAll(s){return s==='input,select,textarea'?['playback','framing','reason','feedback','edit'].map(f=>ids.get('clip-'+f+'-'+key)):Object.values(controls);},addEventListener(n,f){this.handlers[n]=f;},player:element(),requirements:element(),controls};
 ids.set('clip-'+key,card);for(const field of ['playback','framing','reason','feedback','edit','status'])ids.set('clip-'+field+'-'+key,element(field==='edit'?'boundaries':''));
 cards.push(card);saved[asset]={};
}
const document={readyState:'complete',querySelectorAll:()=>cards,getElementById:id=>ids.get(id)};
function updateProgress(d){counter=[d.preview_review_completed,d.preview_review_total];}
function payload(){return {request_id:'request',preview_first:true,preview_review_total:2,
 preview_review_completed:Object.values(saved).filter(x=>x.choice).length,
 clips:cards.map(c=>({asset_id:c.dataset.asset,media_revision:c.dataset.mediaRevision,decision:saved[c.dataset.asset],frozen:false}))};}
async function fetch(url,options){
 if(!options||!options.method){gets++;if(mode==='offline')throw new Error('offline');return {ok:true,status:200,json:async()=>payload()};}
 posts++;const b=JSON.parse(options.body),asset=decodeURIComponent(url.split('/').pop());
 if(mode==='wait')await new Promise(r=>release=r);
 if(mode==='reject')return {ok:false,status:409,json:async()=>({detail:'Stale review version; refresh before saving'})};
 if(mode==='no-commit')throw new Error('connection lost');
 saved[asset]={...b,version:b.version+1,reason:b.choice==='reject'?b.reason:'',framing_reviewed:b.choice==='approve',prior_versions:[]};
 if(mode==='lost-response')throw new Error('response lost after commit');
 return {ok:true,status:200,json:async()=>({status:'saved',decision:saved[asset],copy_review_ready:false})};
}
'''+script+r'''
(async()=>{
 const a=clipParts(cards[0].dataset.asset),b=clipParts(cards[1].dataset.asset);
 assert(a.card.controls.approve.disabled);assert(a.card.controls.reject.disabled);
 a.feedback.value='Entered text 🌱\nkept';a.card.handlers.input();
 await clipDecision(a.card.dataset.asset,'approve');assert.equal(posts,0);assert(a.status.textContent.includes('Confirm'));
 a.playback.checked=true;a.framing.checked=true;a.card.handlers.change();
 assert(!a.card.controls.approve.disabled);
 mode='wait';const first=clipDecision(a.card.dataset.asset,'approve');const duplicate=clipDecision(a.card.dataset.asset,'approve');assert.equal(first,duplicate);
 await new Promise(r=>setImmediate(r));assert(a.card.controls.approve.disabled);assert(a.feedback.disabled);assert.equal(posts,1);release();await first;assert(!a.feedback.disabled);
 assert(a.status.textContent.includes('approve'));assert.equal(a.card.dataset.version,1);assert.deepEqual(counter,[1,2]);
 b.reason.value='weak_hook';b.feedback.value='Specific owner fixture reason';b.card.handlers.input();
 mode='reject';await clipDecision(b.card.dataset.asset,'reject');assert(b.status.textContent.startsWith('Not saved:'));assert(b.status.textContent.includes('Stale'));
 assert.equal(b.feedback.value,'Specific owner fixture reason');assert.deepEqual(counter,[1,2]);assert.equal(saved[b.card.dataset.asset].choice,undefined);
 refreshClipReview(payload());assert(b.status.textContent.startsWith('Not saved:')); // polling cannot hide the failure
 assert.equal(clipStored(b.card).attempt,'reject');initializeClipReviews();
 assert(b.status.textContent.startsWith('Not saved:'));assert.equal(b.feedback.value,'Specific owner fixture reason');
 mode='lost-response';await clipDecision(b.card.dataset.asset,'reject');assert(b.status.textContent.includes('Saved and verified'));
 assert.deepEqual(counter,[2,2]);assert.equal(posts,3);assert.equal(saved[b.card.dataset.asset].version,1);
 initializeClipReviews();assert.equal(b.feedback.value,'Specific owner fixture reason');
 // An uncommitted uncertain request is reconciled before another POST.
 mode='no-commit';await clipDecision(b.card.dataset.asset,'unused');assert(b.status.textContent.includes('No matching saved decision'));
 const previous=posts;assert.equal(clipStored(b.card).pending,null);
 mode='ok';await clipDecision(b.card.dataset.asset,'unused');assert.equal(posts,previous+1);
 // A request whose acknowledgment and readback are unavailable stays pending.
 mode='offline';
 const priorFetch=fetch;fetch=async(u,o)=>{if(o&&o.method)throw new Error('unknown outcome');return priorFetch(u,o);};
 await clipDecision(b.card.dataset.asset,'reject');assert(clipStored(b.card).pending);
 const last=posts;await clipDecision(b.card.dataset.asset,'reject');assert.equal(posts,last);
 localStorage.getItem=()=>{throw new Error('storage unavailable');};localStorage.setItem=()=>{throw new Error('storage unavailable');};
 assert(clipStored(b.card).pending);await clipDecision(b.card.dataset.asset,'reject');assert.equal(posts,last);
 assert.equal(b.feedback.value,'Specific owner fixture reason');
 // Only authoritative state advances counts; no owner action is fabricated.
 assert.deepEqual(counter,[2,2]);assert(gets>=5);assert.equal(reloads,0);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    path=tmp_path/'review.cjs';path.write_text(harness,encoding='utf-8')
    run=subprocess.run([node,str(path)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr
