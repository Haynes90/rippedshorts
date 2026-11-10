import json
from unittest.mock import Mock
import pytest
from fastapi.testclient import TestClient
from test_editorial_framing_review import review
from test_finished_clip_review import render_fixture,read_state,save_state,post
from caption_checkpoints import complete_post,transport_copy,editor_revision


@pytest.fixture
def editor(review,monkeypatch):
    r=review;render_fixture(r)
    assert post(r,choice='approve',version=0,playback_verified=True).status_code==200
    monkeypatch.setattr(r['tg'],'_checkpoint_copy_edit',Mock())
    monkeypatch.setattr(r['tg'],'_refresh_copy_review',Mock(side_effect=RuntimeError('Telegram sendMessage failed (400): Bad Request: BUTTON_DATA_INVALID')))
    monkeypatch.setattr(__import__('intelligence_workflow'),'observe',Mock())
    return r


@pytest.mark.parametrize('kind',['9:16_SHORT','16:9_HIGHLIGHT'])
def test_complete_post_roundtrip_preserves_selected_media_and_exact_copy(editor,kind):
    r=editor;before=read_state(r);before['copy_drafts'][0].update(asset_type=kind,video_title='Owner title',hashtags='#seeds')
    save_state(r,before);base='/api/ripped-shorts/review/'+r['token'];body='🌱 Seeds: “keep them dry!”\nQuestion → answer.\n\n#seeds #jardín #seeds'
    payload=r['client'].get(base).json();draft=payload['items'][0]
    response=r['client'].post(base+'/asset/0',json={'complete_post':body,'copy_revision':draft['copy_revision'],'video_title':'Owner title','action':'save'})
    assert response.status_code==200 and response.json()['status']=='saved'
    after=read_state(r);saved=after['copy_drafts'][0]
    assert complete_post(saved)==body and saved['hashtags']=='#seeds #jardín'
    assert after['website_clip_reviews']==before['website_clip_reviews']
    assert after['result']==before['result'] and after['candidate_reviews']==before['candidate_reviews']
    assert r['client'].get(base).json()['items'][0]['complete_post']==body
    page=r['client'].get('/review/'+r['token']).text
    assert 'id="hashtags-0"' not in page and 'recommendation/framing evidence' not in page
    assert 'Complete '+('caption' if kind=='9:16_SHORT' else 'description') in page
    r['tg']._refresh_copy_review.assert_not_called();r['telegram'].assert_not_called()
    assert not after.get('schedule_requested_at') and not after.get('schedule_master')
    wire=transport_copy({'asset_id':saved['asset_id'],'drive_url':'unchanged',**saved})
    field='social_caption' if kind=='9:16_SHORT' else 'video_description'
    assert wire[field]==body and wire['hashtags']==''
    assert '\n\n'.join(p for p in (wire[field],wire['hashtags']) if p)==body
    from workflow_test_status import copy_revision
    assert copy_revision([saved])==copy_revision([wire])


def test_old_callback_limit_cannot_fail_committed_website_save(editor):
    r=editor;base='/api/ripped-shorts/review/'+r['token']
    _,markup=r['tg']._copy_review_payload('x'*80,read_state(r)['copy_drafts'],0)
    assert any(len(b.get('callback_data','').encode())>64 for line in markup['inline_keyboard'] for b in line)
    response=r['client'].post(base+'/asset/0',json={'social_caption':'Saved without a Telegram menu','action':'save'})
    assert response.status_code==200 and read_state(r)['copy_drafts'][0]['social_caption']=='Saved without a Telegram menu'
    r['tg']._refresh_copy_review.assert_not_called()


def test_learning_service_failure_does_not_erase_committed_copy(editor):
    r=editor;r['tg']._checkpoint_copy_edit.side_effect=RuntimeError('isolated Sheets unavailable')
    response=r['client'].post('/api/ripped-shorts/review/'+r['token']+'/asset/0',json={'social_caption':'Keep this exact copy'})
    assert response.status_code==200 and response.json()['checkpoint_pending']
    assert read_state(r)['copy_drafts'][0]['social_caption']=='Keep this exact copy'


def test_stale_copy_revision_is_not_resubmitted_or_overwritten(editor):
    r=editor;base='/api/ripped-shorts/review/'+r['token'];original=r['client'].get(base).json()['items'][0]
    a={'complete_post':'First saved edit #a','copy_revision':original['copy_revision']}
    assert r['client'].post(base+'/asset/0',json=a).status_code==200
    r['tg']._checkpoint_copy_edit.reset_mock()
    assert r['client'].post(base+'/asset/0',json={**a,'complete_post':'Stale second tab'}).status_code==409
    assert r['client'].get(base).json()['items'][0]['complete_post']==a['complete_post']
    r['tg']._checkpoint_copy_edit.assert_not_called()
    assert not read_state(r).get('schedule_requested_at')


def test_legacy_hashtags_combined_once_and_submitted_post_is_frozen(editor):
    assert complete_post(dict(asset_type='9:16_SHORT',social_caption='Exact #seeds\nWords',hashtags='#seeds #garden #garden'))=='Exact #seeds\nWords\n\n#garden'
    r=editor;s=read_state(r);s['schedule_master']={'shorts_status':'accepted','shorts_response':{'asset_group_ids':['existing:short:1']}};save_state(r,s)
    d=s['copy_drafts'][0]
    response=r['client'].post('/api/ripped-shorts/review/'+r['token']+'/asset/0',json={'complete_post':'Attempt to alter submitted copy','copy_revision':editor_revision(d)})
    assert response.status_code==409 and read_state(r)==s


def test_frontend_uncertain_response_reads_back_without_duplicate_post(editor,tmp_path):
    import shutil,subprocess
    if not shutil.which('node'):pytest.skip('Node required for frontend save regression')
    r=editor;page=r['client'].get('/review/'+r['token']).text
    code='function bodyFor'+page.split('function bodyFor',1)[1].split('async function keepAsset',1)[0]
    harness=r'''
const assert=require('node:assert/strict');
const TOKEN='isolated';let state={};function updateProgress(){}
const entries=new Map();for(const id of ['caption-0','favorite-0','hold-0','feedback-0','saved-0','status-0','playback-0'])entries.set(id,{value:'',checked:false,textContent:''});
entries.set('asset-0',{dataset:{index:'0',asset:'stable-asset',type:'9:16_SHORT',copyRevision:'r0'}});
const document={getElementById:id=>entries.get(id),querySelectorAll:()=>[]};
const local=new Map();const localStorage={setItem:(k,v)=>local.set(k,v),getItem:k=>local.get(k)||null,removeItem:k=>local.delete(k)};
let posts=0,gets=0,mode='committed500',current={index:0,copy_status:'edited',copy_revision:'r0',complete_post:'Old',favorite:false,editorial_hold:false,selection_feedback:''};
const response=(status,data)=>({ok:status===200,status,text:async()=>typeof data==='string'?data:JSON.stringify(data)});
global.fetch=async(url,options={})=>{
 if(options.method==='POST'){
   posts++;if(mode==='offline')throw new Error('isolated network failure');
   const body=JSON.parse(options.body);current={...current,...body,copy_revision:'r'+posts};
   if(mode==='committed500')return response(500,'Internal Server Error');
   return response(200,{status:'saved',review:{items:[current]}});
 }
 gets++;return response(200,{items:[current]});
};
'''+code+r'''
(async()=>{
 entries.get('caption-0').value='🌱 Exact\nwords #seeds';
 const result=await saveAsset(0);
 assert.equal(result.status,'saved_reconciled');assert.equal(posts,1);assert.equal(gets,1);
 assert.match(entries.get('saved-0').textContent,/confirmed by server readback/);assert.equal(local.size,0);
 mode='offline';entries.get('caption-0').value='Unsaved emoji 🌱\n#jardín';
 await assert.rejects(saveAsset(0));assert.equal(posts,2);assert.equal(local.size,1);
 assert.equal(entries.get('caption-0').value,'Unsaved emoji 🌱\n#jardín');
 mode='success';const before=posts,a=saveAsset(0),b=saveAsset(0);assert.equal(a,b);await Promise.all([a,b]);
 assert.equal(posts,before+1);assert.equal(local.size,0);assert.match(entries.get('saved-0').textContent,/Saved/);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    file=tmp_path/'save-behavior.cjs';file.write_text(harness,encoding='utf-8')
    subprocess.run(['node',str(file)],check=True,capture_output=True,text=True)
