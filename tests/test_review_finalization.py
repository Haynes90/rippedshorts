import json
from unittest.mock import Mock
import pytest
from test_editorial_framing_review import review
from test_review_copy_save import editor
from test_finished_clip_review import read_state, save_state


def test_empty_highlight_save_is_durable_but_never_copy_approval(editor,monkeypatch):
    r=editor;s=read_state(r)
    s['copy_drafts'][0].update(asset_type='16:9_HIGHLIGHT',video_title='Repot a Birkin',video_description='',hashtags='',user_edited=False)
    save_state(r,s);base='/api/ripped-shorts/review/'+r['token']
    item=r['client'].get(base).json()['items'][0]
    response=r['client'].post(base+'/asset/0',json={'complete_post':'','video_title':'Repot a Birkin','copy_revision':item['copy_revision']})
    assert response.status_code==200
    assert read_state(r)['copy_drafts'][0]['copy_status']=='pending'
    monkeypatch.setattr(__import__('creator_approval'),'approved_batch',lambda _:dict(approved_at='now',approval_revision=1))
    notify=Mock();monkeypatch.setattr(r['tg'],'_notify_render_queue_complete',notify)
    assert r['client'].post(base+'/finish').status_code==409
    notify.assert_not_called()


def test_missing_copy_uses_clip_transcript_cache_and_never_owner_edits(editor,monkeypatch):
    r=editor;s=read_state(r);s['vid_title']='Plantcon plants';s['copy_drafts'][0].update(asset_type='16:9_HIGHLIGHT',video_title='Birkin',video_description='',transcript='Hold the pot. Backfill gently. Firm with your thumbs.',user_edited=False)
    save_state(r,s);base='/api/ripped-shorts/review/'+r['token']
    generate=Mock(side_effect=lambda assets,*a:[{**assets[0],'video_title':'How to Repot a Philodendron Birkin','video_description':'Backfill and firm the soil gently.','hashtags':'#Philodendron #Repotting'}])
    monkeypatch.setattr(r['tg'],'_generate_schedule_copy',generate)
    # A concurrent owner edit after provider work must win; the generated receipt
    # survives and is reused if the same untouched passage is requested again.
    original=s['copy_drafts'][0].copy()
    def competing(assets,*a):
        latest=read_state(r);latest['copy_drafts'][0].update(video_description='Exact owner words 🌱\n#Birkin',user_edited=True)
        save_state(r,latest)
        return generate(assets,*a)
    monkeypatch.setattr(r['tg'],'_generate_schedule_copy',competing)
    response=r['client'].post(base+'/asset/0/regenerate?missing_only=true')
    assert response.status_code==409
    assert read_state(r)['copy_drafts'][0]['video_description']=='Exact owner words 🌱\n#Birkin'
    assert r['client'].post(base+'/asset/0/regenerate?missing_only=true').json()['status']=='preserved'
    assert generate.call_count==1
    latest=read_state(r);latest['copy_drafts'][0]=original;save_state(r,latest)
    result=r['client'].post(base+'/asset/0/regenerate?missing_only=true')
    assert result.status_code==200 and result.json()['video_title']=='How to Repot a Philodendron Birkin'
    assert result.json()['copy_status']=='pending'
    assert generate.call_count==1
    assert generate.call_args.args[0][0]['transcript']==original['transcript']
    assert read_state(r)['website_clip_reviews']==s['website_clip_reviews']
    assert not read_state(r).get('schedule_requested_at')


def test_edited_highlight_is_never_regenerated(editor,monkeypatch):
    r=editor;s=read_state(r);s['copy_drafts'][0].update(asset_type='16:9_HIGHLIGHT',video_title='Owner title',video_description='Friendw',user_edited=True)
    save_state(r,s);generate=Mock();monkeypatch.setattr(r['tg'],'_generate_schedule_copy',generate)
    base='/api/ripped-shorts/review/'+r['token']+'/asset/0/regenerate'
    assert r['client'].post(base+'?missing_only=true').json()['status']=='preserved'
    assert r['client'].post(base).status_code==409
    assert read_state(r)==s;generate.assert_not_called()


def test_final_approval_survives_telegram_failure_and_reconciles_receipt(editor,monkeypatch):
    r=editor;base='/api/ripped-shorts/review/'+r['token']
    monkeypatch.setattr(__import__('creator_approval'),'approved_batch',lambda _:dict(approved_at='now',approval_revision=1))
    monkeypatch.setattr(r['tg'],'_log_copy_learning',Mock(side_effect=RuntimeError('notification unavailable')))
    calls=[]
    def handoff(rid,chat):
        s=read_state(r);calls.append(rid)
        s['schedule_master']={'shorts_status':'accepted','shorts_response':{'asset_group_ids':['existing:short:1'],'delivery_ids':['isolated-delivery']}}
        save_state(r,s)
    monkeypatch.setattr(r['tg'],'_notify_render_queue_complete',handoff)
    before=read_state(r)
    first=r['client'].post(base+'/finish')
    assert first.status_code==200 and first.json()['status']=='scheduler_accepted'
    assert first.json()['receipt']['delivery_ids']==['isolated-delivery']
    after=read_state(r)
    assert after['schedule_requested_at'] and after['copy_review_completed_at']
    assert after['copy_drafts']==before['copy_drafts'] and after['website_clip_reviews']==before['website_clip_reviews']
    assert r['client'].post(base+'/finish').json()['receipt']==first.json()['receipt']
    assert calls==['existing']
    assert r['client'].get(base).json()['scheduling_result']==first.json()


def test_uncertain_handoff_never_claims_acceptance_or_loses_approval(editor,monkeypatch):
    r=editor;base='/api/ripped-shorts/review/'+r['token']
    monkeypatch.setattr(__import__('creator_approval'),'approved_batch',lambda _:dict(approved_at='now',approval_revision=1))
    monkeypatch.setattr(r['tg'],'_log_copy_learning',Mock())
    monkeypatch.setattr(r['tg'],'send',Mock())
    monkeypatch.setattr(r['tg'],'_notify_render_queue_complete',Mock(side_effect=TimeoutError('uncertain')))
    response=r['client'].post(base+'/finish')
    assert response.status_code==200 and response.json()['status']=='schedule_requested'
    assert response.json()['receipt']=={}
    s=read_state(r);assert s['copy_review_completed_at'] and s['schedule_requested_at']


def test_undecided_sibling_does_not_block_finalizing_approved_video(editor,monkeypatch):
    import copy
    r=editor;s=read_state(r)
    sibling=copy.deepcopy(s['result']['segments'][0]);sibling['candidate_number']=2
    s['result']['segments'].append(sibling)
    s['candidate_reviews']['1']={'status':'rendered','clip_url':'https://drive.google.com/file/d/another_verified_asset/view'}
    s['copy_drafts'].append(dict(asset_id='existing:short:2',asset_type='9:16_SHORT',copy_status='pending',social_caption=''))
    save_state(r,s)
    monkeypatch.setattr(__import__('creator_approval'),'approved_batch',lambda _:dict(approved_at='now',approval_revision=1))
    monkeypatch.setattr(r['tg'],'_log_copy_learning',Mock())
    monkeypatch.setattr(r['tg'],'send',Mock())
    handoff=Mock();monkeypatch.setattr(r['tg'],'_notify_render_queue_complete',handoff)
    result=r['client'].post('/api/ripped-shorts/review/'+r['token']+'/finish')
    assert result.status_code==200 and result.json()['status']=='schedule_requested'
    assert read_state(r)['copy_drafts']==s['copy_drafts']
    assert not read_state(r)['website_clip_reviews'].get('existing:short:2')
    handoff.assert_called_once()


def test_highlight_approval_connects_existing_generator_after_durable_vote(editor,monkeypatch):
    r=editor;s=read_state(r);s['copy_drafts'][0]['asset_type']='16:9_HIGHLIGHT';save_state(r,s)
    result=dict(status='saved',decision={'choice':'approve'},copy_review_ready=True)
    monkeypatch.setattr(__import__('finished_clip_review'),'save',lambda *a:result)
    generate=Mock(side_effect=RuntimeError('copy provider unavailable'))
    monkeypatch.setattr(r['web'],'regenerate_review_asset',generate)
    response=r['client'].post('/api/ripped-shorts/review/'+r['token']+'/clip/existing:short:1',json={})
    assert response.status_code==200 and response.json()['copy_generation_pending']
    generate.assert_called_once_with(r['token'],0,missing_only=True)


def test_valid_unsaved_description_unlocks_save_before_finish(editor,tmp_path):
    import subprocess,shutil
    if not shutil.which('node'):pytest.skip('Node required for frontend regression')
    page=editor['client'].get('/review/'+editor['token']).text
    code='function pageScheduleBlockers'+page.split('function pageScheduleBlockers',1)[1].split('function updateProgress',1)[0]
    harness=r'''
const assert=require('node:assert/strict');
const entries={'asset-4':{},'description-4':{value:''},'title-4':{value:'How to repot a Birkin'},'hold-4':{checked:false}};
const document={getElementById:id=>entries[id]};
const data={schedule_blockers:['Complete description for Birkin.'],schedule_copy_blockers:[{asset_id:'h1',message:'Complete description for Birkin.'}],
 items:[{asset_id:'h1',index:4,asset_type:'16:9_HIGHLIGHT',candidate_number:1,copy_status:'pending'}],clips:[{asset_id:'h1',approval_current:true}]};
'''+code+r'''
assert.deepEqual(pageScheduleBlockers(data),['Enter the description for Highlight 1.']);
entries['description-4'].value='🌱 Exact owner description\n\n#Birkin';
assert.deepEqual(pageScheduleBlockers(data),[]);
data.schedule_blockers.push('Confirm the batch collaborators.');
assert.deepEqual(pageScheduleBlockers(data),['Confirm the batch collaborators.']);
entries['hold-4'].checked=true;
assert.match(pageScheduleBlockers(data).join(' '),/scheduling hold/);
'''
    path=tmp_path/'finalization-gates.cjs';path.write_text(harness,encoding='utf-8')
    subprocess.run(['node',str(path)],check=True,capture_output=True,text=True)


def test_existing_generator_receives_separate_highlight_transcripts(editor,monkeypatch):
    r=editor;tg=r['tg']
    monkeypatch.setenv('OPENAI_API_KEY','isolated-not-a-credential')
    monkeypatch.setattr(tg,'read_google_doc_text',lambda _: 'TCB_CTA: Learn with The Chocolate Botanist')
    monkeypatch.setattr(tg,'get_rows',lambda *a:[])
    response=Mock();response.raise_for_status.return_value=None
    response.json.return_value={'choices':[{'message':{'content':json.dumps({'assets':[
        {'asset_id':'h1','video_title':'How to Repot a Philodendron Birkin','video_description':'Backfill gently around its roots.','hashtags':['#Philodendron','#Repotting']},
        {'asset_id':'h2','video_title':'How to Water Soil Without Floating Perlite','video_description':'Spread water pressure and improve the topsoil.','hashtags':['#Watering']}]})}}]}
    provider=Mock(return_value=response);monkeypatch.setattr(tg.requests,'post',provider)
    assets=[dict(asset_id='h1',asset_type='16:9_HIGHLIGHT',transcript='Hold the Birkin. Backfill gently.',title='Birkin'),
            dict(asset_id='h2',asset_type='16:9_HIGHLIGHT',transcript='Water in a wave. Mix compost into hydrophobic soil.',title='Watering')]
    result=tg._generate_schedule_copy(assets,'TCB','Plantcon plants','')
    prompt=provider.call_args.kwargs['json']['messages'][1]['content']
    assert 'SEO-focused' in prompt and "individual asset's transcript" in prompt
    assert all(a['transcript'] in prompt for a in assets)
    assert result[0]['video_title']=='How to Repot a Philodendron Birkin'
    assert 'Backfill gently' in result[0]['video_description']
    assert result[1]['video_title']=='How to Water Soil Without Floating Perlite'
    assert provider.call_count==1
