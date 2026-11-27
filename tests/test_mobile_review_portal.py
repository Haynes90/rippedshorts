import re
import shutil
import subprocess

import pytest
from test_editorial_framing_review import review
from test_finished_clip_review import render_fixture


def test_existing_review_api_can_return_one_authorized_card_without_scripts(review):
    r=review;render_fixture(r)
    url='/api/ripped-shorts/review/'+r['token']+'?asset_id=existing:short:1'
    result=r['client'].get(url)
    assert result.status_code==200
    assert result.json()['asset_id']=='existing:short:1'
    assert '<script>' not in result.json()['clip_html']
    other=r['web'].ensure_review_session('other')['token']
    assert r['client'].get(url.replace(r['token'],other)).status_code==404


def test_rendered_review_scripts_parse_and_save_paths_never_reload(review,tmp_path):
    if not shutil.which('node'):pytest.skip('Node required for JavaScript regression checks')
    r=review;render_fixture(r)
    page=r['client'].get('/review/'+r['token']).text
    assert 'location.reload()' not in page
    assert 'Approve &amp; Next' in page and 'Previous Clip' in page
    for index,script in enumerate(re.findall(r'<script>(.*?)</script>',page,re.S)):
        path=tmp_path/f'review-{index}.js';path.write_text(script,encoding='utf-8')
        result=subprocess.run(['node','--check',str(path)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr


def test_approve_next_requires_copy_save_and_authoritative_approval_acknowledgment(tmp_path):
    if not shutil.which('node'):pytest.skip('Node required for browser interaction checks')
    from mobile_review_ui import SCRIPT
    script=SCRIPT.removeprefix('<script>').removesuffix('</script>')
    harness='''
const assert=require('node:assert/strict');let moved=0,saves=0,approved=false;
const card={dataset:{asset:'a'},querySelector:s=>s==='[data-navigation-status]'?{textContent:'',setAttribute(){}}:s==='.asset[data-index]'?{dataset:{index:'0'}}:null};
const document={readyState:'loading',addEventListener(){},getElementById(){return card;}};
const TOKEN='isolated',state={},INITIAL={request_id:'r'};
async function saveAsset(){saves++;}async function clipDecision(){return approved;}
async function refreshReviewState(){return {clips:[{asset_id:'a',approval_current:true}]};}
'''+script+'''
navigateReview=()=>moved++;
(async()=>{await approveAndNext('a');assert.equal(saves,1);assert.equal(moved,0);
approved=true;await approveAndNext('a');assert.equal(moved,1);
saveAsset=async()=>{throw new Error('HTTP 409');};await approveAndNext('a');assert.equal(moved,1);
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    path=tmp_path/'approve-next.cjs';path.write_text(harness,encoding='utf-8')
    result=subprocess.run(['node',str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_refresh_keeps_existing_player_and_dirty_editor_nodes(tmp_path):
    if not shutil.which('node'):pytest.skip('Node required for browser interaction checks')
    from mobile_review_ui import SCRIPT
    script=SCRIPT.removeprefix('<script>').removesuffix('</script>')
    harness='''
const assert=require('node:assert/strict');let requests=0,initialized=0;
const player={currentTime:12},editor={value:'Unsaved owner text 🌱'},card={player};
const document={readyState:'loading',addEventListener(){},getElementById:id=>id==='asset-0'?editor:card};
const TOKEN='isolated',state={},INITIAL={request_id:'r'};
async function fetch(){requests++;throw new Error('Existing cards must not be fetched again');}
function initializeClipReviews(){initialized++;}function initializeCopyCards(){}
'''+script+'''
initializeMobileReview=()=>{};
(async()=>{await synchronizeReviewCards({clips:[{asset_id:'a'}],items:[{asset_id:'a',index:0}]});
assert.equal(requests,0);assert.equal(initialized,1);assert.equal(player.currentTime,12);assert.equal(editor.value,'Unsaved owner text 🌱');
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    path=tmp_path/'preserve-nodes.cjs';path.write_text(harness,encoding='utf-8')
    result=subprocess.run(['node',str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
