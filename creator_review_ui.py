"""Batch creator confirmation, separate from per-clip caption editing."""
def section():
    return '''<style>
      #creatorReview .creator-choice {border:1px solid #34505e;border-radius:12px;padding:12px;margin:12px 0;}
      #creatorReview input[type=checkbox] {width:auto;display:inline-block;margin-right:8px;}
      #creatorReview label {display:flex;align-items:center;gap:6px;margin:10px 0;}
      #creatorReview select {width:100%;background:#071b27;color:#eef4f8;border:1px solid #34505e;border-radius:8px;padding:10px;margin:8px 0;white-space:normal;}
      #creatorReview input:read-only {opacity:.85;}
    </style><section class="source" id="creatorReview"><div class="source-copy">
      <h2>Who's featured?</h2><p>Confirm once for all accepted clips in this batch. Source-channel suggestions do not prove someone appears.</p>
      <div id="creatorChoices"></div><button type="button" class="secondary" onclick="addCreator()">Add a person</button>
      <p><label><input type="checkbox" id="creatorConfirmed"> I confirm the selected people, handles and choices (or no featured people).</label></p>
      <button type="button" class="primary" onclick="saveCreators()">Confirm batch creators</button><p id="creatorStatus"></p>
      <p>Choose up to 3 native Instagram collaborators. An invitation request is separate from the person accepting it. If Instagram rejects the collaborators, the delivery is held for review; it will not publish without them. Captions remain unchanged.</p>
    </div></section>'''


SCRIPT = '''
function addCreator(person={}) {
 const row=document.createElement('div'); row.className='creator-choice';
 const selected=document.createElement('input'); selected.type='checkbox'; selected.className='creator-selected'; selected.checked=!!person.selected;selected.disabled=!!person.standing_invite_authorized;selected.setAttribute('aria-label','Include '+(person.display_name||'this person'));
 const name=document.createElement('input'); name.placeholder='Name'; name.className='creator-name'; name.value=person.display_name||'';
 const handle=document.createElement('input'); handle.placeholder='Instagram handle'; handle.className='creator-handle'; handle.value=person.instagram_handle||'';
 const intent=document.createElement('select'); intent.className='creator-intent';
 [['none','Credit only'],['mention','Mention requested — keep approved caption'],['invite','Send native Instagram Collab invitation']].forEach(([value,text])=>{const o=document.createElement('option');o.value=value;o.textContent=text;intent.append(o);});intent.value=person.standing_invite_authorized?'invite':(person.intent||'none');intent.disabled=!!person.standing_invite_authorized;
 const host=document.createElement('input');host.type='checkbox';host.className='creator-host';host.checked=!!person.recurring_host;host.disabled=!!person.standing_invite_authorized;
 const hostLabel=document.createElement('label');hostLabel.append(host,document.createTextNode(' Remember as a recurring TDOG host'));hostLabel.hidden=state.creator_brand!=='TDOG';
 const association=document.createElement('input');association.type='checkbox';association.className='creator-source';
 const sourceLabel=document.createElement('label');sourceLabel.append(association,document.createTextNode(' I verify an association with this YouTube channel'));
 if(person.standing_invite_authorized){name.readOnly=true;handle.readOnly=true;const rule=document.createElement('p');rule.textContent='Confirmed recurring host — Instagram collaboration requested for every accepted clip.';row.append(rule);}
 row.append(selected,name,handle,intent,hostLabel,sourceLabel);document.getElementById('creatorChoices').append(row);
}
async function saveCreators() {
 if(!document.getElementById('creatorConfirmed').checked){document.getElementById('creatorStatus').textContent='Confirm the batch before saving.';return;}
 const entries=[...document.querySelectorAll('.creator-choice')].filter(r=>r.querySelector('.creator-selected').checked).map(r=>({display_name:r.querySelector('.creator-name').value,instagram_handle:r.querySelector('.creator-handle').value,intent:r.querySelector('.creator-intent').value,recurring_host:r.querySelector('.creator-host').checked,confirm_source_association:r.querySelector('.creator-source').checked}));
 if(entries.filter(p=>p.intent==='invite').length>3){document.getElementById('creatorStatus').textContent='Choose at most 3 native Instagram collaborators.';return;}
 const response=await fetch(`/api/ripped-shorts/review/${TOKEN}/creators`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({confirmed:true,entries})});
 const result=await response.json();document.getElementById('creatorStatus').textContent=response.ok?'Batch creators confirmed.':(result.detail||'Could not save creator approval.');
}
if(state.creator_required){
 const saved=state.creator_approval;
 (saved?saved.creators.map(p=>({...p,selected:true})):(state.creator_suggestions||[]).map(p=>({...p,selected:state.creator_brand==='TDOG'&&!!p.recurring_host}))).forEach(addCreator);
 if(saved){document.getElementById('creatorConfirmed').checked=true;document.getElementById('creatorStatus').textContent='Batch creators confirmed.';}
}else{document.getElementById('creatorReview').hidden=true;}
'''
