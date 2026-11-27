"""Editorial and framing sections of the existing review portal."""
import html
import json
from urllib.parse import quote


FINISHED_REVIEW_SCRIPT = r'''<script>
const clipSaves=new Map();
function clipParts(asset){
 const key=asset.replaceAll(':','-'),card=document.getElementById('clip-'+key);
 if(!card)return {card:null};
 return {card,status:document.getElementById('clip-status-'+key),video:card.querySelector('video'),
  playback:document.getElementById('clip-playback-'+key),framing:document.getElementById('clip-framing-'+key),
  reason:document.getElementById('clip-reason-'+key),feedback:document.getElementById('clip-feedback-'+key),edit:document.getElementById('clip-edit-'+key)};
}
function clipDraftKey(card){return 'ripped-clip:'+TOKEN+':'+card.dataset.asset+':'+card.dataset.mediaRevision;}
function clipStored(card){
 let draft={};try{draft=JSON.parse(localStorage.getItem(clipDraftKey(card))||'{}');}catch(_){}
 if(Object.prototype.hasOwnProperty.call(card,'clipPending'))draft.pending=card.clipPending;
 return draft;
}
function rememberClip(p,pending=clipStored(p.card).pending){
 p.card.clipPending=pending; // Reconciliation still works when browser storage is unavailable.
 try{localStorage.setItem(clipDraftKey(p.card),JSON.stringify({reason:p.reason.value,feedback:p.feedback.value,
  edit:p.edit.value,playback:p.playback.checked,framing:p.framing?.checked===true,pending,
  attempt:p.card.clipAttempt||null,error:p.card.clipSaveError||''}));}catch(_){}
}
function clipRequirements(p,choice){
 if(p.card.dataset.mediaChanged==='1')return 'This preview changed. Refresh and review its current media before saving.';
 if(p.card.dataset.frozen==='1')return 'This clip is already submitted; its approval is frozen.';
 if(choice==='approve'){
  if(!p.video||p.video.error||p.video.readyState<2)return 'Wait for this verified video to load before approving.';
  if(!p.playback.checked)return 'Confirm that you watched this finished video and heard its audio.';
  if(p.framing&&!p.framing.checked)return 'Confirm the framing of this exact preview.';
 }
 if(choice==='reject'&&!p.reason.value)return 'Choose a reason above before rejecting this clip.';
 if(choice==='request_adjustment'&&!p.feedback.value.trim())return 'Describe the requested adjustment above.';
 return '';
}
function updateClipControls(p){
 const busy=clipSaves.has(p.card.dataset.asset);
 p.card.querySelectorAll('input,select,textarea').forEach(e=>e.disabled=busy||p.card.dataset.frozen==='1'||p.card.dataset.mediaChanged==='1');
 p.card.querySelectorAll('[data-clip-action]').forEach(b=>b.disabled=busy||!!clipRequirements(p,b.dataset.clipAction));
 const ready=p.video&&!p.video.error&&p.video.readyState>=2,player=p.card.querySelector('[data-player-status]');
 if(player)player.textContent=p.video.error?'Preview could not load. Refresh or report access; do not rerender.':ready?'Preview ready to watch.':'Loading verified preview...';
 p.card.querySelector('[data-clip-requirements]').textContent=[clipRequirements(p,'approve'),clipRequirements(p,'reject'),clipRequirements(p,'request_adjustment')].filter(Boolean).join(' ')||'Your review inputs are ready. Choose an action to save.';
}
function sameClipDecision(saved,intent,unchanged=false){
 return saved&&saved.version===intent.version+(unchanged?0:1)&&saved.media_revision===intent.media_revision&&
  saved.choice===intent.choice&&saved.reason===(intent.choice==='reject'?intent.reason:'')&&
  saved.feedback===intent.feedback&&JSON.stringify(saved.changed_fields||[])===JSON.stringify(intent.changed_fields);
}
async function readClipReview(){
 const r=await fetch('/api/ripped-shorts/review/'+TOKEN,{cache:'no-store'});
 if(!r.ok)throw new Error('Could not check saved review (HTTP '+r.status+').');
 const d=await r.json();if(d.request_id!==INITIAL.request_id)throw new Error('Review request mismatch.');
 refreshClipReview(d);return d;
}
function refreshClipReview(d){
 state=d;updateProgress(d);
 for(const item of d.clips||[]){
  const p=clipParts(item.asset_id);if(!p.card||clipSaves.has(item.asset_id)||clipStored(p.card).pending)continue;
  if(item.media_revision!==p.card.dataset.mediaRevision){p.card.dataset.mediaChanged='1';updateClipControls(p);continue;}
  p.card.dataset.frozen=item.frozen?'1':'0';
  p.card.dataset.version=item.decision.version||0;
  if(p.card.dataset.saveError!=='1')p.status.textContent='Saved clip decision: '+(item.decision.choice||'Undecided');
 }
}
function confirmClipSave(p,decision,message){
 p.card.dataset.version=decision.version;p.card.dataset.saveError='';p.card.clipSaveError='';p.card.clipAttempt=null;rememberClip(p,null);
 p.status.setAttribute('role','status');p.status.textContent=message+' '+decision.choice+'. No rendering, scheduling or invitations started.';
}
async function reconcileClip(p,intent){
 const d=await readClipReview(),item=(d.clips||[]).find(c=>c.asset_id===p.card.dataset.asset);
 if(!item||item.media_revision!==intent.media_revision)throw new Error('Preview revision changed. Your inputs are retained; review the current media before saving.');
 if(sameClipDecision(item.decision,intent)||sameClipDecision(item.decision,intent,true)){confirmClipSave(p,item.decision,'Saved and verified:');return true;}
 p.card.dataset.version=item.decision.version||0;rememberClip(p,null);
 throw new Error(item.decision.version!==undefined&&item.decision.version!==intent.version?
  'A different review version is saved. Your inputs are retained; inspect the saved decision before choosing again.':
  'No matching saved decision is confirmed. Your inputs are retained; choose the action again when ready.');
}
async function performClipDecision(asset,choice){
 const p=clipParts(asset);let intent,knownRejection=false;
 p.card.dataset.saveError='';p.card.clipSaveError='';rememberClip(p);
 try{
  const previous=clipStored(p.card).pending;
  if(previous){await reconcileClip(p,previous);return;}
  p.card.clipAttempt=choice;
  const missing=clipRequirements(p,choice);if(missing){knownRejection=true;throw new Error(missing);}
  intent={choice,version:Number(p.card.dataset.version),media_revision:p.card.dataset.mediaRevision,
   reason:choice==='reject'?p.reason.value:'',feedback:Array.from(p.feedback.value).slice(0,2000).join(''),
   changed_fields:choice==='request_adjustment'?[p.edit.value]:[],playback_verified:p.playback.checked,framing_verified:p.framing?.checked===true};
  rememberClip(p,intent);p.status.textContent='Saving review…';
  const response=await fetch('/api/ripped-shorts/review/'+TOKEN+'/clip/'+encodeURIComponent(asset),
   {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(intent)});
  knownRejection=response.status>=400&&response.status<500&&response.status!==408;
  let result;try{result=await response.json();}catch(error){if(knownRejection)rememberClip(p,null);throw error;}
  if(!response.ok){if(knownRejection)rememberClip(p,null);
   if(knownRejection)try{await readClipReview();}catch(_){}
   throw new Error(result.detail||'Save rejected (HTTP '+response.status+').');}
  if(result.status!=='saved'||!sameClipDecision(result.decision,intent,result.unchanged===true))throw new Error('Save acknowledgment requires verification.');
  confirmClipSave(p,result.decision,'Saved:');
  try{await readClipReview();}catch(_){p.status.textContent='Saved: '+result.decision.choice+'. Review count refresh is pending.';}
  if(result.copy_review_ready){document.querySelectorAll('.clip-card').forEach(c=>rememberClip(clipParts(c.dataset.asset)));location.reload();}
 }catch(error){
  if(intent&&!knownRejection&&clipStored(p.card).pending){
   try{await reconcileClip(p,intent);return;}catch(check){error=check;}
  }
  p.card.dataset.saveError='1';p.card.clipSaveError=(knownRejection?'Not saved: ':'Save not confirmed: ')+error.message;
  rememberClip(p);p.status.setAttribute('role','alert');p.status.textContent=p.card.clipSaveError;
 }finally{updateClipControls(p);}
}
function clipDecision(asset,choice){
 if(clipSaves.has(asset))return clipSaves.get(asset);
 // Schedule the operation after registering its lock, including reconciliation.
 const operation=Promise.resolve().then(()=>performClipDecision(asset,choice)).finally(()=>{clipSaves.delete(asset);updateClipControls(clipParts(asset));});
 clipSaves.set(asset,operation);updateClipControls(clipParts(asset));return operation;
}
function initializeClipReviews(){
 document.querySelectorAll('.clip-card').forEach(card=>{
  const p=clipParts(card.dataset.asset),draft=clipStored(card);
  if('feedback' in draft){p.feedback.value=draft.feedback;p.reason.value=draft.reason;p.edit.value=draft.edit;p.playback.checked=draft.playback;if(p.framing)p.framing.checked=draft.framing;}
  p.card.clipAttempt=draft.attempt;p.card.clipSaveError=draft.error||'';
  if(draft.error){p.card.dataset.saveError='1';p.status.setAttribute('role','alert');p.status.textContent=draft.error+' Your inputs are retained; no review action will be resubmitted automatically.';}
  card.addEventListener('input',()=>{rememberClip(p);updateClipControls(p);});
  card.addEventListener('change',()=>{rememberClip(p);updateClipControls(p);});
  if(p.video)['loadeddata','error','emptied'].forEach(event=>p.video.addEventListener(event,()=>updateClipControls(p)));
  updateClipControls(p);
  if(draft.pending)p.status.textContent='Previous save needs verification. Your inputs are retained; choose an action to check saved state before another submission.';
 });
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initializeClipReviews);else initializeClipReviews();
</script>'''


def candidate_cards(items, token):
    """Use the existing portal for proposals; no finished-video claims here."""
    sections=[]
    for lane,title in [('short','9:16 Shorts candidates'),('highlight','16:9 highlight candidates')]:
        cards=[]
        for item in [c for c in items if c['lane']==lane]:
            disabled='disabled' if item['frozen'] else ''
            framing=(f'<label>Framing<select data-framing {disabled}><option value="auto">Automatic safe framing</option><option value="blurred" '+('selected' if item['framing']=='blurred' else '')+'>Full frame on blurred background</option></select></label>') if lane=='short' else ''
            cards.append(f'''<article class="candidate-card asset" data-asset="{html.escape(item['asset_id'])}" data-revision="{item['revision']}" data-version="{item['version']}">
<span class="pill">{item['format']}</span><h2>{html.escape(item['title'])}</h2>
<p>Proposed passage: {item['start']:.2f}–{item['end']:.2f} seconds ({item['end']-item['start']:.1f}s)</p>
<p>Editorial candidate — rendering starts only after you confirm your selections. Final speech and media checks still apply.</p>
<p><strong>Source excerpt — not a finished render.</strong> Final framing and exported media are reviewed after rendering.</p>
<video data-source-excerpt data-start="{item['start']}" data-end="{item['end']}" controls playsinline preload="none" style="width:100%;max-height:65vh" src="/api/ripped-shorts/review/{quote(token,safe='')}/candidate/{quote(item['asset_id'],safe='')}/source?revision={item['revision']}#t={item['start']},{item['end']}"></video>
<label>Seek within this excerpt<input data-excerpt-seek type="range" min="0" max="{item['end']-item['start']}" step="0.05" value="0" style="width:100%"></label>
<p data-excerpt-status role="status">Play to hear the proposed source passage ({item['start']:.2f}–{item['end']:.2f}s).</p>
<details><summary>Intended transcript and boundaries</summary><p style="white-space:pre-wrap">{html.escape(item['transcript'])}</p></details>
<p><strong>Why this clip?</strong> {html.escape(item['reason'])}</p>{framing}
<label>Adjustment request or feedback<textarea data-feedback {disabled}>{html.escape(item['feedback'])}</textarea></label>
<div class="actions">{''.join(f'<button type="button" data-candidate-choice="{choice}" {disabled}>{label}</button>' for choice,label in [('select','Select'),('reject','Reject'),('request_adjustment','Request Adjustment'),('unused','Leave Unused')])}</div>
<p data-candidate-status role="status">Saved choice: {html.escape(item['choice'] or 'Undecided')}{' — confirmed for this rendering batch' if item['frozen'] else ''}</p></article>''')
        sections.append(f'<section id="{"shorts" if lane=="short" else "highlights"}"><h2>{title}</h2>'+(''.join(cards) if cards else '<p>No candidates available for this format yet. Processing status appears above; this is not a completed review.</p>')+(f'<button type="button" data-confirm-candidates="{lane}">Confirm selected {"Shorts" if lane=="short" else "highlights"} and render</button>' if cards and any(not c['frozen'] for c in items if c['lane']==lane) else '')+'</section>')
    return ''.join(sections)+'''<script>
document.querySelectorAll('[data-source-excerpt]').forEach(video=>{
 const card=video.closest('.candidate-card'),seek=card.querySelector('[data-excerpt-seek]'),status=card.querySelector('[data-excerpt-status]');
 const start=Number(video.dataset.start),end=Number(video.dataset.end),timingEpsilon=.001;let timer;
 const clear=()=>clearTimeout(timer);
 const stop=()=>{clear();video.pause();video.currentTime=end;seek.value=end-start;status.textContent='End of source excerpt. Play to replay.';};
 const arm=()=>{clear();if(!video.paused&&video.currentTime<end)timer=setTimeout(()=>{if(video.currentTime>=end-.01)stop();else arm();},Math.max(10,(end-video.currentTime)/video.playbackRate*1000));};
 video.addEventListener('loadedmetadata',()=>{video.currentTime=start;status.textContent='Source excerpt ready; audio and volume use the player controls.';});
 video.addEventListener('play',()=>{
  document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();});
  if(video.currentTime<start-timingEpsilon||video.currentTime>=end-timingEpsilon)video.currentTime=start;
  arm();
 });
 video.addEventListener('pause',clear);
 video.addEventListener('waiting',clear);video.addEventListener('stalled',clear);video.addEventListener('playing',arm);
 video.addEventListener('seeking',()=>{clear();if(video.currentTime<start-timingEpsilon)video.currentTime=start;if(video.currentTime>end)stop();});
 video.addEventListener('seeked',arm);video.addEventListener('ratechange',arm);
 video.addEventListener('timeupdate',()=>{if(video.currentTime>=end&&!video.paused)stop();seek.value=Math.max(0,Math.min(end-start,video.currentTime-start));});
 video.addEventListener('error',()=>{clear();status.textContent='Source excerpt unavailable. Your decisions are preserved; no render or download was started.';});
 seek.addEventListener('input',()=>{video.currentTime=start+Math.max(0,Math.min(end-start,Number(seek.value)));});
});
document.querySelectorAll('[data-candidate-choice]').forEach(button=>button.onclick=async()=>{
 const card=button.closest('[data-asset]'),status=card.querySelector('[data-candidate-status]');
 const buttons=card.querySelectorAll('button');buttons.forEach(b=>b.disabled=true);
 try{
  const response=await fetch('/api/ripped-shorts/review/'+TOKEN+'/candidate/'+encodeURIComponent(card.dataset.asset),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({choice:button.dataset.candidateChoice,version:Number(card.dataset.version),revision:card.dataset.revision,feedback:card.querySelector('[data-feedback]').value,framing:card.querySelector('[data-framing]')?.value||'auto'})});
  const data=await response.json();if(!response.ok)throw new Error(data.detail||'Save failed');
  card.dataset.version=data.candidate.version;card.dataset.revision=data.candidate.revision;
  status.textContent='Saved: '+data.candidate.choice+'. Rendering and scheduling have not started.';
  await refreshReviewState();
 }catch(e){status.textContent='Save not confirmed: '+e.message+'. Your entered text is still here; refresh the saved state before repeating an uncertain save.';}
 finally{buttons.forEach(b=>b.disabled=false);}
});
document.querySelectorAll('[data-confirm-candidates]').forEach(button=>button.onclick=async()=>{
 button.disabled=true;
 try{
  const r=await fetch('/api/ripped-shorts/review/'+TOKEN+'/candidates/'+button.dataset.confirmCandidates+'/confirm',{method:'POST'});
  const d=await r.json();if(!r.ok)throw new Error(d.detail||'Confirmation failed');location.reload();
 }catch(e){button.textContent=e.message;button.disabled=false;}
});
</script>'''


def preview_status_items(items):
    """Creator-safe projection; detailed diagnostic records remain unchanged."""
    statuses=[]
    for item in items:
        category=(item.get('diagnostics') or {}).get('category') or ''
        if item['finished']:
            label='Ready to watch';reason='The verified rendered preview is available above.'
        elif item['render_status']=='preview_pending':
            label='Preview pending';reason='This candidate has not completed preview preparation.'
        else:
            label='Preview unavailable — awaiting boundary resolution'
            reason={
                'SEMANTIC_EVIDENCE_REFERENCE_INVALID':'The validator could not establish valid setup or payoff references.',
                'NO_EVIDENCE_SUPPORTED_PROGRESS':'Construction could not resolve a complete passage and supported cut points.',
                'EDITORIAL_VALUE_NOT_PRESERVED':'The adjusted passage did not retain the recommended explanation or payoff.',
                'CONSTRUCTION_UNRESOLVED':'The bounded validation attempt could not establish complete speech and safe boundaries.',
            }.get(category,'This candidate remains held until its speech, boundaries or media can be verified.')
        statuses.append({**{k:item.get(k) for k in ('asset_id','format','number','title','transcript','bounds','finished')},
                         'label':label,'explanation':reason})
    return statuses


def preview_status_sections(items):
    """Account for every discovered candidate; only the existing gallery selects media."""
    result=[]
    for aspect,anchor in [('9:16','shorts'),('16:9','highlights')]:
        lane=[c for c in items if c['format']==aspect]
        ready=sum(bool(c['finished']) for c in lane)
        result.append(f'<section id="{anchor}"><h2>{aspect} candidate progress</h2><p>{ready}/{len(lane)} rendered previews available.</p>')
        for item in lane:
            asset=item['asset_id']; title=html.escape(str(item['title']))
            if item['finished']:
                result.append(f'<p><a href="#clip-{html.escape(asset.replace(":","-"),quote=True)}">Clip {item["number"]}: {title}</a> — Ready to watch above.</p>')
                continue
            label=html.escape(item['label']);reason=html.escape(item['explanation'])
            start,end=item.get('bounds') or [None,None]
            result.append(f'<article class="preview-status" data-preview-status="{html.escape(asset,quote=True)}"><h3>Clip {item["number"]}: {title}</h3><p role="status">{label}. {reason}</p><p>Proposed source boundaries: {html.escape(str(start))}–{html.escape(str(end))} seconds.</p><details><summary>Intended transcript</summary><p style="white-space:pre-wrap">{html.escape(item.get("transcript") or "Transcript not available.")}</p></details></article>')
        result.append('</section>')
    return ''.join(result)


def finished_cards(items, token):
    from urllib.parse import quote
    cards = []
    reasons = [('poor_topic','Poor topic choice'),('weak_hook','Weak hook'),('incomplete_thought','Incomplete thought'),('unnecessary_length','Unnecessary length'),('inaccurate_transcript','Inaccurate transcript'),('unsuitable_framing','Unsuitable framing'),('bad_audio','Bad audio')]
    for item in items:
        if item.get('preview') and not item['finished']:
            continue  # Optional failures remain in backend evidence, never selectable cards.
        asset = item['asset_id']; key = asset.replace(':','-'); decision = item['decision']
        video = '<p role="status">Finished video unavailable — '+html.escape(str(item.get('error') or item['render_status']))+'. No source preview is presented as a final export.</p>'
        if item['finished']:
            video = f'<video controls playsinline preload="metadata" style="width:100%;max-height:70vh" src="/api/ripped-shorts/review/{token}/clip/{quote(asset,safe="")}/media"></video>'
        disabled = 'disabled' if item['frozen'] else ''
        unused = f'<button type="button" data-clip-action="unused" {disabled} onclick="clipDecision(\'{asset}\',\'unused\')">Leave unused</button>' if item.get('preview') else ''
        framing = f'<label><input type="checkbox" id="clip-framing-{key}" style="width:auto"> I approve this preview framing</label>' if item.get('preview') else ''
        approve_disabled = 'disabled'  # enable only after this exact player decodes media
        player_status = '<p data-player-status role="status">Loading verified preview...</p>' if item['finished'] else ''
        why = item.get('reason') or 'No selection explanation was saved for this legacy candidate.'
        try:
            seconds = max(0., float(item['bounds'][1])-float(item['bounds'][0]))
            duration_text = f'{int(seconds//60)}:{seconds%60:04.1f}'
        except (ValueError,TypeError,KeyError,IndexError):
            duration_text = 'Duration pending'
        saved_status = 'Previous approval needs renewed review for this render' if decision.get('choice')=='approve' and not item['approval_current'] else str(decision.get('choice') or 'Undecided')
        options = ''.join(f'<option value="{code}" {"selected" if decision.get("reason")==code else ""}>{label}</option>' for code,label in reasons)
        cards.append(f'''<article class="clip-card" id="clip-{key}" data-asset="{html.escape(asset)}" data-version="{decision.get('version',0)}" data-media-revision="{item['media_revision']}" data-preview="{int(bool(item.get('preview')))}" data-frozen="{int(bool(item['frozen']))}" style="border:1px solid #315163;border-radius:16px;padding:14px;margin:16px 0">
<span class="pill">{item['format']}</span><h2>{html.escape(str(item['title']))}</h2>
<p>{duration_text} · Clip {item['number']}</p>
<details><summary>Complete excerpt transcript{' — awaiting final render' if not item['finished'] else ''}</summary><p style="white-space:pre-wrap">{html.escape(item['transcript'])}</p></details>
{video}{player_status}<p><strong>Why this clip?</strong> {html.escape(str(why))}</p>
<label><input type="checkbox" id="clip-playback-{key}" style="width:auto"> I watched this finished clip and heard its audio</label>
{framing}
<label for="clip-reason-{key}">Reason if rejecting</label><select id="clip-reason-{key}"><option value="">Choose a reason</option>{options}</select>
<label for="clip-feedback-{key}">Optional feedback / adjustment request</label><textarea maxlength="2000" id="clip-feedback-{key}">{html.escape(str(decision.get('feedback') or ''))}</textarea>
<label for="clip-edit-{key}">If requesting an adjustment</label><select id="clip-edit-{key}"><option value="boundaries">Boundaries</option><option value="topic_context">Topic/context</option><option value="caption">Caption</option><option value="framing">Framing</option><option value="audio">Audio</option></select>
<p data-clip-requirements role="status">To approve, confirm playback and framing above. To reject, choose a reason.</p>
<div class="actions"><button type="button" data-approve-clip data-clip-action="approve" {approve_disabled} onclick="clipDecision('{asset}','approve')">Approve Clip</button><button type="button" data-clip-action="reject" {disabled} onclick="clipDecision('{asset}','reject')">Reject Clip</button><button type="button" data-clip-action="request_adjustment" {disabled} onclick="clipDecision('{asset}','request_adjustment')">Request Adjustment</button></div>
{unused}<button type="button" class="secondary" onclick="document.getElementById('editing-workflow').open=true;document.getElementById('editing-workflow').scrollIntoView()">Edit / compare alternatives</button>
<p id="clip-status-{key}" role="status">Saved clip decision: {html.escape(saved_status)}</p>
<p class="meta">Clip approval does not approve copy, invite collaborators or start scheduling.</p></article>''')
    return ''.join(cards) + FINISHED_REVIEW_SCRIPT


def sections(reviews, token):
    groups = {'editorial': [], 'framing': []}
    for review in sorted(reviews, key=lambda r: (r['format'] != '9:16', int(r['asset_id'].rsplit(':', 1)[-1]))):
        revision = review['revision_id']
        heading = html.escape(review['asset_id'].rsplit(':', 2)[-2] + ' ' + review['asset_id'].rsplit(':', 1)[-1])
        videos = []
        for variant in ('original', 'proposed', 'opening', 'ending', 'smart', 'reference', 'blurred'):
            if variant not in review['media']:
                continue
            src = f'/api/ripped-shorts/review/{token}/comparison/{revision}/media/{variant}'
            label = 'Previous framing — diagnostic only, not an approval option' if variant == 'reference' else variant.title()
            videos.append(f'<div><h3>{html.escape(label)}</h3><video controls playsinline preload="metadata" style="width:100%;max-height:70vh" src="{src}"></video></div>')
        actions = [('proposed', 'Approve Proposed'), ('original', 'Keep Original'), ('request_edit', 'Request Another Edit')] if review['kind'] == 'editorial' else [('smart', 'Approve Smart Framing'), ('blurred', 'Choose Blurred Full Frame'), ('request_adjustment', 'Request Framing Adjustment')]
        buttons = ''.join(f'<button type="button" onclick="saveComparison(\'{revision}\',\'{choice}\')">{label}</button>' for choice, label in actions if choice.startswith('request_') or choice in review['media'])
        explanation = review['provenance'].get('explanation') or ''
        context = '<details><summary>Nearby transcript context</summary><pre style="white-space:pre-wrap">' + html.escape(json.dumps(review['provenance'].get('transcript_context') or [], indent=2)) + '</pre></details>'
        groups[review['kind']].append(f'''<article class="comparison" data-revision="{revision}" data-version="{review['version']}" style="padding:14px;border:1px solid #315163;border-radius:12px;margin:16px 0">
<h3>{heading} — {html.escape(review['format'])}</h3>
<p>{html.escape(explanation)}</p><p>Original: {html.escape(str(review['original']))}; proposed: {html.escape(str(review['proposed']))}</p>
<p>Timing evidence: {html.escape(str(review['provenance'].get('timing_provenance', 'Review required')))}. Listen to the context; ASR differences alone do not prove an unsafe cut.</p>
{context}<div class="comparison-videos">{''.join(videos)}</div>
<label for="comparison-feedback-{revision}">Feedback</label><textarea id="comparison-feedback-{revision}">{html.escape(review['feedback'])}</textarea>
<div class="actions">{buttons}</div><p id="comparison-status-{revision}" role="status">Saved decision: {html.escape(review['decision'] or 'Awaiting owner')}</p></article>''')
    result = '<style>.comparison-videos{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}@media(max-width:600px){.comparison-videos{grid-template-columns:1fr}}</style>'
    for kind, title in [('editorial', 'Editorial boundary review'), ('framing', '9:16 framing review')]:
        result += f'<section id="{kind}"><h2>{title}</h2><p>These choices save review decisions only. They do not start rendering, schedule content, publish, or send collaborator invitations.</p>{"".join(groups[kind]) or "<p>No comparisons available at this stage.</p>"}</section>'
    return result + '''<script>
async function saveComparison(revision,choice){
 const card=document.querySelector('[data-revision="'+revision+'"]');
 const status=document.getElementById('comparison-status-'+revision);
 const token=TOKEN;
 try{
  const response=await fetch('/api/ripped-shorts/review/'+token+'/comparison/'+revision,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({choice,feedback:document.getElementById('comparison-feedback-'+revision).value,version:Number(card.dataset.version)})});
  const result=await response.json();if(!response.ok)throw new Error(result.detail||'Save failed');
  card.dataset.version=result.version;status.textContent='Saved: '+result.choice+' — rendering and scheduling have not started.';
 }catch(error){status.textContent='Not saved: '+error.message;}
}
</script>'''
