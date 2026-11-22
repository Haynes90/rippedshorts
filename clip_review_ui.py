"""Editorial and framing sections of the existing review portal."""
import html
import json
from urllib.parse import quote


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
 const start=Number(video.dataset.start),end=Number(video.dataset.end);let timer;
 const clear=()=>clearTimeout(timer);
 const stop=()=>{clear();video.pause();video.currentTime=end;seek.value=end-start;status.textContent='End of source excerpt. Play to replay.';};
 const arm=()=>{clear();if(!video.paused&&video.currentTime<end)timer=setTimeout(stop,Math.max(0,(end-video.currentTime)/video.playbackRate*1000));};
 video.addEventListener('loadedmetadata',()=>{video.currentTime=start;status.textContent='Source excerpt ready; audio and volume use the player controls.';});
 video.addEventListener('play',()=>{
  document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();});
  if(video.currentTime<start||video.currentTime>=end)video.currentTime=start;
  arm();
 });
 video.addEventListener('pause',clear);
 video.addEventListener('seeking',()=>{clear();if(video.currentTime<start)video.currentTime=start;if(video.currentTime>end)stop();});
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
        unused = f'<button type="button" {disabled} onclick="clipDecision(\'{asset}\',\'unused\')">Leave unused</button>' if item.get('preview') else ''
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
        cards.append(f'''<article class="clip-card" id="clip-{key}" data-asset="{html.escape(asset)}" data-version="{decision.get('version',0)}" data-frozen="{int(bool(item['frozen']))}" style="border:1px solid #315163;border-radius:16px;padding:14px;margin:16px 0">
<span class="pill">{item['format']}</span><h2>{html.escape(str(item['title']))}</h2>
<p>{duration_text} · Clip {item['number']}</p>
<details><summary>Complete excerpt transcript{' — awaiting final render' if not item['finished'] else ''}</summary><p style="white-space:pre-wrap">{html.escape(item['transcript'])}</p></details>
{video}{player_status}<p><strong>Why this clip?</strong> {html.escape(str(why))}</p>
<label><input type="checkbox" id="clip-playback-{key}" style="width:auto"> I watched this finished clip and heard its audio</label>
{framing}
<label for="clip-reason-{key}">Reason if rejecting</label><select id="clip-reason-{key}"><option value="">Choose a reason</option>{options}</select>
<label for="clip-feedback-{key}">Optional feedback / adjustment request</label><textarea id="clip-feedback-{key}">{html.escape(str(decision.get('feedback') or ''))}</textarea>
<label for="clip-edit-{key}">If requesting an adjustment</label><select id="clip-edit-{key}"><option value="boundaries">Boundaries</option><option value="topic_context">Topic/context</option><option value="caption">Caption</option><option value="framing">Framing</option><option value="audio">Audio</option></select>
<div class="actions"><button type="button" data-approve-clip {approve_disabled} onclick="clipDecision('{asset}','approve')">Approve Clip</button><button type="button" {disabled} onclick="clipDecision('{asset}','reject')">Reject Clip</button><button type="button" {disabled} onclick="clipDecision('{asset}','request_adjustment')">Request Adjustment</button></div>
{unused}<button type="button" class="secondary" onclick="document.getElementById('editing-workflow').open=true;document.getElementById('editing-workflow').scrollIntoView()">Edit / compare alternatives</button>
<p id="clip-status-{key}" role="status">Saved clip decision: {html.escape(saved_status)}</p>
<p class="meta">Clip approval does not approve copy, invite collaborators or start scheduling.</p></article>''')
    return ''.join(cards) + '''<script>
document.querySelectorAll('.clip-card').forEach(card=>{
 const video=card.querySelector('video'),approve=card.querySelector('[data-approve-clip]'),status=card.querySelector('[data-player-status]');
 if(!video)return;
 const update=()=>{
  const ready=!video.error&&video.readyState>=2;
  approve.disabled=card.dataset.frozen==='1'||!ready;
  status.textContent=video.error?'Preview could not load. Refresh or report media access; do not rerender.':ready?'Preview ready to watch.':'Loading verified preview...';
 };
 ['loadeddata','error','emptied'].forEach(event=>video.addEventListener(event,update));update();
});
async function clipDecision(asset,choice){
 const key=asset.replaceAll(':','-'),card=document.getElementById('clip-'+key),status=document.getElementById('clip-status-'+key);
 try{
  const response=await fetch('/api/ripped-shorts/review/'+TOKEN+'/clip/'+encodeURIComponent(asset),{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({choice,version:Number(card.dataset.version),reason:document.getElementById('clip-reason-'+key).value,feedback:document.getElementById('clip-feedback-'+key).value,changed_fields:choice==='request_adjustment'?[document.getElementById('clip-edit-'+key).value]:[],playback_verified:document.getElementById('clip-playback-'+key).checked,framing_verified:document.getElementById('clip-framing-'+key)?.checked===true})});
  const result=await response.json();if(!response.ok)throw new Error(result.detail||'Save failed');
  card.dataset.version=result.decision.version;status.textContent='Saved: '+result.decision.choice+'. No rendering, scheduling or collaborator invitation started.';
  if(result.copy_review_ready||INITIAL.preview_first)location.reload();
 }catch(error){status.textContent='Not saved: '+error.message;}
}
</script>'''


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
