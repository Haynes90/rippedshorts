"""Editorial and framing sections of the existing review portal."""
import html
import json


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
