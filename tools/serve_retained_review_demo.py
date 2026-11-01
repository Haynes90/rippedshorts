"""Existing portal against an isolated DB and retained media; no production access."""
import os
from pathlib import Path
root=Path(__file__).resolve().parents[1]
data=root/'.test-tmp'/'tdog-review-demo'
os.environ['DATA_DIR']=str(data)
os.environ['RIPPED_SHORTS_PUBLIC_URL']='http://127.0.0.1:8767'
os.environ['PUBLISHING_ENABLED']='false'
import hashlib
import json
import shutil
from fastapi import FastAPI
import uvicorn
import audio_master_handoff as storage
import telegram_intake as tg
import caption_review_web as web
import clip_review_state as review

request='94893455-5f48-46de-bdfb-642212ea19a5'
folder=root/'acceptance-media'/'boundary-review'
manifest=json.loads((folder/'manifest.json').read_text())
transcript=json.loads((root/'acceptance-media'/'retained-transcript.json').read_text())
segments=transcript['segments'] if isinstance(transcript,dict) else transcript
state={'acceptance_demo':True,'show_id':'TDOG','parsed':{'video_id':'9mW0bhmUaw0'},'vid_title':'TDOG S2E39 — retained five-clip review',
    'result':{'segments':[{} for _ in range(11)]},'topic_result':{'segments':[{} for _ in range(2)]},
    'candidate_reviews':{},'topic_reviews':{}}
for item in manifest:
    lane,number=item['asset'].split('-');index=int(number)-1
    key,reviews=('result','candidate_reviews') if lane=='short' else ('topic_result','topic_reviews')
    state[key]['segments'][index]={'start':item['original'][0],'end':item['original'][1], 'candidate_number':int(number),'transcript':'Retained selection; final export review remains required.'}
    state[reviews][str(index)]={'status':'render_failed','render_error':'Retained production selection; no retry performed in demo'}
with tg._telegram_db() as db:
    db.execute('INSERT OR IGNORE INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)',
        (request,request,'isolated-no-telegram','isolated-owner','awaiting_render_retry','both','youtube','9mW0bhmUaw0',json.dumps(state),'demo','demo'))

def artifact(path):
    target=data/'review-previews'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        try:os.link(path,target)
        except OSError:shutil.copyfile(path,target)
    with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'local_path':str(target),'size':target.stat().st_size,'sha256':digest}

for item in manifest:
    lane,number=item['asset'].split('-');asset=f'{request}:{lane}:{number}'
    media={key:artifact(folder/(item['asset']+'-'+filename+'.mp4')) for key,filename in [('original','original'),('proposed','proposed'),('opening','start-context'),('ending','end-context')]}
    start,end=item['proposed']
    text=' '.join(s['text'] for s in segments if float(s.get('start',0))>=start-.01 and float(s.get('end',s.get('start',0)+s.get('duration',0)))<=end+.01)
    provenance={'source_video_id':'9mW0bhmUaw0','source_duration':3679.07,'verified_media':True,
        'source_sha256':'45314424b1529f9d169f8d08c4fce31ddab10b17b122ed107c98458376b80a67',
        'timing_provenance':'provider_segment_timing; editorial alternatives require owner listening',
        'transcript_context':item['context'],'explanation':item['reason'],'proposed_transcript':text}
    review.register(request,asset,'editorial',item['original'],item['proposed'],media,provenance,notify=False)
    if lane=='short':
        review.register(request,asset,'framing',item['original'],item['original'],
            {'reference':artifact(folder/(item['asset']+'-original.mp4')), 'blurred':artifact(folder/(item['asset']+'-blurred.mp4'))},
            {**provenance,'preview_bounds':item['original'],'explanation':'Full source preserved in the sharp foreground. Speaker confidence is insufficient for a smart alternative.'},notify=False)
url=web.review_url_for_request(request,'editorial')
(data/'demo-url.txt').write_text(url)
print('ISOLATED_DEMO_URL='+url,flush=True)
app=FastAPI();app.include_router(web.router)
uvicorn.run(app,host='127.0.0.1',port=8767,log_level='warning')
