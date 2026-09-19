'use strict';
const $ = id => document.getElementById(id);
let current=null,lane='shorts',index=0,busy=false;
function notice(text){$('notice').textContent=text;}
function show(view){for(const id of ['login','dashboard','workspace'])$(id).hidden=id!==view;$('logout').hidden=view==='login';}
async function api(path,options={}){
  const res=await fetch('/review/api'+path,{...options,headers:{'Content-Type':'application/json','X-Review-Request':'1'}});
  if(!res.ok){const error=await res.json().catch(()=>({}));if(res.status===401)show('login');throw Error(typeof error.detail==='string'?error.detail:'Please check the form fields.');}
  return res.json();
}
async function action(fn){if(busy)return;busy=true;try{await fn();}catch(e){notice(e.message);}finally{busy=false;}}
function timecode(value){const parts=value.trim().split(':').map(Number);if(!parts.length||parts.length>3||parts.some(x=>!Number.isFinite(x)||x<0))throw Error('Use section times such as 10:00 - 30:00.');return parts.reduce((total,part)=>total*60+part,0);}
function ranges(){return $('sections').value.trim().split('\n').filter(Boolean).map(line=>{const pair=line.split(/\s*[-–]\s*/);if(pair.length!==2)throw Error('Enter one start - end section per line.');return {start:timecode(pair[0]),end:timecode(pair[1])};});}
async function dashboard(){
  current=null;const result=await api('/projects');show('dashboard');$('projects').replaceChildren();
  for(const project of result.projects){const button=document.createElement('button');button.className='project';const title=document.createElement('strong');title.textContent=project.title;const meta=document.createElement('span');meta.textContent=`${project.stage.replaceAll('_',' ')} · ${project.counts.shorts.total} Shorts · ${project.counts.highlights.total} Highlights`;button.append(title,meta);button.onclick=()=>action(()=>openProject(project.project_id));$('projects').append(button);}
  if(!result.projects.length)$('projects').textContent='No projects yet. Verify the order, then submit its selected sections.';
}
async function openProject(id){current=await api('/projects/'+id);index=0;show('workspace');render();}
function clip(){return current?.lanes[lane]?.[index];}
function safeDrive(url){try{const u=new URL(url);return u.protocol==='https:'&&u.hostname==='drive.google.com'?u:null;}catch{return null;}}
function render(){
  const items=current.lanes[lane];index=Math.min(index,Math.max(0,items.length-1));
  $('project-title').textContent=current.title;$('project-status').textContent=`${current.stage.replaceAll('_',' ')} · ${current.eta_label} · Review until ${new Date(current.expires_at).toLocaleDateString()}`;
  $('shorts-tab').classList.toggle('selected',lane==='shorts');$('highlights-tab').classList.toggle('selected',lane==='highlights');
  $('count').textContent=items.length?`${index+1} of ${items.length}`:'No clips yet';$('empty').hidden=!!items.length;$('review-card').hidden=!items.length;
  $('empty').textContent=current.error||'Selections will appear when processing finishes. Only complete, usable moments are returned.';
  $('retry').hidden=current.status!=='error'||current.expired;$('transcript-download').href=`/review/api/projects/${current.project_id}/transcript.txt`;$('transcript-download').hidden=!current.transcript_ready||current.expired;
  const c=clip();if(!c)return;$('clip-title').textContent=c.title;$('clip-state').textContent=c.status.replaceAll('_',' ');$('transcript').textContent=c.transcript;$('clip-error').textContent=c.error||'';
  $('clip-timing').textContent=`Source start ${Math.round(Number(c.start)+Number(c.source_offset||0))}s · ${Math.round(Number(c.duration))} seconds`;
  const player=$('player');player.classList.toggle('wide',lane==='highlights');player.replaceChildren();const drive=safeDrive(c.asset_url);const match=drive?.pathname.match(/\/file\/d\/([\w-]+)/);
  if(match){const frame=document.createElement('iframe');frame.src=`https://drive.google.com/file/d/${match[1]}/preview`;frame.title='Finished video preview';frame.allow='fullscreen';player.append(frame);$('preview-label').textContent='Finished clip. Google may require sign-in.';}
  else if(!current.expired){const video=document.createElement('video');video.controls=true;video.playsInline=true;video.preload='metadata';video.src=`/review/api/projects/${current.project_id}/source?lane=${lane}&index=${index}`;video.onloadedmetadata=()=>{video.currentTime=Number(c.start)||0;};video.ontimeupdate=()=>{if(video.currentTime>=Number(c.start)+Number(c.duration))video.pause();};video.onerror=()=>{const text=document.createElement('p');text.textContent='Source preview unavailable. Review the transcript or finished output.';player.replaceChildren(text);};player.append(video);$('preview-label').textContent='Source section preview. Final framing is generated after approval.';}
  else $('preview-label').textContent='Review access has expired.';
  for(const id of ['approve','reject'])$(id).disabled=current.expired||['queued','rendering','rendered'].includes(c.status);
  $('download').hidden=!drive||current.expired;if(drive)$('download').href=drive.href;$('previous').disabled=index===0;$('next').disabled=index===items.length-1;
}
function move(delta){index=Math.max(0,Math.min(current.lanes[lane].length-1,index+delta));render();}
async function decision(value){const c=clip();if(!c||current.expired||['queued','rendering','rendered'].includes(c.status))return;const result=await api(`/projects/${current.project_id}/${lane}/${index}/decision`,{method:'POST',body:JSON.stringify({decision:value})});c.status=result.status;notice(value==='approve'?'Kept and queued for rendering.':'Skipped.');move(1);}
$('login-form').onsubmit=e=>{e.preventDefault();action(async()=>{await api('/session',{method:'POST',body:JSON.stringify({secret:$('secret').value})});$('secret').value='';notice('');await dashboard();});};
$('logout').onclick=()=>action(async()=>{await api('/session',{method:'DELETE'});current=null;show('login');});
$('source-form').onsubmit=e=>{e.preventDefault();action(async()=>{const result=await api('/projects',{method:'POST',body:JSON.stringify({url:$('source').value,mode:$('mode').value,customer_reference:$('customer').value,selected_ranges:ranges(),plan:$('plan').value,allowance_checked:$('checked').checked})});notice('Selected sections submitted.');await openProject(result.project_id);});};
$('back').onclick=()=>action(dashboard);$('refresh').onclick=()=>action(dashboard);for(const name of ['shorts','highlights'])$(name+'-tab').onclick=()=>{lane=name;index=0;render();};
$('previous').onclick=()=>move(-1);$('next').onclick=()=>move(1);$('approve').onclick=()=>action(()=>decision('approve'));$('reject').onclick=()=>action(()=>decision('reject'));
$('retry').onclick=()=>action(async()=>{await api(`/projects/${current.project_id}/retry`,{method:'POST'});notice('Retry queued.');});
let origin=null;$('swipe').onpointerdown=e=>{origin={x:e.clientX,y:e.clientY};$('swipe').setPointerCapture(e.pointerId);};$('swipe').onpointercancel=()=>{origin=null;};$('swipe').onpointerup=e=>{if(!origin)return;const dx=e.clientX-origin.x,dy=e.clientY-origin.y;origin=null;if(Math.abs(dx)>70&&Math.abs(dx)>Math.abs(dy)*1.5)action(()=>decision(dx>0?'approve':'reject'));};
setInterval(async()=>{if(!current||busy||document.hidden)return;const id=current.project_id;try{const updated=await api('/projects/'+id);if(busy||current?.project_id!==id)return;const changed=JSON.stringify(updated.lanes)!==JSON.stringify(current.lanes);current=updated;if(changed)render();else{$('project-status').textContent=`${current.stage.replaceAll('_',' ')} · ${current.eta_label}`;$('retry').hidden=current.status!=='error'||current.expired;}}catch(e){notice(e.message);}},12000);
action(dashboard);
