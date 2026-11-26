"""Small progressive enhancement of existing review cards; no approval or delivery logic."""


def toolbar():
    return '''<style>
    .review-navigation{position:sticky;top:0;z-index:4;background:#0b202c;display:flex;gap:.5rem;padding:.5rem;flex-wrap:wrap}
    .review-navigation button{min-height:44px;flex:1}.clip-card{scroll-margin-top:5rem}
    .clip-card video{width:100%;max-height:65svh;object-fit:contain;background:#000}
    .copy-editor video{display:none}@media(max-width:600px){input,select,textarea{font-size:16px}.review-navigation button{padding:.5rem}}
    </style><nav class="review-navigation" aria-label="Clip navigation">
    <button type="button" onclick="navigateReview(-1)">Previous Clip</button>
    <button type="button" onclick="saveCurrentCopy()">Save caption</button>
    <button type="button" onclick="navigateReview(1)">Next Clip</button></nav>'''


SCRIPT = r'''<script>
let activeReviewAsset='',portalInitialized=false;const navigationSaves=new Map();
function reviewCard(asset){return document.getElementById('clip-'+asset.replaceAll(':','-'));}
function rememberReview(card){activeReviewAsset=card.dataset.asset;try{localStorage.setItem('ripped-last:'+TOKEN,activeReviewAsset);}catch(_){} }
function navigateReview(delta){
 const cards=[...document.querySelectorAll('.clip-card')],index=cards.findIndex(c=>c.dataset.asset===activeReviewAsset);
 const card=cards[Math.max(0,Math.min(cards.length-1,(index<0?0:index)+delta))];
 if(card){rememberReview(card);card.scrollIntoView({block:'start'});}
}
async function saveCurrentCopy(){
 const card=reviewCard(activeReviewAsset),editor=card?.querySelector('.asset[data-index]');
 if(!editor){document.getElementById('globalStatus').textContent='Approve a playable clip to open its caption editor, then save your copy.';return;}
 try{await saveAsset(Number(editor.dataset.index));}catch(_){} // Existing editor shows the error and preserves text.
}
async function approveAndNext(asset){
 if(navigationSaves.has(asset))return navigationSaves.get(asset);
 const operation=(async()=>{
  const card=reviewCard(asset),status=card.querySelector('[data-navigation-status]');rememberReview(card);
  try{
   const editor=card.querySelector('.asset[data-index]');if(editor)await saveAsset(Number(editor.dataset.index));
   if(await clipDecision(asset,'approve')!==true)return;
   const data=await refreshReviewState();
   if(!(data.clips||[]).some(c=>c.asset_id===asset&&c.approval_current))throw new Error('Approval is not confirmed. Your inputs are retained.');
   navigateReview(1);
  }catch(error){status.textContent=error.message;status.setAttribute('role','alert');}
 })().finally(()=>navigationSaves.delete(asset));navigationSaves.set(asset,operation);return operation;
}
async function synchronizeReviewCards(data){
 for(const clip of data.clips||[]){
  const item=(data.items||[]).find(i=>i.asset_id===clip.asset_id);
  if(reviewCard(clip.asset_id)&&(!item||document.getElementById('asset-'+item.index)))continue;
  const response=await fetch('/api/ripped-shorts/review/'+TOKEN+'?asset_id='+encodeURIComponent(clip.asset_id),{cache:'no-store'});
  if(!response.ok)throw new Error('Card refresh failed (HTTP '+response.status+'). Saved decisions remain intact.');
  const fragment=await response.json();
  if(fragment.request_id!==INITIAL.request_id||fragment.asset_id!==clip.asset_id)throw new Error('Review card identity mismatch.');
  if(!reviewCard(clip.asset_id))document.getElementById('finished-clips').insertAdjacentHTML('beforeend',fragment.clip_html);
  if(item&&!document.getElementById('asset-'+item.index))document.getElementById('assets').insertAdjacentHTML('beforeend',fragment.copy_html);
 }
 initializeClipReviews();initializeCopyCards();initializeMobileReview();
}
function initializeMobileReview(){
 if(!activeReviewAsset){try{activeReviewAsset=localStorage.getItem('ripped-last:'+TOKEN)||'';}catch(_){} }
 document.querySelectorAll('.asset[data-index]').forEach(editor=>{
  const card=reviewCard(editor.dataset.asset);
  if(card&&!editor.closest('.clip-card')){editor.classList.add('copy-editor');editor.querySelector('video')?.remove();card.append(editor);}
 });
 document.querySelectorAll('.clip-card').forEach(card=>{
  if(card.dataset.mobileInitialized)return;card.dataset.mobileInitialized='1';
  card.addEventListener('focusin',()=>rememberReview(card));card.addEventListener('pointerdown',()=>rememberReview(card));
  card.querySelectorAll('video').forEach(video=>{
   const key='ripped-position:'+TOKEN+':'+card.dataset.asset+':'+card.dataset.mediaRevision;let lastSaved=0;
   video.addEventListener('play',()=>{rememberReview(card);document.querySelectorAll('video').forEach(other=>{if(other!==video)other.pause();});});
   const retain=()=>{try{localStorage.setItem(key,String(video.currentTime));}catch(_){} };
   video.addEventListener('pause',retain);video.addEventListener('timeupdate',()=>{if(Date.now()-lastSaved>5000){lastSaved=Date.now();retain();}});
   const restore=()=>{try{const t=Number(localStorage.getItem(key));if(Number.isFinite(t)&&t>0&&t<video.duration)video.currentTime=t;}catch(_){} };
   if(video.readyState>=1)restore();else video.addEventListener('loadedmetadata',restore,{once:true});
  });
 });
 if(!portalInitialized){portalInitialized=true;const card=reviewCard(activeReviewAsset);if(card)requestAnimationFrame(()=>card.scrollIntoView({block:'start'}));else activeReviewAsset=document.querySelector('.clip-card')?.dataset.asset||'';}
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',initializeMobileReview);else initializeMobileReview();
</script>'''
