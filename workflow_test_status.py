"""Evidence projection and durable journal in existing telegram_requests.state_json.

No worker, database or publishing side effect. Unknown legacy evidence stays unknown.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

STAGES = ('SOURCE_ACCEPTED', 'MEDIA_READY', 'TRANSCRIPT_READY', 'CLIPS_SELECTED',
          'CLIPS_APPROVED', 'RENDERS_VERIFIED', 'CREATORS_APPROVED', 'COPY_APPROVED',
          'SCHEDULER_ACCEPTED', 'DELIVERY_VALIDATED', 'READY_FOR_LIVE_CANARY')
HUMAN = {'CLIPS_APPROVED', 'CREATORS_APPROVED', 'COPY_APPROVED'}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def copy_revision(drafts):
    fields = ('asset_id','asset_type','social_caption','video_title','video_description','hashtags','copy_status')
    frozen = sorted(({k:(d.get(k) or '') for k in fields} for d in drafts), key=lambda d:d['asset_id'])
    return hashlib.sha256(json.dumps(frozen,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def context(request_id, state):
    old = state.get('workflow_test') or {}
    parsed = state.get('parsed') or {}
    return dict(test_run_id=old.get('test_run_id') or 'TW-'+request_id, request_id=request_id,
        ripped_shorts_job_id=request_id, brand_id=state.get('show_id'),
        source_video_id=parsed.get('video_id'), source_channel_id=state.get('youtube_channel_id'),
        audio_master_job_id=(state.get('audio_master_ingest') or {}).get('job_id'),
        evidence_mode='SIMULATED' if os.getenv('WORKFLOW_EVIDENCE_MODE')=='SIMULATED' else old.get('evidence_mode','REAL'))


def project(request_id, state, creators=None, scheduler=None):
    creators=creators or state.get('creator_approval')
    identity = context(request_id,state)
    prior = (state.get('workflow_test') or {}).get('checkpoints') or {}
    stages = {}
    def stage(name,status,evidence=None,reason=None,service='rippedshorts'):
        previous = prior.get(name) or {}
        stages[name] = dict(status=status, service=service, evidence=evidence or {},
            reason=reason, failure_category=(reason.split(':')[0] if reason and status in {'FAILED','BLOCKED'} else None), requires_human=name in HUMAN,
            evidence_mode=identity['evidence_mode'],
            observed_at=previous.get('observed_at') if previous.get('status')==status else stamp())
    source_ok = bool(identity['source_video_id'] and identity['brand_id'])
    stage('SOURCE_ACCEPTED','PASSED' if source_ok else 'WAITING_FOR_HUMAN',
          {'source_video_id':identity['source_video_id'],'brand_id':identity['brand_id']})
    media = state.get('source_validation') or state.get('source_video_validation') or {}
    upstream_media = ((state.get('audio_master_ingest') or {}).get('checkpoints') or {}).get('MEDIA_READY') or {}
    if not media and upstream_media.get('status') == 'PASSED':
        media = {**(upstream_media.get('evidence') or {}), 'status': 'verified', 'verified_by': 'audiomaster'}
    stage('MEDIA_READY','PASSED' if media.get('status')=='verified' else 'NOT_STARTED',media)
    transcript = state.get('transcript_evidence') or {}
    stage('TRANSCRIPT_READY','PASSED' if transcript.get('timed_segments',0)>0 and transcript.get('speech_evidence') else 'NOT_STARTED',transcript)
    upstream = state.get('audio_master_ingest') or {}
    if stages['TRANSCRIPT_READY']['status']!='PASSED' and upstream.get('error_class')=='TRANSCRIPTION_AUTHENTICATION_BLOCKED':
        stage('TRANSCRIPT_READY','BLOCKED',{'audio_master_job_id':upstream.get('job_id')},
              'TRANSCRIPTION_AUTHENTICATION_BLOCKED: owner must correct configured provider access','audiomaster')
    raw_error=str(state.get('error') or '')
    if raw_error and stages['TRANSCRIPT_READY']['status']!='PASSED':
        auth=any(word in raw_error for word in ('TRANSCRIPTION_AUTHENTICATION_BLOCKED','invalid_api_key'))
        stage('TRANSCRIPT_READY','BLOCKED' if auth else 'FAILED',{},
              'TRANSCRIPTION_AUTHENTICATION_BLOCKED: owner action required' if auth else 'PROCESSING_FAILED: inspect sanitized service diagnostics')
    candidates = list((state.get('result') or {}).get('segments') or []) + list((state.get('topic_result') or {}).get('segments') or [])
    stage('CLIPS_SELECTED','PASSED' if candidates else 'NOT_STARTED',{'candidate_count':len(candidates)})
    accepted=[]
    for lane,result_key,review_key in [('short','result','candidate_reviews'),('highlight','topic_result','topic_reviews')]:
        clips=(state.get(result_key) or {}).get('segments') or []
        for index,review in (state.get(review_key) or {}).items():
            if review.get('status') in {'queued','rendering','rendered','render_failed'}:
                i=int(index);clip=clips[i] if i<len(clips) else {}
                number=clip.get('candidate_number') or i+1
                accepted.append((f'{request_id}:{lane}:{number}',review,clip))
    ids=[a[0] for a in accepted]
    approval_ok=bool(accepted) and all(r.get('user_id') and (r.get('approved_at') or r.get('reviewed_at')) for _,r,_ in accepted)
    stage('CLIPS_APPROVED','PASSED' if approval_ok else 'WAITING_FOR_HUMAN',{'asset_group_ids':ids})
    renders=[]
    for asset,r,c in accepted:
        proof=c.get('completion_check') or {}
        renders.append(dict(asset_id=asset,status=r.get('status'),
            media_url=r.get('clip_url') or r.get('segment_url'),proof=proof))
    valid=bool(renders) and all(r['status']=='rendered' and r['media_url'] and
        all(r['proof'].get(k) is True for k in ('media_duration','video_stream','audio_stream','format_valid','export_edges')) for r in renders)
    failed=any(r['status']=='render_failed' for r in renders)
    stage('RENDERS_VERIFIED','PASSED' if valid else 'FAILED' if failed else 'RUNNING' if renders else 'NOT_STARTED',{'assets':renders},
          'Render failed; retain prior source/transcript and review affected asset' if failed else None)
    required=identity['brand_id'] in {'TCB','TDOG'}
    creator_ok=bool(creators and creators.get('request_id')==request_id and creators.get('approved_at'))
    stage('CREATORS_APPROVED','PASSED' if creator_ok else 'WAITING_FOR_HUMAN' if required else 'SKIPPED_NOT_APPLICABLE',
          {'approval_revision':(creators or {}).get('approval_revision'),
           'instagram_collaborators':[c['instagram_handle'] for c in (creators or {}).get('creators',[]) if c.get('intent')=='invite']})
    drafts=state.get('copy_drafts') or []
    copied={d.get('asset_id') for d in drafts if d.get('copy_status') in {'approved','edited'} and
        (d.get('social_caption') if d.get('asset_type')=='9:16_SHORT' else d.get('video_title') and d.get('video_description'))}
    copy_ok=bool(ids) and set(ids)<=copied and bool(state.get('copy_review_completed_at'))
    revision=copy_revision(drafts)
    stage('COPY_APPROVED','PASSED' if copy_ok else 'WAITING_FOR_HUMAN',{'approval_revision':revision,'asset_group_ids':sorted(copied)})
    response=scheduler if scheduler is not None else (state.get('schedule_master') or {}).get('shorts_response') or {}
    expected=set(ids)
    response_groups=set(response.get('asset_group_ids') or [])
    ack=bool(expected and expected==response_groups and response.get('request_id')==request_id and
        response.get('test_run_id')==identity['test_run_id'] and response.get('approved_caption_revision')==revision and
        response.get('approved_collaborator_revision')==(creators or {}).get('approval_revision'))
    stage('SCHEDULER_ACCEPTED','PASSED' if ack else 'BLOCKED' if response else 'NOT_STARTED',
          {'delivery_ids':response.get('delivery_ids',[]),'asset_group_ids':sorted(response_groups),
           'acknowledged_at':response.get('acknowledged_at')},None if ack else 'Exact approved batch acknowledgment is not available','schedulemaster')
    validation=response.get('delivery_validation') or {}
    validated=ack and validation.get('status')=='PASSED' and validation.get('asset_group_ids')==sorted(expected)
    stage('DELIVERY_VALIDATED','PASSED' if validated else 'BLOCKED' if response else 'NOT_STARTED',validation,
          None if validated else 'Run authenticated Schedule Master validation; permissions, mapping, media or schedule evidence incomplete','schedulemaster')
    complete=all(v['status'] in {'PASSED','SKIPPED_NOT_APPLICABLE'} for v in stages.values())
    simulated=identity['evidence_mode']!='REAL' or response.get('evidence_mode','REAL')!='REAL' or validation.get('evidence_mode','REAL')!='REAL'
    paused=response.get('publishing_globally_paused')
    ready=complete and not simulated and paused is True
    stage('READY_FOR_LIVE_CANARY','PASSED' if ready else 'BLOCKED' if complete or response else 'NOT_STARTED',{'internal_checks_complete':complete,
          'real_evidence':not simulated,'publishing_globally_paused':paused},
          None if ready else 'Complete real checkpoints with publishing paused; simulated completion cannot authorize live publication')
    blockers=[(n,v) for n,v in stages.items() if v['status'] not in {'PASSED','SKIPPED_NOT_APPLICABLE'}]
    block=blockers[0] if blockers else None
    result={**identity,'checkpoints':stages,'asset_group_ids':ids,'approved_caption_revision':revision,
        'approved_collaborator_revision':(creators or {}).get('approval_revision'),
        'delivery_ids':response.get('delivery_ids',[]),'publishing_globally_paused':paused,
        'instagram_collaborators_in_approved_payload':bool(stages['CREATORS_APPROVED']['evidence'].get('instagram_collaborators')),
        'real_platform_acknowledgment':response.get('real_platform_acknowledgment',False),
        'overall_status':'READY_FOR_LIVE_CANARY' if ready else 'INTERNAL_SIMULATION_PASSED' if complete and simulated else 'BLOCKED' if any(v['status'] in {'BLOCKED','FAILED'} for _,v in blockers) else 'WAITING_FOR_HUMAN' if any(v['status']=='WAITING_FOR_HUMAN' for _,v in blockers) else 'RUNNING',
        'last_successful_stage':next((n for n in reversed(STAGES) if stages[n]['status']=='PASSED'),None),
        'blocking_stage':block[0] if block else None,
        'next_required_action':(block[1].get('reason') or 'Complete '+block[0]) if block else 'Owner approval for the named asset, destinations and publication window is required',
        'human_approvals_required':[n for n in HUMAN if stages[n]['status']!='PASSED'],
        'internal_validation_completed_at':response.get('internal_validation_completed_at'),
        'internal_test_completed_at':response.get('internal_test_completed_at'),
        'public_publication_complete':False}
    from batch_progress import summarize
    result['batch_summary']=summarize(request_id,state,response)
    return result


def journal(request_id,state,creators=None):
    result=project(request_id,state,creators)
    previous=state.get('workflow_test') or {}
    history=list(previous.get('history') or [])
    for name,stage in result['checkpoints'].items():
        old=(previous.get('checkpoints') or {}).get(name) or {}
        if stage['status']!=old.get('status'):
            history.append({'stage':name,'status':stage['status'],'at':stamp(),'service':stage['service']})
    result['history']=history
    state['workflow_test']=result
    return state
