"""Clip identities and full selected-batch accounting; no external operations."""
from datetime import datetime

SELECTED = {'approved','approve','queued','rendering','rendered','render_failed'}


def selected_assets(request_id,state):
    result=[]
    for lane,key,reviews in [('short','result','candidate_reviews'),('highlight','topic_result','topic_reviews')]:
        clips=(state.get(key) or {}).get('segments') or []
        for index,review in (state.get(reviews) or {}).items():
            if review.get('status') not in SELECTED:
                continue
            number=int(index);clip=clips[number] if number<len(clips) else {}
            candidate=(clip.get('candidate_number') or number+1) if lane=='short' else number+1
            result.append((f'{request_id}:{lane}:{candidate}',lane,review,clip))
    return result


def shared_failure(state):
    for _,_,review,_ in selected_assets('batch',state):
        if review.get('status')!='render_failed':
            continue
        error=str(review.get('render_error') or '').lower()
        if any(s in error for s in ('authentication_blocked','invalid_api_key','unauthorized','forbidden','401','403','credential','storage quota','no space left','disk full','source identity mismatch','data integrity','safety hold')):
            return 'Shared authentication, storage, integrity or safety failure; review before continuing'
    return None


def submitted_ids(state):
    schedule=state.get('schedule_master') or {}
    return set((schedule.get('shorts_response') or {}).get('asset_group_ids') or [])


def copy_frozen(state,asset_id):
    schedule=state.get('schedule_master') or {}
    if schedule.get('shorts_status')=='sending':
        return True
    ids=submitted_ids(state)
    return asset_id in ids or (schedule.get('shorts_status')=='accepted' and not ids)


def summarize(request_id,state,scheduler=None):
    selected=selected_assets(request_id,state)
    original=(state.get('schedule_master') or {}).get('shorts_response') or {}
    scheduler=scheduler or original
    manifest=original.get('delivery_manifest') or scheduler.get('delivery_manifest') or []
    expected={}
    for delivery in manifest:
        expected.setdefault(delivery['asset_group_id'],set()).add(delivery['delivery_id'])
    actual={r['delivery_id']:r for r in (scheduler.get('delivery_validation') or {}).get('deliveries',[])}
    drafts={d.get('asset_id'):d for d in state.get('copy_drafts') or []}
    playback=state.get('website_playback_checks') or {}
    counts={lane:dict(proposed=len((state.get(key) or {}).get('segments') or []),selected=0,rendered_audio_verified=0,playable=0,handed_off=0,scheduled=0,completed=0)
            for lane,key in [('short','result'),('highlight','topic_result')]}
    assets=[]
    for asset,lane,review,clip in selected:
        count=counts[lane];count['selected']+=1
        proof=clip.get('completion_check') or {}
        rendered=review.get('status')=='rendered' and all(proof.get(k) is True for k in ('media_duration','video_stream','audio_stream','format_valid','export_edges'))
        url=review.get('clip_url') or review.get('segment_url')
        playable=bool(rendered and url and (playback.get(asset) or {}).get('media_url')==url and (playback.get(asset) or {}).get('verified'))
        copy_ok=drafts.get(asset,{}).get('copy_status') in {'approved','edited'}
        creator=state.get('creator_approval') or {}
        held=bool(drafts.get(asset,{}).get('editorial_hold'))
        approval_ok=copy_ok and not held and (state.get('show_id') not in {'TCB','TDOG'} or bool(creator.get('approved_at')))
        ids=expected.get(asset,set())
        delivery_rows=[actual[i] for i in ids if i in actual]
        scheduled=bool(ids and len(delivery_rows)==len(ids) and all(r.get('scheduled_at') and r.get('destination_id') and r.get('schedule_record_valid') for r in delivery_rows))
        complete=bool(rendered and playable and approval_ok and scheduled)
        for key,value in [('rendered_audio_verified',rendered),('playable',playable),('handed_off',bool(ids)),('scheduled',scheduled),('completed',complete)]:
            count[key]+=int(bool(value))
        cause='HELD_BY_OWNER' if held else 'RENDER_FAILED' if review.get('status')=='render_failed' else 'RENDER_PENDING' if not rendered else 'PLAYBACK_NOT_VERIFIED' if not playable else 'HUMAN_APPROVAL_PENDING' if not approval_ok else 'SCHEDULER_RECORDS_INCOMPLETE' if not scheduled else None
        assets.append(dict(asset_id=asset,format='9:16' if lane=='short' else '16:9',completed=complete,
            cause=cause,render_error=review.get('render_error') if review.get('status')=='render_failed' else None,
            deliveries=delivery_rows))
    total=len(assets);done=sum(a['completed'] for a in assets)
    percent=100*done/total if total else None
    shared=shared_failure(state)
    status='NOT STARTED' if not total else 'BATCH BLOCKED' if shared else 'BATCH COMPLETE' if done==total else 'BATCH MOSTLY SUCCESSFUL' if percent>=80 else 'BATCH PARTIAL FAILURE'
    return dict(status=status,completion_percentage=percent,selected_clips=total,completed_clips=done,
        formats=counts,clips=assets,failures=[a for a in assets if not a['completed']],shared_blocker=shared,
        benchmark_passed=bool(total and percent>=80 and counts['short']['completed'] and counts['highlight']['completed']),
        awaiting_human=any(a['cause']=='HUMAN_APPROVAL_PENDING' for a in assets),public_publishing_required=False)
