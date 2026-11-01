"""Finished-clip decisions in the existing request and intelligence ledger."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json

from audio_master_handoff import connect
from batch_progress import selected_assets, copy_frozen

REASONS = {'poor_topic', 'weak_hook', 'incomplete_thought', 'unnecessary_length',
           'inaccurate_transcript', 'unsuitable_framing', 'bad_audio'}
EDIT_FIELDS = {'boundaries', 'topic_context', 'caption', 'framing', 'audio'}
PROOF = ('media_duration', 'video_stream', 'audio_stream', 'format_valid', 'export_edges')


def clips(request_id, state, comparisons=()):
    result = []
    for asset, lane, review, candidate in selected_assets(request_id, state):
        check = candidate.get('completion_check') or {}
        url = review.get('clip_url') or review.get('segment_url') or ''
        finished = bool(review.get('status') == 'rendered' and url and all(check.get(k) is True for k in PROOF))
        bounds = candidate.get('rendered_bounds') or [candidate.get('start'), candidate.get('end')]
        fingerprint = hashlib.sha256(json.dumps([url, bounds, check, review.get('rendered_review_revisions')], sort_keys=True).encode()).hexdigest()
        decision = (state.get('website_clip_reviews') or {}).get(asset) or {}
        editorial = next((r for r in comparisons if r['asset_id'] == asset and r['kind'] == 'editorial'), None)
        transcript = str(candidate.get('transcript') or '')
        if not finished and editorial and editorial.get('decision') == 'proposed':
            transcript = str(editorial['provenance'].get('proposed_transcript') or transcript)
            bounds = editorial['proposed']
        provenance = candidate.get('selection_provenance') or {}
        result.append(dict(asset_id=asset, format='9:16' if lane == 'short' else '16:9',
            number=asset.rsplit(':', 1)[-1], title=candidate.get('title') or candidate.get('topic') or next(iter(candidate.get('topic_titles') or []),None) or candidate.get('main_theme') or ((state.get('result') or {}).get('analysis') or {}).get('main_theme') or 'Clip '+asset.rsplit(':',1)[-1],
            transcript=transcript, reason=candidate.get('reason') or candidate.get('why_it_will_work'),
            category=candidate.get('category') or candidate.get('highlight_type') or 'Not recorded',
            topic_ids=candidate.get('topic_ids') or [], provenance=provenance,
            learning_influence=candidate.get('learning_influence'), framing=candidate.get('framing_plan'),
            render_status=review.get('status'), error=review.get('render_error'), finished=finished,
            media_url=url if finished else '', bounds=bounds, media_revision=fingerprint,
            decision=decision, approval_current=bool(decision.get('choice') == 'approve' and finished and decision.get('media_revision') == fingerprint),
            frozen=copy_frozen(state, asset)))
    return result


def save(request_id, asset_id, body):
    choice = body.get('choice')
    reason = body.get('reason') or ''
    feedback = str(body.get('feedback') or '')[:2000]
    fields = body.get('changed_fields') or []
    if choice not in {'approve', 'reject', 'request_adjustment'}:
        raise ValueError('Choose Approve, Reject or Request Adjustment')
    if choice == 'reject' and reason not in REASONS:
        raise ValueError('Choose the actual rejection reason')
    if choice == 'request_adjustment' and (not feedback.strip() or not fields or any(f not in EDIT_FIELDS for f in fields)):
        raise ValueError('Describe the adjustment and select what needs changing')
    from clip_review_state import delivery_blockers
    with closing(connect()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM telegram_requests WHERE request_id=?', (request_id,)).fetchone()
        if not row:
            raise ValueError('Request not found')
        state = json.loads(row['state_json'])
        item = next((c for c in clips(request_id, state) if c['asset_id'] == asset_id), None)
        if not item or item['frozen']:
            raise ValueError('Asset is missing or already submitted; existing approvals are frozen')
        current = (state.get('website_clip_reviews') or {}).get(asset_id) or {}
        if type(body.get('version', 0)) is not int or body.get('version', 0) != current.get('version', 0):
            raise ValueError('Stale review version; refresh before saving')
        if choice == 'approve':
            if not item['finished'] or body.get('playback_verified') is not True:
                raise ValueError('Play the verified finished clip with audio before approving')
            if any(b.startswith(asset_id+': ') for b in delivery_blockers(request_id, state) if 'clip approval' not in b):
                raise ValueError('The finished render does not match the editorial/framing approval')
        decision = dict(choice=choice, reason=reason, feedback=feedback, changed_fields=fields,
            media_revision=item['media_revision'], version=current.get('version', 0)+1,
            actor=str(row['user_id']), decided_at=datetime.now(timezone.utc).isoformat(),
            approval_scope='clip_editorial_only', native_invitation_authorized=False)
        if choice != 'reject':
            decision['reason'] = ''
        decision['prior_versions'] = list(current.get('prior_versions') or []) + ([{k:v for k,v in current.items() if k != 'prior_versions'}] if current else [])
        state.setdefault('website_clip_reviews', {})[asset_id] = decision
        state['clip_review_ui_version'] = 2
        db.execute('UPDATE telegram_requests SET state_json=?,updated_at=? WHERE request_id=?',
                   (json.dumps(state), decision['decided_at'], request_id))
    # Same existing learning ledger/Sheets projection, no second learning system.
    try:
        from intelligence_workflow import observe
        observe(request_id, state)
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Saved clip feedback projection pending request=%s', request_id)
    return dict(status='saved', decision=decision, scheduling_started=False, rendering_started=False,
                collaborator_invitation_authorized=False)


def blocked_assets(request_id, state):
    if state.get('clip_review_ui_version') != 2:
        return set()  # preserve prior approved/submitted workflows
    return {c['asset_id'] for c in clips(request_id, state) if not c['approval_current']}
