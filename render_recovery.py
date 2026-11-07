"""Request-scoped render receipts in the existing SQLite evidence/lease tables.

No Sheets reconstruction, intake, provider analysis, scheduling or publishing.
An uncertain or deterministic attempt requires review, not another identical run.
"""
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import hashlib
import json
from datetime import datetime, timezone

import production_intelligence as evidence

LANES = {'short': ('result', 'candidate_reviews'), 'highlight': ('topic_result', 'topic_reviews')}
SELECTED = {'approved', 'approve', 'queued', 'rendering', 'rendered', 'render_failed'}
PROOF = ('media_duration', 'video_stream', 'audio_stream', 'format_valid', 'export_edges')
_CURRENT = ContextVar('render_asset_ticket', default=None)


class RenderReviewRequired(ValueError):
    pass


def digest(value):
    return hashlib.sha256(evidence.encoded(value).encode()).hexdigest()


def _put(db, kind, key, payload):
    body = evidence.encoded(payload)
    prior = db.execute('SELECT kind,payload FROM intelligence_records WHERE record_id=?', (key,)).fetchone()
    if prior:
        if prior['kind'] != kind or prior['payload'] != body:
            raise RenderReviewRequired('Immutable render evidence differs; verification required')
        return key
    db.execute('INSERT INTO intelligence_records(record_id,kind,payload,digest,updated_at) VALUES(?,?,?,?,?)',
               (key, kind, body, evidence.identity(payload), datetime.now(timezone.utc).isoformat()))
    return key


def _get(db, key, kind):
    row = db.execute('SELECT kind,payload,digest FROM intelligence_records WHERE record_id=?', (key,)).fetchone()
    if not row:
        return None
    payload = json.loads(row['payload'])
    if row['kind'] != kind or evidence.identity(payload) != row['digest']:
        raise RenderReviewRequired('Render evidence fingerprint failed')
    return payload


def _state(db, request_id):
    row = db.execute('SELECT * FROM telegram_requests WHERE request_id=?', (request_id,)).fetchone()
    if not row:
        raise RenderReviewRequired('Existing request is required')
    return row, json.loads(row['state_json'])


def asset_id(request_id, lane, clip, index):
    number = clip.get('candidate_number') or index + 1 if lane == 'short' else index + 1
    return f'{request_id}:{lane}:{number}'


def ticket(request_id, state, lane, index):
    key, reviews = LANES[lane]
    clips = (state.get(key) or {}).get('segments') or []
    if type(index) is not int or not 0 <= index < len(clips):
        raise RenderReviewRequired('Selected asset is unavailable')
    clip = clips[index]
    review = (state.get(reviews) or {}).get(str(index)) or {}
    if review.get('status') not in SELECTED:
        raise RenderReviewRequired('Explicit candidate selection is required')
    identity = asset_id(request_id, lane, clip, index)
    from clip_review_state import prepare_segment
    prepared, revisions = prepare_segment(request_id, identity, clip)
    inputs = {k: prepared.get(k) for k in ('start', 'end', 'duration', 'framing_mode')}
    inputs['transcript'] = prepared.get('transcript')
    inputs.update(candidate_id=clip.get('candidate_id') or identity, review_revisions=revisions,
                  source_sha256=(state.get('source_validation') or {}).get('source_sha256'))
    revision = digest(inputs)
    return dict(request_id=request_id, lane=lane, index=index, asset_id=identity,
                candidate_id=inputs['candidate_id'], revision=revision,
                approval_revisions=revisions, source_sha256=inputs['source_sha256'],
                action='asset-render:' + identity + ':' + revision)


def _successful(clip, review):
    return (review.get('status') == 'rendered' and (review.get('clip_url') or review.get('segment_url'))
            and all((clip.get('completion_check') or {}).get(k) is True for k in PROOF))


def failure_kind(review):
    text = str(review.get('render_error') or '').lower()
    diagnostic = review.get('render_diagnostics') or {}
    if (review.get('render_error_type') == 'CompletionReviewRequired' or diagnostic.get('stage')
            or any(v in text for v in ('boundary', 'boundaries', 'speech', 'sentence', 'complete thought',
                                       'approved excerpt', 'framing', 'source identity', 'integrity'))):
        return 'HELD_VALIDATION'
    if (review.get('render_failure_phase') == 'source_access'
            and review.get('render_error_type') in {'Timeout', 'ReadTimeout', 'ConnectTimeout', 'ConnectionError'}):
        return 'FAILED_TRANSIENT_SOURCE_ACCESS'
    return 'HELD_VERIFICATION'  # upload/commit/unknown outcomes must never be blindly replayed


def _assert_local_selection(state):
    if state.get('recovered_approvals_from_sheet') and not state.get('selection_restored_from_receipt'):
        raise RenderReviewRequired('Video-wide retry overwrote this selection; verified request receipt restoration required')


def _reviewed_correction(db, item):
    """A new explicit editorial/framing approval can authorize a new revision.

    A code version, changed source checksum, clock or repeated approval click is
    not evidence of a corrected boundary and cannot release a deterministic hold.
    """
    for row in db.execute('SELECT payload FROM intelligence_records WHERE kind=? AND record_id LIKE ?',
                          ('render_hold', 'asset-render:' + item['request_id'] + ':%')):
        hold = json.loads(row['payload'])
        old = hold.get('ticket') or {}
        if old.get('asset_id') != item['asset_id'] or old.get('candidate_id') != item['candidate_id']:
            continue
        if old.get('source_sha256') != item['source_sha256']:
            continue
        if hold.get('category', hold.get('outcome')) != 'HELD_VALIDATION':
            continue  # uncertain upload/commit outcomes require receipt reconciliation
        kind = 'framing' if 'framing' in str((hold.get('review') or {}).get('render_error')).lower() else 'editorial'
        value = item['approval_revisions'].get(kind)
        if value and value != (old.get('approval_revisions') or {}).get(kind):
            return True
    return False


def _snapshot(db, request_id, state):
    _assert_local_selection(state)
    payload = dict(request_id=request_id, source_video_id=(state.get('parsed') or {}).get('video_id'),
                   source_sha256=(state.get('source_validation') or {}).get('source_sha256'),
                   collections={key: deepcopy(state.get(key) or {}) for key in
                                ('result', 'candidate_reviews', 'topic_result', 'topic_reviews')})
    key = 'selected-batch:' + request_id + ':' + digest(payload)
    _put(db, 'selected_batch', key, payload)
    return key


def prepare_queue(request_id, lane, index):
    """Freeze a ticket before submitting work, never resolve an old index later."""
    with evidence.database() as db:
        _, state = _state(db, request_id)
        result = ticket(request_id, state, lane, index)
        db.execute('BEGIN IMMEDIATE')
        if digest(_state(db, request_id)[1]) != digest(state):
            raise RenderReviewRequired('Selection changed while preparing render ticket')
        _snapshot(db, request_id, state)
        key, reviews = LANES[lane]
        clip = state[key]['segments'][index]
        review = state[reviews][str(index)]
        if _successful(clip, review):
            if not _get(db, result['action'] + ':success', 'render_success'):
                _put(db, 'render_success', result['action'] + ':success',
                     dict(ticket=result, clip=deepcopy(clip), review=deepcopy(review)))
            return None, 'already_rendered'
        if _get(db, result['action'] + ':success', 'render_success'):
            return None, 'successful_receipt_requires_reconciliation'
        if _get(db, result['action'] + ':hold', 'render_hold'):
            return None, 'held_requires_review'
        if (review.get('status') == 'render_failed' and failure_kind(review) != 'FAILED_TRANSIENT_SOURCE_ACCESS'
                and not _reviewed_correction(db, result)):
            _put(db, 'render_hold', result['action'] + ':hold',
                 dict(ticket=result, category=failure_kind(review), review=deepcopy(review)))
            return None, 'held_requires_review'
        return result, 'eligible'


def retry_plan(request_id, chat_id):
    """Only the owning request's approved assets; Sheets is never queried."""
    with evidence.database() as db:
        row, state = _state(db, request_id)
        if str(row['chat_id']) != str(chat_id):
            raise RenderReviewRequired('Retry destination differs from the existing request')
        _assert_local_selection(state)
        positions = [(lane, int(index)) for lane, (_, reviews) in LANES.items()
                     for index, review in (state.get(reviews) or {}).items()
                     if review.get('status') in SELECTED]
    planned, held = [], []
    for lane, index in positions:
        item, reason = prepare_queue(request_id, lane, index)
        if item:
            planned.append(item)
        else:
            held.append(dict(lane=lane, index=index, reason=reason))
    return planned, held


def current_index(state, lane, fallback):
    """Recheck stable asset/revision at every commit; never update a sibling index."""
    expected = _CURRENT.get()
    if expected is None:
        return fallback
    key, _ = LANES[lane]
    for index, clip in enumerate((state.get(key) or {}).get('segments') or []):
        if asset_id(expected['request_id'], lane, clip, index) != expected['asset_id']:
            continue
        candidate = ticket(expected['request_id'], state, lane, index)
        if candidate['candidate_id'] == expected['candidate_id'] and candidate['revision'] == expected['revision']:
            return index
    raise RenderReviewRequired('Selected asset or approval revision changed during rendering')


def current_asset_id(default):
    return (_CURRENT.get() or {}).get('asset_id', default)


def guarded_render(lane):
    def decorate(function):
        @wraps(function)
        def render(request_id, index, chat_id, *, render_ticket=None):
            from audio_master_handoff import DB_PATH
            from durable_jobs import claim, finish, ensure_schema
            item, reason = prepare_queue(request_id, lane, index)
            if item is None:
                return {'status': reason}
            if render_ticket and any(item[k] != render_ticket[k] for k in ('asset_id', 'candidate_id', 'revision')):
                return {'status': 'held_revision_changed'}
            action = item['action']
            with evidence.database() as db:
                ensure_schema(db)
                lease = db.execute('SELECT state FROM durable_job_leases WHERE job_id=? AND action=?', (request_id, action)).fetchone()
                if lease and lease['state'] == 'RUNNING':
                    return {'status': 'running_or_uncertain_requires_verification'}
            if not claim(DB_PATH, request_id, action, lease_seconds=3600):
                return {'status': 'already_claimed'}
            with evidence.database() as db:
                attempt = db.execute('SELECT attempt_count FROM durable_job_leases WHERE job_id=? AND action=?', (request_id, action)).fetchone()[0]
                attempt_id = action + ':' + str(attempt)
                _put(db, 'render_attempt', attempt_id + ':started', dict(ticket=item, attempt=attempt))
            context = _CURRENT.set(item)
            outcome = 'HELD_VERIFICATION'
            try:
                result = function(request_id, index, chat_id)
                with evidence.database() as db:
                    _, state = _state(db, request_id)
                    position = current_index(state, lane, index)
                    key, reviews = LANES[lane]
                    clip, review = state[key]['segments'][position], state[reviews][str(position)]
                    outcome = 'SUCCEEDED' if _successful(clip, review) else failure_kind(review)
                    payload = dict(ticket=item, attempt=attempt, outcome=outcome,
                                   clip=deepcopy(clip), review=deepcopy(review))
                    _put(db, 'render_attempt', attempt_id + ':finished', payload)
                    if outcome == 'SUCCEEDED':
                        _put(db, 'render_success', action + ':success', dict(ticket=item, clip=deepcopy(clip), review=deepcopy(review)))
                    elif outcome != 'FAILED_TRANSIENT_SOURCE_ACCESS':
                        _put(db, 'render_hold', action + ':hold', payload)
                return result
            except Exception as exc:
                with evidence.database() as db:
                    payload = dict(ticket=item, attempt=attempt, outcome='HELD_VERIFICATION', error_type=type(exc).__name__)
                    _put(db, 'render_attempt', attempt_id + ':exception', payload)
                    if not _get(db, action + ':hold', 'render_hold'):
                        _put(db, 'render_hold', action + ':hold', payload)
                raise
            finally:
                _CURRENT.reset(context)
                finish(DB_PATH, request_id, action, '' if outcome == 'SUCCEEDED' else outcome,
                       complete=outcome == 'SUCCEEDED')
        return render
    return decorate


def legacy_selection_receipt(request_id, ordered_evidence):
    """Prepare, but do not apply, an explicitly reviewed legacy recovery manifest.

    Each ordered asset needs its exact request-scoped approval/attempt record and
    a matching immutable PASSED candidate cache. Source-wide Sheets never qualify.
    Ambiguous or incomplete evidence stops reconstruction rather than guessing.
    """
    if not ordered_evidence:
        raise RenderReviewRequired('An explicit ordered selection manifest is required')
    with evidence.database() as db:
        _, state = _state(db, request_id)
        if not (state.get('source_validation') or {}).get('source_sha256'):
            raise RenderReviewRequired('Verified source fingerprint is required')
        rebuilt = deepcopy(state)
        rebuilt.pop('recovered_approvals_from_sheet', None)
        for key, reviews in LANES.values():
            rebuilt[key] = {**(state.get(key) or {}), 'segments': []}
            rebuilt[reviews] = {}
        seen = set()
        for entry in ordered_evidence:
            identity = entry['asset_id']
            if identity in seen:
                raise RenderReviewRequired('Duplicate asset in selection manifest')
            seen.add(identity)
            feedback = _get(db, entry['feedback_id'], 'feedback')
            if not feedback or feedback.get('request_id') != request_id or feedback.get('asset_id') != identity:
                raise RenderReviewRequired('Approval evidence belongs to a different request or asset')
            if feedback.get('source_video_id') != (state.get('parsed') or {}).get('video_id'):
                raise RenderReviewRequired('Approval source differs')
            review = feedback.get('review') or {}
            if review.get('status') not in SELECTED or not review.get('user_id') or not review.get('reviewed_at'):
                raise RenderReviewRequired('Original owner selection receipt is incomplete')
            row = db.execute('SELECT status,result FROM source_analyses WHERE analysis_id=?', (entry['analysis_id'],)).fetchone()
            if not row or row['status'] != 'PASSED':
                raise RenderReviewRequired('Verified candidate cache is required')
            candidates = (json.loads(row['result']) or {}).get('segments') or []
            matches = [c for c in candidates if c.get('candidate_id') == feedback.get('candidate_id')]
            if len(matches) != 1 or {k:matches[0].get(k) for k in ('start','end')} != feedback.get('bounds'):
                raise RenderReviewRequired('Candidate identity or approved bounds differ')
            lane = {'9:16':'short','16:9':'highlight'}.get(feedback.get('format'))
            if lane is None:
                raise RenderReviewRequired('Approval format is invalid')
            key, reviews = LANES[lane]
            clip = deepcopy(matches[0]); index = len(rebuilt[key]['segments'])
            if asset_id(request_id, lane, clip, index) != identity:
                raise RenderReviewRequired('Manifest order changes original asset identities')
            if review.get('status') == 'rendered':
                clip['completion_check'] = deepcopy(feedback.get('render'))
                clip['clip_url' if lane == 'short' else 'segment_url'] = feedback.get('media_url')
                if not _successful(clip, review):
                    raise RenderReviewRequired('Successful-media receipt is incomplete')
            rebuilt[key]['segments'].append(clip)
            rebuilt[reviews][str(index)] = deepcopy(review)
        db.execute('BEGIN IMMEDIATE')
        receipt = _snapshot(db, request_id, rebuilt)
        _put(db, 'selection_manifest', 'manifest:' + receipt,
             dict(request_id=request_id, ordered_evidence=ordered_evidence, receipt_id=receipt))
        return receipt


def restore_from_receipt(request_id, receipt_id, verify_media):
    """Explicit operator operation only. No inference from source-wide history.

    Requires an existing fingerprinted request-scoped selection receipt and an
    independently verified media callback. Never called by a retry or watchdog.
    """
    with evidence.database() as db:
        receipt = _get(db, receipt_id, 'selected_batch')
        _, current = _state(db, request_id)
    if not receipt or receipt.get('request_id') != request_id:
        raise RenderReviewRequired('Verified selection receipt for this request is required')
    if receipt.get('source_video_id') != (current.get('parsed') or {}).get('video_id'):
        raise RenderReviewRequired('Selection source identity differs')
    if receipt.get('source_sha256') != (current.get('source_validation') or {}).get('source_sha256'):
        raise RenderReviewRequired('Selection source fingerprint differs')
    if set(receipt.get('collections') or {}) != {key for pair in LANES.values() for key in pair}:
        raise RenderReviewRequired('Selection receipt must contain exactly the original selection collections')
    restored = {**deepcopy(current), **deepcopy(receipt['collections'])}
    from batch_progress import copy_frozen
    for lane, (key, reviews) in LANES.items():
        for index, clip in enumerate((restored.get(key) or {}).get('segments') or []):
            review = (restored.get(reviews) or {}).get(str(index)) or {}
            if review.get('status') not in SELECTED:
                continue
            item = ticket(request_id, restored, lane, index)
            if copy_frozen(current, item['asset_id']):
                raise RenderReviewRequired('Submitted selection is protected from restoration')
            with evidence.database() as db:
                success = _get(db, item['action'] + ':success', 'render_success')
            if success:
                clip, review = success['clip'], success['review']
            if _successful(clip, review):
                if not verify_media(item, clip, review):
                    raise RenderReviewRequired('Retained successful media needs independent verification')
                restored[key]['segments'][index] = deepcopy(clip)
                restored[reviews][str(index)] = deepcopy(review)
    with evidence.database() as db:
        db.execute('BEGIN IMMEDIATE')
        _, latest = _state(db, request_id)
        if digest(latest) != digest(current):
            raise RenderReviewRequired('Request changed during restoration verification')
        # Restore selection/media only; current copy, creators, schedules and receipts remain authoritative.
        for key in receipt['collections']:
            latest[key] = restored[key]
        latest['selection_restored_from_receipt'] = receipt_id
        _put(db, 'selection_restoration', 'restored:' + request_id + ':' + digest(receipt),
             dict(request_id=request_id, receipt_id=receipt_id, previous_collections={k:current.get(k) for k in receipt['collections']}))
        db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id=?', (json.dumps(latest), request_id))
    return latest
