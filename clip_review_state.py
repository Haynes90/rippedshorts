"""Editorial/framing revisions in the existing job database, never render commands."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
import re

from audio_master_handoff import connect


def website_first(state):
    return state.get('review_interface') == 'website'


def processing_label(status, state, candidates, finished):
    stage=state.get('stage') or status
    if stage in {'error','failed','awaiting_topic_map_review'}:
        return 'Blocked — '+str(state.get('error_type') or 'processing needs review')
    if finished: return 'Finished-video review ready'
    if any(c['frozen'] for c in candidates): return 'Rendering selected clips'
    if candidates: return 'Candidate review ready'
    if (state.get('transcript_evidence') or {}).get('timed_segments') or state.get('topic_source_segments'):
        return 'Transcript ready — discovering candidates'
    if (state.get('source_validation') or {}).get('status')=='verified': return 'Source verified — preparing transcript'
    return 'Preparing source and transcript'


def enable_website_review(request_id, actor):
    """Audited operator transition of an unselected request, never a recovery."""
    if not actor: raise ValueError('Transition actor is required')
    with closing(connect()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM telegram_requests WHERE request_id=?',(request_id,)).fetchone()
        if not row: raise ValueError('Request not found')
        state=json.loads(row['state_json'])
        if website_first(state): return {'status':'already_enabled'}
        if state.get('schedule_master') or state.get('schedule_requested_at') or state.get('copy_review_completed_at'):
            raise ValueError('Submitted history cannot be migrated')
        if any(r.get('status') not in {'pending',''} for key in ('candidate_reviews','topic_reviews') for r in (state.get(key) or {}).values()):
            raise ValueError('Existing selections require separate receipt-based review')
        before=hashlib.sha256(encoded(state).encode()).hexdigest()
        mapping=(state.get('source_intelligence') or {}).get('map_id')
        if row['mode']=='both' and mapping and not state.get('topic_result'):
            cached=db.execute("SELECT analysis_id,result FROM source_analyses WHERE status='PASSED' AND analysis_id LIKE ?",(mapping+':16:9:%',)).fetchall()
            if len(cached)>1: raise ValueError('Multiple highlight revisions require reconciliation')
            if cached:
                result=json.loads(cached[0]['result'])
                if not isinstance(result,dict) or not isinstance(result.get('segments'),list): raise ValueError('Invalid cached highlight result')
                state['topic_result']=result
                state['topic_stage']='awaiting_review' if result['segments'] else 'not_eligible'
        state['review_interface']='website'
        state['website_review_transition']={'actor':actor,'at':datetime.now(timezone.utc).isoformat(),'previous_state_sha256':before,'candidate_discovery_repeated':False}
        db.execute('UPDATE telegram_requests SET state_json=?,updated_at=? WHERE request_id=?',
                   (json.dumps(state),state['website_review_transition']['at'],request_id))
    return {'status':'enabled','candidate_count':len(candidate_items(request_id,state)),'processing_dispatched':False}


def candidate_items(request_id, state):
    """Editorial proposals, separate from finished-media approvals."""
    from render_recovery import LANES, asset_id, digest
    items = []
    for lane, (key, review_key) in LANES.items():
        for index, clip in enumerate((state.get(key) or {}).get('segments') or []):
            review = (state.get(review_key) or {}).get(str(index)) or {}
            revision = digest({'candidate':{k:clip.get(k) for k in ('candidate_id','start','end','transcript','framing_mode')},
                'source_sha256':(state.get('source_validation') or {}).get('source_sha256'),
                'transcript_file_id':(state.get('source_validation') or {}).get('transcript_file_id')})
            items.append(dict(asset_id=asset_id(request_id,lane,clip,index),lane=lane,index=index,
                format='9:16' if lane=='short' else '16:9', revision=revision,
                title=clip.get('title') or clip.get('subject_description') or 'Clip '+str(clip.get('candidate_number') or index+1),
                transcript=str(clip.get('transcript') or ''), reason=str(clip.get('reason') or clip.get('why_it_will_work') or ''),
                start=clip['start'],end=clip['end'],framing=clip.get('framing_mode') or 'auto',
                choice=review.get('selection_choice') or '',version=review.get('selection_version',0),
                feedback=review.get('selection_feedback') or '',
                frozen=bool((state.get('website_selection_confirmed') or {}).get(lane)) or review.get('status') in {'queued','rendering','rendered','render_failed','approved'}))
    return items


def save_candidate(request_id, identity, body):
    """Save one proposal decision; never render or approve finished media here."""
    if body.get('choice') not in {'select','reject','request_adjustment','unused'}:
        raise ValueError('Choose Select, Reject, Request Adjustment or Leave Unused')
    feedback=str(body.get('feedback') or '').strip()[:2000]
    if body['choice']=='request_adjustment' and not feedback:
        raise ValueError('Describe the requested boundary, context or framing adjustment')
    with closing(connect()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM telegram_requests WHERE request_id=?',(request_id,)).fetchone()
        if not row: raise ValueError('Request not found')
        state=json.loads(row['state_json'])
        if state.get('preview_first_version')==1:
            raise ValueError('Use the rendered-preview review for this request; refresh the website')
        if not website_first(state) or state.get('schedule_requested_at') or state.get('schedule_master'):
            raise ValueError('This candidate collection is not editable in this review')
        item=next((c for c in candidate_items(request_id,state) if c['asset_id']==identity),None)
        if not item or item['frozen']: raise ValueError('Candidate is missing or already confirmed for rendering')
        if body.get('revision')!=item['revision'] or type(body.get('version')) is not int or body['version']!=item['version']:
            raise ValueError('Candidate changed; refresh before saving')
        framing=body.get('framing') or item['framing']
        if framing not in {'auto','blurred'} or (item['lane']=='highlight' and framing!='auto'):
            raise ValueError('Unsupported framing option')
        from render_recovery import LANES
        key,review_key=LANES[item['lane']]
        if item['lane']=='short' and body['choice']=='select' and framing!=item['framing']:
            state[key]['segments'][item['index']]['framing_mode']=framing
        current=(state.get(review_key) or {}).get(str(item['index'])) or {}
        status={'select':'selected','reject':'rejected','request_adjustment':'adjustment_requested','unused':'unused'}[body['choice']]
        state.setdefault(review_key,{})[str(item['index'])]={**current,'status':status,
            'selection_choice':body['choice'],'selection_version':item['version']+1,
            'selection_feedback':feedback,'user_id':str(row['user_id']),
            'reviewed_at':datetime.now(timezone.utc).isoformat(),
            'selection_history':list(current.get('selection_history') or [])+([{k:v for k,v in current.items() if k!='selection_history'}] if current else [])}
        db.execute('UPDATE telegram_requests SET state_json=?,updated_at=? WHERE request_id=?',
            (json.dumps(state),datetime.now(timezone.utc).isoformat(),request_id))
    return dict(status='saved',candidate=next(c for c in candidate_items(request_id,state) if c['asset_id']==identity),
                rendering_started=False,scheduling_started=False)


def confirm_candidates(request_id, lane):
    """Commit selected identities once before the existing render dispatcher runs."""
    from render_recovery import LANES
    if lane not in LANES: raise ValueError('Unknown format')
    with closing(connect()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT * FROM telegram_requests WHERE request_id=?',(request_id,)).fetchone()
        if not row: raise ValueError('Request not found')
        state=json.loads(row['state_json'])
        if state.get('preview_first_version')==1:
            raise ValueError('Previews use the existing dispatcher; select the verified rendered videos instead')
        if not website_first(state) or state.get('schedule_requested_at'):
            raise ValueError('Request is not in website candidate review')
        if (state.get('website_selection_confirmed') or {}).get(lane):
            return dict(status='already_confirmed',plan=[],chat_id=str(row['chat_id']))
        selected=[c for c in candidate_items(request_id,state) if c['lane']==lane and c['choice']=='select' and not c['frozen']]
        if not selected: raise ValueError('Select at least one candidate before confirming this format')
        _,review_key=LANES[lane];stamp=datetime.now(timezone.utc).isoformat()
        for item in selected:
            state[review_key][str(item['index'])].update(status='queued',approved_at=stamp)
        state.setdefault('website_selection_confirmed',{})[lane]=stamp
        state['stage']='awaiting_render_completion'
        db.execute('UPDATE telegram_requests SET status=?,state_json=?,updated_at=? WHERE request_id=?',
                   ('awaiting_render_completion',json.dumps(state),stamp,request_id))
    return dict(status='confirmed',plan=[{'lane':lane,'index':c['index']} for c in selected],chat_id=str(row['chat_id']))


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def _schema(db):
    db.execute('CREATE TABLE IF NOT EXISTS clip_review_revisions (revision_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, asset_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, created_at TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS clip_review_decisions (revision_id TEXT PRIMARY KEY, choice TEXT NOT NULL, feedback TEXT NOT NULL, version INTEGER NOT NULL, actor TEXT NOT NULL, decided_at TEXT NOT NULL)')


def _request(db, request_id):
    row = db.execute('SELECT * FROM telegram_requests WHERE request_id=?', (request_id,)).fetchone()
    if not row:
        raise ValueError('Existing request not found')
    return row, json.loads(row['state_json'])


def _asset(state, request_id, asset_id):
    from batch_progress import selected_assets
    for identity, lane, review, clip in selected_assets(request_id, state):
        if identity == asset_id:
            return lane, clip
    if state.get('preview_first_version')==1:
        from render_recovery import LANES, asset_id as candidate_identity
        for lane,(key,_) in LANES.items():
            for index,clip in enumerate((state.get(key) or {}).get('segments') or []):
                if candidate_identity(request_id,lane,clip,index)==asset_id:
                    return lane,clip
    raise ValueError('Asset is not one of the existing selected clips')


def register(request_id, asset_id, kind, original, proposed, media, provenance, *, notify=True):
    """Called by the authenticated processing/operator path, never a public import API."""
    if kind not in {'editorial', 'framing'}:
        raise ValueError('Unknown review kind')
    with closing(connect()) as db, db:
        _schema(db)
        _, state = _request(db, request_id)
        lane, clip = _asset(state, request_id, asset_id)
        from batch_progress import copy_frozen
        if copy_frozen(state, asset_id):
            raise ValueError('Submitted asset review is frozen')
        if list(original) != [float(clip['start']), float(clip['end'])]:
            raise ValueError('Original bounds do not match the protected selection')
        if kind == 'editorial':
            if len(proposed) != 2 or not all(math.isfinite(float(v)) for v in proposed):
                raise ValueError('Invalid proposed bounds')
            start, end = map(float, proposed)
            if start < 0 or end <= start or (lane == 'short' and end-start > 90) or (lane == 'highlight' and end-start < 180):
                raise ValueError('Proposed bounds violate format rules')
            required = {'original', 'proposed'}
        else:
            if lane != 'short':
                raise ValueError('Framing alternatives apply only to 9:16 clips')
            required = {'blurred'}
        if not required <= set(media) or not provenance.get('source_video_id') == (state.get('parsed') or {}).get('video_id'):
            raise ValueError('Missing media or source binding')
        if not provenance.get('verified_media'):
            raise ValueError('Render verification evidence is required')
        duration = float(provenance.get('source_duration') or 0)
        if not math.isfinite(duration) or duration <= 0 or max(float(original[1]), float(proposed[1])) > duration:
            raise ValueError('Verified source duration does not contain the review')
        for name, artifact in media.items():
            if name not in {'original', 'proposed', 'opening', 'ending', 'smart', 'blurred', 'reference'}:
                raise ValueError('Unknown preview variant')
            if not artifact.get('local_path') and not re.fullmatch(r'[A-Za-z0-9_-]{10,}', str(artifact.get('drive_file_id') or '')):
                raise ValueError('A retained local artifact or exact Drive ID is required')
            if artifact.get('local_path'):
                from pathlib import Path
                from audio_master_handoff import DATA_DIR
                path = Path(artifact['local_path']).resolve()
                if not path.is_relative_to(Path(DATA_DIR).resolve()) or not path.is_file() or path.stat().st_size <= 0:
                    raise ValueError('Preview must exist inside retained application storage')
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                if not artifact.get('sha256') or digest != artifact['sha256']:
                    raise ValueError('Preview fingerprint does not match verified artifact')
        body = dict(asset_id=asset_id, kind=kind, format='9:16' if lane == 'short' else '16:9',
                    original=list(original), proposed=proposed, media=media, provenance=provenance)
        revision = hashlib.sha256(encoded([request_id, body]).encode()).hexdigest()
        db.execute('INSERT OR IGNORE INTO clip_review_revisions VALUES (?,?,?,?,?,?)',
                   (revision, request_id, asset_id, kind, encoded(body), datetime.now(timezone.utc).isoformat()))
    if notify:
        __import__('review_navigation').notify_safely(request_id, kind, revision)
    return revision


def reviews(request_id):
    with closing(connect()) as db:
        if not db.execute("SELECT name FROM sqlite_master WHERE name='clip_review_revisions'").fetchone():
            return []
        rows = db.execute('SELECT r.*, d.choice, d.feedback, d.version, d.decided_at FROM clip_review_revisions r LEFT JOIN clip_review_decisions d USING(revision_id) WHERE r.request_id=? ORDER BY r.created_at DESC, r.rowid DESC', (request_id,)).fetchall()
    result, seen = [], set()
    for row in rows:
        key = (row['asset_id'], row['kind'])
        if key in seen:
            continue
        seen.add(key)
        result.append({**json.loads(row['payload']), 'revision_id': row['revision_id'],
                       'decision': row['choice'], 'feedback': row['feedback'] or '', 'version': row['version'] or 0})
    return result


def pending(request_id):
    allowed = {'editorial': {'original', 'proposed'}, 'framing': {'smart', 'blurred'}}
    return [r for r in reviews(request_id) if r['decision'] not in allowed[r['kind']]]


def decide(request_id, revision, choice, feedback, expected_version):
    current = next((r for r in reviews(request_id) if r['revision_id'] == revision), None)
    if not current:
        raise ValueError('Review revision is missing or superseded')
    allowed = {'editorial': {'original', 'proposed', 'request_edit'},
               'framing': {'smart', 'blurred', 'request_adjustment'}}[current['kind']]
    if choice not in allowed or (choice.startswith('request_') and not str(feedback).strip()):
        raise ValueError('Choose a valid action and include adjustment feedback')
    if choice in {'smart', 'blurred', 'original', 'proposed'} and choice not in current['media']:
        raise ValueError('Chosen preview is not available')
    import production_intelligence as intelligence
    with intelligence.database():
        pass  # initialize the existing feedback projection before the transaction
    from batch_progress import copy_frozen
    with closing(connect()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        row, state = _request(db, request_id)
        if copy_frozen(state, current['asset_id']):
            raise ValueError('Submitted asset review is frozen')
        latest = db.execute('SELECT version FROM clip_review_decisions WHERE revision_id=?', (revision,)).fetchone()
        if int(latest['version'] if latest else 0) != int(expected_version):
            raise ValueError('Stale decision version; refresh the review')
        # Recheck active revision under the write lock, without changing job state.
        active = db.execute('SELECT revision_id FROM clip_review_revisions WHERE request_id=? AND asset_id=? AND kind=? ORDER BY created_at DESC,rowid DESC LIMIT 1', (request_id,current['asset_id'],current['kind'])).fetchone()
        if active['revision_id'] != revision:
            raise ValueError('Review revision was superseded')
        version = int(expected_version)+1
        db.execute('INSERT INTO clip_review_decisions VALUES (?,?,?,?,?,?) ON CONFLICT(revision_id) DO UPDATE SET choice=excluded.choice,feedback=excluded.feedback,version=excluded.version,actor=excluded.actor,decided_at=excluded.decided_at',
                   (revision,choice,str(feedback)[:4000],version,str(row['user_id']),datetime.now(timezone.utc).isoformat()))
        lane, clip = _asset(state, request_id, current['asset_id'])
        evidence = dict(candidate_id=clip.get('candidate_id') or current['asset_id'],
            request_id=request_id, brand_id=state.get('show_id'), show_id=state.get('show_id'),
            source_video_id=(state.get('parsed') or {}).get('video_id'), format=current['format'],
            category=clip.get('category') or clip.get('highlight_type'), topic_ids=clip.get('topic_ids') or [],
            decision='edited' if choice == 'proposed' else 'reviewed', review_kind=current['kind'],
            explicit_feedback=bool(str(feedback).strip()), feedback=str(feedback)[:4000],
            before=current['original'], after=current['proposed'] if choice == 'proposed' else current['original'],
            framing_choice=choice if current['kind'] == 'framing' else None,
            approval_revision=revision, approval_version=version, actor=str(row['user_id']))
        body = intelligence.encoded(evidence)
        db.execute('INSERT INTO intelligence_records(record_id,kind,payload,digest,updated_at) VALUES (?,?,?,?,?) ON CONFLICT(record_id) DO UPDATE SET payload=excluded.payload,digest=excluded.digest,updated_at=excluded.updated_at',
                   ('clip-review:'+revision, 'feedback', body, intelligence.identity(evidence), datetime.now(timezone.utc).isoformat()))
    # No selected-boundary, caption, creator, hold, retry or scheduler writes occur.
    return dict(revision_id=revision, choice=choice, version=version, scheduling_authorized=False,
                rendering_started=False, pending_reviews=len(pending(request_id)))


def render_selection(request_id, asset_id):
    """Resolve only for an explicitly requested render; never mutate original selections."""
    chosen = [r for r in reviews(request_id) if r['asset_id'] == asset_id]
    result = {'review_revisions': {}, 'framing_mode': None, 'bounds': None}
    for review in chosen:
        decision = review['decision']
        if review['provenance'].get('optional_reference') is True and decision is None:
            continue  # no fabricated framing approval; primary finished-clip approval is still required
        if decision not in {'original', 'proposed', 'smart', 'blurred'}:
            raise ValueError('Editorial/framing review is pending')
        result['review_revisions'][review['kind']] = review['revision_id'] + ':' + str(review['version'])
        if review['kind'] == 'editorial':
            result['bounds'] = review[decision]
        else:
            result['framing_mode'] = decision
    framing = next((r for r in chosen if r['kind'] == 'framing'), None)
    if framing and framing['provenance'].get('preview_bounds') and result['bounds'] and framing['provenance']['preview_bounds'] != result['bounds']:
        raise ValueError('Framing previews belong to different editorial bounds; revised previews required')
    return result


def delivery_blockers(request_id, state):
    """Old renders cannot satisfy a new editorial/framing approval revision."""
    from batch_progress import selected_assets
    assets = {identity: review for identity, _, review, _ in selected_assets(request_id, state)}
    errors = []
    current = reviews(request_id)
    framing_ids = {r['asset_id'] for r in current if r['kind'] == 'framing'}
    if state.get('preview_first_version')==1:
        framing_ids.update(c['asset_id'] for c in __import__('finished_clip_review').clips(request_id,state)
                           if c['approval_current'] and c['decision'].get('framing_reviewed'))
    for identity, lane, _, clip in selected_assets(request_id, state):
        if lane == 'short' and clip.get('framing_plan') and identity not in framing_ids:
            errors.append(identity + ': framing preview and owner approval required')
    for asset_id in {r['asset_id'] for r in current}:
        try:
            expected = render_selection(request_id, asset_id)['review_revisions']
        except ValueError:
            errors.append(asset_id + ': review pending')
            continue
        if expected and (assets.get(asset_id) or {}).get('rendered_review_revisions') != expected:
            errors.append(asset_id + ': approved revision requires verified rendering')
    from finished_clip_review import blocked_assets
    errors.extend(asset + ': clip approval pending or rendered media changed' for asset in blocked_assets(request_id,state))
    return errors


def prepare_segment(request_id, asset_id, segment):
    """Used only by the pre-existing explicit retry handlers, never decision POSTs."""
    selection = render_selection(request_id, asset_id)
    result = dict(segment)
    if selection['bounds']:
        result['start'], result['end'] = map(float, selection['bounds'])
        result['duration'] = result['end'] - result['start']
        editorial = next(r for r in reviews(request_id) if r['asset_id'] == asset_id and r['kind'] == 'editorial')
        if editorial['decision'] == 'original' and segment.get('original_transcript'):
            result['transcript'] = segment['original_transcript']
        if editorial['decision'] == 'proposed':
            transcript = editorial['provenance'].get('proposed_transcript')
            if not str(transcript or '').strip():
                raise ValueError('Approved proposed cut needs source-bound transcript before rendering')
            result['transcript'] = transcript
            result.pop('words', None)  # old word evidence belongs to original bounds
    result['framing_mode'] = selection['framing_mode']
    return result, selection['review_revisions']


def preserve_selection(original, rendered):
    """Keep the immutable selected interval separately from the export interval."""
    return {**rendered, 'rendered_bounds': [rendered['start'], rendered['end']],
            'start': original['start'], 'end': original['end'],
            'original_transcript': original.get('original_transcript', original.get('transcript'))}


def verify_render_choice(prepared, rendered):
    if prepared.get('framing_mode') == 'smart':
        plan = rendered.get('framing_plan') or {}
        if plan.get('requires_framing_review') or not plan.get('sections') or any(primary is None for _, primary in plan['sections']):
            raise ValueError('Approved smart framing could not be reproduced safely; review required')
    if prepared.get('framing_mode') == 'blurred' and (rendered.get('framing_plan') or {}).get('layout') != 'blurred_full_frame':
        raise ValueError('Approved blurred framing was not rendered')
