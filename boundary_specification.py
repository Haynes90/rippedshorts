"""Persist source-bound speech evidence before the existing selected-asset renderer.

No selection, boundary-edit approval, retry, upload or scheduling operations here.
Unresolved evidence produces an immutable assessment/hold, never repaired references.
"""
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re
import threading
from collections import OrderedDict

import clip_completion as speech
import production_intelligence as evidence
from render_recovery import LANES, asset_id, digest, ticket, _get, _put

VERSION = "pre-render-boundaries-v1"


class BoundaryReviewRequired(speech.CompletionReviewRequired):
    validation_held = True

    def __init__(self, reason, **details):
        super().__init__("Pre-render boundary review required: " + reason,
                         dict(stage="pre_render_boundaries", category=reason, version=VERSION, **details))


def _hash_file(path):
    try:
        with Path(path).open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        raise BoundaryReviewRequired("SOURCE_UNAVAILABLE") from None


def construct_preview_candidate(request_id,lane,index,source):
    """Resolve only new, unapproved passages; freeze only the validated result.

    Each distinct draft gets the existing cached semantic/acoustic assessment.
    A claimed/interrupted construction never silently repeats provider work.
    """
    import json
    from render_recovery import _state, preview_ticket
    from clip_review_state import reviews
    current_reviews=reviews(request_id)  # ensure existing review schema before claiming
    key,review_key=LANES[lane]
    with evidence.database() as db:
        db.execute('BEGIN IMMEDIATE')
        _,state=_state(db,request_id)
        candidate=deepcopy(state[key]['segments'][index])
        identity=asset_id(request_id,lane,candidate,index)
        if (state.get('sentence_first_version')!=1 or state.get('preview_first_version')!=1
                or state.get('show_id') not in {'TCB','TDOG'}
                or (state.get(review_key) or {}).get(str(index))
                or (state.get('website_clip_reviews') or {}).get(identity)
                or state.get('schedule_master') or state.get('schedule_requested_at')
                or any(r['asset_id']==identity for r in current_reviews)):
            raise BoundaryReviewRequired('DRAFT_AUTHORITY_REQUIRED')
        operation='candidate-construction:v1:'+identity
        prior=_get(db,operation+':result','candidate_construction_result')
        if prior:
            if prior['status']=='VALIDATED' and prior['revision']==preview_ticket(request_id,state,identity)['revision']:
                return candidate
            raise BoundaryReviewRequired('CONSTRUCTION_HELD_OR_CHANGED',result_id=operation+':result')
        if _get(db,operation+':intent','candidate_construction_intent'):
            raise BoundaryReviewRequired('CONSTRUCTION_RUNNING_OR_UNCERTAIN')
        _put(db,'candidate_construction_intent',operation+':intent',dict(candidate=deepcopy(candidate)))
    aspect='9:16' if lane=='short' else '16:9'
    duration=speech.media_duration(source)
    units=transcript_units(state.get('topic_source_segments'),duration)
    seen=set(); assessment=None

    def resolve(bounds):
        start,end=map(float,bounds)
        if not math.isfinite(start+end) or not 0<=start<end<=duration:
            raise BoundaryReviewRequired('DRAFT_REFERENCE_INVALID')
        included=[u for u in units if u['end']>start and u['start']<end]
        if not included:
            raise BoundaryReviewRequired('PASSAGE_EVIDENCE_MISSING')
        # Explicit draft construction from actual units, never invented words or
        # a silent repair of an approved cut. Segment units stay whole segments.
        first,last=min(u['start'] for u in included),max(u['end'] for u in included)
        start,end=min(start,first),max(end,last)
        previous=max((u['end'] for u in units if u['end']<=first and u not in included),default=0.)
        following=min((u['start'] for u in units if u['start']>=last and u not in included),default=duration)
        for edge,anchor in [('start',start),('end',end)]:
            measured=acoustic_edge(source,anchor,edge,duration)
            choices=[]
            for left,right in measured['quiet_intervals']:
                low=max(left+.02,previous if edge=='start' else last)
                high=min(right-.02,first if edge=='start' else following)
                if low<=high:
                    choices.append(min(max(anchor,low),high))
            if choices:
                cut=min(choices,key=lambda t:abs(t-anchor))
                if edge=='start': start=cut
                else: end=cut
        length=end-start
        if length<=0 or (aspect=='9:16' and length>90) or (aspect=='16:9' and length<180):
            raise BoundaryReviewRequired('CONSTRUCTED_FORMAT_INELIGIBLE')
        return {**candidate,'start':start,'end':end,'duration':length,
            'transcript':' '.join(u['word'] for u in included),
            'construction_provenance':dict(version='sentence-first-v1',
                source_segments=sorted({u['segment_index'] for u in included}),
                timing_precision=sorted({u['timing_basis'] for u in included}),
                original_proposal=[candidate['start'],candidate['end']])}

    bounds=[candidate['start'],candidate['end']]
    try:
        for step in range(3):
            draft=resolve(bounds)
            signature=digest([draft['start'],draft['end'],draft['transcript']])
            if signature in seen:
                raise BoundaryReviewRequired('NO_EVIDENCE_SUPPORTED_PROGRESS')
            seen.add(signature)
            with evidence.database() as db:
                db.execute('BEGIN IMMEDIATE')
                _,latest=_state(db,request_id)
                if (latest[key]['segments'][index]!=candidate or (latest.get(review_key) or {}).get(str(index)) or (latest.get('website_clip_reviews') or {}).get(identity)
                        or latest.get('schedule_master') or (db.execute("SELECT 1 FROM sqlite_master WHERE name='clip_review_revisions'").fetchone() and db.execute('SELECT 1 FROM clip_review_revisions WHERE request_id=? AND asset_id=?',(request_id,identity)).fetchone())):
                    raise BoundaryReviewRequired('DRAFT_CHANGED_DURING_CONSTRUCTION')
                latest[key]['segments'][index]=draft
                _put(db,'candidate_construction_step',operation+':'+str(step),dict(before=candidate,after=draft))
                db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id=?',(json.dumps(latest),request_id))
            candidate=draft; state=latest
            try:
                prepared=prepare_for_render(request_id,identity,candidate,source,state,aspect,{},purpose='preview')
            except BoundaryReviewRequired as exc:
                assessment_id=exc.diagnostics.get('assessment_id')
                with evidence.database() as db:
                    assessment=_get(db,assessment_id,'boundary_assessment') if assessment_id else None
                if not assessment:
                    raise
                bounds=assessment['suggested_semantic_bounds']
                continue
            with evidence.database() as db:
                db.execute('BEGIN IMMEDIATE')
                _,latest=_state(db,request_id)
                if (latest[key]['segments'][index]!=candidate or (latest.get(review_key) or {}).get(str(index)) or (latest.get('website_clip_reviews') or {}).get(identity)
                        or latest.get('schedule_master') or (db.execute("SELECT 1 FROM sqlite_master WHERE name='clip_review_revisions'").fetchone() and db.execute('SELECT 1 FROM clip_review_revisions WHERE request_id=? AND asset_id=?',(request_id,identity)).fetchone())):
                    raise BoundaryReviewRequired('DRAFT_CHANGED_DURING_CONSTRUCTION')
                latest[key]['segments'][index]=prepared
                db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id=?',(json.dumps(latest),request_id))
                _put(db,'candidate_construction_result',operation+':result',dict(status='VALIDATED',
                    revision=preview_ticket(request_id,latest,identity)['revision'],specification_id=prepared['boundary_specification_id']))
            return prepared
        raise BoundaryReviewRequired('CONSTRUCTION_UNRESOLVED',reasons=assessment['reasons'] if assessment else [])
    except Exception as exc:
        with evidence.database() as db:
            _put(db,'candidate_construction_result',operation+':result',dict(status='HELD',category=type(exc).__name__,diagnostics=getattr(exc,'diagnostics',{})))
        raise


def configuration(candidate, aspect):
    prefixes = ("SPEAKER_", "VERTICAL_", "FFMPEG_", "VIDEO_")
    return dict(aspect=aspect, framing_mode=candidate.get("framing_mode"),
                settings={k:v for k,v in os.environ.items() if k.startswith(prefixes)})


def _selected_ticket(request_id, state, identity):
    for lane, (key, _) in LANES.items():
        for index, clip in enumerate((state.get(key) or {}).get("segments") or []):
            if asset_id(request_id, lane, clip, index) == identity:
                return ticket(request_id, state, lane, index)
    raise BoundaryReviewRequired("SELECTED_IDENTITY_MISSING")


def transcript_units(transcript, duration):
    """Keep original segment/word indices and overlapping speaker evidence intact."""
    if not isinstance(transcript, list) or not transcript:
        raise BoundaryReviewRequired("TRANSCRIPT_MISSING")
    units, previous = [], -1.
    from transcript_timing import normalize_words, TranscriptTimingError
    for index, segment in enumerate(transcript):
        try:
            if any(type(segment[k]) not in (int,float) for k in ("start","end")):
                raise ValueError("Invalid timing type")
            start, end = float(segment["start"]), float(segment["end"])
            text = str(segment["text"]).strip()
        except (KeyError, TypeError, ValueError):
            raise BoundaryReviewRequired("TRANSCRIPT_REFERENCE_INVALID", segment_index=index) from None
        if not text or not math.isfinite(start+end) or not 0 <= start < end <= duration or start < previous:
            raise BoundaryReviewRequired("TRANSCRIPT_REFERENCE_INVALID", segment_index=index)
        previous = start
        if segment.get("words"):
            try:
                if any(type(w.get(k)) not in (int,float) for w in segment["words"] for k in ("start","end")):
                    raise TranscriptTimingError("Invalid timing type")
                words, _ = normalize_words(segment["words"], duration)
            except TranscriptTimingError:
                raise BoundaryReviewRequired("WORD_TIMING_INVALID", segment_index=index) from None
            if any(w["start"] < start or w["end"] > end for w in words):
                raise BoundaryReviewRequired("WORD_PARENT_MISMATCH", segment_index=index)
            for word_index, word in enumerate(words):
                units.append({**word, "segment_index":index, "word_index":word_index,
                              "timing_basis":"provider_word_timing", "speaker":word.get("speaker", segment.get("speaker"))})
        else:
            units.append(dict(word=text, start=start, end=end, segment_index=index,
                              word_index=None, speaker=segment.get("speaker"),
                              timing_basis="provider_segment_timing"))
    return units


def acoustic_edge(source, cut, edge, duration):
    """Measure nearby quiet and propose its nearest midpoint, never fixed padding.

    The bounded scan window is evidence context, not a crop/padding instruction.
    Quiet must span the actual chosen cut; a proposal never alters approved bounds.
    """
    origin, ceiling = max(0., cut-1.5), min(duration, cut+1.5)
    result = speech._run([os.getenv("FFMPEG_BINARY", "ffmpeg"), "-hide_banner", "-nostats",
        "-ss", str(origin), "-i", str(source), "-t", str(ceiling-origin), "-vn", "-af",
        "silencedetect=noise=-35dB:d=0.08", "-f", "null", "-"])
    opened, intervals = None, []
    for kind, value in re.findall(r"silence_(start|end): ([0-9.]+)", result.stderr):
        value = origin + float(value)
        if kind == "start":
            opened = value
        elif opened is not None:
            intervals.append([opened, min(value, ceiling)])
            opened = None
    if opened is not None:
        intervals.append([opened, ceiling])
    # 20ms is measurement uncertainty at the cut, not automatic silence padding.
    clean = [r for r in intervals if r[1]-r[0] >= .08]
    spanning = [r for r in clean if r[0]+.02 <= cut <= r[1]-.02]
    options = [r for r in clean if (r[1] <= cut if edge == "start" else r[0] >= cut)]
    proposed = min(options, key=lambda r:abs(sum(r)/2-cut)) if options else None
    return dict(edge=edge, cut=cut, scan=[origin, ceiling], quiet_intervals=clean,
                measured_clear=bool(spanning), resolution_uncertainty_seconds=.02,
                suggested_cut=cut if spanning else sum(proposed)/2 if proposed else None,
                automatic_adjustment=False)


def prepare_for_render(request_id, identity, candidate, source, state, aspect, approved_revisions, *, purpose='selected'):
    """Existing explicit render action only; persist specification before FFmpeg."""
    source = Path(source)
    from render_recovery import preview_ticket
    selected = preview_ticket(request_id, state, identity) if purpose == 'preview' else _selected_ticket(request_id, state, identity)
    validation = state.get("source_validation") or {}
    source_sha = validation.get("source_sha256")
    source_id = (state.get("parsed") or {}).get("video_id")
    if not source_sha or not source_id or _hash_file(source) != source_sha:
        raise BoundaryReviewRequired("SOURCE_IDENTITY_INVALID")
    if validation.get("source_id", source_id) != source_id:
        raise BoundaryReviewRequired("SOURCE_IDENTITY_INVALID")
    inputs = {k:candidate.get(k) for k in ("start","end","duration","framing_mode","transcript")}
    if purpose == 'preview':
        inputs['duration'] = float(candidate['end'])-float(candidate['start'])
    inputs.update(candidate_id=candidate.get("candidate_id") or identity, review_revisions=approved_revisions, source_sha256=source_sha)
    if digest(inputs) != selected["revision"] or selected["approval_revisions"] != approved_revisions:
        raise BoundaryReviewRequired("APPROVED_INPUT_DIFFERS")
    duration = speech.media_duration(source)
    try:
        start, end = float(candidate["start"]), float(candidate["end"])
    except (KeyError, TypeError, ValueError):
        raise BoundaryReviewRequired("CANDIDATE_INTERVAL_INVALID") from None
    if not math.isfinite(start+end) or not 0 <= start < end <= duration:
        raise BoundaryReviewRequired("CANDIDATE_INTERVAL_INVALID")
    if aspect not in {"9:16", "16:9"} or (aspect == "9:16" and end-start > 90) or (aspect == "16:9" and end-start < 180):
        raise BoundaryReviewRequired("FORMAT_DURATION_INVALID")
    transcript = state.get("topic_source_segments") or []
    units = transcript_units(transcript, duration)
    included = [i for i,u in enumerate(units) if u["end"] > start and u["start"] < end]
    if not included:
        raise BoundaryReviewRequired("PASSAGE_EVIDENCE_MISSING")
    first, last = min(included), max(included)
    left, right = max(0, first-4), min(len(units), last+5)
    context_units = units[left:right]
    binding = dict(request_id=request_id, asset_id=identity, candidate_id=selected["candidate_id"],
        approved_input_revision=selected["revision"], approved_revisions=approved_revisions,
        source_video_id=source_id, source_sha256=source_sha, transcript_revision=digest(transcript),
        aspect=aspect, purpose=purpose, start=start, end=end, transcript=candidate.get("transcript"),
        words_digest=digest(candidate.get("words")), configuration=configuration(candidate, aspect))
    key = "boundary-assessment:" + VERSION + ":" + digest(binding)

    def analyze():
        context = dict(subject=candidate.get("title") or candidate.get("subject_description"),
            selection_reason=candidate.get("reason") or candidate.get("why_it_will_work"),
            intended_excerpt=candidate.get("transcript"), included_unit_indices=[first-left,last-left],
            note="Surrounding units are context, not automatically included speech. Whole segments are not word-precise.")
        semantic = speech.review_thought(context_units, start, end, aspect, context=context)
        if not isinstance(semantic, dict):
            raise BoundaryReviewRequired("SEMANTIC_RESPONSE_INVALID")
        a,b = semantic.get("first_word"), semantic.get("last_word")
        if type(a) is not int or type(b) is not int or not 0 <= a <= b < len(context_units):
            raise BoundaryReviewRequired("SEMANTIC_REFERENCE_INVALID", first_type=type(a).__name__, last_type=type(b).__name__, valid_units=len(context_units))
        reasons = []
        if any(semantic.get(k) is not True for k in ("complete_start","complete_end","complete_thought")):
            reasons.append("SEMANTIC_INCOMPLETE")
        if semantic.get("context_sufficient") is not True:
            reasons.append("CONTEXT_NOT_ESTABLISHED")
        for name in ("setup_units", "payoff_units"):
            references = semantic.get(name)
            if (not isinstance(references, list) or not references or
                    any(type(i) is not int or not a <= i <= b for i in references) or len(set(references)) != len(references)):
                raise BoundaryReviewRequired("SEMANTIC_EVIDENCE_REFERENCE_INVALID", field=name)
        if a != first-left or b != last-left:
            reasons.append("CONTEXT_REQUIRES_EXPLICIT_EDIT")
        chosen = context_units[a:b+1]
        if any(u["start"] < start or u["end"] > end for u in chosen):
            reasons.append("TIMED_SPEECH_EXTENDS_OUTSIDE_CUT")
        edges = {edge:acoustic_edge(source, t, edge, duration) for edge,t in [("start",start),("end",end)]}
        # Silence is one form of evidence, not a compulsory edit requirement.
        # Reliable word-timed speech wholly inside the cut can support a natural
        # transition with background sound. Segment timestamps cannot do this.
        word_timed = all(u['timing_basis']=='provider_word_timing' for u in chosen)
        for edge,cut in [('start',start),('end',end)]:
            margin = chosen[0]['start']-cut if edge=='start' else cut-chosen[-1]['end']
            crossing = any(u['start'] < cut < u['end'] for u in units)
            edges[edge]['timed_capture_supported'] = word_timed and margin >= .1 and not crossing
            edges[edge]['timing_allowance_seconds'] = .1 if word_timed else None
            edges[edge]['captured_margin_seconds'] = margin
        if not all(v["measured_clear"] or v['timed_capture_supported'] for v in edges.values()):
            reasons.append("ACOUSTIC_EDGE_UNRESOLVED")
        return dict(version=VERSION, binding=binding, status="HELD" if reasons else "VALIDATED",
            reasons=reasons, semantic=semantic, acoustic=edges,
            included_segments=sorted({u["segment_index"] for u in units[first:last+1]}),
            speech_units=deepcopy(units[first:last+1]), context_references=[dict(segment_index=u["segment_index"],word_index=u["word_index"]) for u in context_units],
            intended_transcript=candidate.get("transcript"), timing_precision=sorted({u["timing_basis"] for u in units[first:last+1]}),
            selection_provenance=deepcopy(candidate.get("selection_provenance") or {}),
            suggested_semantic_bounds=[chosen[0]["start"],chosen[-1]["end"]],
            boundary_changes_applied=False, approval_scope="existing_selected_asset_only")

    assessment = evidence.cached_analysis(key, analyze)
    with evidence.database() as db:
        _put(db, "boundary_assessment", key, assessment)
    if assessment["status"] != "VALIDATED":
        raise BoundaryReviewRequired("EVIDENCE_UNRESOLVED", assessment_id=key, reasons=assessment["reasons"])
    spec = {**assessment, "assessment_id":key}
    spec_id = "boundary-specification:" + digest(spec)
    with evidence.database() as db:
        _put(db, "boundary_specification", spec_id, spec)
    return {**candidate, "boundary_specification_id":spec_id,
            "boundary_request_id":request_id, "boundary_asset_id":identity,
            "boundary_specification_fingerprint":digest(spec), "requires_boundary_specification":True}


def consume(source, candidate, aspect):
    """Refuse stale/tampered/source-changed inputs immediately before rendering."""
    spec_id = candidate.get("boundary_specification_id")
    with evidence.database() as db:
        spec = _get(db, spec_id, "boundary_specification") if spec_id else None
        if not spec or spec["status"] != "VALIDATED" or digest(spec) != candidate.get("boundary_specification_fingerprint"):
            raise BoundaryReviewRequired("VALIDATED_SPECIFICATION_REQUIRED")
        binding = spec["binding"]
        row = db.execute("SELECT state_json FROM telegram_requests WHERE request_id=?", (binding["request_id"],)).fetchone()
    if not row:
        raise BoundaryReviewRequired("REQUEST_IDENTITY_MISSING")
    import json
    state = json.loads(row["state_json"])
    from render_recovery import preview_ticket
    current = (preview_ticket(binding['request_id'], state, binding['asset_id'])
               if binding.get('purpose')=='preview' else _selected_ticket(binding["request_id"], state, binding["asset_id"]))
    if (candidate.get("boundary_request_id") != binding["request_id"] or candidate.get("boundary_asset_id") != binding["asset_id"]
            or current["candidate_id"] != binding["candidate_id"] or current["revision"] != binding["approved_input_revision"]
            or current["approval_revisions"] != binding["approved_revisions"]
            or digest(state.get("topic_source_segments") or []) != binding["transcript_revision"]
            or (state.get("parsed") or {}).get("video_id") != binding["source_video_id"]):
        raise BoundaryReviewRequired("SPECIFICATION_REVISION_CHANGED")
    if (aspect != binding["aspect"] or candidate.get("start") != binding["start"] or candidate.get("end") != binding["end"]
            or candidate.get("transcript") != binding["transcript"] or digest(candidate.get("words")) != binding["words_digest"]
            or configuration(candidate, aspect) != binding["configuration"]):
        raise BoundaryReviewRequired("RENDER_INPUT_DIFFERS_FROM_SPECIFICATION")
    if _hash_file(source) != binding["source_sha256"]:
        raise BoundaryReviewRequired("SOURCE_IDENTITY_CHANGED")
    return spec


def verify_export(source, output, start, end):
    """Independent exported-audio/source comparison, not a second ASR spelling vote.

    Same exact source passage, no lag search that could conceal a shifted cut.
    AAC/sample uncertainty can still hold a render; never claim phoneme precision
    from a segment timestamp or a model's punctuation.
    """
    from array import array
    import subprocess
    def pcm(path, position, duration):
        result = subprocess.run([os.getenv("FFMPEG_BINARY", "ffmpeg"), "-v", "error",
            "-ss", str(position), "-i", str(path), "-t", str(duration), "-vn",
            # Respect the requested media timeline, including encoded start PTS.
            # Raw PCM otherwise drops a leading timestamp gap on some builds.
            # No lag search or time stretching: changed speech still fails correlation.
            "-ac", "1", "-af", "aresample=8000:first_pts=0", "-ar", "8000",
            "-f", "s16le", "pipe:1"], capture_output=True, timeout=60)
        if result.returncode:
            raise BoundaryReviewRequired("EXPORTED_AUDIO_UNREADABLE")
        samples = array("h"); samples.frombytes(result.stdout)
        if __import__("sys").byteorder != "little": samples.byteswap()
        return samples
    measurements = []
    length = min(2., end-start)
    for edge, position in [("start",0.), ("end",end-start-length)]:
        expected, actual = pcm(source, start+position, length), pcm(output, position, length)
        count = min(len(expected), len(actual))
        if count < 400 or abs(len(expected)-len(actual)) > 16:
            raise BoundaryReviewRequired("EXPORTED_AUDIO_INTERVAL_MISMATCH", edge=edge,
                expected_samples=len(expected), actual_samples=len(actual), position=position,
                sample_rate=8000)
        x,y = expected[:count],actual[:count]
        mx,my = sum(x)/count,sum(y)/count
        xx,yy = sum((v-mx)**2 for v in x),sum((v-my)**2 for v in y)
        if min(xx,yy)/count < 4:
            raise BoundaryReviewRequired("EXPORTED_EDGE_SPEECH_UNVERIFIABLE", edge=edge)
        correlation = sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(xx*yy)
        if correlation < .985:
            raise BoundaryReviewRequired("EXPORTED_AUDIO_DIFFERS_FROM_SOURCE", edge=edge, correlation=round(correlation,5))
        measurements.append(dict(edge=edge, correlation=correlation, samples=count, sample_rate=8000))
    extra = pcm(output, end-start, .3)
    if extra and max(map(abs,extra)) > 164:
        raise BoundaryReviewRequired("AUDIO_OUTSIDE_APPROVED_INTERVAL")
    decoded = speech._run([os.getenv("FFMPEG_BINARY", "ffmpeg"), "-hide_banner",
        "-xerror", "-err_detect", "explode", "-i", str(output), "-af", "volumedetect", "-f", "null", "-"])
    volume = re.search(r"max_volume: ([-0-9.]+) dB", decoded.stderr)
    if not volume or float(volume.group(1)) <= -60:
        raise BoundaryReviewRequired("EXPORTED_AUDIO_INAUDIBLE")
    return dict(source_audio_edges=measurements, full_decode=True, max_volume_db=float(volume.group(1)))
