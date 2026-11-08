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


def prepare_for_render(request_id, identity, candidate, source, state, aspect, approved_revisions):
    """Existing explicit render action only; persist specification before FFmpeg."""
    source = Path(source)
    selected = _selected_ticket(request_id, state, identity)
    validation = state.get("source_validation") or {}
    source_sha = validation.get("source_sha256")
    source_id = (state.get("parsed") or {}).get("video_id")
    if not source_sha or not source_id or _hash_file(source) != source_sha:
        raise BoundaryReviewRequired("SOURCE_IDENTITY_INVALID")
    if validation.get("source_id", source_id) != source_id:
        raise BoundaryReviewRequired("SOURCE_IDENTITY_INVALID")
    inputs = {k:candidate.get(k) for k in ("start","end","duration","framing_mode","transcript")}
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
        aspect=aspect, start=start, end=end, transcript=candidate.get("transcript"),
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
        if not all(v["measured_clear"] for v in edges.values()):
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
    current = _selected_ticket(binding["request_id"], state, binding["asset_id"])
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
