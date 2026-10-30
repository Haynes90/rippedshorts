"""Provider timing validation. Segment timing never becomes invented word timing."""
import math
import re


def cached_response(audio_path, model, fetch, cache_dir=None):
    """Keep successful provider JSON private, including deterministic bad timing."""
    from pathlib import Path
    import hashlib
    import json
    import os
    import uuid
    path = Path(audio_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    folder = Path(cache_dir) if cache_dir else path.parent / '.speech-evidence'
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / (hashlib.sha256((digest + model + ':word-segment-v1').encode()).hexdigest()[:24] + '.json')
    if receipt.is_file():
        data = json.loads(receipt.read_text(encoding='utf-8'))
        if data.get('audio_sha256') != digest or data.get('model') != model:
            raise TranscriptTimingError('cached_transcript_identity_mismatch')
        return data['response']
    data = fetch()
    temporary = receipt.with_suffix('.' + uuid.uuid4().hex[:8] + '.tmp')
    temporary.write_text(json.dumps({'audio_sha256': digest, 'model': model, 'response': data}), encoding='utf-8')
    os.replace(temporary, receipt)
    return data


class TranscriptTimingError(ValueError):
    """Deterministic response-shape failure; retrying the same data cannot fix it."""


def _rows(items, field, duration=None, offset=0):
    if not isinstance(items, list) or not items:
        raise TranscriptTimingError('missing_' + field)
    result, empty = [], 0
    previous = -1.0
    for index, raw in enumerate(items):
        try:
            start, end = float(raw['start']), float(raw['end'])
            text = str(raw.get(field) or '').strip()
        except (KeyError, TypeError, ValueError, OverflowError):
            raise TranscriptTimingError(f'{field}[{index}]: malformed_timestamp') from None
        if not all(map(math.isfinite, (start, end))) or start < 0 or end < start:
            raise TranscriptTimingError(f'{field}[{index}]: invalid_interval')
        # Whisper can emit a blank zero-duration token. It contains no speech or
        # lexical evidence; omitting it does not change any spoken word or time.
        if not text and start == end:
            empty += 1
            continue
        if not text or (field == 'text' and end <= start):
            raise TranscriptTimingError(f'{field}[{index}]: empty_evidence')
        if start < previous or (duration is not None and end > duration + .1):
            raise TranscriptTimingError(f'{field}[{index}]: outside_timeline')
        previous = start
        result.append({**raw, field: text, 'start': start + offset, 'end': end + offset})
    if not result:
        raise TranscriptTimingError('missing_' + field)
    return result, empty


def normalize_words(words, duration=None, offset=0):
    return _rows(words, 'word', duration, offset)


def normalize_response(data, duration=None, offset=0):
    offset = float(offset)
    if not math.isfinite(offset) or offset < 0:
        raise TranscriptTimingError('invalid_chunk_offset')
    if duration is None and data.get('duration') is not None:
        duration = float(data['duration'])
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise TranscriptTimingError('invalid_chunk_duration')
    diagnostics = {}
    try:
        words, empty = normalize_words(data.get('words'), duration, offset)
        diagnostics['discarded_blank_zero_duration_tokens'] = empty
    except TranscriptTimingError as exc:
        words = []
        diagnostics['word_timing_rejection'] = str(exc)
    try:
        segments, empty = _rows(data.get('segments'), 'text', duration, offset)
        diagnostics['discarded_blank_zero_duration_segments'] = empty
    except TranscriptTimingError as exc:
        segments = []
        diagnostics['segment_timing_rejection'] = str(exc)
    if not words and not segments:
        raise TranscriptTimingError('No reliable timing: ' + str(diagnostics))
    if words and not segments:
        # Provider segment records can be absent while word timing is complete.
        # Group only measured words; do not estimate times from text length.
        pending = []
        for word in words:
            pending.append(word)
            if re.search(r'[.!?]["\u201d\u2019\)\]]*$', word['word']):
                segments.append({'text': ' '.join(w['word'] for w in pending),
                                 'start': pending[0]['start'], 'end': pending[-1]['end'],
                                 'words': pending, 'timing_basis': 'provider_word_timing'})
                pending = []
        if pending:
            segments.append({'text': ' '.join(w['word'] for w in pending),
                             'start': pending[0]['start'], 'end': pending[-1]['end'],
                             'words': pending, 'timing_basis': 'provider_word_timing', 'needs_review': True})
    basis = 'provider_word_timing' if words else 'provider_segment_timing'
    return {'words': words, 'segments': segments, 'timing_basis': basis,
            'timing_diagnostics': diagnostics,
            'requires_acoustic_boundary_validation': not bool(words)}


def saved_segments(text, video_id, duration):
    """Read the complete timed TXT emitted only after every chunk has returned."""
    if f'YouTube ID: {video_id}' not in text.splitlines():
        raise TranscriptTimingError('saved_transcript_source_mismatch')
    result = []
    pattern = r'^\[(\d+:\d+:\d+(?:\.\d+)?) - (\d+:\d+:\d+(?:\.\d+)?)\] (.*)$'
    def seconds(clock):
        h, m, s = map(float, clock.split(':'))
        return h * 3600 + m * 60 + s
    for line in text.splitlines():
        match = re.match(pattern, line)
        if match:
            result.append({'start': seconds(match[1]), 'end': seconds(match[2]), 'text': match[3]})
        elif line.startswith('['):
            raise TranscriptTimingError('malformed_saved_segment')
    return normalize_response({'segments': result}, duration=duration)
