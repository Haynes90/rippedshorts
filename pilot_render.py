"""Render the exact semantic unit already shown in the customer review queue."""
from pathlib import Path

import clip_completion as c


def render_approved(video, segment, output, renderer, aspect):
    proof = segment.get('boundary_proof') or {}
    decision = segment.get('semantic_review') or {}
    fail = c.CompletionReviewRequired
    if proof.get('version') != 2 or any(decision.get(k) is not True for k in ('complete_start', 'complete_end', 'complete_thought', 'distinct')):
        raise fail('Rebuild selections to check complete thoughts before approval')
    info = Path(video).stat()
    if info.st_size != proof.get('source_size') or info.st_mtime_ns != proof.get('source_mtime_ns'):
        raise fail('The source changed after review; rebuild selections')
    duration = c.media_duration(video)
    try:
        words = c.validate_words(proof, duration)
        first, last = words[0]['start'], words[-1]['end']
        if abs(first - float(segment['start'])) > .001 or abs(last - float(segment['end'])) > .001:
            raise fail('Approved timestamps no longer match the reviewed words')
        if c._tokens(words) != c._tokens([{'word': segment['transcript']}]):
            raise fail('Approved transcript no longer matches the reviewed words')
        from pilot_selection import END
        if not END.search(words[-1]['word'].strip()):
            raise fail('Approved sentence is unfinished')
        previous, following = float(proof['previous_end']), float(proof['next_start'])
        if not 0 <= previous <= first < last <= following <= duration + .1:
            raise fail('Invalid neighboring word boundaries')
        start = max(previous, first - .15)
        end = c.pause_end(video, last, min(duration, last + 1.5, following))
        if aspect == '9:16' and end - start > 90:
            raise fail('Completed Short plus speech padding exceeds 90 seconds')
        renderer(video, start, end - start, output)
        c.verify_render(output, words, start, end)
    except Exception:
        Path(output).unlink(missing_ok=True)
        raise
    return {**segment, 'start': start, 'end': end, 'duration': end - start,
        'completion_check': {'status': 'verified', 'version': 2, 'semantic_before_approval': True,
            'word_alignment': True, 'audio_pause': True, 'export_edges': True}}
