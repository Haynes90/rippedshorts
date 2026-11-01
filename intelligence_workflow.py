"""Bind source intelligence to the existing two format selectors and review flow."""
import json
import os

import production_intelligence as intelligence


def prepare(state, segments):
    source = {'brand_id': str(state.get('show_id') or '').upper(), 'show_id': state.get('show_id'),
              'source_video_id': (state.get('parsed') or {}).get('video_id'),
              'source_channel_id': state.get('youtube_channel_id'), 'title': state.get('vid_title'),
              'transcript_artifact': (state.get('source_validation') or {}).get('transcript_file_id')}
    # Import existing scoped review decisions into the additive ledger once per revision;
    # never edit, reset or supersede the original request records.
    with intelligence.database() as db:
        table=db.execute("SELECT name FROM sqlite_master WHERE name='telegram_requests'").fetchone()
        rows=db.execute('SELECT request_id,state_json FROM telegram_requests ORDER BY updated_at DESC LIMIT 100').fetchall() if table else []
    for row in rows:
        prior=json.loads(row['state_json'])
        if prior.get('show_id')==source['brand_id']:
            intelligence.feedback_snapshot(row['request_id'],prior)
    mapping = intelligence.topic_map(source, segments, intelligence.model_topic_map)
    return mapping


def analyze_highlights(mapping, segments, video):
    import telegram_intake as tg
    context = intelligence.learning_context(mapping['source'], '16:9', mapping['topics'])
    from selection_specs import highlight_framework
    specification=highlight_framework()
    prompt = specification+'\nCURRENT OWNER WORKFLOW: Analyze both formats concurrently; only presentation and approval wait for Shorts confirmation. Retain all editorial and timing requirements from the specification.\nSHARED FULL-SOURCE TOPIC MAP:\n'+intelligence.encoded(mapping)+intelligence.learning_prompt(context)
    def analyze():
        suggestions = tg._topic_break_suggestions(segments, intelligence_context=prompt)
        topics = tg._preflight_candidates(video, tg._build_contiguous_topic_segments(segments, suggestions))
        if not topics:
            suggestions = tg._topic_break_suggestions(segments, semantic_fallback=True, intelligence_context=prompt)
            topics = tg._preflight_candidates(video, tg._build_contiguous_topic_segments(segments, suggestions, minimum_seconds=180))
        result = intelligence.rank_with_evidence({'segments': topics, 'selection': 'shared_full_source_map'}, context)
        result['segments'].sort(key=lambda c:c['start'])
        return intelligence.candidate_records(mapping, '16:9', result, context)
    return intelligence.cached_analysis(mapping['map_id']+':16:9:'+intelligence.identity(os.getenv('OPENAI_MODEL'), specification, 'minimum-180-v1'), analyze)


def start(state, segments, video, short_prompt):
    import main
    import telegram_intake as tg
    mapping = prepare(state, segments)
    context = intelligence.learning_context(mapping['source'], '9:16', mapping['topics'])
    prompt = short_prompt+'\nSHARED FULL-SOURCE TOPIC MAP:\n'+intelligence.encoded(mapping)+intelligence.learning_prompt(context)
    def short():
        def analyze():
            # Existing semantic and exact transcript-boundary validation is retained.
            from selection_specs import selector_result
            from source_ingestion import select_non_overlapping
            result = tg.validate_complete_candidates(selector_result(main.call_openai_for_clips(segments, prompt)), segments)
            result['segments']=select_non_overlapping(result['segments'],limit=len(result['segments']),allow_overlap=False)
            result = intelligence.rank_with_evidence(result, context)
            return intelligence.candidate_records(mapping, '9:16', result, context)
        return intelligence.cached_analysis(mapping['map_id']+':9:16:'+intelligence.identity(main.OPENAI_MODEL,prompt), analyze)
    futures = intelligence.parallel_analyses(short, lambda: analyze_highlights(mapping, segments, video))
    state['source_intelligence'] = {'map_id': mapping['map_id'], 'version': intelligence.VERSION}
    # The independent highlight result is in SQLite, never merged over live review state.
    return futures['9:16'], mapping


def prepared_highlights(state, segments, video):
    from production_intelligence import database
    key = state['source_intelligence']['map_id']
    with database() as db:
        source = db.execute('SELECT payload FROM intelligence_records WHERE record_id=? AND kind=?', (key,'source')).fetchone()
        rows = db.execute("SELECT payload FROM intelligence_records WHERE kind='topic'").fetchall()
    if not source:
        raise ValueError('Saved source map is unavailable; retain existing selections')
    mapping = json.loads(source['payload'])
    mapping['topics'] = [json.loads(r['payload']) for r in rows if json.loads(r['payload']).get('map_id') == key]
    return analyze_highlights(mapping, segments, video)


def observe(request_id, state):
    """SQLite first. Sheets errors are retained and retried on subsequent progress."""
    if str(state.get('show_id') or '').upper() not in {'TCB','TDOG'}:
        return
    intelligence.feedback_snapshot(request_id, state)


def sync():
    from google_sheets import sheets_service
    sheet = (os.getenv('RIPPED_SHORTS_LOG_SHEET_ID') or os.getenv('PODCAST_SHEET_ID') or '14VruBxjaaE9DyPSdBidMeuPHew3nHUC5sRNGXtLCsis').strip()
    if not sheet:
        return
    return intelligence.sync_sheets(sheets_service(), sheet)
