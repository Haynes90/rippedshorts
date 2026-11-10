"""Bind source intelligence to the existing two format selectors and review flow."""
import json
import os

import production_intelligence as intelligence


def sentence_first(state):
    return state.get('sentence_first_version')==1 or (os.getenv('RIPPED_PREVIEW_FIRST_ENABLED','false').lower()=='true'
        and state.get('show_id') in {'TCB','TDOG'} and not state.get('candidate_reviews'))


def replacement_candidates(state,segments,video,lane,attempted,pass_number):
    """Bounded independent discovery with existing selectors/rubrics/map/learning.

    Rejections are machine evidence, never misattributed as owner preferences.
    """
    import main, telegram_intake as tg
    from selection_specs import selector_result, highlight_framework
    mapping=saved_mapping(state)
    aspect='9:16' if lane=='short' else '16:9'
    learning=intelligence.learning_context(mapping['source'],aspect,mapping['topics'])
    specification=tg._configured_short_framework_prompt() if lane=='short' else highlight_framework()
    context=(specification+'\nSHARED VERIFIED SOURCE MAP:\n'+intelligence.encoded(mapping)
        +intelligence.learning_prompt(learning)+'\nMACHINE ATTEMPTS (not owner rejections):\n'+intelligence.encoded(attempted)
        +'\nFind other worthwhile complete spoken passages across the full recording. Include necessary questions, setup, responses and payoffs. Returning subjects are allowed; substantially identical attempted intervals are not. Quality over silence and quantity. No quota. Timestamps are provisional until sentence/context/acoustic construction validates them. Never invent speech or timing. Return no suggestions if none meet the approved rubric.')
    key='sentence-discovery-v1:'+intelligence.identity(state['discovery_operation_id'],mapping['map_id'],aspect,pass_number)
    def analyze():
        if lane=='short':
            result=selector_result(main.call_openai_for_clips(segments,context))
        else:
            suggestions=tg._topic_break_suggestions(segments,intelligence_context=context)
            result={'segments':tg._build_contiguous_topic_segments(segments,suggestions,minimum_seconds=180)}
        result=intelligence.rank_with_evidence(result,learning)
        for clip in result.get('segments') or []:
            clip['candidate_id']='sentence-'+intelligence.identity(key,clip.get('candidate_id'),clip['start'],clip['end'])
            clip['selection_provenance']={**(clip.get('selection_provenance') or {}),
                'discovery_recipe':'sentence-first-v1','discovery_id':key,
                'map_id':mapping['map_id'],'rubric_sha256':__import__('hashlib').sha256(specification.encode()).hexdigest(),
                'attempted_intervals':len(attempted),'learning_examples':len(learning.get('examples') or [])}
        return intelligence.candidate_records(mapping,aspect,result,learning)
    return intelligence.cached_analysis(key,analyze)


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
    from hierarchical_intelligence import mapping as hierarchical_mapping
    mapping = hierarchical_mapping(source, segments)
    return mapping


def analyze_highlights(mapping, segments, video, *, drafts=False):
    import telegram_intake as tg
    context = intelligence.learning_context(mapping['source'], '16:9', mapping['topics'])
    from selection_specs import highlight_framework
    specification=highlight_framework()
    brand_context = ('TDOG: practical gardening explanations, host/guest discussions, stories and memorable exchanges.' if mapping['source'].get('show_id')=='TDOG' else 'TCB: insightful, useful, funny or memorable source-show discussions with attribution context.')
    prompt = specification+'\nCURRENT OWNER WORKFLOW: Analyze both formats concurrently; only presentation and approval wait for Shorts confirmation. Retain all editorial and timing requirements from the specification.\nSHARED FULL-SOURCE TOPIC MAP:\n'+intelligence.encoded(mapping)+intelligence.learning_prompt(context)
    prompt += '\nSHOW CONTEXT: '+brand_context+'\nDiscover valuable tangents and unusual moments throughout the source, including outside major topics. Never force a quota. Select focused topic arcs (for example a substantive seed-saving technique), not entire chapters by default. Preserve complete conversational context and the established minimum 180 seconds; no arbitrary maximum.'
    def analyze():
        suggestions = tg._topic_break_suggestions(segments, intelligence_context=prompt)
        proposals = tg._build_contiguous_topic_segments(segments, suggestions)
        topics = proposals if drafts else tg._preflight_candidates(video, proposals)
        if not topics:
            suggestions = tg._topic_break_suggestions(segments, semantic_fallback=True, intelligence_context=prompt)
            proposals = tg._build_contiguous_topic_segments(segments, suggestions, minimum_seconds=180)
            topics = proposals if drafts else tg._preflight_candidates(video, proposals)
        result = intelligence.rank_with_evidence({'segments': topics, 'selection': 'shared_full_source_map'}, context)
        from selection_specs import framework_provenance
        result['selection_provenance']={'specification_sha256':__import__('hashlib').sha256(specification.encode()).hexdigest(), 'specification':framework_provenance()}
        result['segments'].sort(key=lambda c:c['start'])
        return intelligence.candidate_records(mapping, '16:9', result, context)
    # Freeze the completed lane for this unchanged source/specification. New
    # feedback applies to future sources, not a silent rerun of this batch.
    return intelligence.cached_analysis(mapping['map_id']+':16:9:'+intelligence.identity(os.getenv('OPENAI_MODEL'), specification, brand_context, mapping.get('analysis_version'), 'sentence-first-v1' if drafts else 'minimum-180-v2'), analyze)


def start(state, segments, video, short_prompt):
    import main
    import telegram_intake as tg
    mapping = prepare(state, segments)
    context = intelligence.learning_context(mapping['source'], '9:16', mapping['topics'])
    prompt = short_prompt+'\nSHARED FULL-SOURCE TOPIC MAP:\n'+intelligence.encoded(mapping)+intelligence.learning_prompt(context)
    prompt += '\nSENTENCE-LEVEL CLIP FARMING: Examine the full transcript for strong individual explanations, funny exchanges, useful statements and complete question-answer moments. Expand verified boundaries for complete thoughts, setup, context and payoff. Sentence punctuation alone is not acoustic evidence. Do not constrain clips to chapter/topic boundaries or force quotas; returning topics and cross-topic exchanges are eligible. Preserve the approved rubric and speech-boundary validation.'
    def short():
        def analyze():
            # Existing semantic and exact transcript-boundary validation is retained.
            from selection_specs import selector_result
            from source_ingestion import select_non_overlapping
            try:
                response=main.call_openai_for_clips(segments, prompt)
            except RuntimeError as exc:
                if str(exc).startswith('OpenAI response was not valid JSON:'):
                    raise ValueError('Shorts response failed JSON validation') from None
                raise  # authentication/provider failures must not be disguised as editorial rejection
            result = (selector_result(response) if sentence_first(state) else
                      tg.validate_complete_candidates(selector_result(response), segments, maximum=None))
            result['selection_provenance'] = intelligence.prompt_evidence(main.OPENAI_MODEL, main.openai_clip_prompt(segments,prompt), '9:16')
            result['selection_provenance']['specification_sha256'] = __import__('hashlib').sha256(short_prompt.encode()).hexdigest()
            live = getattr(tg,'_SHORT_FRAMEWORK_LIVE_CACHE','')
            result['selection_provenance']['specification'] = {'source':'google_doc' if live and live in short_prompt else 'packaged_or_supplied_approved_rules',
                'document_id':getattr(tg,'_SHORT_FRAMEWORK_LIVE_REFERENCE',None) if live and live in short_prompt else None,
                'content_sha256':__import__('hashlib').sha256((live if live and live in short_prompt else short_prompt).encode()).hexdigest()}
            result['segments']=select_non_overlapping(result['segments'],limit=len(result['segments']),allow_overlap=False)
            result = intelligence.rank_with_evidence(result, context)
            return intelligence.candidate_records(mapping, '9:16', result, context)
        revision=(intelligence.identity(main.OPENAI_MODEL,prompt,'sentence-first-v1') if sentence_first(state)
                  else intelligence.identity(main.OPENAI_MODEL,prompt))
        return intelligence.cached_analysis(mapping['map_id']+':9:16:'+revision, analyze)
    futures = intelligence.parallel_analyses(short, lambda: analyze_highlights(mapping, segments, video, drafts=sentence_first(state)))
    state['source_intelligence'] = {'map_id': mapping['map_id'], 'analysis_id': mapping['analysis_id'],
                                  'version': intelligence.VERSION}
    # The independent highlight result is in SQLite, never merged over live review state.
    return futures['9:16'], mapping


def shorts_for_review(state, future, segments, video):
    """Expose a valid sibling without retrying or approving a rejected format."""
    try:
        return future.result()
    except ValueError as exc:
        if (getattr(exc,'operational_failure',None) or {}).get('category')=='AUTHENTICATION_FAILED':
            raise
        if ':9:16:' not in str(getattr(exc,'analysis_id','')):
            raise
        highlights=prepared_highlights(state,segments,video)
        if not highlights.get('segments'):
            raise exc
        state['topic_result']=highlights
        state['format_failures']={**state.get('format_failures',{}),'9:16':{
            'analysis_id':exc.analysis_id,'category':'FORMAT_VALIDATION_FAILED',
            'status':'held','retry_requires_owner':True}}
        return {'segments':[], 'format_failure':state['format_failures']['9:16']}


def prepared_highlights(state, segments, video):
    mapping=saved_mapping(state)
    return analyze_highlights(mapping, segments, video, drafts=sentence_first(state))


def saved_mapping(state):
    from production_intelligence import database
    key = state['source_intelligence']['map_id']
    with database() as db:
        source = db.execute('SELECT payload FROM intelligence_records WHERE record_id=? AND kind=?', (key,'source')).fetchone()
        rows = db.execute("SELECT payload FROM intelligence_records WHERE kind='topic'").fetchall()
    if not source:
        raise ValueError('Saved source map is unavailable; retain existing selections')
    mapping = json.loads(source['payload'])
    mapping['topics'] = [json.loads(r['payload']) for r in rows if json.loads(r['payload']).get('map_id') == key]
    return mapping


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

