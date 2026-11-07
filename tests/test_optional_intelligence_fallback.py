"""Complete isolated episodes through real selector transports and persistence."""
import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import hierarchical_intelligence as hi
import intelligence_workflow as workflow
import production_intelligence as pi
from test_hierarchical_intelligence import isolated, generate as legacy_generate, topic


@pytest.fixture
def episode(isolated,monkeypatch):
    import main
    import telegram_intake as tg
    import selection_specs
    texts=[
        'Welcome to our seed saving discussion.', 'Which ripe tomatoes make good seed stock?',
        'Use healthy open-pollinated fruit.', 'How do you remove the gel coating?',
        'Ferment briefly, rinse and dry the seeds.', 'I labelled my seeds dinner.',
        'That certainly keeps your family guessing!', 'Can damp seeds go in a closed jar?',
        'No, moisture can ruin the entire collection.', 'Let the seeds dry thoroughly first.',
        'PlantCon will include a propagation workshop.', 'Bring questions to the workshop.',
        'We are giving away plants this weekend.', 'I am carving a dinosaur from wood.',
        'That is an unexpected gardening project!', 'Now let us discuss seed storage.',
        'Store fully dry seeds in a cool dark location.', 'Why does dry storage matter?',
        'Moisture permits mold and reduces viability.', 'Use a sealed jar after drying.',
        'A desiccant can help control humidity.', 'Label the variety and harvest year.',
        'Test a small sample before sowing the rest.', 'Those steps complete the storage technique.',
        'Returning to tomato seeds, should we ferment every variety?',
        'Use a process appropriate for the crop.', 'Ask your local growers for regional advice.',
        'What is the simplest germination test?', 'Try a small labelled sample on a damp towel.',
        'Count sprouts and compare the result.', 'Thank you for joining this episode.',
        'We will explain the next technique because',
    ]
    segments=[{'start':i*30.,'end':(i+1)*30.,'duration':30.,'text':t} for i,t in enumerate(texts)]
    state={'show_id':'TDOG','parsed':{'video_id':'complete-episode'},'vid_title':'Seed saving fixture'}
    monkeypatch.setattr(main,'OPENAI_API_KEY','isolated-fake-not-a-credential')
    monkeypatch.setenv('OPENAI_API_KEY','isolated-fake-not-a-credential')
    monkeypatch.setattr(selection_specs,'highlight_framework',lambda:'APPROVED HIGHLIGHT CONTRACT: coherent full explanations, minimum 180 seconds.')
    monkeypatch.setattr(tg,'_preflight_candidates',lambda video,clips:clips)  # media gate tested separately; no live file
    def generator(mode):
        calls=[]
        def run(stage,transcript,context,bounds,count):
            calls.append((stage,bounds))
            if stage=='episode':
                v=legacy_generate(stage,transcript,context,bounds,count)
                if mode=='bad_episode':v['summary']=''
                return v
            if stage=='relationships':
                if mode=='bad_conversation_links':return {'relationships':None}
                nodes=context['verified_conversations']
                return {'relationships':[{'conversation_id':nodes[0]['conversation_id'],'related_ids':[nodes[-1]['conversation_id']]}]}
            if stage=='moments':
                return {'moments':[{'title':'Seeds labelled dinner','summary':'A complete humorous exchange.',
                                   'category':'humor','first_segment':5,'last_segment':99 if mode=='bad_moments' else 6}]}
            if stage=='conversations':
                nodes=[topic('conversation-'+str(i),i*8) for i in range(4)]
                for node in nodes:node.pop('related_keys')
                if mode=='bad_conversations':nodes[0]['first_segment']=1
            else:
                nodes=[topic('question-answer',bounds[0]),topic('focused-explanation',bounds[0]+3),topic('returning-subject',bounds[0]+6)]
                nodes[0]['related_keys']=['returning-subject'];nodes[-1]['related_keys']=['question-answer']
                if bounds[0]==16:
                    if mode=='self':nodes[1]['related_keys']=[nodes[1]['key']]
                    if mode=='unknown':nodes[1]['related_keys']=['private unverified target']
                    if mode=='malformed':nodes[1]['related_keys']={'private':'content'}
                    if mode=='bad_topics':nodes[0]['first_segment']=0
                    if mode=='bad_classification':nodes[0]['tags']={'private':'invalid'}
                if mode=='missing_links':
                    for node in nodes:node.pop('related_keys')
            return {'range_contract':pi.TOPIC_RECIPE,'topics':nodes,
                    'provenance':{'prompt_version':hi.stage_version(stage)+':'+stage,'prompt_sha256':'fixture-prompt-sha'}}
        return run,calls
    transport=[]
    def post(url,**kwargs):
        assert url=='https://api.openai.com/v1/chat/completions'
        prompt=kwargs['json']['messages'][-1]['content']; lane='16:9' if '16:9 Highlight Editor' in prompt else '9:16'
        transport.append((lane,prompt))
        if lane=='9:16':
            result={'segments':[{'start':210.,'end':270.,'transcript':' '.join(texts[7:9]),'category':'education',
                                 'score':90,'reason':'A complete moisture question with its useful answer.'},
                                {'start':930.,'end':960.,'transcript':texts[-1],'category':'education','score':99}]}
        else:
            result={'segments':[{'start':480.,'end':720.,'title':'Keeping saved seeds viable','summary':'Dry storage technique.',
                                  'highlight_type':'explanation','reason':'A complete focused technique and rationale.','score':90},
                                 {'start':0.,'end':120.,'title':'Too short','score':95}]}
        class Response:
            status_code=200
            def json(self):return {'choices':[{'message':{'content':json.dumps(result)}}]}
        return Response()
    monkeypatch.setattr(main.requests,'post',post)
    for lane in ('9:16','16:9'):
        pi.record('feedback','owner-'+lane,{'candidate_id':'previous-strong-'+lane,'brand_id':'TDOG','show_id':'TDOG',
                  'format':lane,'decision':'approved','category':'education' if lane=='9:16' else 'explanation',
                  'feedback':'Preserve the whole answer and practical payoff.'})
    with pi.database() as db:db.execute("UPDATE intelligence_records SET synced_digest=digest WHERE kind='feedback'")
    return state,segments,generator,transport,tg


@pytest.mark.parametrize('mode,path',[
    ('full','full-hierarchy'),('self','topics-only'),('unknown','topics-only'),('malformed','topics-only'),
    ('missing_links','full-hierarchy'),('bad_topics','conversations-only'),
    ('bad_conversation_links','topics-only'),('bad_moments','topics-only'),
    ('bad_classification','topics-only'),('bad_conversations','transcript-direct'),('bad_episode','transcript-direct'),
])
def test_fallbacks_execute_real_selectors_quality_checks_learning_and_idempotent_persistence(episode,monkeypatch,mode,path):
    state,segments,generator,transport,tg=episode
    gen,calls=generator(mode);monkeypatch.setattr(hi,'provider',gen)
    future,mapping=workflow.start(state,segments,'isolated-media','APPROVED SHORT RUBRIC: complete context and meaningful payoff; no quota or forced novelty.')
    short=workflow.shorts_for_review(state,future,segments,'isolated-media')
    long=workflow.prepared_highlights(state,segments,'isolated-media')
    assert mapping['analysis_path']==path
    assert len(short['segments'])==len(long['segments'])==1
    assert short['segments'][0]['start']==210 and short['segments'][0]['end']==270
    assert short['validation_rejections']  # incomplete final sentence was rejected
    assert long['segments'][0]['duration']==240  # 120-second suggestion rejected
    assert {lane for lane,_ in transport}=={'9:16','16:9'} and len(transport)==2
    assert all('Preserve the whole answer and practical payoff.' in p for _,p in transport)
    assert all('SHARED FULL-SOURCE TOPIC MAP' in p and path in p for _,p in transport)
    for lane,result in [('9:16',short),('16:9',long)]:
        item=result['segments'][0]; evidence=item['selection_provenance']
        assert evidence['analysis_path']==path and evidence['learning_evidence_ids']==['owner-'+lane]
        assert item['learning_influence']['score_adjustment']>0
        assert evidence['transcript_revision'] and evidence['specification_sha256']
        assert item['candidate_id'] and item['reason'] and item['transcript']
        assert bool(item['topic_ids'])==(path!='transcript-direct')
    if mode=='self':
        failure=next(f for f in mapping['metadata_failures'] if f['stage']=='topic_links')
        assert failure['diagnostics']['failure_kind']=='SELF_REFERENCE'
        assert failure['diagnostics']['topic_key']==failure['diagnostics']['rejected_reference']
        with pi.database() as db:
            core=[json.loads(r[0]) for r in db.execute("select result from source_analyses where status='PASSED' and analysis_id like '%:topics:%'")]
        assert len(core)==4 and all('related_keys' not in t for value in core for t in value['topics'])
        assert any('focused-explanation' in r['related_keys'] for value in core for r in value['unverified_relationships'])
    if path=='transcript-direct':assert mapping['topics']==[] and mapping['discovery_windows'][0]['last_segment']==31
    elif mode=='bad_topics':assert any(f['stage']=='topics' for f in mapping['metadata_failures'])
    else:assert len([s for s,b in calls if s=='topics'])==4
    before=list(calls)
    retry,_=workflow.start(state,segments,'isolated-media','APPROVED SHORT RUBRIC: complete context and meaningful payoff; no quota or forced novelty.')
    assert retry.result()==short and workflow.prepared_highlights(state,segments,'isolated-media')==long
    assert calls==before and len(transport)==2
    with pi.database() as db:
        assert db.execute("select count(*) from intelligence_records where kind='candidate'").fetchone()[0]==2
    assert not state.get('candidate_reviews') and not state.get('topic_reviews')


def test_invalid_transcript_and_shared_authentication_do_not_fallback(episode,monkeypatch):
    import requests
    state,segments,generator,transport,_=episode
    bad=copy.deepcopy(segments);bad[4]['start']=float('nan')
    spy=Mock();monkeypatch.setattr(hi,'provider',spy)
    with pytest.raises(ValueError,match='valid ordered timed transcript'):workflow.start(state,bad,'media','rubric')
    spy.assert_not_called(); assert not transport
    def denied(*args):
        response=requests.Response();response.status_code=401
        raise requests.HTTPError(response=response)
    monkeypatch.setattr(hi,'provider',denied)
    with pytest.raises(requests.HTTPError):workflow.start(state,segments,'media','rubric')
    assert not transport


def test_five_existing_checkpoints_survive_new_relationship_failure_and_no_provider_replay(episode,monkeypatch):
    state,segments,generator,_,_=episode
    timeline=pi.normalize_transcript(segments)
    source={'brand_id':'TDOG','show_id':'TDOG','source_video_id':state['parsed']['video_id'],
            'source_channel_id':None,'title':state['vid_title'],'transcript_artifact':None}
    good,_=generator('full')
    def old_failure(stage,transcript,context,bounds,count):
        if stage=='topics' and bounds[0]==16:
            raise pi.TopicValidationError('Invalid chronological topic reference',failure_kind='SELF_REFERENCE',topic_position=5)
        return good(stage,transcript,context,bounds,count)
    with monkeypatch.context() as m:
        m.setattr(hi,'STAGE_VERSIONS',{**hi.STAGE_VERSIONS,'topics':'chronological-topics-v2'})
        with pytest.raises(pi.TopicValidationError):hi.analyze(timeline,source,old_failure)
    with pi.database() as db:before={r['analysis_id']:dict(r) for r in db.execute('select * from source_analyses')}
    assert sum(r['status']=='PASSED' for r in before.values())==5
    corrected,calls=generator('self')
    result=hi.analyze(timeline,source,corrected,allow_fallback=True)
    assert calls==[('topics',(16,23)),('topics',(24,31)),('moments',(0,31))]
    assert result['analysis_path']=='topics-only'
    assert result['metadata_failures'][0]['diagnostics']['failure_kind']=='SELF_REFERENCE'
    with pi.database() as db:after={r['analysis_id']:dict(r) for r in db.execute('select * from source_analyses')}
    assert all(after[key]==value for key,value in before.items())
    assert hi.analyze(timeline,source,Mock(side_effect=AssertionError('unexpected provider call')),allow_fallback=True)==result


def test_provider_timeout_is_bounded_cached_fallback_but_storage_failure_is_not(episode,monkeypatch):
    import requests
    import sqlite3
    state,segments,generator,transport,_=episode
    good,calls=generator('full')
    attempts=[]
    def timeout(stage,*args):
        if stage=='topics' and args[2][0]==16:
            attempts.append(stage); raise requests.Timeout('private provider message')
        return good(stage,*args)
    monkeypatch.setattr(hi,'provider',timeout)
    a=workflow.prepare(state,segments);b=workflow.prepare(state,segments)
    assert a==b and a['analysis_path']=='conversations-only' and len(attempts)==1
    assert a['metadata_failures'][0]['diagnostics']['category']=='PROVIDER_UNAVAILABLE'
    assert 'private provider message' not in json.dumps(a['metadata_failures'])
    other={**state,'parsed':{'video_id':'different-fixture'}}
    monkeypatch.setattr(hi,'provider',Mock(side_effect=sqlite3.OperationalError('storage failure')))
    with pytest.raises(sqlite3.OperationalError):workflow.prepare(other,segments)
    assert not transport


def test_failed_shorts_exposes_valid_highlights_without_retry_or_owner_selection(episode,monkeypatch):
    import main
    state,segments,generator,transport,_=episode
    gen,_=generator('self');monkeypatch.setattr(hi,'provider',gen)
    normal_post=main.requests.post
    def malformed_short(url,**kw):
        if '16:9 Highlight Editor' in kw['json']['messages'][-1]['content']:
            return normal_post(url,**kw)
        class BadResponse:
            status_code=200
            def json(self):return {'choices':[{'message':{'content':'private malformed Shorts response'}}]}
        return BadResponse()
    denied=Mock(side_effect=malformed_short)
    monkeypatch.setattr(main.requests,'post',denied)
    future,_=workflow.start(state,segments,'media','rubric')
    result=workflow.shorts_for_review(state,future,segments,'media')
    assert result['segments']==[] and result['format_failure']['status']=='held'
    assert state['topic_result']['segments'][0]['duration']>=180
    assert len(transport)==1 and transport[0][0]=='16:9'
    assert denied.call_count==2 and not state.get('candidate_reviews') and not state.get('topic_reviews')
    assert 'private malformed' not in json.dumps(state['format_failures'])


def test_failed_highlights_does_not_invalidate_successful_shorts(episode,monkeypatch):
    state,segments,generator,transport,tg=episode
    gen,_=generator('self');monkeypatch.setattr(hi,'provider',gen)
    fail=Mock(side_effect=ValueError('invalid highlight boundaries'))
    monkeypatch.setattr(tg,'_topic_break_suggestions',fail)
    future,_=workflow.start(state,segments,'media','rubric')
    short=workflow.shorts_for_review(state,future,segments,'media')
    assert short['segments'] and short['segments'][0]['candidate_id']
    with pytest.raises(ValueError):workflow.prepared_highlights(state,segments,'media')
    with pytest.raises(ValueError):workflow.prepared_highlights(state,segments,'media')
    assert fail.call_count==1 and len(transport)==1
    with pi.database() as db:
        assert db.execute("select count(*) from intelligence_records where kind='candidate'").fetchone()[0]==1


def test_degraded_candidates_reach_existing_request_specific_telegram_website_once(episode,monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from urllib.parse import urlsplit
    import caption_review_web as web
    import review_navigation as navigation
    state,segments,generator,_,tg=episode
    gen,_=generator('self');monkeypatch.setattr(hi,'provider',gen)
    future,_=workflow.start(state,segments,'isolated-media','approved rubric')
    state['result']=workflow.shorts_for_review(state,future,segments,'isolated-media')
    state['topic_result']=workflow.prepared_highlights(state,segments,'isolated-media')
    state['candidate_reviews']={};state['topic_reviews']={}
    with tg._telegram_db() as db:
        db.execute('INSERT INTO telegram_requests VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                   ('fixture-request','key','saved-chat','owner','awaiting_review','both','youtube','source',json.dumps(state),'now','now'))
    monkeypatch.setenv('RIPPED_SHORTS_PUBLIC_URL','https://review.example.test')
    monkeypatch.setattr(tg,'_authorized',lambda chat,user:chat=='saved-chat' and user=='owner')
    telegram=Mock(return_value={'ok':True,'result':{'message_id':77}})
    monkeypatch.setattr(tg,'telegram',telegram)
    first=navigation.notify('fixture-request','shorts')
    assert first=={'status':'delivered','message_id':77}
    assert navigation.notify('fixture-request','shorts')['status']=='already_delivered'
    assert telegram.call_count==1
    payload=telegram.call_args.args[1]
    assert payload['chat_id']=='saved-chat'
    button=payload['reply_markup']['inline_keyboard'][0][0]
    assert button['text']=='🌐 Open Clip Review' and '?stage=shorts' in button['url']
    url=urlsplit(button['url']);token=url.path.rsplit('/',1)[-1]
    app=FastAPI();app.include_router(web.router)
    with TestClient(app) as client:
        page=client.get(url.path+'?'+url.query)
        assert page.status_code==200 and 'A complete moisture question' in page.text
        evidence=web._payload_for_session(token)['candidate_evidence']
        assert len(evidence)==2 and all(c['selection_provenance']['analysis_path']=='topics-only' for c in evidence)
        assert all(c['candidate_id'] and c['transcript'] and c['reason'] and c['title'] for c in evidence)
        assert client.post('/api/ripped-shorts/review/'+token+'/finish').status_code==409
    with tg._telegram_db() as db:
        saved=json.loads(db.execute('select state_json from telegram_requests where request_id=?',('fixture-request',)).fetchone()[0])
        receipt=json.loads(db.execute("select receipt from durable_outbox where destination='telegram_review_navigation'").fetchone()[0])
    assert saved==state and receipt['message_id']==77 and receipt['url']==button['url']


def test_fallback_provenance_never_bypasses_acoustic_complete_thought_gate(monkeypatch,tmp_path):
    import clip_completion as speech
    monkeypatch.setattr(speech,'media_duration',lambda _:10)
    monkeypatch.setattr(speech,'review_thought',lambda *a:{'complete_start':True,'complete_end':True,'complete_thought':False})
    renderer=Mock()
    with pytest.raises(speech.CompletionReviewRequired,match='complete thought'):
        speech.render_complete_clip('source',{'start':0,'end':10,'transcript':'But what happened next?',
            'selection_provenance':{'analysis_path':'transcript-direct'}},tmp_path/'render.mp4',renderer,'9:16')
    renderer.assert_not_called()


def test_quality_qualified_intelligence_results_are_not_truncated_to_twenty(episode,monkeypatch):
    import main
    state,segments,generator,_,tg=episode
    gen,_=generator('bad_conversations');monkeypatch.setattr(hi,'provider',gen)
    proposals={'segments':[{'start':s['start'],'end':s['end'],'transcript':s['text'],'score':80,'category':'education',
                            'reason':'A complete source-supported statement.'} for s in segments[:25]]}
    monkeypatch.setattr(main,'call_openai_for_clips',lambda *a:copy.deepcopy(proposals))
    future,_=workflow.start(state,segments,'media','approved rubric; no quota')
    short=future.result(timeout=5)
    assert len(short['segments'])==25
    assert len(tg.validate_complete_candidates(proposals,segments)['segments'])==20  # other legacy callers unchanged
    assert len(tg.validate_complete_candidates(short,segments,maximum=None)['segments'])==25


def test_cached_authentication_failure_cannot_be_misclassified_as_a_rejected_format(monkeypatch):
    from concurrent.futures import Future
    error=ValueError('Saved analysis failed: HTTPError')
    error.analysis_id='source:9:16:revision'
    error.operational_failure={'category':'AUTHENTICATION_FAILED','http_status':401}
    future=Future();future.set_exception(error)
    sibling=Mock();monkeypatch.setattr(workflow,'prepared_highlights',sibling)
    with pytest.raises(ValueError):workflow.shorts_for_review({},future,[],'media')
    sibling.assert_not_called()
