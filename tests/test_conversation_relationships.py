import copy
import json
from unittest.mock import Mock

import pytest
import hierarchical_intelligence as hi
import production_intelligence as pi
from test_hierarchical_intelligence import SOURCE,TIMELINE,generate,isolated


def verified():
    return [{'conversation_id':'c_'+'a'*32,'position':0,'key':'Private seed conversation'},
            {'conversation_id':'c_'+'b'*32,'position':1,'key':'Private closing conversation'}]


def payload(source=None,target=None):
    nodes=verified()
    return {'relationships':[{'conversation_id':source or nodes[0]['conversation_id'],
            'related_ids':[target or nodes[1]['conversation_id']]}],
            'provenance':{'prompt_version':'actual-link-prompt','prompt_sha256':'saved-request-sha'}}


def test_forward_and_reverse_references_use_full_verified_identity_set():
    nodes=verified(); value=payload()
    value['relationships'].append({'conversation_id':nodes[1]['conversation_id'],'related_ids':[nodes[0]['conversation_id']]})
    original=copy.deepcopy(value)
    assert hi.validate_relationships(value,nodes)==original
    assert value==original  # neither renamed, silently dropped nor symmetrized
    assert hi.validate_relationships({'relationships':[]},nodes)=={'relationships':[]}


@pytest.mark.parametrize('kind,mutate',[
    ('UNKNOWN_TARGET',lambda p:p['relationships'][0].update(related_ids=['Private unknown source text'])),
    ('SELF_REFERENCE',lambda p:p['relationships'][0].update(related_ids=[p['relationships'][0]['conversation_id']])),
    ('DUPLICATE_LINK',lambda p:p['relationships'][0]['related_ids'].append(p['relationships'][0]['related_ids'][0])),
    ('REFERENCE_TYPE',lambda p:p['relationships'][0].update(related_ids=[False])),
    ('REFERENCE_TYPE',lambda p:p['relationships'][0].update(related_ids=[None])),
    ('REFERENCE_TYPE',lambda p:p['relationships'][0].update(related_ids=[{'private':'transcript'}])),
    ('COLLECTION_TYPE',lambda p:p['relationships'][0].update(related_ids='Private string')),
    ('UNKNOWN_SOURCE',lambda p:p['relationships'][0].update(conversation_id='missing')),
    ('SOURCE_TYPE',lambda p:p['relationships'][0].update(conversation_id=7)),
    ('DUPLICATE_SOURCE',lambda p:p['relationships'].append(copy.deepcopy(p['relationships'][0]))),
    ('ITEM_TYPE',lambda p:p.update(relationships=['Private provider content'])),
    ('COLLECTION_TYPE',lambda p:p.update(relationships=None)),
    ('ITEM_FIELDS',lambda p:p['relationships'][0].update(unapproved_field='Private content')),
    ('ITEM_FIELDS',lambda p:p['relationships'][0].pop('related_ids')),
    ('SOURCE_TYPE',lambda p:p['relationships'][0].pop('conversation_id')),
    ('RESPONSE_FIELDS',lambda p:p.update(unapproved_field='Private content')),
])
def test_invalid_relationship_shape_has_specific_safe_diagnostics(isolated,kind,mutate):
    value=payload(); mutate(value); original=copy.deepcopy(value)
    with pytest.raises(pi.TopicValidationError) as failure:
        pi.cached_analysis('isolated-links',lambda:hi.validate_relationships(value,verified()))
    assert value==original
    diagnostic=failure.value.diagnostics
    assert diagnostic['failure_kind']==kind
    assert diagnostic['valid_reference_count']==2
    assert diagnostic['prompt_version']=='actual-link-prompt' and diagnostic['prompt_sha256']=='saved-request-sha'
    assert diagnostic['relationship_schema']['known_ids_only']
    assert 'Private' not in json.dumps(diagnostic) and 'transcript' not in json.dumps(diagnostic)
    with pi.database() as db:
        persisted=json.loads(db.execute('select result from source_analyses').fetchone()[0])['validation_failure']
    assert persisted==diagnostic
    with pytest.raises(ValueError) as cached:
        pi.cached_analysis('isolated-links',Mock(side_effect=AssertionError('duplicate call')))
    state={}
    assert pi.hold_analysis_failure(state,cached.value) and pi.analysis_recovery_blocked(state)


def test_schema_only_offers_verified_ids_and_boundaries_cannot_generate_links():
    contract=hi.schema('relationships',1189,(0,1188),{'verified_conversations':verified()})
    item=contract['properties']['relationships']['items']['properties']
    assert item['conversation_id']['enum']==[v['conversation_id'] for v in verified()]
    assert item['related_ids']['items']['enum']==item['conversation_id']['enum']
    boundary=hi.schema('conversations',1189,(0,1188))['properties']['topics']['items']
    assert 'related_keys' not in boundary['properties'] and 'related_keys' not in boundary['required']


def test_unsolicited_relationships_in_boundary_response_are_held_not_dropped(isolated):
    def invalid(stage,*args):
        value=generate(stage,*args)
        if stage=='conversations':value['topics'][1]['related_keys']=['Private unknown label']
        return value
    with pytest.raises(pi.TopicValidationError) as failure:hi.analyze(TIMELINE,SOURCE,invalid)
    evidence=failure.value.diagnostics
    assert evidence['failure_kind']=='UNEXPECTED_RELATIONSHIPS'
    assert evidence['conversation_position']==1 and evidence['stage']=='conversations'
    assert 'Private' not in json.dumps(evidence)


def test_link_failure_preserves_boundaries_episode_and_recovers_only_links(isolated):
    def bad(stage,*args):
        if stage=='relationships':return payload(source=args[1]['verified_conversations'][1]['conversation_id'],target='unknown target')
        return generate(stage,*args)
    calls=Mock(side_effect=bad)
    with pytest.raises(pi.TopicValidationError) as failure:hi.analyze(TIMELINE,SOURCE,calls)
    diagnostic=failure.value.diagnostics
    assert diagnostic['stage']=='relationships' and diagnostic['analysis_version']==hi.stage_version('relationships')
    assert diagnostic['failure_kind']=='UNKNOWN_TARGET' and diagnostic['conversation_position']==1
    assert diagnostic['conversation_key']['sha256'] and diagnostic['rejected_reference']['sha256']
    assert diagnostic['stage_analysis_id']==failure.value.analysis_id
    with pi.database() as db:
        passed={r['analysis_id']:r['result'] for r in db.execute("select * from source_analyses where status='PASSED'")}
    assert len(passed)==2 and any(':episode:' in k for k in passed) and any(':conversations:' in k for k in passed)
    before=calls.call_count
    with pytest.raises(ValueError):hi.analyze(TIMELINE,SOURCE,calls)
    assert calls.call_count==before
    state={}; assert pi.hold_analysis_failure(state,failure.value)
    from test_semantic_boundary_recovery import watchdog,job
    run,save,executor=watchdog([job(state)])
    for _ in range(4):assert run()==0
    save.assert_not_called(); executor.submit.assert_not_called()
    pi.retry_failed(failure.value.analysis_id)  # explicit isolated test only
    recovery=Mock(side_effect=generate)
    result=hi.analyze(TIMELINE,SOURCE,recovery)
    assert [c.args[0] for c in recovery.call_args_list]==['relationships','topics','topics','moments']
    assert result['topics'] and result['episode_moments']
    with pi.database() as db:
        for key,body in passed.items():
            assert db.execute('select result from source_analyses where analysis_id=?',(key,)).fetchone()[0]==body


def test_related_metadata_updates_existing_compiled_topic_ids_without_inventing_targets(isolated,monkeypatch):
    def linked(stage,*args):
        if stage=='relationships':
            context=args[1]; nodes=context['verified_conversations']
            assert args[0]==[]  # supplemental pass receives verified summaries, not duplicated transcript
            return payload(source=nodes[0]['conversation_id'],target=nodes[1]['conversation_id'])
        return generate(stage,*args)
    monkeypatch.setattr(hi,'provider',linked)
    mapping=hi.mapping(SOURCE,TIMELINE)
    roots=[t for t in mapping['topics'] if not t.get('parent_key')]
    assert roots[0]['related_topic_ids']==[roots[1]['topic_id']]
    assert roots[1]['related_topic_ids']==[]
    assert roots[0]['first_segment']==0 and roots[1]['first_segment']==8


def test_exact_legacy_episode_identity_reused_with_new_conversation_recipe(isolated,monkeypatch):
    timeline=pi.normalize_transcript(TIMELINE)
    root=pi.identity(pi.VERSION,SOURCE['brand_id'],SOURCE['source_video_id'],pi.identity(timeline))
    context={'source':SOURCE}; bounds=(0,len(timeline)-1)
    inputs={'source':SOURCE,'bounds':bounds,'context':context,'transcript_revision':pi.identity(timeline)}
    import os
    old_key=root+':hierarchical-source-v1:episode:'+pi.identity(inputs,os.getenv('OPENAI_MODEL','gpt-4o-mini'))
    saved=generate('episode',timeline,context,bounds,len(timeline))
    pi.cached_analysis(old_key,lambda:saved)
    # The historical failed root is retained and is never reclassified as passed.
    with pytest.raises(pi.TopicValidationError):
        pi.cached_analysis(root+':hierarchical-source-v1',lambda:(_ for _ in ()).throw(pi.TopicValidationError('Historical failure')))
    calls=Mock(side_effect=generate); monkeypatch.setattr(hi,'provider',calls)
    result=hi.mapping(SOURCE,TIMELINE)
    assert result['episode']==saved
    assert calls.call_args_list[0].args[0]=='conversations'
    assert all(c.args[0]!='episode' for c in calls.call_args_list)
    assert result['stage_evidence'][0]['analysis_id']==old_key
    with pi.database() as db:
        assert db.execute('select status from source_analyses where analysis_id=?',(root+':hierarchical-source-v1',)).fetchone()[0]=='FAILED'
    assert hi.stage_version('conversations')!=hi.stage_version('relationships')
