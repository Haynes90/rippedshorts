import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import pytest
import production_intelligence as pi


@pytest.fixture
def db(tmp_path, monkeypatch):
    import audio_master_handoff
    monkeypatch.setattr(audio_master_handoff,'DB_PATH',tmp_path/'existing.db')
    return tmp_path/'existing.db'


@pytest.fixture
def timeline():
    return [{'start':0,'end':190,'text':'A complete explanation.'},
            {'start':190,'end':400,'text':'A question and full response.'}]


def mapped(timeline):
    return {'topics':[{'key':'a','title':'Seed saving','first_segment':0,'last_segment':0,'tags':['seeds'],'classification':'explanation'},
                      {'key':'b','title':'Questions','first_segment':1,'last_segment':1,'related_keys':['a']}]}


SOURCE={'brand_id':'TDOG','show_id':'TDOG','source_video_id':'video'}


def test_full_map_is_durable_stable_and_reused(db,timeline):
    analyze=Mock(return_value=mapped(timeline))
    first=pi.topic_map(SOURCE,timeline,analyze)
    again=pi.topic_map(SOURCE,timeline,analyze)
    assert first==again
    analyze.assert_called_once()
    assert first['topics'][1]['related_topic_ids']==[first['topics'][0]['topic_id']]
    with pi.database() as c:
        assert c.execute('SELECT count(*) FROM intelligence_records').fetchone()[0]==3


@pytest.mark.parametrize('bad',['omitted','invalid','duplicate','parent'])
def test_unverified_or_incomplete_map_is_rejected(db,timeline,bad):
    response=mapped(timeline)
    if bad=='omitted':response['topics'].pop()
    if bad=='invalid':response['topics'][0]['first_segment']=-1
    if bad=='duplicate':response['topics'][1]['key']='a'
    if bad=='parent':response['topics'][0]['parent_key']='nonexistent'
    with pytest.raises(ValueError):pi.topic_map(SOURCE,timeline,lambda _:response)


def test_changed_source_creates_revision_without_replacing_original(db,timeline):
    first=pi.topic_map(SOURCE,timeline,mapped)
    changed=[{**s,'text':s['text']+' Corrected.'} for s in timeline]
    second=pi.topic_map(SOURCE,changed,mapped)
    assert first['map_id']!=second['map_id']
    with pi.database() as c:
        assert c.execute("SELECT count(*) FROM intelligence_records WHERE kind='source'").fetchone()[0]==2


def test_candidate_links_are_stable_without_full_transcript_in_records(db,timeline):
    mapping=pi.topic_map(SOURCE,timeline,mapped)
    result={'segments':[{'start':30,'end':60,'score':80,'transcript':'Some speech.'}]}
    pi.candidate_records(mapping,'9:16',result,{'examples':[]})
    candidate=result['segments'][0]
    assert candidate['topic_ids']==[mapping['topics'][0]['topic_id']]
    with pi.database() as c:
        body=json.loads(c.execute("SELECT payload FROM intelligence_records WHERE kind='candidate'").fetchone()[0])
        topic=json.loads(c.execute('SELECT payload FROM intelligence_records WHERE record_id=?',(mapping['topics'][0]['topic_id'],)).fetchone()[0])
    assert 'transcript' not in body and body['transcript_ref']['revision']
    assert topic['candidate_ids']==[candidate['candidate_id']]
    pi.topic_map(SOURCE,timeline,Mock(side_effect=AssertionError('unchanged map must be reused')))
    with pi.database() as c:
        reloaded=json.loads(c.execute('SELECT payload FROM intelligence_records WHERE record_id=?',(mapping['topics'][0]['topic_id'],)).fetchone()[0])
    assert reloaded['candidate_ids']==[candidate['candidate_id']]


def test_parallel_lanes_really_overlap_and_failure_preserves_sibling(db):
    short_running,highlight_running=Event(),Event()
    def short():
        short_running.set();assert highlight_running.wait(2)
        return pi.cached_analysis('short',lambda:{'segments':[{'id':'success'}]})
    def long():
        highlight_running.set();assert short_running.wait(2)
        raise ValueError('bad long response')
    futures=pi.parallel_analyses(short,long)
    assert futures['9:16'].result(timeout=3)['segments']
    with pytest.raises(ValueError):futures['16:9'].result(timeout=3)
    assert pi.cached_analysis('short',Mock(side_effect=AssertionError('duplicate analysis')))['segments']


def test_concurrent_same_analysis_runs_once(db):
    calls=[]
    def analyze():
        calls.append(1);return {'done':True}
    with ThreadPoolExecutor(max_workers=2) as e:
        results=list(e.map(lambda _:pi.cached_analysis('same',analyze),range(2)))
    assert results==[{'done':True}]*2 and len(calls)==1


def state(decision,brand='TDOG'):
    return {'show_id':brand,'parsed':{'video_id':'old'},'result':{'segments':[{'start':0,'end':20,'category':'humor','score':85}]},
            'candidate_reviews':{'0':{'status':decision,'user_id':'owner'}}}


def test_approval_and_rejection_actually_change_future_ranking_without_removing_new_candidates(db):
    pi.feedback_snapshot('old',state('approved'))
    context=pi.learning_context(SOURCE,'9:16')
    candidates={'segments':[{'category':'education','score':84},{'category':'humor','score':84}]}
    ranked=pi.rank_with_evidence(candidates,context)
    assert ranked['segments'][0]['category']=='humor'
    assert ranked['segments'][0]['learning_influence']['evidence_ids']
    pi.feedback_snapshot('old',state('rejected'))
    rejected=pi.learning_context(SOURCE,'9:16')
    assert rejected['examples'][0]['sign']==-1
    assert len(pi.rank_with_evidence({'segments':[{'category':'humor','score':84},{'category':'new','score':83}]},rejected)['segments'])==2


def test_holds_and_provider_failures_are_not_negative_editorial_evidence(db):
    pi.feedback_snapshot('one',state('held'))
    pi.feedback_snapshot('two',state('render_failed'))
    assert not pi.learning_context(SOURCE,'9:16')['examples']


def test_brand_and_format_learning_are_scoped(db):
    pi.feedback_snapshot('old',state('approved','TCB'))
    assert not pi.learning_context(SOURCE,'9:16')['examples']
    assert not pi.learning_context({**SOURCE,'brand_id':'TCB'},'16:9')['examples']


def test_approval_remains_positive_when_later_render_failure_is_not_editorial_feedback(db):
    pi.feedback_snapshot('same',state('approved'))
    pi.feedback_snapshot('same',state('render_failed'))
    assert pi.learning_context(SOURCE,'9:16')['examples'][0]['sign']==1


def test_manual_boundary_edit_is_durable_and_used_by_future_discovery(db):
    original=state('approved')
    edit={'lane':'short','action':'end_later','before':[{'start':10,'end':20}], 'after':[{'start':10,'end':25}]}
    pi.record_edits('request',original,edit,'Include the full answer.','owner')
    context=pi.learning_context(SOURCE,'9:16')
    example=context['examples'][0]
    assert example['feedback']=='Include the full answer.'
    assert example['boundary_changes']['before'][0]['end']==20
    assert example['boundary_changes']['after'][0]['end']==25
    assert 'Include the full answer.' in pi.learning_prompt(context)


def test_feedback_links_rendered_asset_and_acknowledged_delivery_ids(db):
    source=state('rendered');source['schedule_master']={'shorts_response':{
        'delivery_manifest':[{'asset_group_id':'request:short:1','delivery_id':'ig'}],
        'delivery_validation':{'deliveries':[{'asset_group_id':'request:short:1','delivery_id':'ig','scheduled_at':'2030-01-01T12:00:00Z','destination_id':'verified-tcb','schedule_record_valid':True}]}}}
    pi.feedback_snapshot('request',source)
    with pi.database() as c:
        data=json.loads(c.execute("SELECT payload FROM intelligence_records WHERE kind='feedback'").fetchone()[0])
    assert data['asset_id']=='request:short:1' and data['delivery_ids']==['ig']
    assert data['scheduled_deliveries'][0]['scheduled_at']=='2030-01-01T12:00:00Z'


class FakeSheets:
    def __init__(self):self.tabs={};self.writes=[];self.fail_after_write=False;self.batch_writes=[];self.batch_reads=[]
    def spreadsheets(self):return self
    def values(self):return self
    def result(self,value):return Mock(execute=Mock(return_value=value))
    def get(self,**kwargs):
        if 'range' not in kwargs:
            return self.result({'sheets':[{'properties':{'title':t}} for t in self.tabs]})
        tab,area=kwargs['range'].split('!');tab=tab.strip("'")
        rows=self.tabs[tab]
        import re
        nums=re.findall(r'\d+',area)
        return self.result({'values':rows[int(nums[0])-1:int(nums[-1])]})
    def batchUpdate(self,**kwargs):
        if 'data' in kwargs['body']:
            self.batch_writes.append(kwargs)
            for item in kwargs['body']['data']:
                self.update(spreadsheetId=kwargs['spreadsheetId'],range=item['range'],
                    valueInputOption=kwargs['body']['valueInputOption'],body={'values':item['values']}).execute()
            return self.result({})
        for r in kwargs['body']['requests']:self.tabs[r['addSheet']['properties']['title']]=[]
        return self.result({})
    def batchGet(self,**kwargs):
        self.batch_reads.append(kwargs)
        return self.result({'valueRanges':[{'range':area,**self.get(
            spreadsheetId=kwargs['spreadsheetId'],range=area,
            valueRenderOption=kwargs.get('valueRenderOption')).execute()} for area in kwargs['ranges']]})
    def update(self,**kwargs):
        import re
        tab,area=kwargs['range'].split('!');tab=tab.strip("'");number=int(re.search(r'\d+',area)[0])
        rows=self.tabs[tab]
        while len(rows)<number:rows.append([])
        rows[number-1]=kwargs['body']['values'][0];self.writes.append(kwargs)
        if self.fail_after_write and number>1:
            self.fail_after_write=False
            return Mock(execute=Mock(side_effect=TimeoutError('uncertain Sheets write')))
        return self.result({})


def test_sheet_write_readback_and_uncertain_retry_does_not_duplicate(db,timeline):
    pi.topic_map(SOURCE,timeline,mapped)
    service=FakeSheets();service.tabs['Existing approvals']=[['protected'],['leave unchanged']]
    service.fail_after_write=True
    with pytest.raises(TimeoutError):pi.sync_sheets(service,'isolated')
    assert pi.sync_sheets(service,'isolated')==3
    assert pi.sync_sheets(service,'isolated')==0
    assert service.tabs['Existing approvals']==[['protected'],['leave unchanged']]
    for tab,rows in service.tabs.items():
        if tab.startswith('Ripped '):assert len({r[0] for r in rows[1:]})==len(rows)-1
    assert all(w['valueInputOption']=='RAW' for w in service.writes)


def test_sheet_readback_uses_exact_stored_timestamps_not_display_rounding(db):
    class FormattedSheets(FakeSheets):
        def get(self, **kwargs):
            result = super().get(**kwargs)
            rows = result.execute().get('values', [])
            if kwargs.get('range', '').endswith(':Y2'):
                if kwargs.get('valueRenderOption') != 'UNFORMATTED_VALUE':
                    rows = [[f'{v:.6f}' if isinstance(v, float) else v for v in row] for row in rows]
                return self.result({'values': rows})
            return result

    # Exact November 25 production readback discrepancy, candidate row 9.
    pi.record('candidate', 'precision-case', {'start':2553.3999938964844,
                                          'end':2580.0599975585938})
    service = FormattedSheets()
    assert pi.sync_sheets(service, 'isolated') == 1
    assert pi.sync_sheets(service, 'isolated') == 0
    row = service.tabs[pi.SHEET_TABS['candidate']][1]
    assert row[14:16] == [2553.3999938964844, 2580.0599975585938]


def test_sheet_readback_still_rejects_changed_stored_value(db):
    class CorruptReadback(FakeSheets):
        def get(self, **kwargs):
            result = super().get(**kwargs)
            if kwargs.get('range', '').endswith(':Y2'):
                rows = [list(row) for row in result.execute()['values']]
                rows[0][14] = 2553.4
                return self.result({'values': rows})
            return result

    pi.record('candidate', 'precision-case', {'start':2553.3999938964844})
    with pytest.raises(ValueError, match='readback mismatch'):
        pi.sync_sheets(CorruptReadback(), 'isolated')
    with pi.database() as connection:
        assert connection.execute('SELECT synced_digest FROM intelligence_records').fetchone()[0] is None


def test_retained_backlog_uses_one_batch_write_and_readback_per_kind(db):
    for i in range(150):
        pi.record('feedback',f'old-feedback-{i:03}',{'decision':'reject','brand_id':'TDOG'})
    pi.record('boundary_assessment','internal-preserved',{'complete':False})
    service=FakeSheets()
    assert pi.sync_sheets(service,'isolated')==150
    assert len(service.batch_writes)==len(service.batch_reads)==1
    assert len(service.batch_writes[0]['body']['data'])==150
    assert service.batch_reads[0]['valueRenderOption']=='UNFORMATTED_VALUE'
    assert pi.sync_sheets(service,'isolated')==0
    rows=service.tabs[pi.SHEET_TABS['feedback']][1:]
    assert len({row[0] for row in rows})==150
    with pi.database() as connection:
        record=connection.execute("SELECT payload,synced_digest FROM intelligence_records WHERE record_id='internal-preserved'").fetchone()
        assert json.loads(record[0])=={'complete':False} and record[1] is None


@pytest.mark.parametrize('supported',[False,True])
def test_sheet_sync_preserves_internal_evidence_without_blocking_supported_records(db,supported):
    internal={'boundary_assessment':{'complete':False,'diagnostic':'held'},
              'boundary_specification':{'revision':'immutable','start':10,'end':30},
              'render_attempt':{'status':'verified'},
              'future_internal_kind':{'private_evidence':'retain unchanged'}}
    for kind,payload in internal.items():
        pi.record(kind,kind,payload)
    with pi.database() as c:
        before=[tuple(r) for r in c.execute('SELECT * FROM intelligence_records ORDER BY record_id')]
    service=FakeSheets()
    if supported:
        for kind in pi.SHEET_TABS:
            pi.record(kind,'supported-'+kind,{'title':kind,'request_id':'isolated'})
    assert pi.sync_sheets(service,'isolated')==len(pi.SHEET_TABS)*int(supported)
    assert pi.sync_sheets(service,'isolated')==0
    assert set(service.tabs)==(set(pi.SHEET_TABS.values()) if supported else set())
    with pi.database() as c:
        after=[tuple(r) for r in c.execute("SELECT * FROM intelligence_records WHERE record_id NOT LIKE 'supported-%' ORDER BY record_id")]
    assert after==before  # Payloads, digests and unsynced markers all preserved.


def test_sheet_schema_conflict_holds_projection_not_processing(db,timeline):
    result=pi.topic_map(SOURCE,timeline,mapped)
    service=FakeSheets();service.tabs[pi.SHEET_TABS['source']]=[['unrelated schema']]
    with pytest.raises(ValueError,match='schema mismatch'):pi.sync_sheets(service,'isolated')
    assert pi.topic_map(SOURCE,timeline,Mock(side_effect=AssertionError()))==result


@pytest.mark.parametrize('brand',['AGAPE_CHURCH','OTHER_SHOW'])
def test_each_brand_feedback_influences_future_ranking_without_cross_brand_leak(db,brand):
    import intelligence_workflow as workflow
    workflow.observe('retained-approval',state('approved',brand))
    source={**SOURCE,'brand_id':brand,'show_id':brand}
    context=pi.learning_context(source,'9:16')
    result=pi.rank_with_evidence({'segments':[{'category':'education','score':84},{'category':'humor','score':84}]},context)
    assert result['segments'][0]['category']=='humor'
    assert context['examples'][0]['evidence_id']
    assert not pi.learning_context(SOURCE,'9:16')['examples']
    edit={'lane':'short','action':'end_later','before':[{'start':10,'end':20}],'after':[{'start':10,'end':25}]}
    pi.record_edits('retained-edit',state('approved',brand),edit,'Include the full answer.','owner')
    assert any(e['boundary_changes'] for e in pi.learning_context(source,'9:16')['examples'])
