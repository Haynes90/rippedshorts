from copy import deepcopy
import pytest
from batch_progress import summarize, shared_failure


def batch(total=10,complete=8):
    state={'show_id':'TDOG','creator_approval':{'approved_at':'now'},'result':{'segments':[]},'topic_result':{'segments':[]},
        'candidate_reviews':{},'topic_reviews':{},'copy_drafts':[],'website_playback_checks':{}}
    manifest=[];deliveries=[]
    for i in range(total):
        lane='short' if i%2==0 else 'highlight';key='result' if lane=='short' else 'topic_result';review='candidate_reviews' if lane=='short' else 'topic_reviews'
        index=len(state[key]['segments']);asset=f'job:{lane}:{index+1}';url='https://fixture.invalid/'+asset
        proof={k:True for k in ('media_duration','video_stream','audio_stream','format_valid','export_edges')}
        state[key]['segments'].append({'completion_check':proof,'candidate_number':index+1})
        state[review][str(index)]={'status':'rendered' if i<complete else 'render_failed','clip_url':url,'segment_url':url,'render_error':'local clip render failed' if i>=complete else None}
        state['copy_drafts'].append({'asset_id':asset,'copy_status':'approved'})
        state['website_playback_checks'][asset]={'media_url':url,'verified':True}
        if i<complete:
            for platform in (['INSTAGRAM','FACEBOOK','YOUTUBE'] if lane=='short' else ['FACEBOOK','YOUTUBE']):
                d={'delivery_id':asset+platform,'asset_group_id':asset,'platform':platform,'destination_id':platform,
                    'scheduled_at':'2030-01-01T12:00:00+00:00','schedule_record_valid':True}
                manifest.append(d);deliveries.append(d)
    state['schedule_master']={'shorts_response':{'delivery_manifest':manifest,'delivery_validation':{'deliveries':deliveries}}}
    return state


@pytest.mark.parametrize('total,complete,status',[(10,10,'BATCH COMPLETE'),(10,8,'BATCH MOSTLY SUCCESSFUL'),(10,7,'BATCH PARTIAL FAILURE')])
def test_full_selected_denominator_includes_failed_clips_in_both_formats(total,complete,status):
    result=summarize('job',batch(total,complete))
    assert result['selected_clips']==total and result['completed_clips']==complete
    assert result['completion_percentage']==complete/total*100
    assert result['status']==status
    assert len(result['failures'])==total-complete


def test_missing_one_required_companion_delivery_means_clip_is_not_complete():
    state=batch(2,2)
    state['schedule_master']['shorts_response']['delivery_validation']['deliveries'].pop()
    result=summarize('job',state)
    assert result['selected_clips']==2 and result['completed_clips']==1


def test_render_url_without_actual_playback_confirmation_is_not_counted_playable():
    state=batch(2,2);state['website_playback_checks']={}
    result=summarize('job',state)
    assert result['completed_clips']==0
    assert result['formats']['short']['rendered_audio_verified']==1
    assert result['formats']['short']['playable']==0


def test_shared_authentication_failure_blocks_the_batch_without_losing_selection():
    state=batch(10,8);state['candidate_reviews']['4']['render_error']='TRANSCRIPTION_AUTHENTICATION_BLOCKED'
    assert summarize('job',state)['status']=='BATCH BLOCKED'
    assert summarize('job',state)['selected_clips']==10


def test_no_selections_has_no_success_percentage():
    assert summarize('job',{})['status']=='NOT STARTED'
    assert summarize('job',{})['completion_percentage'] is None


def test_owner_hold_keeps_selected_denominator_and_blocks_success():
    state=batch(2,2)
    state['copy_drafts'][0]['editorial_hold']=True
    result=summarize('job',state)
    assert result['selected_clips']==2 and result['completed_clips']==1
    assert result['clips'][0]['cause']=='HELD_BY_OWNER'


@pytest.mark.parametrize('lane,error,expected',[('short','clip-specific encoder failure',1),('highlight','clip-specific encoder failure',1),('highlight','TRANSCRIPTION_AUTHENTICATION_BLOCKED',0)])
def test_actual_render_completion_gate_isolates_clip_failure_but_holds_shared_failure(tmp_path,lane,error,expected):
    import ast,json,os,sqlite3,threading
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import Mock
    path=tmp_path/'gate.db'
    state={'schedule_requested_at':'now','candidate_reviews':{'0':{'status':'rendered'}},'topic_reviews':{}}
    target='candidate_reviews' if lane=='short' else 'topic_reviews'
    state[target]['1']={'status':'render_failed','render_error':error}
    def connect():
        db=sqlite3.connect(path);db.row_factory=sqlite3.Row;return db
    with connect() as db:
        db.execute('CREATE TABLE telegram_requests(request_id TEXT,status TEXT,state_json TEXT,updated_at TEXT)')
        db.execute('INSERT INTO telegram_requests VALUES(?,?,?,?)',('job','review',json.dumps(state),'now'))
    tree=ast.parse(Path('telegram_intake.py').read_text(encoding='utf-8'))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_notify_render_queue_complete')
    submit=Mock()
    ns={'_LOCK':threading.RLock(),'_telegram_db':connect,'json':json,'Any':object,'now':lambda:'now',
        '_save':Mock(),'os':os,'telegram':Mock(),'send':Mock(),
        'RENDER_EXECUTOR':SimpleNamespace(submit=submit),'_handoff_shorts_to_schedule_master':Mock()}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'real-render-gate','exec'),ns)
    ns[fn.name]('job','chat')
    assert submit.call_count==expected
    with connect() as db:saved=json.loads(db.execute('SELECT state_json FROM telegram_requests').fetchone()[0])
    assert saved[target]['1']['status']=='render_failed' and saved[target]['1']['render_error']==error
