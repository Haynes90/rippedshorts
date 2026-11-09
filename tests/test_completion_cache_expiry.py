import copy,json,os,time
from datetime import datetime,timezone,timedelta
import pytest
from test_boundary_specification import passage


@pytest.fixture
def owned(passage,monkeypatch):
    p=passage;tg=p['tg'];source=p['tmp']/'sources'/'shared'/'source.mp4';source.parent.mkdir(parents=True);source.write_bytes(b'verified cache')
    os.utime(source,(time.time()-7*86400,)*2)
    monkeypatch.setattr(tg,'SOURCE_DIR',p['tmp']/'sources');monkeypatch.setenv('DATA_DIR',str(p['tmp']))
    monkeypatch.setattr(tg,'_active_local_source_dirs',lambda:set())
    state=p['read']();state.update(video_path=str(source),source_cache_closed_at='owner-closed',source_cache_closure_actor='owner');p['write'](state)
    clock=[datetime.now(timezone.utc)]
    class Clock:
        @staticmethod
        def now(tz=None):return clock[0]
    monkeypatch.setattr(tg,'datetime',Clock)
    return p,tg,source,clock


@pytest.mark.parametrize('emergency',[False,True])
def test_old_source_expiry_starts_at_completion_and_emergency_cannot_shorten_it(owned,emergency):
    p,tg,source,clock=owned
    tg._purge_expired_local_artifacts(retention_hours=12,emergency=emergency)
    assert source.exists()
    clock[0]+=timedelta(hours=2);tg._purge_expired_local_artifacts(retention_hours=12,emergency=emergency)
    assert source.exists()
    clock[0]+=timedelta(hours=11);tg._purge_expired_local_artifacts(retention_hours=12,emergency=emergency)
    assert not source.exists()


def test_shared_unresolved_request_and_reopened_work_reset_the_clock(owned):
    p,tg,source,clock=owned;state=p['read']();tg._source_cache_eligibility(set())
    other=copy.deepcopy(state);other.pop('source_cache_closed_at');other.pop('source_cache_closure_actor')
    with tg._telegram_db() as db:
        db.execute('INSERT INTO telegram_requests VALUES(?,?,?,?,?,?,?,?,?,?,?)',('other','other','chat','owner','awaiting_review','both','youtube','source',json.dumps(other),'now','now'))
    clock[0]+=timedelta(days=8)
    assert tg._source_cache_eligibility(set())[source.parent.resolve()] is None
    tg._purge_expired_local_artifacts(emergency=True);assert source.exists()
    other.update(source_cache_closed_at='closed',source_cache_closure_actor='owner')
    with tg._telegram_db() as db:db.execute('UPDATE telegram_requests SET state_json=? WHERE request_id=?',(json.dumps(other),'other'))
    started=tg._source_cache_eligibility(set())[source.parent.resolve()]
    assert started==clock[0].timestamp()
    state.pop('source_cache_closed_at');state.pop('source_cache_closure_actor');p['write'](state)
    assert tg._source_cache_eligibility(set())[source.parent.resolve()] is None


def test_active_source_job_prevents_starting_a_completion_clock(owned):
    p,tg,source,clock=owned
    with tg._telegram_db() as db:
        db.execute('INSERT INTO source_jobs VALUES(?,?,?,?,?,?)',('shared','processing','{}','{}','now','now'))
    assert tg._source_cache_eligibility({source.parent.resolve()})[source.parent.resolve()] is None
