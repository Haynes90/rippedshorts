"""Held-source retention tests use disposable local media only."""
import os
from pathlib import Path
import time

import pytest


@pytest.fixture
def cache(tmp_path,monkeypatch):
    import telegram_intake as tg
    monkeypatch.setenv('DATA_DIR',str(tmp_path))
    monkeypatch.setenv('LOCAL_ARTIFACT_RETENTION_HOURS','12')
    database=tmp_path/'ripped_shorts.db';database.write_bytes(b'isolated database')
    monkeypatch.setattr(tg,'DB_PATH',database)
    active=tmp_path/'sources'/'telegram-held';active.mkdir(parents=True)
    monkeypatch.setattr(tg,'_active_local_source_dirs',lambda:{active.resolve()})
    def write(path,age=13):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'isolated retained artifact')
        stamp=time.time()-age*3600;os.utime(path,(stamp,stamp));return path
    return tg,tmp_path,active,write


@pytest.mark.parametrize('age',[13,48,168])
@pytest.mark.parametrize('emergency',[False,True])
def test_held_source_survives_age_cleanup_and_emergency(cache,emergency,age):
    tg,root,active,write=cache
    source=write(active/'verified-source.mp4',age)
    stale=write(root/'sources'/'completed'/'expired.mp4')
    assert tg._purge_expired_local_artifacts(emergency=emergency)==1
    assert source.read_bytes()==b'isolated retained artifact' and not stale.exists()


@pytest.mark.parametrize('emergency',[False,True])
def test_release_backups_are_not_temp_cache(cache,emergency):
    tg,root,active,write=cache
    backup=write(root/'release-backups'/'prior-release'/'ripped_shorts.db',48)
    assert tg._purge_expired_local_artifacts(emergency=emergency)==0
    assert backup.exists()


def test_failed_active_source_discovery_fails_closed(cache,monkeypatch):
    tg,root,active,write=cache
    source=write(active/'source.mp4')
    def fail():raise RuntimeError('Database source discovery unavailable')
    monkeypatch.setattr(tg,'_active_local_source_dirs',fail)
    with pytest.raises(RuntimeError):tg._purge_expired_local_artifacts()
    assert source.exists()


@pytest.mark.parametrize('age',[48,168])
@pytest.mark.parametrize('emergency',[False,True])
@pytest.mark.parametrize('status',['active','processing','awaiting_review','approved','awaiting_topic_map_review'])
def test_real_request_discovery_protects_unresolved_media(tmp_path,monkeypatch,age,emergency,status):
    import sqlite3
    import telegram_intake as tg
    root=tmp_path/'sources';root.mkdir()
    db_path=tmp_path/'state.db'
    def connection():
        db=sqlite3.connect(db_path);db.row_factory=sqlite3.Row;return db
    source=root/'telegram-retained'/'source.mp4';source.parent.mkdir()
    source.write_bytes(b'retained unresolved source')
    stamp=time.time()-age*3600;os.utime(source,(stamp,stamp))
    with connection() as db:
        db.execute('CREATE TABLE telegram_requests(request_id TEXT,status TEXT,state_json TEXT)')
        db.execute('CREATE TABLE source_jobs(source_job_id TEXT,status TEXT)')
        db.execute('INSERT INTO telegram_requests VALUES(?,?,?)',('retained',status,'{}'))
    monkeypatch.setenv('DATA_DIR',str(tmp_path))
    monkeypatch.setenv('LOCAL_ARTIFACT_RETENTION_HOURS','12')
    monkeypatch.setattr(tg,'DB_PATH',db_path)
    monkeypatch.setattr(tg,'SOURCE_DIR',root)
    monkeypatch.setattr(tg,'_telegram_db',connection)
    monkeypatch.setattr(tg,'connect',connection)
    assert source.parent.resolve() in tg._active_local_source_dirs()
    assert tg._purge_expired_local_artifacts(emergency=emergency)==0
    assert source.read_bytes()==b'retained unresolved source'
