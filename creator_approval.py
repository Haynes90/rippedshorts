"""Verified creator directory and one durable decision per accepted batch."""
import hashlib
import json
import re
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from audio_master_handoff import connect
DEFAULTS_PATH = Path(__file__).with_name('creator_defaults.json')


def _db():
    db = connect()
    db.executescript('''
      CREATE TABLE IF NOT EXISTS creator_directory (
        creator_id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
        instagram_handle TEXT NOT NULL UNIQUE, verified_by TEXT NOT NULL, verified_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS creator_brands (
        creator_id TEXT NOT NULL, brand_id TEXT NOT NULL, recurring_host INTEGER NOT NULL DEFAULT 0,
        standing_invite_authorized INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(creator_id,brand_id));
      CREATE TABLE IF NOT EXISTS creator_channels (
        creator_id TEXT NOT NULL, brand_id TEXT NOT NULL, channel_id TEXT NOT NULL,
        PRIMARY KEY(creator_id,brand_id,channel_id));
      CREATE TABLE IF NOT EXISTS batch_creator_approvals (
        request_id TEXT PRIMARY KEY, approval_json TEXT NOT NULL);
    ''')
    columns = {r[1] for r in db.execute('PRAGMA table_info(creator_brands)')}
    if 'standing_invite_authorized' not in columns:
        db.execute('ALTER TABLE creator_brands ADD COLUMN standing_invite_authorized INTEGER NOT NULL DEFAULT 0')
    if DEFAULTS_PATH.is_file():
        defaults = json.loads(DEFAULTS_PATH.read_text(encoding='utf-8'))
        for brand, entries in defaults.get('brands',{}).items():
            for person in entries:
                handle = person['instagram_handle']
                cid = hashlib.sha256(handle.encode()).hexdigest()[:24]
                db.execute('INSERT OR IGNORE INTO creator_directory VALUES(?,?,?,?,?)',
                    (cid,person['display_name'],handle,defaults['verification_source'],'2026-10-29T00:00:00Z'))
                db.execute('INSERT OR IGNORE INTO creator_brands VALUES(?,?,?,?)',
                    (cid,brand,bool(person.get('recurring_host')),bool(person.get('standing_invite_authorized'))))
        db.commit()
    return db


def suggestions(brand, channel_id=''):
    with closing(_db()) as db:
        rows = db.execute('''SELECT DISTINCT d.*,b.recurring_host,b.standing_invite_authorized,
          EXISTS(SELECT 1 FROM creator_channels c WHERE c.creator_id=d.creator_id
                 AND c.brand_id=b.brand_id AND c.channel_id=?) AS source_match
          FROM creator_directory d JOIN creator_brands b USING(creator_id)
          WHERE b.brand_id=? ORDER BY b.recurring_host DESC,d.display_name''',
          (channel_id, brand.upper())).fetchall()
        return [dict(r) for r in rows if brand.upper() == 'TDOG' or r['source_match']]


def approved_batch(request_id):
    with closing(_db()) as db:
        row = db.execute('SELECT approval_json FROM batch_creator_approvals WHERE request_id=?', (request_id,)).fetchone()
        return json.loads(row[0]) if row else None


def confirm_batch(request_id, brand, source_video_id, channel_id, entries, reviewer, confirmed=False):
    brand = brand.upper()
    if brand not in {'TCB', 'TDOG'} or confirmed is not True or not reviewer:
        raise ValueError('Explicit identity and batch confirmation is required')
    if not isinstance(entries, list) or len(entries) > 30:
        raise ValueError('Invalid creator selection')
    if any(not isinstance(e,dict) for e in entries):
        raise ValueError('Each creator must have confirmed structured identity fields')
    if sum(e.get('intent')=='invite' for e in entries) > 3:
        raise ValueError('Instagram supports at most 3 native collaborators per Reel')
    stamp = datetime.now(timezone.utc).isoformat()
    with closing(_db()) as db, db:
        db.execute('BEGIN IMMEDIATE')
        required = db.execute('SELECT d.instagram_handle FROM creator_directory d JOIN creator_brands b USING(creator_id) WHERE b.brand_id=? AND b.standing_invite_authorized=1',(brand,)).fetchall()
        choices = {str(e.get('instagram_handle') or '').strip().lstrip('@').lower():e.get('intent','none') for e in entries}
        if any(choices.get(r[0]) != 'invite' for r in required):
            raise ValueError('Keep the confirmed recurring hosts selected for Instagram collaboration under this brand standing rule')
        previous = db.execute('SELECT approval_json FROM batch_creator_approvals WHERE request_id=?', (request_id,)).fetchone()
        creators, seen = [], set()
        for entry in entries:
            name = str(entry.get('display_name') or '').strip()
            handle = str(entry.get('instagram_handle') or '').strip().lstrip('@').lower()
            intent = entry.get('intent', 'none')
            if not name or len(name) > 120 or not re.fullmatch(r'[a-z0-9_](?:[a-z0-9._]{0,28}[a-z0-9_])?', handle) or '..' in handle or intent not in {'none','mention','invite'}:
                raise ValueError('Confirm a name, valid Instagram handle and collaboration choice')
            creator_id = hashlib.sha256(handle.encode()).hexdigest()[:24]
            if creator_id in seen:
                raise ValueError('A creator may only appear once in the batch')
            seen.add(creator_id)
            db.execute('INSERT INTO creator_directory VALUES(?,?,?,?,?) ON CONFLICT(creator_id) DO UPDATE SET display_name=excluded.display_name,verified_by=excluded.verified_by,verified_at=excluded.verified_at',
                       (creator_id,name,handle,reviewer,stamp))
            db.execute('INSERT INTO creator_brands(creator_id,brand_id,recurring_host) VALUES(?,?,?) ON CONFLICT(creator_id,brand_id) DO UPDATE SET recurring_host=excluded.recurring_host',
                       (creator_id,brand,(bool(entry.get('recurring_host')) or handle in {r[0] for r in required}) if brand=='TDOG' else False))
            # Channel owner need not be featured. Save an association only when explicitly confirmed.
            if channel_id and entry.get('confirm_source_association') is True:
                db.execute('INSERT OR IGNORE INTO creator_channels VALUES(?,?,?)', (creator_id,brand,channel_id))
            creators.append({'creator_id':creator_id,'display_name':name,'instagram_handle':handle,
                             'intent':intent,'verification_status':'human_confirmed',
                             'recurring_host':(bool(entry.get('recurring_host')) or handle in {r[0] for r in required}) if brand=='TDOG' else False,
                             'source_association_confirmed':bool(entry.get('confirm_source_association')),
                             'standing_invite_authorized':handle in {r[0] for r in required}})
        selection = {'request_id':request_id,'brand_id':brand,'source_video_id':source_video_id,
                     'source_channel_id':channel_id,'scope':'approved_batch','approved_by':reviewer,
                     'approved_at':stamp,'approval_revision':json.loads(previous[0])['approval_revision']+1 if previous else 1,
                     'creators':creators}
        # Identical retries retain the exact snapshot/revision.
        if previous:
            old = json.loads(previous[0])
            if all(old.get(k)==selection.get(k) for k in ['brand_id','source_video_id','source_channel_id','creators']):
                return old
        db.execute('INSERT INTO batch_creator_approvals VALUES(?,?) ON CONFLICT(request_id) DO UPDATE SET approval_json=excluded.approval_json',
                   (request_id,json.dumps(selection)))
        return selection
