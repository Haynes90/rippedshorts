"""Customer accounts, verified Stripe access, and atomic pilot allowances."""
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlencode

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import review_hub as hub

router = APIRouter(prefix='/app')
COOKIE = 'r3_session'


@contextmanager
def db():
    path = Path(os.getenv('DATA_DIR', 'data'))
    path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path / 'customers.db', timeout=30)
    conn.row_factory = sqlite3.Row
    conn.executescript('''
    CREATE TABLE IF NOT EXISTS accounts(id TEXT PRIMARY KEY,email TEXT UNIQUE NOT NULL,password TEXT NOT NULL,customer TEXT UNIQUE,subscription TEXT UNIQUE);
    CREATE TABLE IF NOT EXISTS sessions(token TEXT PRIMARY KEY,account TEXT NOT NULL,expires INTEGER NOT NULL);
    CREATE TABLE IF NOT EXISTS attempts(key TEXT PRIMARY KEY,count INTEGER,expires INTEGER);
    CREATE TABLE IF NOT EXISTS usage(project TEXT PRIMARY KEY,account TEXT,bucket TEXT,seconds REAL,shorts INTEGER,highlights INTEGER,status TEXT);
    CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY);
    ''')
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def guard(request):
    if request.headers.get('x-r3-request') != '1' or request.headers.get('sec-fetch-site') == 'cross-site':
        raise HTTPException(403, 'Refresh the page and try again')


def account(request: Request):
    if request.method != 'GET':
        guard(request)
    token = hashlib.sha256(request.cookies.get(COOKIE, '').encode()).hexdigest()
    with db() as conn:
        row = conn.execute('SELECT a.* FROM accounts a JOIN sessions s ON a.id=s.account WHERE s.token=? AND s.expires>?', (token, time.time())).fetchone()
    if not row:
        raise HTTPException(401, 'Please sign in')
    return dict(row)


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    return salt + ':' + hashlib.scrypt(password.encode(), salt=salt.encode(), n=16384, r=8, p=1).hex()


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=12, max_length=128)


@router.get('')
@router.get('/')
def page():
    return FileResponse(hub.STATIC / 'customer.html', headers={'Cache-Control': 'no-store'})


@router.post('/api/auth/{action}')
def auth(action: str, body: Credentials, request: Request, response: Response):
    guard(request)
    if action not in ('signup', 'login'):
        raise HTTPException(404)
    email = body.email.strip().lower()
    if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        raise HTTPException(422, 'Enter a valid email address')
    # Shared persistent limit also covers nonexistent accounts and restarts.
    key = hashlib.sha256((request.client.host if request.client else 'unknown').encode()).hexdigest()
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute('DELETE FROM attempts WHERE expires<?', (time.time(),))
        conn.execute('INSERT INTO attempts VALUES (?,1,?) ON CONFLICT(key) DO UPDATE SET count=count+1', (key, time.time()+900))
        count = conn.execute('SELECT count FROM attempts WHERE key=?', (key,)).fetchone()[0]
    if count > 20:
        raise HTTPException(429, 'Too many attempts. Try again in 15 minutes.')
    with db() as conn:
        row = conn.execute('SELECT * FROM accounts WHERE email=?', (email,)).fetchone()
        if action == 'signup':
            if row:
                raise HTTPException(409, 'Unable to create account. Try signing in instead.')
            account_id = str(uuid.uuid4())
            try:
                conn.execute('INSERT INTO accounts(id,email,password) VALUES (?,?,?)', (account_id,email,password_hash(body.password)))
            except sqlite3.IntegrityError:
                raise HTTPException(409, 'Unable to create account. Try signing in instead.')
        else:
            stored = row['password'] if row else password_hash('dummy-password-value', '0'*32)
            valid = hmac.compare_digest(password_hash(body.password, stored.split(':')[0]), stored)
            if not row or not valid:
                raise HTTPException(401, 'Email or password is incorrect')
            account_id = row['id']
        token = secrets.token_urlsafe(32)
        conn.execute('DELETE FROM sessions WHERE expires<?', (time.time(),))
        conn.execute('INSERT INTO sessions VALUES (?,?,?)', (hashlib.sha256(token.encode()).hexdigest(), account_id, time.time()+604800))
    response.set_cookie(COOKIE, token, max_age=604800, httponly=True, secure=request.url.scheme=='https', samesite='strict', path='/app')
    return {'ok': True}


@router.delete('/api/session')
def logout(request: Request, response: Response, user=Depends(account)):
    with db() as conn:
        conn.execute('DELETE FROM sessions WHERE token=?', (hashlib.sha256(request.cookies.get(COOKIE,'').encode()).hexdigest(),))
    response.delete_cookie(COOKIE, path='/app')
    return {'ok':True}


def stripe_get(path):
    key = os.getenv('STRIPE_SECRET_KEY')
    if not key:
        raise HTTPException(503, 'Billing connection is being prepared')
    try:
        result = requests.get('https://api.stripe.com/v1/' + path, auth=(key,''), timeout=20)
        result.raise_for_status()
        return result.json()
    except (requests.RequestException, ValueError):
        raise HTTPException(503, 'Could not verify billing. Please try again shortly.')


def entitlement(user):
    if not user.get('subscription'):
        raise HTTPException(402, 'Start your trial, then refresh billing status')
    sub = stripe_get('subscriptions/' + user['subscription'])
    items = sub.get('items',{}).get('data',[])
    if sub.get('customer') != user['customer'] or len(items)!=1 or items[0].get('price',{}).get('id') != os.getenv('STRIPE_PRICE_ID') or items[0].get('quantity') != 1:
        raise HTTPException(403, 'Subscription does not match the Starter plan')
    trial = sub.get('status') == 'trialing'
    if sub.get('status') not in ('active','trialing'):
        raise HTTPException(402, 'An active trial or subscription is required')
    end = sub.get('trial_end') if trial else items[0].get('current_period_end',sub.get('current_period_end',0))
    start = items[0].get('current_period_start',sub.get('current_period_start'))
    if not end or end <= time.time() or (not trial and start is None):
        raise HTTPException(402, 'Your billing period has ended. Refresh after renewal.')
    return {'plan':'trial' if trial else 'starter','bucket':'trial' if trial else str(start),'seconds':900 if trial else 3600,'videos':1 if trial else 2,'shorts':2 if trial else 10,'highlights':1 if trial else 2}


@router.get('/api/me')
def me(user=Depends(account)):
    access, issue = None, None
    try:
        access = entitlement(user)
    except HTTPException as exc:
        issue = exc.detail
    usage = {'seconds':0,'videos':0,'shorts':0,'highlights':0}
    if access:
        with db() as conn:
            row = conn.execute("SELECT COALESCE(SUM(seconds),0) seconds,COUNT(*) videos,COALESCE(SUM(shorts),0) shorts,COALESCE(SUM(highlights),0) highlights FROM usage WHERE account=? AND bucket=? AND status!='released'", (user['id'],access['bucket'])).fetchone()
            usage = dict(row)
    return {'email':user['email'],'access':access,'usage':usage,'billing_message':issue}


@router.post('/api/checkout')
def checkout(user=Depends(account)):
    if user.get('subscription'):
        raise HTTPException(409, 'Use Manage billing for your existing subscription')
    if not all(os.getenv(k) for k in ('STRIPE_SECRET_KEY','STRIPE_WEBHOOK_SECRET','STRIPE_PRICE_ID','STRIPE_PAYMENT_LINK_ID')):
        raise HTTPException(503, 'Checkout opens after billing setup is complete')
    link = os.getenv('PILOT_CHECKOUT_URL') or 'https://buy.stripe.com/3cI14p3xb0fK7Ted3Y5ZC00'
    if not link.startswith('https://buy.stripe.com/') or '?' in link:
        raise HTTPException(503, 'Checkout configuration needs attention')
    return {'url':link+'?'+urlencode({'client_reference_id':user['id']})}


@router.post('/stripe/webhook')
async def webhook(request: Request):
    secret = os.getenv('STRIPE_WEBHOOK_SECRET')
    if not secret:
        raise HTTPException(503)
    raw = bytearray()
    async for chunk in request.stream():
        raw.extend(chunk)
        if len(raw)>262144:
            raise HTTPException(413)
    try:
        parts = [part.split('=',1) for part in request.headers.get('stripe-signature','').split(',')]
        timestamp = next(v for k,v in parts if k=='t')
        signed = hmac.new(secret.encode(), timestamp.encode()+b'.'+raw, hashlib.sha256).hexdigest()
        if abs(time.time()-int(timestamp))>300 or not any(k=='v1' and hmac.compare_digest(v,signed) for k,v in parts):
            raise ValueError()
        event = json.loads(raw)
    except (ValueError, StopIteration, TypeError):
        raise HTTPException(400, 'Invalid Stripe signature')
    if event.get('type') != 'checkout.session.completed':
        return {'received':True}
    obj = event['data']['object']
    if obj.get('mode')!='subscription' or obj.get('payment_link')!=os.getenv('STRIPE_PAYMENT_LINK_ID') or not obj.get('subscription') or not obj.get('customer'):
        raise HTTPException(400, 'Unexpected checkout')
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('SELECT 1 FROM events WHERE id=?',(event['id'],)).fetchone():
            return {'received':True}
        row = conn.execute('SELECT * FROM accounts WHERE id=?',(obj.get('client_reference_id'),)).fetchone()
        if not row:
            raise HTTPException(400, 'Checkout was not started from a customer account')
        if row['subscription'] and row['subscription']!=obj['subscription']:
            raise HTTPException(409, 'An account already has a subscription')
        try:
            conn.execute('UPDATE accounts SET customer=?,subscription=? WHERE id=?',(obj['customer'],obj['subscription'],row['id']))
        except sqlite3.IntegrityError:
            raise HTTPException(409, 'Subscription is already linked')
        conn.execute('INSERT INTO events VALUES (?)',(event['id'],))
    return {'received':True}


class Submission(BaseModel):
    url: str = Field(min_length=10,max_length=2048)
    selected_ranges: list[dict] = Field(min_length=1,max_length=4)
    shorts: int = Field(ge=0,le=10)
    highlights: int = Field(ge=0,le=2)
    request_id: str = Field(pattern=r'^[a-zA-Z0-9-]{20,80}$')


@router.post('/api/projects',status_code=202)
def submit(body: Submission, tasks: BackgroundTasks, user=Depends(account)):
    from review_basic import normalize_ranges
    from urllib.parse import urlparse
    if not body.shorts and not body.highlights:
        raise HTTPException(422,'Choose at least one clip')
    parsed_url = urlparse(body.url)
    if parsed_url.scheme!='https' or parsed_url.hostname not in ('youtube.com','www.youtube.com','m.youtube.com','youtu.be') or parsed_url.username or parsed_url.port not in (None,443):
        raise HTTPException(422,'Use an HTTPS YouTube video link')
    try:
        ranges = normalize_ranges(body.selected_ranges)
    except (ValueError,KeyError,TypeError):
        raise HTTPException(422,'Check section start and end times')
    seconds = sum(r['end']-r['start'] for r in ranges)
    access = entitlement(user)
    project = str(uuid.uuid5(uuid.UUID(user['id']),body.request_id))
    # Reserve budget before dispatch. Held reservations fail closed after crashes.
    with db() as conn:
        conn.execute('BEGIN IMMEDIATE')
        previous = conn.execute('SELECT * FROM usage WHERE project=?',(project,)).fetchone()
        if previous and previous['status']=='accepted':
            return {'project_id':project,'status':'accepted'}
        if previous and previous['status']=='held':
            raise HTTPException(409,'This submission is already pending. Check your projects.')
        rows = conn.execute("SELECT * FROM usage WHERE account=? AND bucket=? AND status!='released'",(user['id'],access['bucket'])).fetchall()
        if len(rows)>=access['videos'] or sum(r['seconds'] for r in rows)+seconds>access['seconds'] or sum(r['shorts'] for r in rows)+body.shorts>access['shorts'] or sum(r['highlights'] for r in rows)+body.highlights>access['highlights']:
            raise HTTPException(409,'This request exceeds your remaining allowance. Reduce sections or clip counts.')
        conn.execute('INSERT OR REPLACE INTO usage VALUES (?,?,?,?,?,?,?)',(project,user['id'],access['bucket'],seconds,body.shorts,body.highlights,'held'))
    try:
        result = hub.submit_source(hub.Source(url=body.url,mode='both' if body.shorts and body.highlights else 'shorts' if body.shorts else 'topics',customer_reference=user['id'],selected_ranges=ranges,plan=access['plan'],allowance_checked=True),tasks,hub.config()[1:], project_id_override=project, clip_limits={'shorts':body.shorts,'highlights':body.highlights})
    except Exception:
        with db() as conn:
            conn.execute("UPDATE usage SET status='released' WHERE project=?",(project,))
        raise
    with db() as conn:
        conn.execute("UPDATE usage SET status='accepted' WHERE project=?",(project,))
    return result


@router.get('/api/projects')
def projects(user=Depends(account)):
    e = hub.engine()
    with e._LOCK,e._telegram_db() as conn:
        rows = conn.execute('SELECT * FROM telegram_requests WHERE chat_id=? AND user_id=? ORDER BY created_at DESC LIMIT 100',hub.config()[1:]).fetchall()
    result=[]
    for row in rows:
        state=json.loads(row['state_json'])
        if state.get('account_id')!='pilot:'+user['id']:
            continue
        item=hub.summary(row)
        item.pop('error',None)
        item['downloads']=[]
        if not item['expired']:
            for lane,(results,reviews) in hub.LANES.items():
                for index,clip in enumerate(state.get(results,{}).get('segments',[])):
                    url=clip.get('clip_url') or clip.get('segment_url')
                    if state.get(reviews,{}).get(str(index),{}).get('status')=='rendered' and url and url.startswith('https://'):
                        item['downloads'].append({'title':clip.get('title') or f'{lane} {index+1}','url':url})
            item['transcript_ready']=bool(state.get('transcript_text'))
        result.append(item)
    return {'projects':result}


@router.get('/api/projects/{project_id}/transcript')
def transcript(project_id: str,user=Depends(account)):
    e=hub.engine()
    with e._LOCK,e._telegram_db() as conn:
        row=hub.owned(conn,project_id,hub.config()[1:])
    if json.loads(row['state_json']).get('account_id')!='pilot:'+user['id']:
        raise HTTPException(404)
    return hub.transcript_download(project_id,hub.config()[1:])
