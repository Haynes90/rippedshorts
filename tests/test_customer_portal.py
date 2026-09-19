import hashlib, hmac, json, time, uuid
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
import customer_portal as p

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('DATA_DIR',str(tmp_path))
    monkeypatch.setenv('STRIPE_WEBHOOK_SECRET','whsec_test')
    monkeypatch.setenv('STRIPE_PRICE_ID','price_starter')
    monkeypatch.setenv('STRIPE_PAYMENT_LINK_ID','plink_starter')
    app=FastAPI();app.include_router(p.router)
    return TestClient(app)

def signup(c,email='one@example.com'):
    r=c.post('/app/api/auth/signup',json={'email':email,'password':'a-long-test-password'},headers={'x-r3-request':'1'})
    assert r.status_code==200
    with p.db() as db:
        return dict(db.execute('SELECT * FROM accounts WHERE email=?',(email,)).fetchone())

def signed(c,event,stamp=None):
    raw=json.dumps(event).encode();stamp=str(stamp or int(time.time()))
    sig=hmac.new(b'whsec_test',stamp.encode()+b'.'+raw,hashlib.sha256).hexdigest()
    return c.post('/app/stripe/webhook',content=raw,headers={'stripe-signature':f't={stamp},v1={sig}'})

def event(user):
    return {'id':'evt_one','type':'checkout.session.completed','data':{'object':{'mode':'subscription','payment_link':'plink_starter','customer':'cus_one','subscription':'sub_one','client_reference_id':user['id']}}}

def test_auth_csrf_logout(client):
    assert client.get('/app/api/me').status_code==401
    assert client.post('/app/api/auth/signup',json={'email':'a@b.com','password':'long-password-here'}).status_code==403
    signup(client)
    assert client.get('/app/api/me').json()['email']=='one@example.com'
    assert client.delete('/app/api/session',headers={'x-r3-request':'1'}).status_code==200
    assert client.get('/app/api/me').status_code==401

def test_webhook_signature_replay_and_binding(client):
    user=signup(client);evt=event(user)
    assert client.post('/app/stripe/webhook',json=evt).status_code==400
    assert signed(client,evt,int(time.time())-600).status_code==400
    assert signed(client,evt).status_code==200
    assert signed(client,evt).status_code==200
    evt['id']='evt_two';evt['data']['object']['subscription']='sub_other'
    assert signed(client,evt).status_code==409
    with p.db() as db:
        assert db.execute('SELECT subscription FROM accounts').fetchone()[0]=='sub_one'

def test_wrong_product_rejected(client):
    user=signup(client);evt=event(user);evt['data']['object']['payment_link']='plink_other'
    assert signed(client,evt).status_code==400

def test_entitlements_refresh_from_stripe(client,monkeypatch):
    user={'subscription':'sub_one','customer':'cus_one'}
    sub={'customer':'cus_one','status':'trialing','trial_end':time.time()+100,'items':{'data':[{'price':{'id':'price_starter'},'quantity':1}]}}
    monkeypatch.setattr(p,'stripe_get',lambda path:sub)
    assert p.entitlement(user)['seconds']==900
    sub['status']='past_due'
    with pytest.raises(p.HTTPException):p.entitlement(user)
    sub['status']='active';sub['items']['data'][0].update(current_period_start=100,current_period_end=time.time()+100)
    assert p.entitlement(user)['seconds']==3600
    sub['items']['data'][0]['price']['id']='price_wrong'
    with pytest.raises(p.HTTPException):p.entitlement(user)

def test_allowance_and_duplicate_submit(client,monkeypatch):
    user=signup(client)
    monkeypatch.setattr(p,'entitlement',lambda user:{'plan':'trial','bucket':'trial','seconds':900,'videos':1,'shorts':2,'highlights':1})
    monkeypatch.setattr(p.hub,'config',lambda:('secret','pilot','operator'))
    calls=[]
    def dispatch(body,tasks,owner,**kw):
        calls.append(kw);return {'project_id':kw['project_id_override'],'status':'accepted'}
    monkeypatch.setattr(p.hub,'submit_source',dispatch)
    body={'url':'https://youtu.be/abcdefghijk','selected_ranges':[{'start':0,'end':600}],'shorts':2,'highlights':1,'request_id':str(uuid.uuid4())}
    headers={'x-r3-request':'1'}
    assert client.post('/app/api/projects',json=body,headers=headers).status_code==202
    assert client.post('/app/api/projects',json=body,headers=headers).status_code==202
    assert len(calls)==1 and calls[0]['clip_limits']=={'shorts':2,'highlights':1}
    body['request_id']=str(uuid.uuid4())
    assert client.post('/app/api/projects',json=body,headers=headers).status_code==409

def test_dispatch_failure_releases_reservation(client,monkeypatch):
    signup(client)
    monkeypatch.setattr(p,'entitlement',lambda user:{'plan':'trial','bucket':'trial','seconds':900,'videos':1,'shorts':2,'highlights':1})
    monkeypatch.setattr(p.hub,'config',lambda:('secret','pilot','operator'))
    def failed(*a,**kw):raise p.HTTPException(503,'not configured')
    monkeypatch.setattr(p.hub,'submit_source',failed)
    body={'url':'https://youtu.be/abcdefghijk','selected_ranges':[{'start':0,'end':600}],'shorts':2,'highlights':1,'request_id':str(uuid.uuid4())}
    assert client.post('/app/api/projects',json=body,headers={'x-r3-request':'1'}).status_code==503
    with p.db() as db:assert db.execute('SELECT status FROM usage').fetchone()[0]=='released'
