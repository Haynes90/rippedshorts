"""Receipt-backed navigation through the existing bot, database and review portal."""
from contextlib import closing
import hashlib
import json
from audio_master_handoff import connect


def notify(request_id, stage, revision='ready'):
    import telegram_intake as tg
    from caption_review_web import review_url_for_request, _public_base_url
    from durable_jobs import ensure_schema
    if not _public_base_url():
        return {'status': 'not_configured'}
    with closing(connect()) as db:
        row = db.execute('SELECT chat_id,user_id FROM telegram_requests WHERE request_id=?', (request_id,)).fetchone()
        if not row or not tg._authorized(str(row['chat_id']), str(row['user_id'])):
            raise ValueError('Review destination does not match an authorized existing request')
    url = review_url_for_request(request_id, stage)
    if not url:
        return {'status': 'not_configured'}
    event = 'clip-review:' + hashlib.sha256(f'{request_id}:{stage}:{revision}'.encode()).hexdigest()
    with closing(connect()) as db, db:
        ensure_schema(db)
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM telegram_requests WHERE request_id=?', (request_id,)).fetchone()
        if not row or not tg._authorized(str(row['chat_id']), str(row['user_id'])):
            raise ValueError('Review destination does not match an authorized existing request')
        previous = db.execute('SELECT * FROM durable_outbox WHERE event_id=?', (event,)).fetchone()
        receipt = json.loads(previous['receipt'] or '{}') if previous else {}
        if previous and previous['state'] != 'COMPLETE':
            return {'status': 'verification_required'}  # never resend an uncertain send
        if previous and receipt.get('url') == url:
            return {'status': 'already_delivered', 'message_id': receipt.get('message_id')}
        timestamp = tg.now()
        db.execute("INSERT INTO durable_outbox(event_id,job_id,destination,state,created_at,updated_at) VALUES (?,?,?,'SENDING',?,?) ON CONFLICT(event_id) DO UPDATE SET state='SENDING',updated_at=excluded.updated_at",
                   (event, request_id, 'telegram_review_navigation', timestamp, timestamp))
        chat_id = str(row['chat_id'])
    payload = {'chat_id': chat_id, 'text': f'Clip review ready: {stage}. Your saved decisions are preserved. Editorial and framing choices do not schedule or publish.',
               'reply_markup': {'inline_keyboard': [[{'text': '🌐 Open Clip Review', 'url': url}]]}}
    method = 'sendMessage'
    if receipt.get('message_id'):
        method = 'editMessageText'
        payload['message_id'] = receipt['message_id']
    try:
        response = tg.telegram(method, payload)
        message_id = (response.get('result') or {}).get('message_id') or receipt.get('message_id')
        if not response.get('ok') or not message_id:
            raise ValueError('Telegram acknowledgment missing')
    except Exception:
        with closing(connect()) as db, db:
            db.execute("UPDATE durable_outbox SET state='UNCERTAIN',last_error='Telegram outcome requires verification',updated_at=? WHERE event_id=?", (tg.now(), event))
        return {'status': 'verification_required'}
    with closing(connect()) as db, db:
        db.execute("UPDATE durable_outbox SET state='COMPLETE',receipt=?,updated_at=? WHERE event_id=?",
                   (json.dumps({'message_id': message_id, 'url': url, 'stage': stage}), tg.now(), event))
    return {'status': 'delivered', 'message_id': message_id}


def notify_safely(request_id, stage, revision='ready'):
    try:
        return notify(request_id, stage, revision)
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Review navigation unavailable request=%s stage=%s', request_id, stage)
        return {'status': 'unavailable'}
