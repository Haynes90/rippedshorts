import ast
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import BackgroundTasks
from test_editorial_framing_review import review
from test_finished_clip_review import render_fixture, read_state, save_state


@pytest.mark.parametrize('action',['approve','reject','topic_approve','topic_reject','shorts_confirm','shorts_page','topic_page','schedule_now','copy_finish','copy_approve','copy_keep','copy_page','copy_regenerate','copy_edit_caption','copy_edit_title','copy_edit_description','options','rerip','reuse','retry_missing','fresh_start'])
def test_previously_issued_buttons_cannot_change_legacy_records(review,monkeypatch,action):
    r=review;state=render_fixture(r);state.pop('review_interface',None)
    state['copy_drafts'][0]['social_caption']='Exact retained owner wording 🌱'
    save_state(r,state);before=read_state(r)
    for name in ['_handoff_shorts_to_schedule_master','_render_approved','_render_topic_approved']:
        monkeypatch.setattr(r['tg'],name,Mock(side_effect=AssertionError('Telegram must not invoke review/delivery mutations')))
    tasks=BackgroundTasks()
    result=r['tg']._accept_update({'callback_query':{'id':'old-control','data':f'rs:{action}:existing:0','from':{'id':'owner'},'message':{'chat':{'id':'saved-chat'},'message_id':1}}},tasks)
    assert result['status']=='website_review_required'
    assert read_state(r)==before and tasks.tasks==[]
    links=[call.args[1] for call in r['telegram'].call_args_list if call.args[0] in {'sendMessage','editMessageText'}]
    assert len(links)==1
    button=links[0]['reply_markup']['inline_keyboard'][0][0]
    assert button['text']=='Review Clips' and '/review/' in button['url'] and 'callback_data' not in button


def test_legacy_caption_reply_does_not_overwrite_saved_copy(review):
    r=review;render_fixture(r)
    with r['tg']._telegram_db() as db:
        db.execute('INSERT INTO telegram_copy_inputs(prompt_message_id,request_id,asset_index,field,chat_id,user_id,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)',('42','existing',0,'caption','saved-chat','owner','now','now'))
    before=read_state(r)
    result=r['tg']._accept_update({'message':{'from':{'id':'owner'},'chat':{'id':'saved-chat'},'text':'Do not replace my saved caption','reply_to_message':{'message_id':42}}},BackgroundTasks())
    assert result['status']=='website_review_required' and read_state(r)==before


def test_review_notification_is_one_job_link_across_stages(review):
    r=review;render_fixture(r)
    import review_navigation
    review_navigation.notify('existing','shorts','one')
    review_navigation.notify('existing','highlights','two')
    review_navigation.notify('existing','final','three')
    calls=[c.args[0] for c in r['telegram'].call_args_list]
    assert calls.count('sendMessage')==1 and calls.count('editMessageText')==2
    assert 'existing' not in r['telegram'].call_args.args[1]['reply_markup']['inline_keyboard'][0][0]['url'] # bearer session, not another job's token


def test_obsolete_review_builders_and_mutating_callback_branches_are_removed():
    source=(Path(__file__).resolve().parents[1]/'telegram_intake.py').read_text(encoding='utf-8')
    tree=ast.parse(source);names={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
    assert not names&{'_copy_review_payload','_short_review_page_payload','_short_control_payload','_topic_review_payload'}
    assert 'callback_data": f"rs:' not in source
    assert 'Continue to 16:9' not in source and 'Finish & Schedule' not in source


def test_old_button_cannot_leak_another_owners_review_link(review):
    r=review;render_fixture(r);before=read_state(r)
    result=r['tg']._accept_update({'callback_query':{'id':'wrong-owner','data':'rs:approve:existing:0','from':{'id':'owner'},'message':{'chat':{'id':'other-chat'},'message_id':1}}},BackgroundTasks(),trusted_source=True)
    assert result['status']=='unauthorized' and read_state(r)==before
    r['telegram'].assert_not_called()
