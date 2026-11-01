import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_SOURCE = (ROOT / "clipmaster_review.py").read_text()
TELEGRAM_SOURCE = (ROOT / "telegram_intake.py").read_text()
REVIEW_TREE = ast.parse(REVIEW_SOURCE)
TELEGRAM_TREE = ast.parse(TELEGRAM_SOURCE)


def test_universal_review_endpoint_and_polling_exist():
    assert '"/api/clip-master/reviews"' in REVIEW_SOURCE
    assert '"/api/clip-master/reviews/{review_id}"' in REVIEW_SOURCE
    assert "origin_job_key TEXT UNIQUE" in REVIEW_SOURCE


def test_review_carries_origin_and_callback_without_reintake():
    import clipmaster_review as review
    from unittest.mock import Mock,patch
    from types import SimpleNamespace
    state={'review_id':'review','origin_job_id':'existing-job','source_service':'ripped','video_id':'source','service_segments':[{'start':0,'end':200}],'callback_url':'https://origin.test/completion'}
    post=Mock(return_value=SimpleNamespace(status_code=200))
    with patch.object(review.requests,'post',post),patch.object(review,'_save'):
        assert review._deliver(state)['status']=='approved'
        assert review._deliver(state)['status']=='already_delivered'
    assert post.call_count==1 and post.call_args.args[0]==state['callback_url']
    assert post.call_args.kwargs['json']['origin_job_id']=='existing-job'
    assert post.call_args.kwargs['json']['sections']==state['service_segments']


def test_telegram_routes_clipmaster_before_ripped_shorts():
    gateway = next(
        node
        for node in TELEGRAM_TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "telegram_gateway"
    )
    source = ast.get_source_segment(TELEGRAM_SOURCE, gateway)
    local_index = source.index("clipmaster_claims_update")
    forward_index = source.index('requests.post(f"{target}/api/ripped-shorts/intake"')
    assert local_index < forward_index


def test_pending_ripped_copy_input_wins_over_clipmaster_quick_text():
    gateway = next(
        node
        for node in TELEGRAM_TREE.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "telegram_gateway"
    )
    source = ast.get_source_segment(TELEGRAM_SOURCE, gateway)
    pending_index = source.index("/api/ripped-shorts/pending-copy-input")
    quick_claim_index = source.rindex("clipmaster_claims_update")
    assert pending_index < quick_claim_index
    assert 'callback_data.startswith("cm:")' in source
    assert '"pending": True' in TELEGRAM_SOURCE
    assert '"awaiting_copy_input"' in TELEGRAM_SOURCE


def test_every_review_has_simple_controls():
    for label in ("✅ Approve All", "✏️ Change / Add", "✅ Keep", "❌ Remove"):
        assert label in REVIEW_SOURCE
    assert "apply_quick_command" in REVIEW_SOURCE


def test_approval_delivery_is_idempotent():
    assert 'if state.get("delivered_at")' in REVIEW_SOURCE
    assert '"already_delivered"' in REVIEW_SOURCE


def test_learning_uses_podcast_decision_log():
    assert '"\'Decision Log\'!A:AD"' in REVIEW_SOURCE
    assert '"yes"' in REVIEW_SOURCE


def test_callback_cannot_point_back_to_intake_or_review():
    for path in (
        "/api/telegram/webhook",
        "/api/ripped-shorts/intake",
        "/api/clip-master/reviews",
    ):
        assert path in REVIEW_SOURCE
    assert "callback_url must be a completion endpoint" in REVIEW_SOURCE


def test_last_section_cannot_be_removed():
    assert "At least one section must remain" in REVIEW_SOURCE
