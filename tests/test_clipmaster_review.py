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
    assert '"event": "clipmaster_review_completed"' in REVIEW_SOURCE
    assert '"origin_job_id": state.get("origin_job_id")' in REVIEW_SOURCE
    assert "callback_url" in REVIEW_SOURCE
    assert "/api/ripped-shorts/intake" not in REVIEW_SOURCE


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
