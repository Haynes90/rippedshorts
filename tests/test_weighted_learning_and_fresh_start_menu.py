from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
GATE=(ROOT/"schedule_route_gate.py").read_text(encoding="utf-8")
INTAKE=(ROOT/"telegram_intake.py").read_text(encoding="utf-8")


def test_reviewed_shorts_menu_offers_fresh_start():
    assert "🧼 Fresh Start — Keep Learning" in GATE
    assert "rs:fresh_start:{request_id}" in GATE


def test_short_learning_uses_weighted_wins_and_losses():
    from acceptance_fixtures import learning_rows
    with learning_rows() as (tg,_):prompt=tg._short_learning_prompt('source')
    assert 'useful: +' in prompt and 'weak: -' in prompt
    assert 'weight=6.00' in prompt and 'weight=2.00' in prompt
    assert 'Learn the qualities, not exact wording or timestamps' in prompt


def test_complete_thought_validation_checks_opening_and_ending():
    assert "complete_start = bool(" in INTAKE
    assert "and complete_start" in INTAKE
    assert "and complete_end" in INTAKE
    assert "incomplete opening/ending thought boundary" in INTAKE


def test_prompt_requires_complete_standalone_thought():
    assert "A candidate must be a complete standalone thought" in INTAKE
    assert "Never cut in mid-sentence" in INTAKE
