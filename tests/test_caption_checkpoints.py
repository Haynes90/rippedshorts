from caption_checkpoints import caption_checkpoints
import ast
from pathlib import Path


def row(**changes):
    return dict(request_id="r", asset_id="a", decision="ACCEPTED",
                recorded_at="2026-09-01T00:00:00Z", **changes)


def test_edit_beats_later_generated_snapshot():
    edited = row(final_caption="Mine", ai_caption="Hook")
    edited["decision"] = "EDITED"
    generated = row(final_caption="Hook", ai_caption="Hook")
    generated["recorded_at"] = "2026-09-30T00:00:00Z"
    assert caption_checkpoints([edited, generated], "r")["a"]["social_caption"] == "Mine"


def test_latest_edit_wins_not_last_physical_row():
    old = row(final_caption="Old")
    old["decision"] = "EDITED"
    new = dict(old, final_caption="New", recorded_at="2026-09-02T00:00:00Z")
    assert caption_checkpoints([new, old], "r")["a"]["social_caption"] == "New"


def test_empty_fields_do_not_erase_caption_or_other_metadata():
    edited = row(final_caption="  ", final_title="My title")
    edited["decision"] = "EDITED"
    result = caption_checkpoints([row(final_caption="Fallback"), edited], "r")["a"]
    assert result["social_caption"] == "Fallback"
    assert result["video_title"] == "My title"
    assert "video_description" not in result
    assert caption_checkpoints([row(final_caption=" ")], "r") == {}


def test_manual_sheet_edit_without_decision_update_is_used():
    manual = row(final_caption="My sheet edit", ai_caption="Generated")
    stale = row(final_caption="Generated", ai_caption="Generated")
    assert caption_checkpoints([manual, stale], "r")["a"]["social_caption"] == "My sheet edit"


def test_request_and_asset_isolation_and_rejected_rows():
    rows = [row(final_caption="Right"), dict(row(final_caption="Wrong"), request_id="other"),
            dict(row(final_caption="Rejected"), decision="REJECTED"),
            dict(row(final_caption="Other clip"), asset_id="b")]
    result = caption_checkpoints(rows, "r")
    assert result["a"]["social_caption"] == "Right"
    assert result["b"]["social_caption"] == "Other clip"


def test_timezones_and_missing_timestamps():
    first = row(final_caption="Later instant")
    first["recorded_at"] = "2026-09-01T01:00:00-04:00"
    second = row(final_caption="Earlier instant")
    missing = dict(second, recorded_at="invalid")
    assert caption_checkpoints([first, second, missing], "r")["a"]["social_caption"] == "Later instant"


def test_formatting_is_preserved():
    assert caption_checkpoints([row(final_caption="First\n\nSecond")], "r")["a"]["social_caption"] == "First\n\nSecond"


def test_actual_sheet_reader_uses_resolver():
    source = Path(__file__).resolve().parents[1] / "telegram_intake.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_caption_learning_checkpoint_map")
    namespace = dict(caption_checkpoints=caption_checkpoints, RIPPED_LOG_SHEET_ID="sheet",
                     get_rows=lambda *_: [row(final_caption="Saved")])
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["_caption_learning_checkpoint_map"]("r")["a"]["social_caption"] == "Saved"


def test_legacy_handoff_applies_sheet_copy_after_generation():
    source = Path(__file__).resolve().parents[1] / "telegram_intake.py"
    text = source.read_text(encoding="utf-8")
    import textwrap
    start = text.index("    reviewed_copy = {", text.index("def _handoff_shorts_to_schedule_master"))
    end = text.index("    payload = {", start)
    namespace = dict(
        state={}, assets=[{"asset_id": "a"}], request_id="r",
        _state_vid_title=lambda state: "Title",
        _generate_schedule_copy=lambda assets, *args: [{"asset_id": "a", "social_caption": "Generated"}],
        _caption_learning_checkpoint_map=lambda request: {"a": {"social_caption": "Saved sheet wording"}},
    )
    exec(textwrap.dedent(text[start:end]), namespace)
    assert namespace["assets"][0]["social_caption"] == "Saved sheet wording"
