import copy

from telegram_quick_edits import apply_quick_command, is_quick_command


def state():
    return {
        "service_segments": [
            {"start": 60, "end": 180, "title": "Sermon"},
            {"start": 200, "end": 260, "title": "Prayer"},
        ]
    }


def test_options_are_real_commands():
    for command in [
        "Show options",
        "Show sections",
        "Approve all",
        "Remove 2",
        "Add 04:30–05:10",
        "Start 10 seconds earlier",
        "End 10 seconds later",
        "Undo",
    ]:
        assert is_quick_command(command)


def test_add_remove_and_undo():
    current = state()
    added = apply_quick_command(current, "Add 04:30–05:10")
    assert added["changed"]
    assert len(current["service_segments"]) == 3
    removed = apply_quick_command(current, "Remove 2")
    assert removed["changed"]
    assert len(current["service_segments"]) == 2
    undone = apply_quick_command(current, "Undo")
    assert undone["changed"]
    assert len(current["service_segments"]) == 3


def test_boundary_adjustments():
    current = state()
    apply_quick_command(current, "Start 30 seconds earlier")
    apply_quick_command(current, "End 20 seconds later")
    assert current["service_segments"][0]["start"] == 30
    assert current["service_segments"][-1]["end"] == 280


def test_approve_all():
    current = state()
    apply_quick_command(current, "Approve all")
    assert all(item["review_status"] == "approved" for item in current["service_segments"])
