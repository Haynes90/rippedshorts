"""Fast, state-aware Telegram section editing shared by Clip Master review lanes."""
from __future__ import annotations

import copy
import re
from typing import Any

_TIME = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})")
_RANGE = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:-|–|to|through)\s*(\d{1,2}:\d{2}(?::\d{2})?)", re.I)

OPTIONS_TEXT = (
    "Quick options:\n"
    "• Approve all\n"
    "• Remove 3\n"
    "• Add 24:10–27:30\n"
    "• Start 30 seconds earlier\n"
    "• End 20 seconds later\n"
    "• Undo\n"
    "• Show sections\n"
    "You can use the buttons or type these commands."
)


def clock_seconds(value: str) -> float:
    match = _TIME.fullmatch(value.strip())
    if not match:
        raise ValueError("Use MM:SS or HH:MM:SS.")
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2))
    seconds = int(match.group(3))
    if minutes > 59 when hours else False:
        raise ValueError("Invalid time.")
    if seconds > 59:
        raise ValueError("Invalid time.")
    return float(hours * 3600 + minutes * 60 + seconds)


def _fmt(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _collection(state: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(state.get("service_segments"), list):
        return "service", state["service_segments"]
    topics = (state.get("topic_result") or {}).get("segments")
    if isinstance(topics, list):
        return "topic", topics
    shorts = (state.get("result") or {}).get("segments")
    if isinstance(shorts, list):
        return "short", shorts
    raise ValueError("There are no sections ready to edit yet.")


def _bounds(item: dict[str, Any]) -> tuple[float, float]:
    start = float(item.get("start", item.get("start_seconds", 0)) or 0)
    end = item.get("end", item.get("end_seconds"))
    if end is None:
        end = start + float(item.get("duration", 0) or 0)
    return start, float(end)


def sections_text(state: dict[str, Any]) -> str:
    lane, items = _collection(state)
    if not items:
        return "No sections are currently listed."
    lines = [f"{lane.title()} sections:"]
    for index, item in enumerate(items, 1):
        start, end = _bounds(item)
        title = str(item.get("title") or item.get("display_title") or item.get("category") or f"Section {index}")
        status = str(item.get("review_status") or item.get("status") or "").strip()
        suffix = f" — {status}" if status else ""
        lines.append(f"{index}. {title} | {_fmt(start)}–{_fmt(end)}{suffix}")
    return "\n".join(lines)


def is_quick_command(text: str) -> bool:
    clean = " ".join((text or "").lower().split())
    return bool(
        clean in {"options", "show options", "what can i do", "what are my options", "show sections", "sections", "approve all", "undo"}
        or re.fullmatch(r"(?:remove|delete|reject)\s+(?:section\s+|short\s+|segment\s+)?\d+", clean)
        or re.fullmatch(r"add\s+.+", clean)
        or re.fullmatch(r"(?:start|end)\s+\d+\s+seconds?\s+(?:earlier|later)", clean)
    )


def apply_quick_command(state: dict[str, Any], text: str) -> dict[str, Any]:
    clean = " ".join((text or "").strip().split())
    lowered = clean.lower()
    if lowered in {"options", "show options", "what can i do", "what are my options"}:
        return {"changed": False, "message": OPTIONS_TEXT, "action": "options", "state": state}
    if lowered in {"show sections", "sections"}:
        return {"changed": False, "message": sections_text(state), "action": "show_sections", "state": state}

    lane, items = _collection(state)
    before = copy.deepcopy(items)

    if lowered == "undo":
        history = state.get("quick_edit_history") or []
        if not history:
            return {"changed": False, "message": "Nothing to undo.", "action": "undo_empty", "state": state}
        previous = history.pop()
        restored = copy.deepcopy(previous["segments"])
        if lane == "service":
            state["service_segments"] = restored
        elif lane == "topic":
            state.setdefault("topic_result", {})["segments"] = restored
        else:
            state.setdefault("result", {})["segments"] = restored
        state["quick_edit_history"] = history
        return {"changed": True, "message": "↩️ Last change undone.\n\n" + sections_text(state), "action": "undo", "state": state, "before": before, "after": restored, "lane": lane}

    action = ""
    if lowered == "approve all":
        for item in items:
            item["review_status"] = "approved"
        action = "approve_all"
    else:
        remove = re.fullmatch(r"(?:remove|delete|reject)\s+(?:section\s+|short\s+|segment\s+)?(\d+)", lowered)
        add = re.fullmatch(r"add\s+(.+)", clean, re.I)
        adjust = re.fullmatch(r"(start|end)\s+(\d+)\s+seconds?\s+(earlier|later)", lowered)
        if remove:
            index = int(remove.group(1)) - 1
            if index < 0 or index >= len(items):
                raise ValueError(f"Choose a number from 1 to {len(items)}.")
            items.pop(index)
            action = "remove"
        elif add:
            match = _RANGE.search(add.group(1))
            if not match:
                raise ValueError("To add a section, use: Add 24:10–27:30")
            start, end = clock_seconds(match.group(1)), clock_seconds(match.group(2))
            if end <= start:
                raise ValueError("The added section must end after it starts.")
            items.append({"start": start, "end": end, "duration": end - start, "title": "Added section", "review_status": "pending"})
            items.sort(key=lambda item: _bounds(item)[0])
            action = "add"
        elif adjust:
            edge, amount_text, direction = adjust.groups()
            if not items:
                raise ValueError("There are no sections to adjust.")
            amount = float(amount_text) * (-1 if direction == "earlier" else 1)
            target = items[0] if edge == "start" else items[-1]
            start, end = _bounds(target)
            if edge == "start":
                start = max(0, start + amount)
            else:
                end = end + amount
            if end <= start:
                raise ValueError("That change would make the section empty.")
            target["start"], target["end"], target["duration"] = start, end, end - start
            action = f"{edge}_{direction}"
        else:
            raise ValueError("That quick command is not supported. Send “Show options”.")

    state.setdefault("quick_edit_history", []).append({"action": action, "segments": before})
    state["quick_edit_history"] = state["quick_edit_history"][-20:]
    return {
        "changed": True,
        "message": "✅ Updated.\n\n" + sections_text(state),
        "action": action,
        "state": state,
        "before": before,
        "after": copy.deepcopy(items),
        "lane": lane,
    }
