"""Resolve durable copy without letting generated snapshots erase human edits."""
from datetime import datetime, timezone


def _recorded_time(value):
    try:
        parsed = datetime.fromisoformat(str(value or "").strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).timestamp()
    except (ValueError, TypeError, OverflowError):
        return float("-inf")


def caption_checkpoints(rows, request_id):
    result, ranks = {}, {}
    fields = (
        ("social_caption", "final_caption", "ai_caption"),
        ("video_title", "final_title", "ai_title"),
        ("video_description", "final_description", "ai_description"),
        ("hashtags", "hashtags", None),
    )
    for index, row in enumerate(rows):
        if str(row.get("request_id") or "").strip() != str(request_id).strip():
            continue
        asset_id = str(row.get("asset_id") or "").strip()
        decision = str(row.get("decision") or "").strip().upper()
        if not asset_id or decision not in {"", "EDITED", "ACCEPTED"}:
            continue
        for target, final_field, ai_field in fields:
            value = str(row.get(final_field) or "")
            if not value.strip():
                continue
            ai = str(row.get(ai_field) or "") if ai_field else ""
            human = decision == "EDITED" or bool(ai.strip() and value.strip() != ai.strip())
            rank = (human, _recorded_time(row.get("recorded_at")), index)
            key = (asset_id, target)
            if key not in ranks or rank > ranks[key]:
                ranks[key] = rank
                result.setdefault(asset_id, {})[target] = value
                result[asset_id]["copy_source"] = "CAPTION_LEARNING_CHECKPOINT"
    return result
