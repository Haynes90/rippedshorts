"""Resolve durable copy without letting generated snapshots erase human edits."""
from datetime import datetime, timezone
import hashlib
import json
import re


def complete_post(draft):
    if draft.get('post_editor_version') == 1:
        return str(draft.get('post_text') or '')
    field = 'social_caption' if draft.get('asset_type') == '9:16_SHORT' or (not draft.get('asset_type') and draft.get('social_caption')) else 'video_description'
    text = str(draft.get(field) or '')
    present = {tag.casefold() for tag in re.findall(r'(?<!\w)#\w+', text, re.UNICODE)}
    missing = []
    for tag in re.findall(r'(?<!\w)#\w+', str(draft.get('hashtags') or ''), re.UNICODE):
        if tag.casefold() not in present:
            missing.append(tag); present.add(tag.casefold())
    return text + ('\n\n' if text and missing else '') + ' '.join(missing)


def editor_revision(draft):
    values = [draft.get(k) for k in ('asset_id','social_caption','video_title','video_description','hashtags','post_text','post_editor_version','web_review_saved_at','copy_status','editorial_hold')]
    return hashlib.sha256(json.dumps(values,ensure_ascii=False).encode()).hexdigest()


def transport_copy(asset, draft=None):
    """The single editor already includes its hashtags; never append them twice."""
    draft = draft or asset
    if draft.get('post_editor_version') != 1:
        return asset
    field = 'social_caption' if draft.get('asset_type') == '9:16_SHORT' else 'video_description'
    return {**asset, field:complete_post(draft), 'hashtags':'',
            'post_editor_version':1, 'post_text':complete_post(draft),
            **({'video_title':draft.get('video_title','')} if field=='video_description' else {})}


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
