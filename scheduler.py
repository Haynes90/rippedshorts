from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, Optional

DAY_INDEX = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def parse_publish_time(value: Any) -> tuple[int, int]:
    text = str(value or "").strip()
    if not text:
        return 7, 0
    if ":" in text:
        hour, minute = text.split(":", 1)
        return int(hour), int(float(minute))
    return int(float(text)), 0


def next_publish_datetime(show_config: Dict[str, Any], now: Optional[datetime] = None) -> str:
    timezone_name = str(show_config.get("time_zone") or "America/New_York").strip()
    tz = ZoneInfo(timezone_name)
    now_local = (now or datetime.now(tz)).astimezone(tz)
    publish_day = str(show_config.get("publish_day") or "").strip().lower()
    publish_time = show_config.get("publish_time") or "7:00"
    hour, minute = parse_publish_time(publish_time)

    if publish_day in DAY_INDEX:
        target_weekday = DAY_INDEX[publish_day]
        days_ahead = (target_weekday - now_local.weekday()) % 7
        candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=days_ahead)
        if candidate <= now_local:
            frequency_days = int(float(show_config.get("frequency_days") or 7))
            candidate += timedelta(days=frequency_days)
    else:
        frequency_days = int(float(show_config.get("frequency_days") or 7))
        candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0) + timedelta(days=frequency_days)

    return candidate.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def queue_id(show_id: str, video_id: str, scheduled_date: str) -> str:
    yymmdd = scheduled_date[:10].replace("-", "")[2:]
    return f"{yymmdd}-{show_id}-{video_id}"
