import os
from typing import Any, Dict, List, Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build

GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL")
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")
GOOGLE_PRIVATE_KEY_ID = os.getenv("GOOGLE_PRIVATE_KEY_ID")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


def _credentials(scopes: List[str]):
    if GOOGLE_CREDENTIALS:
        import json
        info = json.loads(GOOGLE_CREDENTIALS)
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)
    if GOOGLE_SERVICE_ACCOUNT_FILE:
        return service_account.Credentials.from_service_account_file(GOOGLE_SERVICE_ACCOUNT_FILE, scopes=scopes)
    if GOOGLE_CLIENT_EMAIL and GOOGLE_PRIVATE_KEY:
        private_key = GOOGLE_PRIVATE_KEY.replace("\\n", "\n")
        info = {
            "type": "service_account",
            "client_email": GOOGLE_CLIENT_EMAIL,
            "private_key": private_key,
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        if GOOGLE_PRIVATE_KEY_ID:
            info["private_key_id"] = GOOGLE_PRIVATE_KEY_ID
        if GOOGLE_PROJECT_ID:
            info["project_id"] = GOOGLE_PROJECT_ID
        if GOOGLE_CLIENT_ID:
            info["client_id"] = GOOGLE_CLIENT_ID
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)
    raise RuntimeError("Google credentials not configured")


def sheets_service():
    creds = _credentials(["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def normalize_header(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def get_rows(sheet_id: str, tab_name: str, cell_range: str = "A1:Z1000") -> List[Dict[str, Any]]:
    service = sheets_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range=f"'{tab_name}'!{cell_range}",
    ).execute()
    values = result.get("values", [])
    if not values:
        return []
    headers = [normalize_header(h) for h in values[0]]
    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(values[1:], start=2):
        item = {headers[i]: row[i] if i < len(row) else "" for i in range(len(headers))}
        item["_row_number"] = index
        rows.append(item)
    return rows


def find_show_config(sheet_id: str, playlist_id: Optional[str], show_id: Optional[str] = None) -> Dict[str, Any]:
    rows = get_rows(sheet_id, "Show Config", "A1:Z1000")
    playlist_id = (playlist_id or "").strip()
    show_id = (show_id or "").strip().upper()
    active_rows = [r for r in rows if str(r.get("active", "TRUE")).strip().upper() != "FALSE"]
    for row in active_rows:
        if show_id and str(row.get("show_id", "")).strip().upper() == show_id:
            return row
    for row in active_rows:
        candidates = [
            row.get("youtube_channel_id", ""),
            row.get("playlist_id", ""),
            row.get("youtube_playlist_id", ""),
        ]
        if playlist_id and any(str(c).strip() == playlist_id for c in candidates):
            return row
    raise RuntimeError(f"No active Show Config row matched playlist_id={playlist_id!r} show_id={show_id!r}")


def get_or_create_headers(sheet_id: str, tab_name: str, desired_headers: List[str]) -> List[str]:
    service = sheets_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range=f"'{tab_name}'!1:1",
    ).execute()
    headers = result.get("values", [[]])[0]
    if not headers:
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"'{tab_name}'!A1",
            valueInputOption="RAW",
            body={"values": [desired_headers]},
        ).execute()
        return desired_headers
    changed = False
    for header in desired_headers:
        if header not in headers:
            headers.append(header)
            changed = True
    if changed:
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=f"'{tab_name}'!A1",
            valueInputOption="RAW",
            body={"values": [headers]},
        ).execute()
    return headers


def append_queue_row(sheet_id: str, queue_row: Dict[str, Any], tab_name: str = "Queue") -> Dict[str, Any]:
    headers = get_or_create_headers(sheet_id, tab_name, list(queue_row.keys()))
    values = [[queue_row.get(header, "") for header in headers]]
    result = sheets_service().spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=f"'{tab_name}'!A1",
        valueInputOption="RAW",
        insertDataOption="INSERT_ROWS",
        body={"values": values},
    ).execute()
    return {"updated_range": result.get("updates", {}).get("updatedRange"), "headers": headers}
