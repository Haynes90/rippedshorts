import json
import os
from pathlib import Path
from typing import Optional
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io

GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS")
GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
GOOGLE_CLIENT_EMAIL = os.getenv("GOOGLE_CLIENT_EMAIL")
GOOGLE_PRIVATE_KEY = os.getenv("GOOGLE_PRIVATE_KEY")
GOOGLE_PRIVATE_KEY_ID = os.getenv("GOOGLE_PRIVATE_KEY_ID")
GOOGLE_PROJECT_ID = os.getenv("GOOGLE_PROJECT_ID")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")


def _credentials(scopes):
    if GOOGLE_CREDENTIALS:
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


def drive_service():
    creds = _credentials(["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/documents"])
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def docs_service():
    creds = _credentials(["https://www.googleapis.com/auth/documents", "https://www.googleapis.com/auth/drive"])
    return build("docs", "v1", credentials=creds, cache_discovery=False)


def extract_doc_id(value: str) -> Optional[str]:
    value = (value or "").strip()
    if not value:
        return None
    if "/document/d/" in value:
        return value.split("/document/d/", 1)[1].split("/", 1)[0]
    if len(value) > 20 and " " not in value and "/" not in value:
        return value
    return None


def find_file_by_name(name: str, mime_type: Optional[str] = None) -> Optional[dict]:
    service = drive_service()
    safe_name = name.replace("'", "\\'")
    query = f"name = '{safe_name}' and trashed = false"
    if mime_type:
        query += f" and mimeType = '{mime_type}'"
    result = service.files().list(
        q=query,
        fields="files(id,name,mimeType,webViewLink,webContentLink,size)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        pageSize=5,
    ).execute()
    files = result.get("files", [])
    return files[0] if files else None


def read_google_doc_text(doc_id_or_url_or_name: str) -> str:
    value = (doc_id_or_url_or_name or "").strip()
    if not value:
        raise RuntimeError("Prompt document value is blank")
    doc_id = extract_doc_id(value)
    if not doc_id:
        found = find_file_by_name(value, "application/vnd.google-apps.document")
        if not found:
            raise RuntimeError(f"Could not find Google Doc named {value!r}")
        doc_id = found["id"]
    doc = docs_service().documents().get(documentId=doc_id).execute()
    parts = []
    for item in doc.get("body", {}).get("content", []):
        for element in item.get("paragraph", {}).get("elements", []):
            text_run = element.get("textRun")
            if text_run:
                parts.append(text_run.get("content", ""))
    return "".join(parts).strip()


def download_drive_file(file_id: str, destination: Path) -> Path:
    service = drive_service()
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return destination


def direct_download_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}&export=download"
