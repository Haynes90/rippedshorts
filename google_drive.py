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


def _export_google_doc_text_via_drive(doc_id: str) -> str:
    """Read a native Google Doc through Drive export when Docs API access misbehaves."""
    service = drive_service()
    request = service.files().export_media(
        fileId=doc_id,
        mimeType="text/plain",
    )
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buffer.getvalue().decode("utf-8", errors="replace").strip()


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

    # Prefer Drive export for prompt documents. The same service account already relies
    # on Drive for source/media workflows, and Drive export avoids recurring Docs API
    # permission failures seen for otherwise-accessible shared native Docs.
    drive_error = None
    try:
        exported = _export_google_doc_text_via_drive(doc_id)
        if exported:
            print(
                f"GOOGLE_DOC_READ path=drive_export status=success doc_id={doc_id}",
                flush=True,
            )
            return exported
    except Exception as exc:
        drive_error = exc
        print(
            f"GOOGLE_DOC_READ path=drive_export status=failed doc_id={doc_id} "
            f"error={type(exc).__name__}: {exc}",
            flush=True,
        )

    try:
        doc = docs_service().documents().get(documentId=doc_id).execute()
        parts = []
        for item in doc.get("body", {}).get("content", []):
            for element in item.get("paragraph", {}).get("elements", []):
                text_run = element.get("textRun")
                if text_run:
                    parts.append(text_run.get("content", ""))
        text = "".join(parts).strip()
        if text:
            print(
                f"GOOGLE_DOC_READ path=docs_api status=success doc_id={doc_id}",
                flush=True,
            )
            return text
    except Exception as docs_exc:
        if drive_error is not None:
            raise RuntimeError(
                f"Drive export failed ({drive_error}); "
                f"Google Docs read also failed ({docs_exc})"
            ) from docs_exc
        raise

    if drive_error is not None:
        raise RuntimeError(
            f"Google Doc {doc_id} was empty through Docs API after Drive export failed "
            f"({drive_error})"
        )
    raise RuntimeError(f"Google Doc {doc_id} was empty through both Drive and Docs APIs")


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


def retain_verified_clip(service, path, metadata, clip, create):
    """Retain one verified revision; reconcile uncertain outcomes before replay.

    Uses existing immutable evidence, not a new database or expiring retry lease.
    Never selects, approves, schedules, deletes or publishes media.
    """
    import hashlib
    from production_intelligence import database
    from render_recovery import _get, _put, digest, RenderReviewRequired, PROOF
    path = Path(path)
    with database() as db:
        spec = _get(db, clip.get('boundary_specification_id'), 'boundary_specification')
    binding = (spec or {}).get('binding') or {}
    if (not spec or spec.get('status') != 'VALIDATED'
            or digest(spec) != clip.get('boundary_specification_fingerprint')
            or clip.get('boundary_request_id') != binding.get('request_id')
            or clip.get('boundary_asset_id') != binding.get('asset_id')
            or [clip.get('start'), clip.get('end')] != [binding.get('start'), binding.get('end')]
            or not all((clip.get('completion_check') or {}).get(k) is True for k in PROOF)
            or (clip.get('boundary_export_evidence') or {}).get('full_decode') is not True):
        raise RenderReviewRequired('Verified exact-revision media is required before retaining a preview')
    if not path.is_file() or path.stat().st_size <= 0 or len(metadata.get('parents') or []) != 1:
        raise RenderReviewRequired('Available output and one retained destination folder are required')
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    with path.open('rb') as stream:
        md5 = hashlib.file_digest(stream, 'md5').hexdigest()
    expected = dict(specification_id=clip['boundary_specification_id'],
        specification_fingerprint=digest(spec), asset_id=binding['asset_id'],
        source_sha256=binding['source_sha256'], output_sha256=sha, md5=md5,
        bytes=path.stat().st_size, folder_id=metadata['parents'][0])
    token = digest({k:expected[k] for k in ('specification_id', 'specification_fingerprint',
                                         'asset_id', 'source_sha256', 'folder_id')})
    key = 'retained-media:' + token
    with database() as db:
        receipt = _get(db, key+':success', 'retained_media')
        prior = _get(db, key+':intent', 'media_upload_intent')
        if (prior and prior != expected) or (receipt and receipt['expected'] != expected):
            raise RenderReviewRequired('Different output for this retained revision; reuse or review existing media')

    def verified(remote):
        if (not remote.get('id') or remote.get('trashed')
                or remote.get('mimeType') != 'video/mp4'
                or str(remote.get('size')) != str(expected['bytes'])
                or remote.get('md5Checksum') != md5
                or expected['folder_id'] not in (remote.get('parents') or [])
                or (remote.get('appProperties') or {}).get('ripped_media_receipt') != token):
            raise RenderReviewRequired('Retained media identity changed; reconcile without another upload')
        result = dict(clip_id=remote['id'],
            clip_url=remote.get('webViewLink') or 'https://drive.google.com/file/d/'+remote['id']+'/view',
            folder_id=expected['folder_id'], media_receipt_id=key+':success')
        with database() as db:
            existing = _get(db, key+':success', 'retained_media')
            if existing:
                if existing['remote']['clip_id'] != result['clip_id']:
                    raise RenderReviewRequired('Conflicting retained media receipt; review required')
                return existing['remote']
            _put(db, 'retained_media', key+':success', dict(expected=expected, remote=result))
        return result

    fields = 'id,size,mimeType,md5Checksum,trashed,parents,appProperties,webViewLink'
    if receipt:
        return verified(service.files().get(fileId=receipt['remote']['clip_id'],
            fields=fields, supportsAllDrives=True).execute())
    found = service.files().list(q="trashed=false and appProperties has { key='ripped_media_receipt' and value='"+token+"' }",
        spaces='drive', fields='files('+fields+'),nextPageToken', pageSize=2,
        includeItemsFromAllDrives=True, supportsAllDrives=True).execute()
    matches = found.get('files') or []
    if len(matches) > 1 or found.get('nextPageToken'):
        raise RenderReviewRequired('Multiple retained media matches; reconcile without uploading')
    if matches:
        return verified(matches[0])
    if prior:
        raise RenderReviewRequired('Prior upload outcome uncertain; review before another upload')
    # Read-only API failure before this claim is safe to retry: creation has not
    # begun. Atomically recheck after listing to prevent concurrent creators.
    with database() as db:
        db.execute('BEGIN IMMEDIATE')
        late_receipt = _get(db, key+':success', 'retained_media')
        if late_receipt and late_receipt['expected'] != expected:
            raise RenderReviewRequired('Different output for this retained revision')
        if not late_receipt:
            if _get(db, key+':intent', 'media_upload_intent'):
                raise RenderReviewRequired('Prior upload outcome uncertain; review before another upload')
            _put(db, 'media_upload_intent', key+':intent', expected)
    if late_receipt:
        return verified(service.files().get(fileId=late_receipt['remote']['clip_id'],
            fields=fields, supportsAllDrives=True).execute())
    # Intent survives timeout/interruption and receipt-write failure. Later calls
    # only reconcile; an empty listing is not proof an uncertain upload failed.
    uploaded = create({**metadata, 'appProperties':{'ripped_media_receipt':token}})
    return verified(service.files().get(fileId=uploaded['id'], fields=fields,
                                        supportsAllDrives=True).execute())
