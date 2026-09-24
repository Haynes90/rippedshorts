from pathlib import Path

SOURCE=(Path(__file__).resolve().parents[1]/"google_drive.py").read_text(encoding="utf-8")


def test_google_doc_reader_falls_back_to_drive_export():
    assert "def _export_google_doc_text_via_drive(" in SOURCE
    assert 'service.files().export_media(' in SOURCE
    assert 'mimeType="text/plain"' in SOURCE
    assert "GOOGLE_DOC_DRIVE_EXPORT_FALLBACK success" in SOURCE


def test_docs_api_failure_is_not_terminal_if_drive_export_works():
    reader=SOURCE.split("def read_google_doc_text(",1)[1]
    assert "docs_error = None" in reader
    assert "_export_google_doc_text_via_drive(doc_id)" in reader
    assert "Drive export fallback also failed" in reader
