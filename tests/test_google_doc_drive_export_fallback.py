from pathlib import Path

SOURCE=(Path(__file__).resolve().parents[1]/"google_drive.py").read_text(encoding="utf-8")


def test_google_doc_reader_prefers_drive_export():
    reader=SOURCE.split("def read_google_doc_text(",1)[1].split("def download_drive_file",1)[0]
    assert reader.index("_export_google_doc_text_via_drive(doc_id)") < reader.index("docs_service().documents().get")
    assert "GOOGLE_DOC_READ path=drive_export status=success" in reader


def test_docs_api_is_secondary_fallback():
    reader=SOURCE.split("def read_google_doc_text(",1)[1].split("def download_drive_file",1)[0]
    assert "drive_error = None" in reader
    assert "GOOGLE_DOC_READ path=docs_api status=success" in reader
    assert "Drive export failed" in reader
