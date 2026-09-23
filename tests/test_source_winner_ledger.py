from unittest.mock import MagicMock

import workflow_reliability as ledger


def test_record_source_winner_updates_latest_matching_video(monkeypatch):
    sheets = MagicMock()
    sheets.spreadsheets().get().execute.return_value = {
        "sheets": [{"properties": {"title": ledger.WORKFLOW_JOBS_TAB}}]
    }
    values = sheets.spreadsheets().values()
    values.get().execute.side_effect = [
        {"values": [ledger.HEADERS]},
        {"values": [
            ledger.HEADERS,
            ["job-old", "", "abc123"],
            ["job-new", "", "abc123"],
        ]},
    ]
    monkeypatch.setattr(ledger, "_services", lambda: (None, None, sheets))

    assert ledger.record_source_winner(
        "sheet-id", "abc123", "rapidapi", "rapidapi_video_download", 123456
    )

    update = values.update.call_args
    assert update.kwargs["range"] == "'Workflow Jobs'!S3:V3"
    row = update.kwargs["body"]["values"][0]
    assert row[0] == "rapidapi"
    assert row[1] == "rapidapi_video_download"
    assert row[3] == 123456


def test_source_winner_columns_are_in_workflow_ledger():
    assert ledger.HEADERS[-4:] == [
        "source_provider",
        "source_profile",
        "source_acquired_at",
        "source_bytes",
    ]
