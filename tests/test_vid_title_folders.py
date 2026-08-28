from unittest.mock import MagicMock

import main
import telegram_intake


def test_drive_title_folder_reuses_existing_child(monkeypatch):
    drive = MagicMock()
    drive.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "vid-title-folder", "name": "How Plants Really Grow"}]
    }
    monkeypatch.setattr(main, "DRIVE_FOLDER_ID", "ripped-shorts-root")
    monkeypatch.setattr(main, "get_google_services", lambda: (drive, None, None))

    folder_id = main._drive_title_folder("How Plants Really Grow")

    assert folder_id == "vid-title-folder"
    query = drive.files.return_value.list.call_args.kwargs["q"]
    assert "name = 'How Plants Really Grow'" in query
    assert "'ripped-shorts-root' in parents" in query


def test_drive_title_folder_creates_child_under_existing_destination(monkeypatch):
    drive = MagicMock()
    drive.files.return_value.list.return_value.execute.return_value = {"files": []}
    drive.files.return_value.create.return_value.execute.return_value = {
        "id": "new-vid-title-folder",
        "name": "Plant Science Live",
    }
    monkeypatch.setattr(main, "DRIVE_FOLDER_ID", "ripped-shorts-root")
    monkeypatch.setattr(main, "get_google_services", lambda: (drive, None, None))

    folder_id = main._drive_title_folder("Plant Science Live")

    assert folder_id == "new-vid-title-folder"
    body = drive.files.return_value.create.call_args.kwargs["body"]
    assert body == {
        "name": "Plant Science Live",
        "mimeType": "application/vnd.google-apps.folder",
        "parents": ["ripped-shorts-root"],
    }


def test_youtube_vid_title_uses_oembed_title(monkeypatch):
    response = MagicMock()
    response.json.return_value = {"title": "The Actual Vid Title"}
    monkeypatch.setattr(telegram_intake.requests, "get", lambda *args, **kwargs: response)

    assert (
        telegram_intake._youtube_vid_title("https://youtu.be/abc12345")
        == "The Actual Vid Title"
    )
    response.raise_for_status.assert_called_once()


def test_drive_vid_title_removes_video_extension_only():
    assert telegram_intake._drive_vid_title({"name": "My Video.Title.mp4"}) == "My Video.Title"
