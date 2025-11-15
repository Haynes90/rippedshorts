def upload_to_drive(file_path, title):
    try:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(os.environ["GOOGLE_CREDENTIALS"]),
            scopes=["https://www.googleapis.com/auth/drive"]
        )

        drive = build("drive", "v3", credentials=creds)

        # This must be the Shared Drive folder ID
        folder_id = os.environ.get("DRIVE_FOLDER_ID")

        file_metadata = {
            "name": title + ".mp4",
            "parents": [folder_id]
        }

        media = MediaFileUpload(file_path, mimetype="video/mp4", resumable=True)

        request = drive.files().create(
            body=file_metadata,
            media_body=media,
            fields="id, webViewLink",
            supportsAllDrives=True      # REQUIRED for Shared Drives
        )

        response = None
        while response is None:
            status, response = request.next_chunk(num_retries=5)
            if status:
                print(f"Uploaded {int(status.progress() * 100)}%")

        return response

    except Exception as e:
        print("Upload failed:", e)
        raise
