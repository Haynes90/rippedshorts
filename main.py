@app.post("/analyze-and-clip")
async def analyze_and_clip(request: Request, background_tasks: BackgroundTasks):

    data = await request.json()

    raw_segments = data.get("segments_json")
    normalized_segments = None

    # ---------------------------------------------------
    # CASE 1: Already a correct list
    # ---------------------------------------------------
    if isinstance(raw_segments, list):
        normalized_segments = raw_segments

    # ---------------------------------------------------
    # CASE 2: Zapier sends {"segments_array": [...]}
    # ---------------------------------------------------
    elif isinstance(raw_segments, dict) and "segments_array" in raw_segments:
        normalized_segments = raw_segments["segments_array"]

    # ---------------------------------------------------
    # CASE 3: Zapier sends {"0": {...}, "1": {...}}
    # ---------------------------------------------------
    elif isinstance(raw_segments, dict):
        numeric_items = []
        for key, value in raw_segments.items():
            try:
                int(key)  # numeric key
                numeric_items.append(value)
            except:
                continue

        if numeric_items:
            # Sort by numeric key to retain order
            numeric_items = [v for k, v in sorted(
                ((int(k), v) for k, v in raw_segments.items() if k.isdigit()),
                key=lambda x: x[0]
            )]
            normalized_segments = [v for _, v in numeric_items]

    # ---------------------------------------------------
    # CASE 4: Stringified JSON
    # ---------------------------------------------------
    elif isinstance(raw_segments, str):
        try:
            possible = json.loads(raw_segments)
            if isinstance(possible, list):
                normalized_segments = possible
            elif isinstance(possible, dict) and "segments_array" in possible:
                normalized_segments = possible["segments_array"]
        except:
            pass

    # ---------------------------------------------------
    # FINAL CHECK
    # ---------------------------------------------------
    if not normalized_segments or not isinstance(normalized_segments, list):
        return JSONResponse(
            {"error": "segments_json must be a JSON array"},
            status_code=400
        )

    # ---------------------------------------------------
    # Extract fields
    # ---------------------------------------------------
    video_total_len_sec = data["video_total_len_sec"]
    min_clip_len_sec = data["min_clip_len_sec"]
    max_clip_len_sec = data["max_clip_len_sec"]
    max_clip_count = data["max_clip_count"]
    drive_file_id = data["drive_file_id"]
    folder_id = data["folder_id"]
    video_t_
