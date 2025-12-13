@app.post("/analyze-and-clip")
async def analyze_and_clip(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    # -------------------------------
    # FIX: segments_json may be:
    # - a list (correct)
    # - a string containing JSON
    # - a string representation from Zapier
    # -------------------------------
    raw_segments = data.get("segments_json")

    if raw_segments is None:
        return JSONResponse({"error": "segments_json missing"}, status_code=400)

    # If string → try to parse
    if isinstance(raw_segments, str):
        try:
            segments_json = json.loads(raw_segments)
        except Exception:
            return JSONResponse(
                {"error": "segments_json is a string but not valid JSON"},
                status_code=400
            )
    else:
        segments_json = raw_segments

    # Must now be a list
    if not isinstance(segments_json, list):
        return JSONResponse({"error": "segments_json must be a JSON array"}, status_code=400)

    # -------------------------------
    # Normalize text fields safely
    # -------------------------------
    def normalize_text(t):
        if not isinstance(t, str):
            return t
        t = (
            t.replace("&quot;", '"')
             .replace("&#39;", "'")
             .replace("&gt;", ">")
             .replace("&lt;", "<")
             .replace("&amp;", "&")
        )
        return t

    for seg in segments_json:
        if isinstance(seg, dict) and "text" in seg:
            seg["text"] = normalize_text(seg["text"])

    # Extract other fields
    video_total_len_sec = data.get("video_total_len_sec")
    min_clip_len_sec = data.get("min_clip_len_sec")
    max_clip_len_sec = data.get("max_clip_len_sec")
    max_clip_count = data.get("max_clip_count")
    drive_file_id = data.get("drive_file_id")
    folder_id = data.get("folder_id")
    video_title = data.get("video_title") or "video"
    callback_url = data.get("callback_url")

    # -------------------------------
    # Build OpenAI prompt
    # -------------------------------
    prompt = f"""
You are an expert short-form editor.
Choose up to {max_clip_count} clips.

Rules:
- Use segments_json (provided separately below)
- Use only time ranges within {video_total_len_sec} seconds
- Each clip must be between {min_clip_len_sec} and {max_clip_len_sec} seconds
Return ONLY valid JSON with:
{{
  "clips": [
    {{
      "title": "string",
      "hook": "string",
      "summary": "string",
      "keywords_csv": "string",
      "start_sec": 0,
      "end_sec": 0,
      "duration_sec": 0,
      "confidence": 0.0,
      "reason": "string"
    }}
  ]
}}
"""

    import time

    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return {"error": "Missing OPENAI_API_KEY"}

    result = None
    last_err = None

    for attempt in range(4):
        try:
            r = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {"role": "system", "content": "Return ONLY valid JSON."},
                        {"role": "user", "content": prompt},
                        {"role": "user", "content": json.dumps(segments_json)}
                    ]
                },
                timeout=45
            )
            if r.status_code == 429:
                last_err = r.text
                time.sleep(1 + attempt)
                continue

            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            result = json.loads(content)
            break

        except Exception as e:
            last_err = str(e)
            time.sleep(1)

    if result is None:
        return {"error": f"OpenAI failed: {last_err}"}

    clips = result.get("clips", [])
    if not clips:
        return {"error": "OpenAI returned no clips"}

    formatted = []
    for i, clip in enumerate(clips, start=1):
        formatted.append({
            "index": i,
            "start": clip["start_sec"],
            "duration": clip["duration_sec"],
            "name": f"{video_title}_clip_{i}"
        })

    payload = {
        "drive_file_id": drive_file_id,
        "folder_id": folder_id,
        "video_title": video_title,
        "clips": formatted,
        "callback_url": callback_url
    }

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued"}

    background_tasks.add_task(process_clip_job, job_id, payload)

    return {"status": "queued", "job_id": job_id, "clips_found": len(formatted)}
