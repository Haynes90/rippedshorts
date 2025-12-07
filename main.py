# --------- /ANALYZE-AND-CLIP ----------
@app.post("/analyze-and-clip")
async def analyze_and_clip(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()

    # --------------------------------------------------------
    # NORMALIZE segments_json FROM ANY ZAPIER FORMAT
    # --------------------------------------------------------
    raw = data.get("segments_json")

    if raw is None:
        return JSONResponse({"error": "missing segments_json"}, status_code=400)

    segments_json = None

    # Case 1: Already a list (correct)
    if isinstance(raw, list):
        segments_json = raw

    # Case 2: Zapier sends a string containing JSON
    elif isinstance(raw, str):
        try:
            segments_json = json.loads(raw)
        except Exception:
            return JSONResponse(
                {"error": "segments_json string could not be parsed to JSON"},
                status_code=400
            )

    # Case 3: Zapier sends an object like {"segments_array":[...]}
    elif isinstance(raw, dict):
        # Accept the key "segments_array"
        if "segments_array" in raw and isinstance(raw["segments_array"], list):
            segments_json = raw["segments_array"]

        # Accept the key "segments" if present
        elif "segments" in raw and isinstance(raw["segments"], list):
            segments_json = raw["segments"]

        else:
            return JSONResponse(
                {"error": "segments_json dict not in acceptable format"},
                status_code=400
            )

    # Still nothing valid?
    if not isinstance(segments_json, list):
        return JSONResponse(
            {"error": "segments_json must be (list | JSON string | dict containing list)"},
            status_code=400
        )

    # --------------------------------------------------------
    # NORMALIZE NUMERIC FIELDS (Zapier often sends strings)
    # --------------------------------------------------------
    def to_float(x, name):
        try:
            return float(x)
        except:
            raise HTTPException(status_code=400, detail=f"{name} must be numeric")

    video_total_len_sec = to_float(data.get("video_total_len_sec"), "video_total_len_sec")
    min_clip_len_sec    = to_float(data.get("min_clip_len_sec"), "min_clip_len_sec")
    max_clip_len_sec    = to_float(data.get("max_clip_len_sec"), "max_clip_len_sec")
    max_clip_count      = int(float(data.get("max_clip_count")))  # cast safely

    drive_file_id = str(data.get("drive_file_id"))
    folder_id     = str(data.get("folder_id"))
    video_title   = data.get("video_title") or "video"
    callback_url  = data.get("callback_url")

    # --------------------------------------------------------
    # BUILD PROMPT FOR OPENAI
    # --------------------------------------------------------
    prompt = f"""
You are an expert short-form editor.
Choose up to {max_clip_count} clips.

Rules:
- segments_json (provided as JSON) has `text`, `offset`, `duration`
- Total video length = {video_total_len_sec} seconds
- Each clip must be between {min_clip_len_sec} and {max_clip_len_sec} seconds
- Prefer hooks, strong educational beats, clean takeaways
- Avoid intros/outros
Return ONLY valid JSON EXACTLY matching:
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

    # --------------------------------------------------------
    # CALL OPENAI WITH RETRIES
    # --------------------------------------------------------
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        return {"error": "Missing OPENAI_API_KEY on server"}

    import time
    retries = 4
    last_err = None
    result = None

    for attempt in range(retries):
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
            result = json.loads(r.json()["choices"][0]["message"]["content"])
            break

        except Exception as e:
            last_err = str(e)
            time.sleep(1.5)

    if result is None:
        return {"error": f"OpenAI failed: {last_err}"}

    clips = result.get("clips", [])
    if not clips:
        return {"error": "OpenAI returned no usable clips"}

    # --------------------------------------------------------
    # FORMAT CLIPS FOR INTERNAL PROCESSOR
    # --------------------------------------------------------
    formatted = []
    for i, clip in enumerate(clips, start=1):
        formatted.append({
            "index": i,
            "start": float(clip["start_sec"]),
            "duration": float(clip["duration_sec"]),
            "name": f"{video_title}_clip_{i}"
        })

    # --------------------------------------------------------
    # TRIGGER BACKGROUND CLIPPING JOB
    # --------------------------------------------------------
    payload = {
        "drive_file_id": drive_file_id,
        "folder_id": folder_id,
        "video_title": video_title,
        "clips": formatted,
        "callback_url": callback_url
    }

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {"job_id": job_id, "status": "queued", "kind": "auto_clips"}

    background_tasks.add_task(process_clip_job, job_id, payload)

    return {
        "status": "queued",
        "job_id": job_id,
        "clips_found": len(formatted)
    }
