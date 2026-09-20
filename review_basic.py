"""Selected-range processing for the separate, operator-run customer pilot."""
import json
import math
import os
import subprocess
from pathlib import Path


def normalize_ranges(ranges):
    if not 1 <= len(ranges) <= 4:
        raise ValueError("Select between one and four video sections")
    merged = []
    for item in sorted(ranges, key=lambda x: float(x["start"])):
        start, end = float(item["start"]), float(item["end"])
        if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
            raise ValueError("Each section needs a valid start and end time")
        if merged and start <= merged[-1]["end"]:
            merged[-1]["end"] = max(end, merged[-1]["end"])
        else:
            merged.append({"start": start, "end": end})
    if sum(x["end"] - x["start"] for x in merged) > 3600:
        raise ValueError("Selected sections exceed 60 minutes")
    return merged


def cut_section(source, output, start, end):
    # Re-encode rather than keyframe-copy so completion analysis cannot expand
    # into unselected speech. Process every selected section independently.
    result = subprocess.run([os.getenv("FFMPEG_BINARY", "ffmpeg"), "-nostdin", "-y",
        "-ss", str(start), "-i", str(source), "-t", str(end-start),
        "-map", "0:v:0", "-map", "0:a:0", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", str(output)],
        capture_output=True, text=True, timeout=3600)
    if result.returncode or not output.is_file():
        raise RuntimeError("Selected section extraction failed: " + result.stderr[-500:])


def learning_examples(db, account_id):
    examples = []
    for row in db.execute("SELECT state_json FROM telegram_requests ORDER BY created_at DESC LIMIT 500"):
        state = json.loads(row["state_json"])
        if state.get("account_id") != account_id:
            continue
        for snapshot in [state, *reversed(state.get("selection_history", []))]:
            for lane, review_key in (("result", "candidate_reviews"), ("topic_result", "topic_reviews")):
                for index, item in enumerate(snapshot.get(lane, {}).get("segments", [])):
                    review = snapshot.get(review_key, {}).get(str(index), {})
                    status = review.get("status")
                    if status not in {"rendered", "rejected", "reject"}:
                        continue
                    examples.append({"choice": "kept" if status == "rendered" else "rejected",
                        "format": "9:16" if lane == "result" else "16:9",
                        "topic": item.get("topic", ""), "reason": review.get("reason", ""),
                        "text": str(item.get("transcript", ""))[:500]})
                    if len(examples) >= 12:
                        return examples
    return examples


def process(project_id):
    from pilot_download import download_youtube
    from pilot_selection import select, VERSION
    import telegram_intake as e
    from clip_completion import media_duration, transcribe_source
    from durable_jobs import claim, finish
    if not claim(e.DB_PATH, project_id, "basic_pilot", lease_seconds=14400):
        return
    state = {}
    error = ""
    try:
        with e._LOCK, e._telegram_db() as db:
            row = db.execute("SELECT * FROM telegram_requests WHERE request_id=?", (project_id,)).fetchone()
        if not row:
            raise RuntimeError("Project not found")
        state = json.loads(row["state_json"])
        with e._LOCK, e._telegram_db() as db:
            examples = learning_examples(db, state["account_id"])
        # Once candidates exist, retry failed individual renders; never replace decisions.
        if state.get("basic_selection_complete"):
            return
        ranges = normalize_ranges(state["selected_ranges"])
        work = e.SOURCE_DIR / f"pilot-{project_id}"
        work.mkdir(parents=True, exist_ok=True)
        parsed = state["parsed"]
        state["stage"] = "source_download"
        e._save(project_id, "processing", state)
        sections_ready = all((work / f"section-{i}.mp4").is_file() and abs(media_duration(work / f"section-{i}.mp4") - (b["end"] - b["start"])) < .5 for i, b in enumerate(ranges))
        if sections_ready:
            video = None
            title = parsed.get("video_id") or "Source"
        elif parsed["source_kind"] == "youtube":
            video = download_youtube(parsed["video_id"], parsed["source_value"], work)
            title = parsed["video_id"]
        else:
            metadata = e.drive_metadata(parsed["drive_ids"][0])
            video = e.download_drive(parsed["drive_ids"][0], work / "source.mp4", metadata)
            title = metadata.get("name", "Source")
        duration = max(b["end"] for b in ranges) if sections_ready else media_duration(video)
        if any(item["end"] > duration + .05 for item in ranges):
            raise ValueError("A selected section ends after the source video")
        # Each customer project gets its own output folder; do not reuse another order.
        state["vid_title"] = f"{state['customer_reference']} - {title} - {project_id[:8]}"
        sections, transcript = [], []
        for section_index, bounds in enumerate(ranges):
            section = work / f"section-{section_index}.mp4"
            state["stage"] = f"transcribing_section_{section_index + 1}_of_{len(ranges)}"
            e._save(project_id, "processing", state)
            cache = work / f"section-{section_index}-punctuated.json"
            if not section.is_file() or abs(media_duration(section) - (bounds["end"] - bounds["start"])) >= .5:
                cut_section(video, section, bounds["start"], bounds["end"])
                cache.unlink(missing_ok=True)
            if cache.is_file():
                timed = json.loads(cache.read_text(encoding="utf-8"))
            else:
                timed = transcribe_source(section)
                cache.write_text(json.dumps(timed), encoding="utf-8")
            for line in timed:
                absolute = bounds["start"] + float(line["start"])
                transcript.append(f"[{int(absolute)//3600:02}:{int(absolute)//60%60:02}:{int(absolute)%60:02}] {line['text']}")
            sections.append({"timed": timed, "path": section, "offset": bounds["start"], "duration": bounds["end"] - bounds["start"]})
        def progress(stage):
            state["stage"] = stage
            state["transcript_text"] = "\n\n".join(transcript)
            e._save(project_id, "selecting", state)
        selection = select(sections, row["mode"], examples, work, progress)
        common = {"project_id": project_id, "account_id": state["account_id"]}
        shorts = [{**item, **common} for item in selection["shorts"]]
        highlights = [{**item, **common} for item in selection["highlights"]]
        state["source_understanding"] = selection["understanding"]
        state["selection_audit"] = {"proposals": selection["proposals"], "semantic_reviews": selection["semantic_reviews"]}
        state["selection_version"] = VERSION
        state.pop("error", None)
        state.update({"stage": "awaiting_review", "video_path": str(work / "section-0.mp4"),
            "result": {"segments": shorts}, "topic_result": {"segments": highlights},
            "candidate_reviews": {}, "topic_reviews": {}, "topic_stage": "awaiting_review" if highlights else "not_eligible",
            "transcript_text": "\n\n".join(transcript), "basic_selection_complete": True})
        e._save(project_id, "awaiting_review", state)
    except Exception as exc:
        error = str(exc)
        state.update({"stage": "error", "error": error, "retryable": True})
        e._save(project_id, "error", state)
        e.logger.exception("Basic pilot processing failed project=%s", project_id)
    finally:
        finish(e.DB_PATH, project_id, "basic_pilot", error, complete=not bool(error))
