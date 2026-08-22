import json
import os
import requests
from typing import Any, Dict

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
METADATA_MODEL = os.getenv("METADATA_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
METADATA_TRANSCRIPT_MAX_CHARS = int(os.getenv("METADATA_TRANSCRIPT_MAX_CHARS", "120000"))


def _extract_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise RuntimeError(f"OpenAI metadata response was not valid JSON: {raw[:1000]}")


def generate_episode_metadata(prompt_text: str, transcript_text: str, show_config: Dict[str, Any], source_title: str = "") -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not configured")
    transcript_text = (transcript_text or "")[:METADATA_TRANSCRIPT_MAX_CHARS]
    show_context = json.dumps(show_config, ensure_ascii=False, indent=2)
    user_prompt = f"""
{prompt_text}

Return valid JSON only using this exact structure:
{{
  "episode_title": "",
  "episode_description": "",
  "question_of_the_day": "",
  "keywords": [],
  "chapters": []
}}

SOURCE_TITLE:
{source_title}

SHOW_CONFIG:
{show_context}

TRANSCRIPT:
{transcript_text}
""".strip()
    payload = {
        "model": METADATA_MODEL,
        "messages": [
            {"role": "system", "content": "You create podcast episode metadata and return strict JSON only."},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=(10, 240),
    )
    if response.status_code != 200:
        raise RuntimeError(f"OpenAI metadata error ({response.status_code}): {response.text}")
    content = response.json()["choices"][0]["message"]["content"]
    data = _extract_json(content)
    if not data.get("episode_title"):
        raise RuntimeError(f"OpenAI metadata missing episode_title: {data}")
    if not data.get("episode_description"):
        raise RuntimeError(f"OpenAI metadata missing episode_description: {data}")
    return data
