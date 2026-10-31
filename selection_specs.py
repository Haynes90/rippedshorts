"""Reuse the owner's editable 16:9 specification with a reviewed packaged fallback."""
import os
import math
from pathlib import Path
import threading
import time

REFERENCE='1QyDKGKJscc107U38QsWtLP6fhqv8VPZhdoSwIC-bQdM'
_lock=threading.Lock()
_cache=None
_at=0.
_disabled=False


def highlight_framework():
    global _cache,_at,_disabled
    with _lock:
        if _cache is not None and (time.monotonic()-_at<600 or _disabled):
            return _cache
        reference=os.getenv('RIPPED_16X9_SELECTION_DOC',REFERENCE).strip()
        try:
            if not reference:
                raise ValueError('No editable highlight reference configured')
            from google_drive import read_google_doc_text
            text=read_google_doc_text(reference).strip()
            if not text:
                raise ValueError('Blank highlight specification')
        except Exception as exc:
            # Do not repeatedly request a document that this service cannot read.
            _disabled=any(t in str(exc).lower() for t in ('403','404','permission','forbidden','not found'))
            text=(Path(__file__).parent/'ripped_shorts_16x9_framework.txt').read_text(encoding='utf-8')
        _cache,_at=text,time.monotonic()
        return text


def selector_result(response):
    """Accept the approved Doc's clips/score_total names without losing its fields."""
    if not isinstance(response,dict):
        raise ValueError('Selector must return a JSON object')
    segments=response.get('segments',response.get('clips'))
    if not isinstance(segments,list):
        raise ValueError('Selector response omitted its candidate list')
    result={**response,'segments':[]}
    for rank,candidate in enumerate(segments,1):
        if not isinstance(candidate,dict):
            raise ValueError('Selector candidate must be an object')
        score=candidate.get('score',candidate.get('score_total'))
        if score is not None and (not math.isfinite(float(score)) or not 0<=float(score)<=100):
            raise ValueError('Candidate score must be finite and between zero and 100')
        result['segments'].append({**candidate,
            'score':float(score) if score is not None else 0,
            'ai_score_available':score is not None,'original_ai_rank':rank,
            'reason':candidate.get('reason',candidate.get('why_it_will_work',''))})
    return result
