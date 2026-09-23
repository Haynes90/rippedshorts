from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "telegram_intake.py").read_text(encoding="utf-8")


def test_framework_permission_failure_opens_circuit_breaker():
    assert "_SHORT_FRAMEWORK_LIVE_DISABLED = False" in SOURCE
    assert "permission denied once; disabling" in SOURCE
    assert '"403" in lowered' in SOURCE
    assert '"permission_denied" in lowered' in SOURCE
    assert '"does not have permission" in lowered' in SOURCE


def test_disabled_framework_skips_future_google_doc_reads():
    function = SOURCE.split("def _configured_short_framework_prompt()", 1)[1].split(
        "def _brand_short_prompt", 1
    )[0]
    assert "if _SHORT_FRAMEWORK_LIVE_DISABLED:" in function
    assert "return packaged" in function


def test_successful_live_framework_is_cached():
    assert "_SHORT_FRAMEWORK_LIVE_CACHE" in SOURCE
    assert "SHORT_FRAMEWORK_CACHE_SECONDS" in SOURCE
    assert "caching for %ss" in SOURCE
