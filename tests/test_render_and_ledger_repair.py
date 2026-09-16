import ast
import shutil
import subprocess
import traceback
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests
import active_speaker
import workflow_reliability as ledger


def test_mixed_layouts_render_with_square_pixels(monkeypatch):
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        ffmpeg = pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    monkeypatch.setattr(active_speaker, '_detect_tracks', lambda *args: [])
    monkeypatch.setattr(active_speaker, '_audio_rms', lambda *args: [])
    monkeypatch.setattr(active_speaker, '_stable_layout_sections', lambda *args: [
        (0, 'A', .25, .75), (.2, 'STACKED', .25, .75), (.4, 'B', .25, .75)])
    graph = active_speaker.build_active_speaker_filter(Path('unused'), 0, .6, 640, 360, None)
    result = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'info',
        '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=30:duration=0.6',
        '-filter_complex', graph + ';[v]showinfo[out]', '-map', '[out]',
        '-f', 'null', '-'], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'sar:1/1 s:1080x1920' in result.stderr


@pytest.mark.parametrize('existing', [False, True])
def test_ledger_writes_all_columns(monkeypatch, existing):
    sheets = MagicMock()
    sheets.spreadsheets().get().execute.return_value = {'sheets': [{'properties': {'title': ledger.WORKFLOW_JOBS_TAB}}]}
    values = sheets.spreadsheets().values()
    values.get().execute.side_effect = [{'values': [ledger.HEADERS[:-1]]},
        {'values': [ledger.HEADERS] + ([['job']] if existing else [])}]
    monkeypatch.setattr(ledger, '_services', lambda: (None, None, sheets))
    ledger.upsert_job('sheet', 'job', 'accepted', {'stage': 'accepted'})
    writes = values.update.call_args_list + values.append.call_args_list
    assert len(writes) == 2
    for call in writes:
        assert ':R' in call.kwargs['range']
        assert len(call.kwargs['body']['values'][0]) == 18


def test_telegram_error_traceback_does_not_expose_token(monkeypatch):
    # Isolate the transport helper without booting the service or its workers.
    tree = ast.parse(Path('telegram_intake.py').read_text(encoding='utf-8'))
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_telegram_api')
    namespace = {'requests': requests, 'Any': object}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), 'telegram_helper', 'exec'), namespace)
    def fail(*args, **kwargs):
        raise requests.HTTPError('https://api.telegram.org/botTEST_SECRET/getMe')
    monkeypatch.setattr(requests, 'post', fail)
    with pytest.raises(RuntimeError) as caught:
        namespace['_telegram_api']('TEST_SECRET', 'getMe')
    rendered = ''.join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert 'TEST_SECRET' not in str(caught.value)
    assert 'HTTPError: https://' not in rendered
    assert caught.value.__suppress_context__
