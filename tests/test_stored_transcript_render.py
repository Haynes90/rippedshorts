"""Offline regression tests; no accounts, transcription, or production jobs."""
import ast
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parent
if not (ROOT / 'clip_completion.py').exists():
    ROOT = ROOT.parent


def load_function(filename, name, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), filename, 'exec'), namespace)
    return namespace[name]


class StoredTranscriptRenderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.output = Path(self.folder.name) / 'clip.mp4'
        self.api = Mock(side_effect=AssertionError('Rendering must not call speech services'))
        self.probe = Mock(return_value=SimpleNamespace(stdout='video\n'))
        self.duration = Mock(side_effect=[100, 5])
        self.env = dict(Path=Path, math=math, os=os, CompletionReviewRequired=RuntimeError,
                        media_duration=self.duration, _run=self.probe,
                        transcribe_window=self.api, review_thought=self.api,
                        pause_end=self.api, verify_render=self.api)
        self.render = load_function('clip_completion.py', 'render_complete_clip', self.env)
        self.renderer = Mock(side_effect=lambda *args: self.output.write_bytes(b'video fixture'))
        self.segment = {'start': 10, 'end': 15, 'transcript': 'This is a complete sentence.'}

    def test_both_aspects_use_stored_bounds_without_speech_calls(self):
        for aspect in ('9:16', '16:9'):
            with self.subTest(aspect=aspect):
                self.duration.side_effect = [100, 5]
                result = self.render(Path('silent-source.mp4'), self.segment, self.output, self.renderer, aspect)
                self.renderer.assert_called_with(Path('silent-source.mp4'), 10, 5, self.output)
                self.assertEqual(result['completion_check']['basis'], 'stored_transcript_and_media_bounds')
                self.assertFalse(result['completion_check']['audio_pause'])
        self.api.assert_not_called()

    def test_bad_bounds_fail_before_render(self):
        for start, end in ((-1, 5), (2, 1), (0, 101), (float('nan'), 4)):
            self.duration.side_effect = [100]
            with self.assertRaises(RuntimeError):
                self.render(Path('source'), {**self.segment, 'start': start, 'end': end}, self.output, self.renderer, '16:9')
        self.renderer.assert_not_called()

    def test_incomplete_transcript_does_not_render(self):
        with self.assertRaisesRegex(RuntimeError, 'mid-sentence'):
            self.render(Path('source'), {**self.segment, 'transcript': 'unfinished'}, self.output, self.renderer, '9:16')
        self.renderer.assert_not_called()

    def test_short_limit_is_enforced(self):
        with self.assertRaisesRegex(RuntimeError, '90-second'):
            self.render(Path('source'), {**self.segment, 'start': 0, 'end': 91}, self.output, self.renderer, '9:16')

    def test_wrong_export_duration_removes_output(self):
        self.duration.side_effect = [100, 2]
        with self.assertRaisesRegex(RuntimeError, 'duration differs'):
            self.render(Path('source'), self.segment, self.output, self.renderer, '9:16')
        self.assertFalse(self.output.exists())

    def test_audio_only_export_is_rejected(self):
        self.probe.return_value.stdout = ''
        with self.assertRaisesRegex(RuntimeError, 'no video stream'):
            self.render(Path('source'), self.segment, self.output, self.renderer, '9:16')
        self.assertFalse(self.output.exists())

    def test_missing_output_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'no video'):
            self.render(Path('source'), self.segment, self.output, lambda *args: None, '9:16')

    def test_preflight_accepts_punctuation_and_closing_quotes_without_audio(self):
        preflight = load_function('telegram_intake.py', '_preflight_candidates',
                                  dict(Path=Path, Any=object, re=re, logger=logging.getLogger('test')))
        for ending in ('.', '!', '?', '."', '.”', ".’", '.)'):
            clip = {'transcript': 'Complete sentence' + ending}
            self.assertEqual(preflight(Path('no-audio.mp4'), [clip]), [clip])
        self.assertEqual(preflight(Path('source'), [{'transcript': 'unfinished'}]), [])

    def test_existing_render_source_does_not_require_audio(self):
        self.output.write_bytes(b'existing source')
        recover = load_function('telegram_intake.py', '_ensure_render_source', dict(Path=Path, Any=object))
        self.assertEqual(recover('job', {'video_path': str(self.output)}), self.output)

    def test_download_fallback_is_imported(self):
        tree = ast.parse((ROOT / 'telegram_intake.py').read_text(encoding='utf-8'))
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == 'source_ingestion'
                            and any(a.name == 'download_youtube_resilient' for a in n.names) for n in tree.body))


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'FFmpeg tools required')
class RealVideoRenderTests(unittest.TestCase):
    def test_actual_renderers_with_and_without_audio(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            env = dict(Path=Path, math=math, os=os, subprocess=subprocess,
                       CompletionReviewRequired=RuntimeError)
            load_function('clip_completion.py', '_run', env)
            load_function('clip_completion.py', 'media_duration', env)
            render = load_function('clip_completion.py', 'render_complete_clip', env)
            render_env = dict(Path=Path, subprocess=subprocess,
                              _build_vertical_filter=lambda *args: ('crop=134:240', False))
            for with_audio in (False, True):
                source = folder / ('sound.mp4' if with_audio else 'silent.mp4')
                command = ['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i',
                           'color=c=blue:s=320x240:r=25:d=2']
                if with_audio:
                    command += ['-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', '-c:a', 'aac']
                subprocess.run(command + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(source)], check=True)
                for aspect, name in (('9:16', 'create_clip_file'), ('16:9', 'create_topic_segment_file')):
                    renderer = load_function('main.py', name, render_env)
                    output = folder / (aspect.replace(':', '-') + source.name)
                    render(source, {'start': .2, 'end': 1.2, 'transcript': 'Stored sentence.'},
                           output, renderer, aspect)
                    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                                            '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', str(output)],
                                           check=True, capture_output=True, text=True)
                    self.assertEqual('audio' in probe.stdout, with_audio)


if __name__ == '__main__':
    unittest.main()

