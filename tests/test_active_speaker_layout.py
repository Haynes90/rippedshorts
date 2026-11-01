from pathlib import Path
import unittest

ACTIVE = Path("active_speaker.py").read_text(encoding="utf-8")
MAIN = Path("main.py").read_text(encoding="utf-8")


class ActiveSpeakerLayoutTests(unittest.TestCase):
    def test_audio_and_mouth_motion_are_combined(self):
        self.assertIn("def _audio_rms(", ACTIVE)
        self.assertIn("Lower half of the detected face", ACTIVE)
        self.assertIn("speaking_audio", ACTIVE)

    def test_layout_contract_supports_a_b_and_stacked(self):
        self.assertIn('layout = "A"', ACTIVE)
        self.assertIn('layout = "B"', ACTIVE)
        self.assertIn('layout = "STACKED"', ACTIVE)
        self.assertIn("ACTIVE_SPEAKER_CONFIRMATIONS", ACTIVE)
        self.assertIn("ACTIVE_SPEAKER_MIN_HOLD_SECONDS", ACTIVE)

    def test_dynamic_sections_are_concatenated(self):
        from unittest.mock import patch
        import camera_layout
        layout={'regions':[{'id':'0','kind':'camera','rect':[0,0,480,540]}, {'id':'1','kind':'camera','rect':[480,0,480,540]}]}
        with patch.object(camera_layout,'analyze',return_value=(layout,[])), patch.object(camera_layout,'speaker_sections',return_value=[(0.,'0'),(5.,'1')]):
            graph,proof=camera_layout.build('isolated',0,10,640,360,None,[])
        self.assertIn('trim=start=5.000:end=10.000',graph)
        self.assertIn('concat=n=2',graph)
        self.assertEqual(proof['sections'],[(0.,'0'),(5.,'1')])

    def test_active_speaker_crops_respect_participant_lanes(self):
        self.assertIn("def _participant_horizontal_zones(", ACTIVE)
        self.assertIn("PARTICIPANT_BOUNDARY_GUTTER_RATIO", ACTIVE)
        self.assertIn("def _bounded_crop_x(", ACTIVE)
        self.assertIn("forbidden from crossing", ACTIVE)


    def test_clear_speaker_handoff_can_bypass_normal_hold(self):
        self.assertIn('ACTIVE_SPEAKER_MIN_HOLD_SECONDS", "3.0"', ACTIVE)
        self.assertIn("ACTIVE_SPEAKER_SWITCH_CONFIRMATIONS", ACTIVE)
        self.assertIn("ACTIVE_SPEAKER_STRONG_CONFIDENCE_RATIO", ACTIVE)
        self.assertIn("clear_speaker_handoff", ACTIVE)
        self.assertIn("if clear_speaker_handoff or normal_transition", ACTIVE)

    def test_ambiguous_state_still_uses_stacked_and_hold(self):
        self.assertIn('layout = "STACKED"', ACTIVE)
        self.assertIn("normal_transition", ACTIVE)
        self.assertIn("timestamp - last_change >= minimum_hold", ACTIVE)

    def test_failure_retains_deterministic_fallback(self):
        from unittest.mock import patch
        import active_speaker
        with patch('camera_layout.build',return_value=None), patch.object(active_speaker,'_audio_rms',return_value=[]):
            graph=active_speaker.build_active_speaker_filter(Path('isolated'),0,10,640,360,lambda:object())
        self.assertIn('gblur=',graph)
        self.assertIn('force_original_aspect_ratio=decrease',graph)
        self.assertTrue(active_speaker.last_framing_plan()['requires_framing_review'])
        self.assertNotIn('_build_crop_filter(video_path, start, duration)', MAIN)


if __name__ == "__main__":
    unittest.main()
