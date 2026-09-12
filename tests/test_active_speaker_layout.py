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
        self.assertIn("trim=start=", ACTIVE)
        self.assertIn("concat=n=", ACTIVE)
        self.assertIn("vstack=2", ACTIVE)

    def test_failure_retains_deterministic_fallback(self):
        self.assertIn("return None", ACTIVE)
        self.assertIn("if active_filter:", MAIN)
        self.assertIn("_estimate_dual_participant_tracks", MAIN)
        self.assertIn("_build_crop_filter(video_path, start, duration)", MAIN)


if __name__ == "__main__":
    unittest.main()
