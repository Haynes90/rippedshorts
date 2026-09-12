from pathlib import Path
import unittest

SOURCE = Path("main.py").read_text(encoding="utf-8")


class DualParticipantFramingTests(unittest.TestCase):
    def test_two_persistent_faces_enable_stacked_layout(self):
        self.assertIn("def _estimate_dual_participant_tracks(", SOURCE)
        self.assertIn('DUAL_TRACK_MIN_FRACTION", "0.50"', SOURCE)
        self.assertIn("if right - left < 0.22", SOURCE)

    def test_participants_are_cropped_independently(self):
        self.assertIn("def _stacked_participant_filter(", SOURCE)
        self.assertIn("[p0]crop=", SOURCE)
        self.assertIn("[p1]crop=", SOURCE)
        self.assertIn("[top][bottom]vstack=inputs=2[v]", SOURCE)

    def test_single_speaker_path_remains_fallback(self):
        self.assertIn("return _build_crop_filter(video_path, start, duration), False", SOURCE)

    def test_horizontal_renderer_is_not_changed_to_vertical_layout(self):
        horizontal = SOURCE[SOURCE.index("def create_topic_segment_file("):]
        self.assertNotIn("_build_vertical_filter(", horizontal.split("\ndef ", 1)[0])


if __name__ == "__main__":
    unittest.main()
