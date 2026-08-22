from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


def test_renderer_does_not_use_removed_mediapipe_package_attribute():
    assert "mp.solutions" not in SOURCE
    assert "from mediapipe.python.solutions import face_detection" in SOURCE


def test_renderer_has_opencv_and_center_crop_fallbacks():
    assert "CascadeClassifier" in SOURCE
    assert "haarcascade_frontalface_default.xml" in SOURCE
    assert "return 0.5" in SOURCE
    assert "MediaPipe face detection unavailable" in SOURCE
