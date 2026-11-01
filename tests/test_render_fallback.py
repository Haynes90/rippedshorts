from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")


def test_opencv_detector_loads_and_missing_model_falls_back():
    import main,cv2
    from unittest.mock import patch
    from pathlib import Path
    # Existing OpenCV4 detector remains loadable; no removed MediaPipe APIs required.
    with patch.object(main.Path,'is_file',return_value=False):
        assert main._load_face_cascade() is None
    assert callable(cv2.CascadeClassifier)
    cascade=cv2.CascadeClassifier(str(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'))
    assert not cascade.empty()


def test_renderer_has_opencv_and_center_crop_fallbacks():
    import main,cv2,numpy as np
    from unittest.mock import Mock,patch
    assert main._motion_center(None,np.zeros((32,32),np.uint8)) is None
    assert main._plan_framing_sections([],20)==[(0.,.5)]
    capture=Mock();capture.get.return_value=30;capture.read.return_value=(False,None)
    with patch.object(cv2,'VideoCapture',return_value=capture),patch.object(main,'_load_face_cascade',return_value=None):
        assert main._estimate_speaker_track('missing-test-media',0,20)==[(0.,.5),(20,.5)]
    capture.release.assert_called_once()
