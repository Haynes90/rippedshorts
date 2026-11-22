"""Run in the clean production Docker image; synthetic media, no credentials."""
from importlib.metadata import distributions
import json
from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np
import main

packages={d.metadata['Name'].lower():d.version for d in distributions() if d.metadata['Name'].lower().startswith('opencv-')}
assert packages=={'opencv-python-headless':'4.11.0.86'}, packages
assert cv2.__version__=='4.11.0', cv2.__version__
cascade=main._load_face_cascade()
assert cascade is not None and not cascade.empty()
assert main._motion_center(None,np.zeros((32,32),np.uint8)) is None
with tempfile.TemporaryDirectory() as work:
    root=Path(work);source=root/'source.mp4'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=640x360:rate=12',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=44100','-t','2','-c:v','libx264',
        '-pix_fmt','yuv420p','-c:a','aac',str(source)],check=True)
    for aspect,render,dimensions in [('9:16',main.create_clip_file,(1080,1920)),('16:9',main.create_topic_segment_file,(1920,1080))]:
        output=root/(aspect.replace(':','-')+'.mp4')
        render(source,0,1,output)
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(output)]))
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        assert (video['width'],video['height'])==dimensions
        assert any(s['codec_type']=='audio' for s in probe['streams'])
        assert abs(float(probe['format']['duration'])-1)<.2
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(output),'-f','null','-'],check=True)
        print('VERIFIED_SYNTHETIC_RENDER',aspect,dimensions)
print('VERIFIED_SINGLE_OPENCV',packages)
