"""Isolated retained-media acceptance, not a production render/retry command."""
import hashlib
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
import imageio_ffmpeg
from camera_layout import blurred_full_frame

root = Path(__file__).resolve().parents[1] / "acceptance-media"
source = root / '9mW0bhmUaw0-source.mp4'
target = root / 'boundary-review'
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
results = []
for item in json.loads((target/'manifest.json').read_text()):
    if not item['asset'].startswith('short-'):
        continue
    start, end = item['original']
    output = target/(item['asset']+'-blurred.mp4')
    if not output.exists():
        subprocess.run([ffmpeg,'-v','error','-y','-ss',str(start),'-t',str(end-start),'-i',str(source),
            '-filter_complex_threads','2','-filter_complex',blurred_full_frame(),'-map','[v]','-map','0:a:0',
            '-c:v','libx264','-threads','2','-preset','ultrafast','-crf','23','-c:a','aac','-movflags','+faststart',str(output)],check=True,timeout=900)
    capture=cv2.VideoCapture(str(output));frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));fps=capture.get(cv2.CAP_PROP_FPS)
    dimensions=[int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    samples=[]
    for t in [0, min(8,end-start-1),(end-start)/2,end-start-1]:
        capture.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,frame=capture.read();assert ok
        image=output.with_name(output.stem+'-'+str(round(t,1))+'.jpg');cv2.imwrite(str(image),frame)
        samples.append(str(image))
    capture.release()
    decoded=subprocess.run([ffmpeg,'-v','error','-i',str(output),'-vn','-ac','1','-ar','16000','-f','f32le','-'],capture_output=True,check=True,timeout=120)
    sound=np.frombuffer(decoded.stdout,np.float32)
    assert dimensions==[1080,1920] and abs(frames/fps-(end-start))<.25 and np.sqrt(np.mean(sound**2))>.001
    subprocess.run([ffmpeg,'-v','error','-i',str(output),'-f','null','-'],capture_output=True,check=True,timeout=180)
    with output.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    results.append(dict(asset=item['asset'],bounds=item['original'],dimensions=dimensions,duration=frames/fps,
        audio_rms=float(np.sqrt(np.mean(sound**2))),audio_samples=len(sound),decoded=True,sha256=digest,
        preview=str(output),frames=samples,production_record_modified=False,boundary_approved=False))
    print(item['asset']+' blurred complete',flush=True)
(target/'blurred-media-validation.json').write_text(json.dumps(results,indent=2))
