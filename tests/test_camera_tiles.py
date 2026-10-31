import json
from pathlib import Path
import subprocess
import shutil

import pytest

cv2=pytest.importorskip('cv2')
np=pytest.importorskip('numpy')

from camera_layout import composition_graph,detect_regions,stable_regions,speaker_sections


def fixture(layout):
    frame=np.zeros((360,640,3),np.uint8)
    if layout=='side_by_side':
        frame[:,:320]=[20,20,200];frame[:,320:]=[200,200,20]
        faces=[(80,80,70,70),(430,80,70,70)]
    elif layout=='stacked':
        frame[:180]=[20,20,200];frame[180:]=[200,200,20]
        faces=[(120,40,50,50),(120,230,50,50)]
    else:
        frame[:180,:320]=[0,0,180];frame[:180,320:]=[180,180,30]
        frame[180:,:320]=[240,0,240];frame[180:,320:]=[20,240,240]
        faces=[(80,50,50,50),(430,50,50,50),(80,230,50,50),(430,230,50,50)]
        if layout=='three':
            faces=faces[:3]
            frame[180:,320:]=[30,30,30]
        if layout=='mixed':
            faces=faces[:3]
            frame[180:,320:,0]=np.arange(320,dtype=np.uint8)[None,:]
    return frame,faces


@pytest.mark.parametrize('layout',['side_by_side','stacked','three','grid','mixed'])
def test_actual_visual_separator_not_face_midpoint_defines_tiles(layout):
    frame,faces=fixture(layout)
    result=detect_regions(frame,faces)
    assert result['layout']==('mixed' if layout=='three' else layout)
    assert result['confidence']>=.8
    assert all(r['rect'][0] in (0,320) and r['rect'][1] in (0,180) for r in result['regions'])
    if layout=='mixed':assert result['regions'][-1]['kind']=='context'


def test_single_and_unknown_do_not_invent_tile_boundaries():
    frame=np.zeros((360,640,3),np.uint8)
    assert detect_regions(frame,[(70,70,50,50)])['confidence']==0
    assert detect_regions(frame,[(70,70,50,50),(400,70,50,50)])['confidence']==0


def test_layout_changes_or_inconsistent_faces_use_existing_fallback():
    frame,faces=fixture('grid');grid=detect_regions(frame,faces)
    assert stable_regions([grid]*5,640,360)
    assert stable_regions([grid,None,None,None,grid],640,360) is None


def test_mixed_screen_share_is_preserved_in_full_source_view():
    frame,faces=fixture('mixed');regions=detect_regions(frame,faces)['regions']
    graph=composition_graph(regions,primary='0')
    assert 'crop=' not in graph and 'force_original_aspect_ratio=decrease' in graph


def test_audio_spike_without_distinct_mouth_evidence_never_identifies_a_speaker():
    samples=[(i*.5,{'0':.02,'1':.02,'2':.02,'3':.02}) for i in range(30)]
    assert speaker_sections(samples,[0.01 if i%3 else .3 for i in range(300)],15)==[(0.,None)]


def test_correlated_distinct_mouth_motion_has_stable_speaker_handoff():
    samples=[];energy=[.01]*600
    for i in range(40):
        t=i*.5;level=.05 if i%2 else .01
        tile='0' if i<20 else '1'
        samples.append((t,{tile:level,'2':.001}))
        energy[int(t*20)]=level
    sections=speaker_sections(samples,energy,20)
    assert any(primary=='0' for _,primary in sections)
    assert any(primary=='1' for _,primary in sections)
    assert all(b[0]-a[0]>=3 for a,b in zip(sections,sections[1:]))


@pytest.mark.parametrize('layout',['side_by_side','stacked','three','grid','mixed'])
def test_real_ffmpeg_output_preserves_tiles_dimensions_audio_and_pixels(tmp_path,layout):
    executable=shutil.which('ffmpeg') or pytest.importorskip('imageio_ffmpeg').get_ffmpeg_exe()
    frame,faces=fixture(layout);source=tmp_path/'source.png';cv2.imwrite(str(source),frame)
    regions=detect_regions(frame,faces)['regions']
    graph=composition_graph(regions)
    target=tmp_path/'clip.mp4'
    subprocess.run([executable,'-v','error','-y','-loop','1','-i',str(source),'-f','lavfi','-i','sine=frequency=440:sample_rate=48000',
                    '-t','1','-filter_complex',graph,'-map','[v0]','-map','1:a:0','-c:v','libx264','-preset','ultrafast','-c:a','aac',str(target)],check=True,capture_output=True,timeout=60)
    cap=cv2.VideoCapture(str(target));ok,image=cap.read();cap.release()
    assert ok and image.shape[:2]==(1920,1080)
    audio=subprocess.run([executable,'-v','error','-i',str(target),'-vn','-f','f32le','-ac','1','-ar','16000','-'],capture_output=True,check=True,timeout=30).stdout
    assert np.sqrt(np.mean(np.frombuffer(audio,np.float32)**2))>.01
    if layout in {'side_by_side','stacked'}:
        assert image[480,540,2]>image[480,540,0]+80
        assert image[1440,540,0]>image[1440,540,2]+80
    elif layout=='grid':
        assert image[480,270,2]>100 and image[480,270,0]<60
        assert image[480,810,0]>100 and image[480,810,2]<90
        assert image[1440,270,0]>150 and image[1440,270,1]<60
        assert image[1440,810,1]>150 and image[1440,810,2]>150
    elif layout=='three':
        assert image[540,540,2]>100 and image[540,540,0]<60
        assert image[1500,270,0]>100 and image[1500,270,2]<90
        assert image[1500,810,0]>150 and image[1500,810,1]<60
    elif layout=='mixed':
        # Original presentation/context quadrant remains visible, not discarded.
        assert image[1080,810,1]>150
    assert target.stat().st_size>1000
