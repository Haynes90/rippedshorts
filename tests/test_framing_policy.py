import hashlib
import json
from unittest.mock import Mock
import pytest
from test_editorial_framing_review import review

@pytest.mark.parametrize('count',[3,4])
def test_group_policy_never_labels_a_forced_blurred_section_as_smart(monkeypatch,count):
    import camera_layout as camera
    regions=[{'id':str(i),'kind':'camera','rect':[i*200,0,200,540]} for i in range(count)]
    monkeypatch.setattr(camera,'analyze',lambda *a:({'confidence':.8,'regions':regions,'layout':'composite'},[]))
    monkeypatch.setattr(camera,'speaker_sections',lambda *a:[(0.,'0'),(4.,'1')])
    graph,proof=camera.build('isolated',0,8,960,540,None,[])
    assert proof['sections']==[(0.,None),(4.,None)]
    assert proof['framing_method']=='blurred_full_frame' and proof['requires_framing_review']
    assert proof['speaker_confidence']=='uncertain' and 'gblur=' in graph
    assert 'xstack=' not in graph


def test_single_complete_camera_and_unknown_speaker_preserve_source():
    from camera_layout import composition_graph
    region=[{'id':'camera','kind':'camera','rect':[0,0,640,360]}]
    smart=composition_graph(region,primary='camera')
    assert 'crop=640:360:0:0' in smart and 'force_original_aspect_ratio=decrease' in smart
    uncertain=composition_graph(region,primary='not-a-verified-track')
    assert '[fg0]scale=1080:1920:force_original_aspect_ratio=decrease' in uncertain


