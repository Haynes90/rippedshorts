import ast
from pathlib import Path
import unittest


SOURCE = (Path(__file__).resolve().parents[1] / "audio_master_handoff.py").read_text()
TREE = ast.parse(SOURCE)
FUNCTIONS = {
    node.name: node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name in {"normalize_sermon"}
}
namespace = {}
exec(compile(ast.Module(body=list(FUNCTIONS.values()), type_ignores=[]), "handoff-functions", "exec"), namespace)
normalize_sermon = namespace["normalize_sermon"]


class TimelineTests(unittest.TestCase):
    def test_original_service_timeline_is_filtered_and_rebased(self):
        segments = [
            {"start": 90, "end": 99, "text": "before"},
            {"start": 100, "end": 110, "text": "opening"},
            {"start": 125, "end": 140, "text": "message"},
            {"start": 171, "end": 180, "text": "after"},
        ]
        manifest = {"sermon_start_seconds": 100, "sermon_end_seconds": 170, "transcript_timeline": "original_service"}
        result = normalize_sermon(segments, manifest)
        self.assertEqual([item["text"] for item in result], ["opening", "message"])
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[1]["start"], 25)

    def test_empty_sermon_range_is_retryable_error(self):
        with self.assertRaisesRegex(RuntimeError, "no segments"):
            normalize_sermon(
                [{"start": 10, "end": 20, "text": "outside"}],
                {"sermon_start_seconds": 100, "sermon_end_seconds": 200, "transcript_timeline": "original_service"},
            )

    def test_contract_has_idempotent_primary_key_and_duplicate_response(self):
        self.assertIn("source_job_id TEXT PRIMARY KEY", SOURCE)
        self.assertIn('{"status": "duplicate"', SOURCE)


if __name__ == "__main__":
    unittest.main()


def test_rich_transcript_preserved_and_only_measured_times_rebased(tmp_path):
    import json, copy
    from audio_master_handoff import parse_transcript, normalize_sermon
    words=[{'word':'Save','start':100.,'end':101.,'speaker':'pastor'}, {'word':'seeds.','start':101.,'end':102.,'speaker':'pastor'}]
    segment={'start':100.,'end':110.,'text':'Save seeds.', 'speaker':'pastor','words':words,
             'sentences':[{'start':100.,'end':102.,'text':'Save seeds.'}], 'timing_basis':'provider_word_timing'}
    before=copy.deepcopy(segment);path=tmp_path/'retained.json';path.write_text(json.dumps({'segments':[segment]}),encoding='utf-8')
    parsed=parse_transcript(path)
    assert parsed==[segment]
    manifest={'sermon_start_seconds':100,'sermon_end_seconds':170,'transcript_timeline':'original_service'}
    result=normalize_sermon(parsed,manifest)[0]
    assert result['speaker']=='pastor' and result['timing_basis']=='provider_word_timing'
    assert [(w['start'],w['end']) for w in result['words']]==[(0.,1.),(1.,2.)]
    assert result['sentences'][0]['start']==0 and result['sentences'][0]['end']==2
    assert segment==before
    relative=normalize_sermon([result],{**manifest,'transcript_timeline':'sermon_relative'})[0]
    assert relative['words']==result['words']


def test_boundary_overlap_retains_original_evidence_and_requires_review():
    from audio_master_handoff import normalize_sermon
    source={'start':99.,'end':103.,'text':'Keep the full thought.', 'speaker':'pastor',
            'words':[{'word':'Keep','start':99.,'end':100.5},{'word':'the','start':100.6,'end':101.}, {'word':'thought.','start':101.,'end':103.}]}
    result=normalize_sermon([source],{'sermon_start_seconds':100,'sermon_end_seconds':170,'transcript_timeline':'original_service'})[0]
    assert result['needs_review'] and result['source_words']==source['words']
    assert result['source_segment_bounds']==[99.,103.]
    assert abs(result['words'][0]['start']-.6)<1e-8
    assert result['text']==source['text'] and result['speaker']=='pastor'
