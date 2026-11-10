"""Isolated actual persistence/render paths; semantic provider explicitly injected.

Generated media proves signal/FFmpeg behavior, not live TDOG sentence acceptance.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest
import boundary_specification as boundary
import clip_completion as speech
REAL_DURATION = speech.media_duration
REAL_EDGE = boundary.acoustic_edge
REAL_SOURCE_AUDIO = __import__("main")._assert_source_audio


@pytest.fixture
def passage(tmp_path, monkeypatch):
    import audio_master_handoff as storage
    import telegram_intake as tg
    import main
    monkeypatch.setattr(main,"_assert_source_audio",lambda *a,**k:-1.)
    monkeypatch.setattr(storage,"DB_PATH",tmp_path/"isolated.db")
    monkeypatch.setattr(storage,"DATA_DIR",tmp_path)
    monkeypatch.setattr(tg,"DB_PATH",storage.DB_PATH)
    source=tmp_path/"verified-source";source.write_bytes(b"isolated source identity, not playable media")
    transcript=[dict(start=0.,end=8.,text="What is the best way to store tiny seeds?",speaker="host"),
                dict(start=10.2,end=15.8,text="Dry tiny seeds gently, label them, and store them cool.",speaker="guest"),
                dict(start=20.,end=25.,text="The labels disappear faster than the seeds!",speaker="host"),
                dict(start=30.2,end=229.8,text="Here is a complete, substantial seed-storage explanation.",speaker="guest")]
    short=dict(candidate_id="selected-short",candidate_number=12,start=10.,end=16.,duration=6.,
               transcript=transcript[1]["text"],title="Keep tiny seeds viable",reason="A practical complete answer.",
               selection_provenance={"analysis_path":"topics_only","historical_examples":14})
    sibling=dict(candidate_id="completed-sibling",candidate_number=13,start=20.,end=25.,duration=5.,transcript=transcript[2]["text"])
    highlight=dict(candidate_id="selected-highlight",start=30.,end=230.,duration=200.,transcript=transcript[3]["text"])
    state=dict(vid_title="Isolated fixture",show_id="TDOG",parsed={"video_id":"verified-source-id"},
        source_validation={"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"source_id":"verified-source-id"},
        topic_source_segments=transcript,result={"segments":[short,sibling]},
        candidate_reviews={"0":{"status":"queued","user_id":"owner","reviewed_at":"approved"},"1":{"status":"rendered","clip_url":"https://drive.google.com/file/d/finished-sibling/view"}},
        topic_result={"segments":[highlight]},topic_reviews={"0":{"status":"queued","user_id":"owner","reviewed_at":"approved"}},
        copy_drafts=[{"social_caption":"Human words"}],creator_approval={"revision":"existing"})
    with tg._telegram_db() as db:
        db.execute("INSERT INTO telegram_requests VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                   ("request","key","chat","owner","awaiting_review","both","youtube","source",json.dumps(state),"now","now"))
    def read():
        with tg._telegram_db() as db:return json.loads(db.execute("SELECT state_json FROM telegram_requests WHERE request_id=?",("request",)).fetchone()[0])
    def write(s):
        with tg._telegram_db() as db:db.execute("UPDATE telegram_requests SET state_json=? WHERE request_id=?",(json.dumps(s),"request"))
    monkeypatch.setattr(speech,"media_duration",lambda p:300.)
    def review(units,start,end,aspect,*,context):
        a,b=context["included_unit_indices"]
        return dict(complete_start=True,complete_end=True,complete_thought=True,first_word=a,last_word=b,
                    context_sufficient=True,setup_units=[a],payoff_units=[b],reason="The answer identifies its subject and finishes.",
                    provenance={"provider":"isolated-fixture","prompt_version":"fixture"})
    reviewer=Mock(side_effect=review);monkeypatch.setattr(speech,"review_thought",reviewer)
    edge=Mock(side_effect=lambda source,t,side,duration:dict(edge=side,cut=t,measured_clear=True,quiet_intervals=[[t-.2,t+.2]],suggested_cut=t,automatic_adjustment=False))
    monkeypatch.setattr(boundary,"acoustic_edge",edge)
    def prepare(lane="short"):
        s=read();key="result" if lane=="short" else "topic_result";clip=deepcopy(s[key]["segments"][0]);clip["framing_mode"]=None
        return boundary.prepare_for_render("request","request:"+lane+(":12" if lane=="short" else ":1"),clip,source,s,"9:16" if lane=="short" else "16:9",{})
    return dict(read=read,write=write,state=state,source=source,reviewer=reviewer,edge=edge,prepare=prepare,tmp=tmp_path,tg=tg)


def test_specification_is_immutable_cached_source_bound_and_does_not_change_selections(passage):
    p=passage;before=p["read"]();first=p["prepare"]();second=p["prepare"]()
    assert first==second and p["read"]()==before and p["reviewer"].call_count==1 and p["edge"].call_count==2
    spec=boundary.consume(p["source"],first,"9:16")
    assert spec["status"]=="VALIDATED" and spec["included_segments"]==[1]
    assert spec["timing_precision"]==["provider_segment_timing"]
    assert all(u["word_index"] is None for u in spec["speech_units"])
    assert spec["selection_provenance"]["historical_examples"]==14
    assert spec["binding"]["source_sha256"]==before["source_validation"]["source_sha256"]
    with boundary.evidence.database() as db:
        assert db.execute("SELECT count(*) FROM intelligence_records WHERE kind=?",("boundary_specification",)).fetchone()[0]==1
        with pytest.raises(Exception,match="Immutable"):
            boundary._put(db,"boundary_specification",first["boundary_specification_id"],{**spec,"start":999})


def test_renderer_consumes_persisted_exact_bounds_without_another_semantic_or_asr_call(passage,monkeypatch):
    p=passage;clip=p["prepare"]();out=p["tmp"]/"export.mp4";calls=[]
    monkeypatch.setattr(speech,"media_duration",lambda path:6. if path==out else 300.)
    monkeypatch.setattr(speech,"transcribe_window",Mock(side_effect=AssertionError("Do not repeat ASR or vote on spelling")))
    monkeypatch.setattr(boundary,"verify_export",Mock(return_value={"full_decode":True,"source_audio_edges":["independent fixture"]}))
    monkeypatch.setattr(speech,"_run",lambda args:SimpleNamespace(stdout=json.dumps({"streams":[dict(codec_type="video",width=1080,height=1920),dict(codec_type="audio")]})))
    def renderer(source,start,duration,destination):
        with boundary.evidence.database() as db:
            assert boundary._get(db,clip["boundary_specification_id"],"boundary_specification")
        calls.append((source,start,duration));destination.write_bytes(b"isolated renderer output")
    result=speech.render_complete_clip(p["source"],clip,out,renderer,"9:16")
    assert calls==[(p["source"],10.,6.)] and p["reviewer"].call_count==1
    assert result["boundary_specification_id"]==clip["boundary_specification_id"]
    assert result["completion_check"]["basis"]=="immutable_specification_and_source_export_audio"
    assert p["read"]()["candidate_reviews"]["1"]["status"]=="rendered"


@pytest.mark.parametrize("lane",["short","highlight"])
@pytest.mark.parametrize("semantic_complete",[True,False])
def test_existing_explicit_render_path_persists_contract_before_attach_or_holds_individually(passage,monkeypatch,lane,semantic_complete):
    p=passage;tg=p["tg"];before=p["read"]();import main
    monkeypatch.setattr(tg,"_ensure_render_source",lambda *a:p["source"])
    for name in ("send","_refresh_short_review_message","_refresh_topic_review","_send_short_confirmation","_notify_render_queue_complete","_safe_log_candidate"):
        monkeypatch.setattr(tg,name,Mock())
    monkeypatch.setattr(__import__("workflow_test_status"),"journal",Mock())
    monkeypatch.setattr(__import__("review_navigation"),"notify_safely",Mock())
    monkeypatch.setattr(main,"prepare_framing_review",Mock())
    if not semantic_complete:
        valid=p["reviewer"].side_effect
        def incomplete(*a,**kw):
            result=valid(*a,**kw);result["complete_thought"]=False;return result
        p["reviewer"].side_effect=incomplete
    def attached(candidate,*args,**kwargs):
        clip=candidate["segments"][0] if lane=="short" else candidate
        assert boundary.consume(p["source"],clip,"9:16" if lane=="short" else "16:9")["status"]=="VALIDATED"
        result={**clip,"completion_check":{k:True for k in ("media_duration","video_stream","audio_stream","format_valid","export_edges")},
                "clip_url":"https://drive.google.com/file/d/fixture_verified_output/view","segment_url":"https://drive.google.com/file/d/fixture_verified_output/view"}
        return {"segments":[result]} if lane=="short" else result
    attachment=Mock(side_effect=attached)
    monkeypatch.setattr(main,"attach_clip_assets" if lane=="short" else "attach_topic_segment_asset",attachment)
    renderer=tg._render_approved if lane=="short" else tg._render_topic_approved
    renderer("request",0,"chat");after=p["read"]()
    reviews="candidate_reviews" if lane=="short" else "topic_reviews"
    assert after[reviews]["0"]["status"]==("rendered" if semantic_complete else "render_failed")
    assert after["candidate_reviews"]["1"]==before["candidate_reviews"]["1"]
    assert after["copy_drafts"]==before["copy_drafts"] and after["creator_approval"]==before["creator_approval"]
    renderer("request",0,"chat")
    assert attachment.call_count==(1 if semantic_complete else 0) and p["reviewer"].call_count==1


def test_punctuation_does_not_override_missing_context_or_incomplete_thought(passage):
    p=passage;valid=p["reviewer"].side_effect
    def uncertain(*a,**kw):
        result=valid(*a,**kw);result.update(context_sufficient=False,complete_thought=False);return result
    p["reviewer"].side_effect=uncertain
    with pytest.raises(boundary.BoundaryReviewRequired):p["prepare"]()
    with boundary.evidence.database() as db:
        result=json.loads(db.execute("SELECT payload FROM intelligence_records WHERE kind=?",("boundary_assessment",)).fetchone()[0])
    assert {"SEMANTIC_INCOMPLETE","CONTEXT_NOT_ESTABLISHED"} <= set(result["reasons"])


@pytest.mark.parametrize("change",["start","transcript","configuration","source","transcript_revision","owner_revision","cross_asset"])
def test_changed_input_or_identity_is_held_before_renderer(passage,monkeypatch,change):
    p=passage;clip=p["prepare"]();render=Mock()
    if change=="start":clip["start"]+=.1
    elif change=="transcript":clip["transcript"]="Different approved words."
    elif change=="configuration":monkeypatch.setenv("VERTICAL_TEST_SETTING","changed")
    elif change=="source":p["source"].write_bytes(b"corrupt or replaced source")
    elif change=="cross_asset":clip["boundary_asset_id"]="another-request:short:12"
    else:
        state=p["read"]()
        if change=="transcript_revision":state["topic_source_segments"][0]["text"]="Changed source context."
        else:state["result"]["segments"][0]["end"]+=1
        p["write"](state)
    with pytest.raises(boundary.BoundaryReviewRequired):
        speech.render_complete_clip(p["source"],clip,p["tmp"]/"no-output",render,"9:16")
    render.assert_not_called()


def test_required_specification_cannot_be_omitted_and_foreign_bounds_cannot_be_registered(passage):
    p=passage
    with pytest.raises(boundary.BoundaryReviewRequired,match="VALIDATED_SPECIFICATION_REQUIRED"):
        boundary.consume(p["source"],{"requires_boundary_specification":True},"9:16")
    state=p["read"]();fake={**state["result"]["segments"][0],"start":0,"framing_mode":None}
    with pytest.raises(boundary.BoundaryReviewRequired,match="APPROVED_INPUT_DIFFERS"):
        boundary.prepare_for_render("request","request:short:12",fake,p["source"],state,"9:16",{})
    p["reviewer"].assert_not_called()


def test_missing_question_or_payoff_records_proposal_without_changing_approved_cut(passage):
    p=passage;before=p["read"]();original=p["reviewer"].side_effect
    def needs_question(*args,**kwargs):
        result=original(*args,**kwargs);result.update(first_word=0,setup_units=[0]);return result
    p["reviewer"].side_effect=needs_question
    for _ in range(2):
        with pytest.raises(boundary.BoundaryReviewRequired,match="EVIDENCE_UNRESOLVED"):p["prepare"]()
    assert p["reviewer"].call_count==1 and p["read"]()==before
    with boundary.evidence.database() as db:
        record=json.loads(db.execute("SELECT payload FROM intelligence_records WHERE kind=?",("boundary_assessment",)).fetchone()[0])
    assert record["status"]=="HELD" and "CONTEXT_REQUIRES_EXPLICIT_EDIT" in record["reasons"]
    assert record["suggested_semantic_bounds"]==[0.,15.8] and not record["boundary_changes_applied"]


@pytest.mark.parametrize("value",[None,"1",True,999,-1,1.5])
def test_invalid_semantic_references_persist_sanitized_deterministic_failure(passage,value):
    p=passage;original=p["reviewer"].side_effect
    def invalid(*args,**kwargs):
        result=original(*args,**kwargs);result["first_word"]=value;return result
    p["reviewer"].side_effect=invalid
    with pytest.raises(boundary.BoundaryReviewRequired):p["prepare"]()
    with pytest.raises(ValueError,match="Saved analysis failed"):p["prepare"]()
    assert p["reviewer"].call_count==1
    with boundary.evidence.database() as db:
        saved=db.execute("SELECT result,status FROM source_analyses WHERE analysis_id LIKE ?",("boundary-assessment:%",)).fetchone()
    assert saved["status"]=="FAILED" and "SEMANTIC_REFERENCE_INVALID" in saved["result"]
    assert "Dry tiny seeds" not in saved["result"]


def test_uncertain_acoustics_holds_one_asset_and_preserves_independent_highlight(passage):
    p=passage;before=p["read"]();clean=p["edge"].side_effect
    def uncertain(source,t,side,duration):
        result=clean(source,t,side,duration)
        if t==16:result.update(measured_clear=False,suggested_cut=16.147)
        return result
    p["edge"].side_effect=uncertain
    with pytest.raises(boundary.BoundaryReviewRequired):p["prepare"]()
    highlight=p["prepare"]("highlight")
    assert boundary.consume(p["source"],highlight,"16:9")["status"]=="VALIDATED"
    assert p["read"]()==before


@pytest.mark.parametrize("bad",["reversed","nan","bool","word_parent"])
def test_malformed_source_timing_never_becomes_specification(passage,bad):
    p=passage;s=p["read"]()
    if bad=="reversed":s["topic_source_segments"][1]["end"]=9
    elif bad=="nan":s["topic_source_segments"][1]["start"]=float("nan")
    elif bad=="bool":s["topic_source_segments"][1]["start"]=True
    else:s["topic_source_segments"][1]["words"]=[dict(word="Dry",start=9.,end=11.)]
    p["write"](s)
    with pytest.raises(boundary.BoundaryReviewRequired):p["prepare"]()
    p["reviewer"].assert_not_called()


def test_overlap_and_speaker_turns_are_preserved_without_invented_word_timing(passage):
    p=passage;s=p["read"]();s["topic_source_segments"].insert(2,dict(start=14.,end=15.5,text="Yes, keep the label!",speaker="host"));p["write"](s)
    spec=boundary.consume(p["source"],p["prepare"](),"9:16")
    assert [u["speaker"] for u in spec["speech_units"]]==["guest","host"]
    assert [u["segment_index"] for u in spec["speech_units"]]==[1,2]
    assert all(u["word_index"] is None for u in spec["speech_units"])


def test_real_word_timing_keeps_original_word_and_segment_references(passage):
    p=passage;s=p["read"]();s["topic_source_segments"][1]["words"]=[dict(word="Dry",start=10.2,end=11.),dict(word="seeds.",start=11.5,end=15.8)];p["write"](s)
    spec=boundary.consume(p["source"],p["prepare"](),"9:16")
    assert spec["timing_precision"]==["provider_word_timing"]
    assert [(u["segment_index"],u["word_index"]) for u in spec["speech_units"]]==[(1,0),(1,1)]


def test_highlight_has_minimum_without_arbitrary_maximum(passage,monkeypatch):
    p=passage;s=p["read"]();s["topic_result"]["segments"][0].update(end=209.,duration=179.);p["write"](s)
    with pytest.raises(boundary.BoundaryReviewRequired,match="FORMAT_DURATION_INVALID"):p["prepare"]("highlight")
    monkeypatch.setattr(speech,"media_duration",lambda _:1000.)
    s["topic_result"]["segments"][0].update(end=930.,duration=900.);s["topic_source_segments"][3]["end"]=929.8;p["write"](s)
    assert boundary.consume(p["source"],p["prepare"]("highlight"),"16:9")["binding"]["end"]==930.


def test_adaptive_suggestions_come_from_quiet_intervals_not_uniform_padding(monkeypatch):
    monkeypatch.setattr(speech,"_run",lambda args:SimpleNamespace(stderr="silence_start: 1.7\nsilence_end: 1.9"))
    a=boundary.acoustic_edge(Path("source"),10.,"end",100.)
    monkeypatch.setattr(speech,"_run",lambda args:SimpleNamespace(stderr="silence_start: 1.6\nsilence_end: 2.3"))
    b=boundary.acoustic_edge(Path("source"),10.,"end",100.)
    assert not a["measured_clear"] and not b["measured_clear"]
    assert a["suggested_cut"]==pytest.approx(10.3) and b["suggested_cut"]==pytest.approx(10.45)
    assert not a["automatic_adjustment"] and not b["automatic_adjustment"]


@pytest.fixture
def real_source(tmp_path,monkeypatch):
    ffmpeg=pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe();monkeypatch.setenv("FFMPEG_BINARY",ffmpeg)
    audio=tmp_path/"source.wav"
    from array import array
    samples=array("h")
    for i in range(6*16000):
        t=i/16000;gate=max(0.,min(1.,(t-1.2)/.04,(3.8-t)/.04))
        value=gate*(.25*math.sin(2*math.pi*(200*t+23*t*t))+.1*math.sin(2*math.pi*(370*t+13*t*t)))
        samples.append(int(32767*value))
    with wave.open(str(audio),"wb") as out:out.setparams((1,2,16000,0,"NONE","not compressed"));out.writeframes(samples.tobytes())
    video=tmp_path/"source.mp4"
    subprocess.run([ffmpeg,"-v","error","-y","-f","lavfi","-i","color=c=blue:s=320x240:r=25:d=6","-i",str(audio),"-c:v","libx264","-c:a","aac","-b:a","192k","-shortest",str(video)],check=True)
    return video,ffmpeg,tmp_path


def test_actual_source_acoustics_and_export_audio_match_without_asr(real_source):
    source,ffmpeg,tmp=real_source;out=tmp/"verified.mp4"
    assert boundary.acoustic_edge(source,.9,"start",6.)["measured_clear"]
    assert boundary.acoustic_edge(source,4.1,"end",6.)["measured_clear"]
    assert not boundary.acoustic_edge(source,2.,"start",6.)["measured_clear"]
    subprocess.run([ffmpeg,"-v","error","-y","-ss","0.9","-i",str(source),"-t","3.2","-c:v","libx264","-c:a","aac","-b:a","192k",str(out)],check=True)
    verified=boundary.verify_export(source,out,.9,4.1)
    assert verified["full_decode"] and all(m["correlation"]>=.985 for m in verified["source_audio_edges"])
    assert verified["max_volume_db"]>-60


def test_actual_wrong_passage_is_held_even_with_matching_duration(real_source):
    source,ffmpeg,tmp=real_source;wrong=tmp/"wrong.mp4"
    subprocess.run([ffmpeg,"-v","error","-y","-ss","1.2","-i",str(source),"-t","3.2","-c:v","libx264","-c:a","aac",str(wrong)],check=True)
    with pytest.raises(boundary.BoundaryReviewRequired):boundary.verify_export(source,wrong,.9,4.1)


@pytest.mark.parametrize("aspect",["9:16","16:9"])
def test_persisted_contract_through_actual_ffmpeg_export_in_both_formats(passage,monkeypatch,aspect):
    if not shutil.which("ffprobe"):pytest.skip("FFprobe required; GitHub installs FFmpeg tools")
    p=passage;ffmpeg=pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    monkeypatch.setenv("FFMPEG_BINARY",ffmpeg);monkeypatch.setattr(speech,"media_duration",REAL_DURATION)
    monkeypatch.setattr(__import__("main"),"_assert_source_audio",REAL_SOURCE_AUDIO)
    monkeypatch.setattr(boundary,"acoustic_edge",REAL_EDGE)
    source=p["tmp"]/"real-source.mp4";audio=p["tmp"]/"source-audio.wav";output=p["tmp"]/"actual-export.mp4"
    total,release,end=(6.,3.8,4.1) if aspect=="9:16" else (183.,181.2,181.5)
    from array import array
    samples=array("h")
    for i in range(int(total*16000)):
        t=i/16000;gate=max(0.,min(1.,(t-1.2)/.04,(release-t)/.04))
        samples.append(int(32767*gate*(.25*math.sin(2*math.pi*(220*t+20*math.sin(t*.3)))+.1*math.sin(2*math.pi*(370*t+7*math.sin(t*.7))))))
    with wave.open(str(audio),"wb") as out:out.setparams((1,2,16000,0,"NONE","not compressed"));out.writeframes(samples.tobytes())
    subprocess.run([ffmpeg,"-v","error","-y","-f","lavfi","-i",f"color=c=blue:s=320x180:r=5:d={total}","-i",str(audio),"-c:v","libx264","-c:a","aac","-b:a","192k","-shortest",str(source)],check=True)
    state=p["read"]();state["source_validation"]["source_sha256"]=hashlib.sha256(source.read_bytes()).hexdigest()
    state["topic_source_segments"]=[dict(start=1.2,end=release,text="Synthetic complete test passage.",speaker="fixture")]
    key="result" if aspect=="9:16" else "topic_result";lane="short" if aspect=="9:16" else "highlight"
    state[key]["segments"][0].update(start=.9,end=end,duration=end-.9,transcript="Synthetic complete test passage.")
    p["write"](state);clip={**state[key]["segments"][0],"framing_mode":None}
    prepared=boundary.prepare_for_render("request","request:"+lane+(":12" if lane=="short" else ":1"),clip,source,state,aspect,{})
    monkeypatch.setattr(speech,"transcribe_window",Mock(side_effect=AssertionError("No repeat ASR")))
    width,height=(1080,1920) if aspect=="9:16" else (640,360)
    def renderer(video,start,duration,destination):
        subprocess.run([ffmpeg,"-v","error","-y","-ss",str(start),"-i",str(video),"-t",str(duration),"-vf",f"scale={width}:{height}","-r","25" if aspect=="9:16" else "5","-c:v","libx264","-preset","ultrafast","-c:a","aac","-b:a","192k",str(destination)],check=True)
    try:
        result=speech.render_complete_clip(source,prepared,output,renderer,aspect)
    except boundary.BoundaryReviewRequired as exc:
        pytest.fail(str(exc)+" "+json.dumps(exc.diagnostics))
    assert result["boundary_export_evidence"]["full_decode"] and result["completion_check"]["export_edges"]
    assert result["boundary_specification_id"]==prepared["boundary_specification_id"] and p["reviewer"].call_count==1
