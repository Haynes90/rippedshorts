"""Real SQLite receipts, isolated Drive API, no production media or uploads."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import pytest
import google_drive as drive
import production_intelligence as evidence
from render_recovery import _put, digest, PROOF, RenderReviewRequired


@pytest.fixture
def retained(tmp_path, monkeypatch):
    import audio_master_handoff as storage
    monkeypatch.setattr(storage, 'DB_PATH', tmp_path/'test.db')
    path=tmp_path/'preview.mp4'; path.write_bytes(b'isolated verified-render fixture')
    spec=dict(status='VALIDATED',binding=dict(request_id='request',asset_id='request:short:13',
        source_sha256='verified-source',start=10.,end=35.))
    with evidence.database() as db:_put(db,'boundary_specification','spec',spec)
    clip=dict(boundary_specification_id='spec',boundary_specification_fingerprint=digest(spec),
        boundary_request_id='request',boundary_asset_id='request:short:13',start=10.,end=35.,
        completion_check={k:True for k in PROOF},boundary_export_evidence={'full_decode':True})
    files=Mock(); service=Mock();service.files.return_value=files
    objects=[]
    files.list.side_effect=lambda **kw:SimpleNamespace(execute=lambda:{'files':deepcopy(objects)})
    files.get.side_effect=lambda **kw:SimpleNamespace(execute=lambda:deepcopy(next(x for x in objects if x['id']==kw['fileId'])))
    def create(body):
        obj=dict(id='retained-'+str(len(objects)),size=str(path.stat().st_size),mimeType='video/mp4',
            md5Checksum=hashlib.md5(path.read_bytes()).hexdigest(),trashed=False,
            parents=body['parents'],appProperties=body['appProperties'])
        objects.append(obj);return obj
    creator=Mock(side_effect=create)
    def call():return drive.retain_verified_clip(service,path,{'name':'preview.mp4','parents':['library']},clip,creator)
    return SimpleNamespace(path=path,clip=clip,objects=objects,files=files,create=creator,call=call)


def test_repeat_retains_same_file_and_receipt_without_changes_or_second_upload(retained):
    r=retained;before=deepcopy(r.clip);a=r.call();b=r.call()
    assert a==b and r.create.call_count==1 and r.clip==before
    with evidence.database() as db:
        assert db.execute("SELECT count(*) FROM intelligence_records WHERE kind='retained_media'").fetchone()[0]==1
        assert db.execute("SELECT count(*) FROM intelligence_records WHERE kind='media_upload_intent'").fetchone()[0]==1
    assert r.files.delete.call_count==0 and r.files.update.call_count==0


@pytest.mark.parametrize('change',['missing_spec','fingerprint','request','asset','bounds','proof','decode'])
def test_unverified_or_wrong_revision_never_uploads(retained,change):
    r=retained
    if change=='missing_spec':r.clip['boundary_specification_id']='unknown'
    elif change=='fingerprint':r.clip['boundary_specification_fingerprint']='tampered'
    elif change=='request':r.clip['boundary_request_id']='other'
    elif change=='asset':r.clip['boundary_asset_id']='request:short:12'
    elif change=='bounds':r.clip['end']+=1
    elif change=='proof':r.clip['completion_check']['export_edges']=False
    else:r.clip['boundary_export_evidence']['full_decode']=False
    with pytest.raises(RenderReviewRequired):r.call()
    r.create.assert_not_called()


def test_successful_upload_with_lost_response_is_reconciled_without_repeating(retained):
    r=retained;original=r.create.side_effect
    def uncertain(body):original(body);raise TimeoutError('response lost')
    r.create.side_effect=uncertain
    with pytest.raises(TimeoutError):r.call()
    assert r.call()['clip_id']=='retained-0' and r.create.call_count==1


def test_empty_listing_after_uncertain_attempt_does_not_authorize_replay(retained):
    r=retained;r.create.side_effect=TimeoutError('uncertain')
    with pytest.raises(TimeoutError):r.call()
    with pytest.raises(RenderReviewRequired,match='uncertain'):r.call()
    assert r.create.call_count==1


def test_read_only_listing_failure_can_retry_before_creation_begins(retained):
    r=retained;original=r.files.list.side_effect;r.files.list.side_effect=TimeoutError('read failed')
    with pytest.raises(TimeoutError):r.call()
    r.create.assert_not_called();r.files.list.side_effect=original
    assert r.call()['clip_id']=='retained-0' and r.create.call_count==1


def test_receipt_write_failure_reconciles_the_uploaded_object(retained,monkeypatch):
    r=retained;original=_put
    import render_recovery
    def fail(db,kind,key,payload):
        if kind=='retained_media':raise OSError('isolated commit failure')
        return original(db,kind,key,payload)
    monkeypatch.setattr(render_recovery,'_put',fail)
    with pytest.raises(OSError):r.call()
    monkeypatch.setattr(render_recovery,'_put',original)
    assert r.call()['clip_id']=='retained-0' and r.create.call_count==1


@pytest.mark.parametrize('change',['checksum','missing_checksum','size','folder','tag','trashed'])
def test_existing_receipt_requires_authoritative_readback_not_replacement(retained,change):
    r=retained;r.call();obj=r.objects[0]
    if change=='checksum':obj['md5Checksum']='changed'
    elif change=='missing_checksum':obj.pop('md5Checksum')
    elif change=='size':obj['size']='1'
    elif change=='folder':obj['parents']=['other']
    elif change=='tag':obj['appProperties']={}
    else:obj['trashed']=True
    with pytest.raises(RenderReviewRequired):r.call()
    assert r.create.call_count==1 and r.files.delete.call_count==0


def test_changed_encoding_cannot_duplicate_the_same_specification(retained):
    r=retained;r.call();r.path.write_bytes(b'different encoded bytes for the same speech revision')
    with pytest.raises(RenderReviewRequired,match='Different output'):r.call()
    assert r.create.call_count==1


def test_multiple_remote_matches_hold_without_choosing_newest_or_uploading(retained):
    r=retained;original=r.create.side_effect
    def uncertain(body):original(body);raise TimeoutError()
    r.create.side_effect=uncertain
    with pytest.raises(TimeoutError):r.call()
    r.objects.append({**r.objects[0],'id':'duplicate'})
    with pytest.raises(RenderReviewRequired,match='Multiple'):r.call()
    assert r.create.call_count==1


def test_concurrent_request_observes_intent_and_cannot_start_second_upload(retained):
    r=retained;original=r.create.side_effect
    def competing(body):
        with pytest.raises(RenderReviewRequired,match='uncertain'):r.call()
        return original(body)
    r.create.side_effect=competing
    assert r.call()['clip_id']=='retained-0' and r.create.call_count==1


def test_existing_uploader_uses_verified_revision_receipt_path(retained,monkeypatch):
    import main
    r=retained;service=Mock();service.files.return_value=r.files
    monkeypatch.setattr(main,'get_google_services',lambda:(service,None,None))
    monkeypatch.setattr(main,'_drive_title_folder',lambda title:'library')
    monkeypatch.setattr(main,'MediaFileUpload',Mock())
    r.files.create.side_effect=lambda **kw:SimpleNamespace(execute=lambda:r.create(kw['body']))
    a=main.upload_clip_to_drive(r.path,'preview.mp4',vid_title='Fixture',verified_clip=r.clip)
    b=main.upload_clip_to_drive(r.path,'preview.mp4',vid_title='Fixture',verified_clip=r.clip)
    assert a==b and a['media_receipt_id'] and r.create.call_count==1
