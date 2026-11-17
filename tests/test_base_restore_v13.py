import hashlib
import zipfile
from pathlib import Path
import pytest
from omega_geodepth.base_restore import extract_pinned_member


def test_extract_and_check(tmp_path):
    z=tmp_path/'joined.zip'; out=tmp_path/'out';out.mkdir()
    payload=b'base checkpoint'
    with zipfile.ZipFile(z,'w') as f:f.writestr('model (2).safetensors',payload)
    dest=out/'model.safetensors'
    extract_pinned_member(z,dest,expected_size=len(payload),expected_sha256=hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes()==payload
    with pytest.raises(FileExistsError):extract_pinned_member(z,dest,expected_size=len(payload),expected_sha256=hashlib.sha256(payload).hexdigest())


@pytest.mark.parametrize('name,extra,digest',[('../model.safetensors',False,True),('model.safetensors',True,True),('model.safetensors',False,False)])
def test_bad_archive_never_publishes(tmp_path,name,extra,digest):
    z=tmp_path/'joined.zip';dest=tmp_path/'model.safetensors';payload=b'checkpoint'
    with zipfile.ZipFile(z,'w') as f:
        f.writestr(name,payload)
        if extra:f.writestr('extra.txt','unknown')
    with pytest.raises(ValueError):extract_pinned_member(z,dest,expected_size=len(payload),expected_sha256=hashlib.sha256(payload).hexdigest() if digest else '0'*64)
    assert not dest.exists()
