"""Restore official DA3 Base from the six supplied Info-ZIP split volumes."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, tempfile, zipfile
from pathlib import Path
from .da3_adapter import OFFICIAL_WEIGHT_SHA256, file_sha256, register_da3_checkpoint

BASE_SIZE=541_518_028
CONFIG_SHA256='5e34115ebc17bd2d8d43033c5f72e9446ac8833fd61d3fa160b7e67e0bb5b7b5'


def extract_pinned_member(joined: Path, dest: Path, *, expected_size: int, expected_sha256: str) -> None:
    """Read one allowlisted regular member, CRC + byte count + digest, no overwrite."""
    if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
    with zipfile.ZipFile(joined) as z:
        members=z.infolist()
        if len(members)!=1:raise ValueError('expected exactly one archive member')
        info=members[0]
        mode=(info.external_attr>>16)&0o170000
        if (info.filename not in ('model (2).safetensors','model.safetensors') or mode not in (0,0o100000)
            or info.is_dir() or info.file_size!=expected_size or info.flag_bits&1):
            raise ValueError('invalid checkpoint member')
        with tempfile.TemporaryDirectory(prefix='.base-restore-',dir=dest.parent) as temp:
            candidate=Path(temp)/'weights';h=hashlib.sha256();count=0
            with z.open(info) as src,candidate.open('xb') as out:
                while chunk:=src.read(1024*1024):
                    count+=len(chunk)
                    if count>expected_size:raise ValueError('checkpoint exceeds size')
                    h.update(chunk);out.write(chunk)
                out.flush();os.fsync(out.fileno())
            if count!=expected_size or h.hexdigest()!=expected_sha256:
                raise ValueError('checkpoint integrity mismatch')
            os.link(candidate,dest) # atomic publish; fails if a target appears


def restore(split_zip: Path, model_dir: Path) -> dict:
    if model_dir.is_symlink() or not model_dir.is_dir():raise ValueError('existing model directory required')
    cfg=model_dir/'config.json'
    if cfg.is_symlink() or file_sha256(cfg)!=CONFIG_SHA256:raise ValueError('Base config pin mismatch')
    dest=model_dir/'model.safetensors';pin=OFFICIAL_WEIGHT_SHA256['DA3-BASE']
    if dest.is_symlink():raise ValueError('unsafe checkpoint path')
    if dest.exists():
        if dest.stat().st_size!=BASE_SIZE or file_sha256(dest)!=pin:raise ValueError('existing different checkpoint; refusing overwrite')
    else:
        volumes=[split_zip.with_suffix(f'.z{i:02d}') for i in range(1,6)]+[split_zip]
        if split_zip.suffix!='.zip' or any(not p.is_file() or p.is_symlink() for p in volumes):
            raise ValueError('expected .z01 through .z05 and .zip alongside each other')
        if sum(p.stat().st_size for p in volumes)>BASE_SIZE+1_000_000:
            raise ValueError('archive size exceeds limit')
        with tempfile.TemporaryDirectory(prefix='.base-join-',dir=model_dir) as tmp:
            joined=Path(tmp)/'joined.zip'
            subprocess.run(['zip','-s','0',str(split_zip.resolve()),'--out',str(joined.resolve())],check=True,timeout=180,capture_output=True)
            extract_pinned_member(joined,dest,expected_size=BASE_SIZE,expected_sha256=pin)
    return register_da3_checkpoint(model_dir,'DA3-BASE','6544606')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--split-zip',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
    a=p.parse_args();print(json.dumps(restore(a.split_zip,a.model_dir),indent=2))

if __name__=='__main__':main()
