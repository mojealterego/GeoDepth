"""Fail-closed provenance and extraction of the explicitly supplied DA3 source ZIP.

This module never executes upstream code and never downloads weights. The recorded
SHA-256 identifies the archive supplied for this release, not an independently
verified official upstream commit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import zipfile

BUNDLED_DA3_SHA256 = '6d2ba1d07a69f3d260394e49c7545aed8f3cc7df8633469ae79782ee34978d42'
BUNDLE_ROOT = 'Depth-Anything-3-main'
SOURCE_MANIFEST = 'omega_source_integrity.json'
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_UNPACKED_BYTES = 256 * 1024 * 1024
MAX_SINGLE_BYTES = 64 * 1024 * 1024
MAX_ENTRIES = 2000
REQUIRED = (
    'LICENSE', 'pyproject.toml',
    'src/depth_anything_3/api.py',
    'src/depth_anything_3/cfg.py',
    'src/depth_anything_3/registry.py',
    'src/depth_anything_3/model/da3.py',
    'src/depth_anything_3/configs/da3-small.yaml',
    'src/depth_anything_3/configs/da3-base.yaml',
)
WEIGHT_SUFFIXES = {'.safetensors','.pt','.pth','.bin','.gguf','.onnx','.ckpt'}
EXECUTABLE_SUFFIXES = {'.py','.pth','.so','.pyd','.dll','.dylib'}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _clean_name(name: str) -> str:
    if '\\' in name or '\x00' in name or name.startswith('/'):
        raise ValueError('DA3_SOURCE_UNSAFE_PATH')
    raw = name.rstrip('/')
    parts = raw.split('/')
    if any(x in ('','.','..') or ':' in x for x in parts) or parts[0] != BUNDLE_ROOT:
        raise ValueError('DA3_SOURCE_UNSAFE_PATH')
    if len(parts) == 1: return ''
    p = PurePosixPath(*parts[1:])
    if p.is_absolute() or '..' in p.parts:
        raise ValueError('DA3_SOURCE_UNSAFE_PATH')
    return p.as_posix()


def _precheck(z: zipfile.ZipFile) -> tuple[list[tuple[zipfile.ZipInfo,str]],int]:
    info = z.infolist()
    if len(info)>MAX_ENTRIES: raise ValueError('DA3_SOURCE_TOO_MANY_ENTRIES')
    seen = set()
    out=[];total=0
    for entry in info:
        rel = _clean_name(entry.filename)
        if rel:
            case = rel.casefold()
            if case in seen: raise ValueError('DA3_SOURCE_DUPLICATE')
            seen.add(case)
        mode = entry.external_attr >> 16
        ftype = stat.S_IFMT(mode)
        if ftype == stat.S_IFLNK: raise ValueError('DA3_SOURCE_SYMLINK')
        if ftype not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise ValueError('DA3_SOURCE_UNSUPPORTED_ENTRY')
        if entry.flag_bits & 1: raise ValueError('DA3_SOURCE_ENCRYPTED')
        if not entry.is_dir():
            if not rel: raise ValueError('DA3_SOURCE_UNSAFE_PATH')
            total+=entry.file_size
            if entry.file_size>MAX_SINGLE_BYTES or total>MAX_UNPACKED_BYTES:
                raise ValueError('DA3_SOURCE_UNPACKED_RESOURCE_LIMIT')
        out.append((entry,rel))
    return out,total


def _check_api_contract(z:zipfile.ZipFile) -> None:
    path = BUNDLE_ROOT + '/src/depth_anything_3/api.py'
    try: tree=ast.parse(z.read(path).decode('utf-8'))
    except (KeyError,UnicodeDecodeError,SyntaxError) as e:
        raise ValueError('DA3_SOURCE_INVALID_API') from e
    classes={c.name:c for c in tree.body if isinstance(c,ast.ClassDef)}
    da=classes.get('DepthAnything3')
    if da is None or not {'inference','forward'} <= {x.name for x in da.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))}:
        raise ValueError('DA3_SOURCE_INVALID_API')


def inspect_da3_source_archive(archive:Path,*,expected_sha256:str=BUNDLED_DA3_SHA256)->dict:
    path=Path(archive)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_ARCHIVE_BYTES:
        raise ValueError('DA3_SOURCE_ARCHIVE_MISSING_OR_UNSAFE')
    sha=_sha256_file(path)
    if sha!=expected_sha256: raise ValueError('DA3_SOURCE_ARCHIVE_HASH_MISMATCH')
    with zipfile.ZipFile(path) as z:
        entries,total=_precheck(z)
        files={rel for e,rel in entries if rel and not e.is_dir()}
        for required in REQUIRED:
            if required not in files:raise ValueError('DA3_SOURCE_REQUIRED_FILE_MISSING')
        _check_api_contract(z)
        source_only=not any(Path(f).suffix.lower() in WEIGHT_SUFFIXES for f in files)
        return {
            'sha256':sha,'entry_count':len(entries),'file_count':len(files),
            'unpacked_bytes':total,'source_only':source_only,'api_contract_verified':True,
            'source_provenance':'user_supplied_source_zip_not_independently_verified',
            'upstream_commit_verified':False,'license_file_present':True,
            'contains_pretrained_weights':not source_only,
        }


def extract_da3_source(archive:Path,destination:Path,*,expected_sha256:str=BUNDLED_DA3_SHA256)->dict:
    report=inspect_da3_source_archive(archive,expected_sha256=expected_sha256)
    target=Path(destination).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError('DA3_SOURCE_TARGET_EXISTS')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.parent.is_symlink(): raise ValueError('DA3_SOURCE_UNSAFE_TARGET')
    staging=Path(tempfile.mkdtemp(prefix='.da3-staging-',dir=target.parent))
    try:
        digests={}
        with zipfile.ZipFile(archive) as z:
            entries,_=_precheck(z)
            for info,rel in entries:
                if not rel:continue
                dest=staging/rel
                if info.is_dir():dest.mkdir(parents=True,exist_ok=True);continue
                dest.parent.mkdir(parents=True,exist_ok=True)
                h=hashlib.sha256()
                with z.open(info) as source, dest.open('xb') as sink:
                    while True:
                        data=source.read(1024*1024)
                        if not data:break
                        h.update(data)
                        sink.write(data)
                digests[rel]=h.hexdigest()
        record={'schema_version':1,'archive_sha256':report['sha256'],
                'provenance':report['source_provenance'],'upstream_commit_verified':False,
                'files_sha256':digests}
        (staging/SOURCE_MANIFEST).write_text(json.dumps(record,sort_keys=True,indent=2)+'\n',encoding='utf-8')
        if target.exists() or target.is_symlink(): raise FileExistsError('DA3_SOURCE_TARGET_EXISTS')
        os.replace(staging,target)
        return report
    finally:
        if staging.exists():shutil.rmtree(staging)


def verify_extracted_da3_source(directory:Path)->dict:
    root=Path(directory)
    if not root.is_dir() or root.is_symlink():raise ValueError('DA3_SOURCE_DIRECTORY_MISSING_OR_UNSAFE')
    path=root/SOURCE_MANIFEST
    if not path.is_file() or path.is_symlink():raise ValueError('DA3_SOURCE_MANIFEST_MISSING')
    try:record=json.loads(path.read_text(encoding='utf-8'))
    except (ValueError,UnicodeDecodeError) as e:raise ValueError('DA3_SOURCE_INVALID_MANIFEST') from e
    listed=record.get('files_sha256')
    if (record.get('archive_sha256')!=BUNDLED_DA3_SHA256 or not isinstance(listed,dict)
            or not listed or any(r not in listed for r in REQUIRED)):
        raise ValueError('DA3_SOURCE_INVALID_MANIFEST')
    for rel,expected in listed.items():
        if (not isinstance(rel,str) or not isinstance(expected,str)
                or len(expected)!=64 or _clean_name(BUNDLE_ROOT+'/'+rel)!=rel):
            raise ValueError('DA3_SOURCE_INVALID_MANIFEST')
        p=root/rel
        for ancestor in (p,*p.parents):
            if ancestor==root.parent:break
            if ancestor.is_symlink():raise ValueError('DA3_SOURCE_SYMLINK')
        if not p.is_file() or _sha256_file(p)!=expected:
            raise ValueError('DA3_SOURCE_TAMPERED')
    for p in root.rglob('*'):
        if p.is_symlink():raise ValueError('DA3_SOURCE_SYMLINK')
        if p.is_file() and p.name!=SOURCE_MANIFEST and p.suffix.lower() in EXECUTABLE_SUFFIXES:
            rel=p.relative_to(root).as_posix()
            if rel not in listed and '__pycache__' not in p.parts:
                raise ValueError('DA3_SOURCE_UNDECLARED_FILE')
    return {'ok':True,'archive_sha256':BUNDLED_DA3_SHA256,'verified_source_files':len(listed),
            'upstream_commit_verified':False}


def main() -> None:
    parser=argparse.ArgumentParser(description='Offline unpack/check of user-supplied DA3 source; NOT pretrained weights')
    parser.add_argument('command',choices=('inspect','extract','verify'))
    parser.add_argument('--archive',type=Path,default=Path(__file__).resolve().parents[1]/'vendor'/f'{BUNDLE_ROOT}.zip')
    parser.add_argument('--target',type=Path)
    a=parser.parse_args()
    if a.command=='inspect':out=inspect_da3_source_archive(a.archive)
    else:
        if a.target is None:parser.error('--target is required for extract/verify')
        out=extract_da3_source(a.archive,a.target) if a.command=='extract' else verify_extracted_da3_source(a.target)
    print(json.dumps(out,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
