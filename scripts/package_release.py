"""Build two independently verifiable ZIP files; AQ add-on overlays the core."""
from pathlib import Path
import hashlib,json,sys,zipfile
import tomllib
ROOT=Path(__file__).resolve().parents[1]
VERSION=tomllib.loads((ROOT/'pyproject.toml').read_text(encoding='utf-8'))['project']['version']
EXCLUDED={'__pycache__','.pytest_cache','build','omega_geodepth.egg-info','.git'}
CORE_MANIFEST='ARTIFACTS_SHA256.json'
AQ_MANIFEST='AQ_ADDON_SHA256.json'
def is_aq(name): return name.startswith(('models/da3_profiles/aq4/','models/da3_profiles/aq6/'))
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def select(addon):
    files=[]
    for p in ROOT.rglob('*'):
        rel=p.relative_to(ROOT).as_posix()
        if p.is_symlink():raise ValueError('PACKAGE_UNSAFE_SYMLINK: '+rel)
        if not p.is_file() or any(part in EXCLUDED for part in p.relative_to(ROOT).parts) or p.suffix=='.pyc' or rel in (CORE_MANIFEST,AQ_MANIFEST):continue
        if rel == 'models/da3_base/model.safetensors': continue  # separately restored from pinned split ZIP
        if is_aq(rel)==addon: files.append(p)
    return sorted(files)
def build(destination,addon):
    files=select(addon); name=AQ_MANIFEST if addon else CORE_MANIFEST
    metadata={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    manifest={'release':VERSION,'component':'experimental-aq-addon' if addon else 'fp32-fp16-core','file_count':len(files),'files':metadata}
    mp=ROOT/name;mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files+[mp]: z.write(p,f'{ROOT.name}/{p.relative_to(ROOT).as_posix()}')
    with zipfile.ZipFile(destination) as z:
        assert len(z.namelist())==len(files)+1 and z.testzip() is None
        for rel,sha in metadata.items():
            with z.open(f'{ROOT.name}/{rel}') as f:
                assert hashlib.file_digest(f,'sha256').hexdigest()==sha,rel
    result={'zip':str(destination),'bytes':destination.stat().st_size,'sha256':digest(destination),'verified_files':len(files)}
    print(json.dumps(result),flush=True)
    if result['bytes']>=250_000_000: raise ValueError('ARCHIVE_OVER_250_MB')
if __name__=='__main__':
    out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT.parent
    out.mkdir(parents=True,exist_ok=True)
    build(out/f'OMEGA_GeoDepth_Omega_v{VERSION}_DA3_FP32_FP16_5D.zip',False)
    build(out/f'OMEGA_GeoDepth_Omega_v{VERSION}_AQ6_AQ4_Addon.zip',True)
