"""Explicit, bounded and pickle-free prediction contract."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import zipfile
import re
import math
import os
import numpy as np

_REQUIRED = {'depth.npy', 'conf.npy', 'extrinsics.npy', 'intrinsics.npy'}


@dataclass
class Prediction:
    depth: np.ndarray
    conf: np.ndarray
    extrinsics: np.ndarray  # world-to-camera, [N,3,4]
    intrinsics: np.ndarray  # [N,3,3]
    scale_status: str = 'relative'

    def __post_init__(self) -> None:
        for name in ('depth', 'conf', 'extrinsics', 'intrinsics'):
            arr = np.asarray(getattr(self, name))
            if arr.size > 20_000_000:
                raise ValueError('array exceeds resource limit')
            if arr.dtype.kind != 'f':
                raise ValueError(f'{name}: floating point array required')
            setattr(self, name, arr.astype(np.float32, copy=False))
        self.validate()

    def validate(self, max_views: int = 64, max_pixels: int = 20_000_000) -> None:
        if self.depth.ndim != 3:
            raise ValueError('depth must have shape [N,H,W]')
        n, h, w = self.depth.shape
        if n < 1 or n > max_views or h < 1 or w < 1 or n*h*w > max_pixels:
            raise ValueError('depth exceeds shape or resource limits')
        if self.conf.shape != self.depth.shape:
            raise ValueError('conf shape does not match depth')
        if self.extrinsics.shape != (n,3,4):
            raise ValueError('extrinsics must have shape [N,3,4]')
        if self.intrinsics.shape != (n,3,3):
            raise ValueError('intrinsics must have shape [N,3,3]')
        if not np.isfinite(self.extrinsics).all():
            raise ValueError('extrinsics must be finite')
        if not np.isfinite(self.intrinsics).all():
            raise ValueError('intrinsics must be finite')
        k = self.intrinsics
        if (np.any(k[:,0,0] <= 0) or np.any(k[:,1,1] <= 0) or
            not np.allclose(k[:,2,:], [0,0,1], atol=1e-3) or
            np.any(np.abs(np.linalg.det(k)) < 1e-8)):
            raise ValueError('intrinsics are invalid or singular')
        rot = self.extrinsics[:,:,:3]
        orth = rot @ np.swapaxes(rot,1,2)
        if not np.allclose(orth, np.eye(3), atol=1e-4) or not np.allclose(np.linalg.det(rot), 1, atol=1e-4):
            raise ValueError('camera rotation must be approximately orthonormal and right-handed')
        if self.scale_status not in ('relative','metric_calibrated'):
            raise ValueError('invalid scale status')

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.depth) & (self.depth > 0) & np.isfinite(self.conf)


_NATIVE = {'depth_z.npy', 'confidence_raw.npy', 'pose_c2w.npy', 'intrinsics_k.npy'}


def _bounded_header(archive, name, *, max_elements, kinds='f'):
    """Validate shape, dtype and payload size before NumPy can allocate an array."""
    info=archive.getinfo(name)
    with archive.open(info) as fp:
        version=np.lib.format.read_magic(fp)
        if version==(1,0):
            shape,order,dtype=np.lib.format.read_array_header_1_0(fp,max_header_size=10000)
        elif version==(2,0):
            shape,order,dtype=np.lib.format.read_array_header_2_0(fp,max_header_size=10000)
        else:
            raise ValueError('unsupported NPY header version')
        count=math.prod(shape)
        if dtype.kind not in kinds or dtype.hasobject or not 0<count<=max_elements or dtype.itemsize>256:
            raise ValueError('NPY shape or dtype exceeds limit')
        if fp.tell()+count*dtype.itemsize!=info.file_size:
            raise ValueError('NPY payload length mismatch')
    return shape


def load_prediction_npz(path: str | Path, *, max_views: int = 64,
                        max_pixels: int = 20_000_000,
                        max_uncompressed_bytes: int = 1_000_000_000) -> Prediction:
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('NPZ file not found or is a symlink')
    if not 1<=max_views<=64 or not 1<=max_pixels<=20_000_000 or not 1<=max_uncompressed_bytes<=1_000_000_000:
        raise ValueError('invalid NPZ resource limits')
    with path.open('rb') as raw:
        if os.fstat(raw.fileno()).st_size > max_uncompressed_bytes:
            raise ValueError('NPZ file exceeds size limit')
        with zipfile.ZipFile(raw) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            native=_NATIVE.issubset(names)
            external=_REQUIRED.issubset(names)
            if (native==external or len(names)!=len(set(names)) or len(names)>32 or
                any(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}\.npy',name) is None for name in names)):
                raise ValueError('Unexpected NPZ members; ambiguous or missing required arrays')
            if any(i.flag_bits & 1 for i in infos):
                raise ValueError('Encrypted NPZ members are not allowed')
            if sum(i.file_size for i in infos) > max_uncompressed_bytes:
                raise ValueError('NPZ uncompressed size exceeds limit')
            if any(i.file_size > max_uncompressed_bytes or (i.file_size > 100_000_000 and i.file_size > 1000*max(1,i.compress_size)) for i in infos):
                raise ValueError('NPZ member exceeds size or compression limit')
            keys=('depth_z','confidence_raw','pose_c2w','intrinsics_k') if native else ('depth','conf','extrinsics','intrinsics')
            shapes=[_bounded_header(archive,k+'.npy',max_elements=max_pixels if i<2 else max_views*16) for i,k in enumerate(keys)]
            n,h,w=shapes[0] if len(shapes[0])==3 else (0,0,0)
            if not 1<=n<=max_views or h<1 or w<1 or n*h*w>max_pixels or shapes[1]!=(n,h,w) or shapes[2]!=((n,4,4) if native else (n,3,4)) or shapes[3]!=(n,3,3):
                raise ValueError('NPZ array shapes exceed limits or do not match contract')
            def read(key):
                with archive.open(key+'.npy') as f:
                    return np.load(f,allow_pickle=False,max_header_size=10000)
            for key, allowed in [('units',{'arbitrary_scale','arbitrary_relative_unit'}),('provenance',{'relative_estimated'}),('scale_status',{'relative'})]:
                if key+'.npy' in names:
                    if _bounded_header(archive,key+'.npy',max_elements=1,kinds='U')!=():
                        raise ValueError('invalid relative metadata')
                    if str(read(key).item()) not in allowed:
                        raise ValueError('unverified metric/relative claim in NPZ')
            d,c,e,k=[read(key) for key in keys]
    if native:
        from ..geometry import validate_pose_c2w
        e=np.stack([np.linalg.inv(validate_pose_c2w(v))[:3] for v in e])
    pred=Prediction(d,c,e,k)
    pred.validate(max_views=max_views,max_pixels=max_pixels)
    return pred
