"""Separate-process CPU benchmark; relative comparison, not metric accuracy."""
import json, sys, time, resource
from pathlib import Path
import numpy as np
from omega_geodepth.reconstruction5d import build_from_da3_sequence,build_scene5d
ROOT=Path(__file__).resolve().parents[1]
profile=sys.argv[1]; out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
start=time.perf_counter()
frames,meta=build_from_da3_sequence(ROOT/'examples/v08_real_da3/manifest.json',ROOT/'models/da3','DA3-SMALL',source_dir=ROOT/'vendor/da3_src',runtime='native_network',process_res=280,profile=profile,allow_experimental=profile=='int8',verify_reprojection=True)
elapsed=time.perf_counter()-start
scene=build_scene5d(frames,voxel_size=.08,neighbor_radius=.12,max_points_per_frame=24000)
np.savez_compressed(out/f'{profile}.npz',depth=np.stack([f.depth_z for f in frames]),intrinsics=np.stack([f.intrinsics for f in frames]),pose=np.stack([f.pose_c2w for f in frames]),xyz=scene.xyz)
(out/f'{profile}.json').write_text(json.dumps({'profile':meta['weight_profile'],'reprojection':meta['multiview_reprojection_audit'],'load_and_infer_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'points':len(scene.xyz)},indent=2))
