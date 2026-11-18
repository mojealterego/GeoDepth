"""Global Sim(3) synchronization of overlapping static DA3 camera chunks.

Camera 0's chunk fixes the gauge. No external metric scale is manufactured.
Optimizes shared camera centres and orientations, not image bundle adjustment.
"""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .visual_benchmark import _poses,_umeyama,_positive

@dataclass(frozen=True)
class CameraChunk:
    camera_ids: tuple[int,...]
    poses_c2w: np.ndarray

@dataclass(frozen=True)
class ChunkAlignment:
    camera_ids: tuple[int,...]
    poses_c2w: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    translations: np.ndarray
    initial_cost: float
    final_cost: float
    max_center_error: float
    max_rotation_error_deg: float
    passed: bool
    edge_count: int
    units: str='arbitrary_scale'

    def transform_depth(self,chunk_index,depth_z):
        if not self.passed:raise ValueError('CHUNK_GRAPH_REJECTED')
        if not isinstance(chunk_index,int) or not 0<=chunk_index<len(self.scales):raise ValueError('CHUNK_INDEX')
        a=np.asarray(depth_z)
        if a.dtype.kind not in 'fiu' or a.size>24_000_000 or np.isinf(a).any() or np.any(a[np.isfinite(a)]<=0):raise ValueError('CHUNK_DEPTH')
        out=a.astype(np.float64)*self.scales[chunk_index]
        if np.isinf(out).any():raise ValueError('CHUNK_DEPTH_OVERFLOW')
        return out


def align_chunks(chunks,*,center_tolerance,rotation_tolerance_deg,max_nfev=100):
    """Joint robust pose agreement in a fixed first-chunk gauge; fail closed.

    Every edge with >=3 non-collinear common camera centres initializes a
    relative similarity; all shared observations (including 1/2-camera overlaps)
    enter the global optimization. Every observation pair is gated afterwards.
    Chunk IDs identify the SAME immutable camera capture, not temporal matches.
    """
    ct=_positive(center_tolerance);rt=_positive(rotation_tolerance_deg)
    if rt>45 or not isinstance(max_nfev,int) or not 10<=max_nfev<=300 or not 2<=len(chunks)<=16:raise ValueError('CHUNK_GRAPH_LIMIT')
    poses=[];ids=[];lookup=[]
    for ch in chunks:
        p=_poses(ch.poses_c2w);k=tuple(ch.camera_ids)
        if not 3<=len(k)<=8 or len(k)!=len(p) or len(set(k))!=len(k) or any(type(v) is not int or not 0<=v<=1_000_000 for v in k):raise ValueError('CHUNK_GRAPH_IDS')
        poses.append(p);ids.append(k);lookup.append({v:i for i,v in enumerate(k)})
    n=len(chunks);edges=[];initial_edges=[]
    for i,j in combinations(range(n),2):
        shared=sorted(set(ids[i])&set(ids[j]));
        if not shared:continue
        a=poses[i][[lookup[i][v] for v in shared]];b=poses[j][[lookup[j][v] for v in shared]]
        edges.append((i,j,a,b))
        if len(shared)>=3:
            try: s,R,t=_umeyama(b[:,:3,3],a[:,:3,3])
            except ValueError:continue
            if .01<=s<=100:initial_edges.append((i,j,s,R,t))
    scales=[None]*n;rots=[None]*n;trans=[None]*n;scales[0]=1.;rots[0]=np.eye(3);trans[0]=np.zeros(3)
    # Propagate along a connected graph of nondegenerate overlap fits.
    while any(v is None for v in scales):
        progress=False
        for i,j,s,R,t in initial_edges:
            if scales[i] is None and scales[j] is not None:
                i,j,s,R,t=j,i,1/s,R.T,-R.T@t/s
            if scales[i] is not None and scales[j] is None:
                scales[j]=scales[i]*s;rots[j]=rots[i]@R;trans[j]=scales[i]*rots[i]@t+trans[i];progress=True
        if not progress:raise ValueError('CHUNK_GRAPH_DISCONNECTED_OR_DEGENERATE')
    if any(not 1e-4<q<1e4 for q in scales):raise ValueError('CHUNK_GRAPH_SCALE_RANGE')
    x0=np.concatenate([np.r_[np.log(scales[i]),Rotation.from_matrix(rots[i]).as_rotvec(),trans[i]] for i in range(1,n)])
    def unpack(x):
        z=x.reshape(-1,7);return np.r_[1.,np.exp(z[:,0])],np.concatenate([np.eye(3)[None],Rotation.from_rotvec(z[:,1:4]).as_matrix()]),np.vstack([np.zeros(3),z[:,4:7]])
    def apply(s,R,t,p):
        return s*p[:,:3,3]@R.T+t,R@p[:,:3,:3]
    def residual(x):
        s,R,t=unpack(x);parts=[]
        for i,j,a,b in edges:
            ac,ar=apply(s[i],R[i],t[i],a);bc,br=apply(s[j],R[j],t[j],b)
            parts.extend([(ac-bc).ravel()/ct,Rotation.from_matrix(ar@br.transpose(0,2,1)).as_rotvec().ravel()/np.radians(rt)])
        return np.concatenate(parts)
    def cost(x):return float(np.sum(2*(np.sqrt(1+residual(x)**2)-1)))
    initial=cost(x0);lo=np.tile([-np.log(1e4),-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],n-1);hi=-lo
    opt=least_squares(residual,x0,bounds=(lo,hi),loss='soft_l1',max_nfev=max_nfev)
    if not np.isfinite(opt.x).all() or cost(opt.x)>initial+1e-8:raise ValueError('CHUNK_GRAPH_OPTIMIZATION_FAILED')
    s,R,t=unpack(opt.x);maxc=0.;maxr=0.
    for i,j,a,b in edges:
        ac,ar=apply(s[i],R[i],t[i],a);bc,br=apply(s[j],R[j],t[j],b);maxc=max(maxc,float(np.linalg.norm(ac-bc,axis=1).max()));maxr=max(maxr,float(np.degrees(Rotation.from_matrix(ar@br.transpose(0,2,1)).magnitude()).max()))
    # Use first occurrence rather than silently averaging inconsistent cameras.
    merged={}
    for i,p in enumerate(poses):
        c,r=apply(s[i],R[i],t[i],p)
        for j,v in enumerate(ids[i]):
            if v not in merged:
                m=np.eye(4);m[:3,:3]=r[j];m[:3,3]=c[j];merged[v]=m
    order=tuple(sorted(merged));return ChunkAlignment(order,np.stack([merged[v] for v in order]),s,R,t,initial,cost(opt.x),maxc,maxr,bool(opt.success and maxc<=ct and maxr<=rt),len(edges))


def align_da3_chunks(chunks,*,center_tolerance,rotation_tolerance_deg,depth_log_p90_limit=.20,intrinsics_relative_limit=.05,min_depth_overlap=.70):
    """Accept only same-model/same-image chunks with audited camera/depth overlap.

    A passing result returns first-occurrence views (one per capture ID), scaled
    depths and transformed poses in the first chunk's arbitrary world gauge.
    No dense blending, temporal inference, or assumed metric units.
    """
    from dataclasses import replace
    from .da3_adapter import DA3Output
    if not 2<=len(chunks)<=16 or not np.isfinite(depth_log_p90_limit) or not 0<depth_log_p90_limit<=1 or not np.isfinite(intrinsics_relative_limit) or not 0<intrinsics_relative_limit<=.5 or not np.isfinite(min_depth_overlap) or not .5<=min_depth_overlap<=1:raise ValueError('CHUNK_BRIDGE_LIMIT')
    first=chunks[0][1]
    if not isinstance(first,DA3Output):raise ValueError('CHUNK_BRIDGE_TYPE')
    identity=(first.model_id,first.weights_sha256,first.pose_decoder,first.pose_conditioned)
    seen={};groups=[];total=0
    for ids,p in chunks:
        if not isinstance(p,DA3Output) or p.units!='arbitrary_scale' or p.provenance!='relative_estimated' or (p.model_id,p.weights_sha256,p.pose_decoder,p.pose_conditioned)!=identity:raise ValueError('CHUNK_BRIDGE_PROVENANCE')
        ids=tuple(ids);z=p.depth_z;k=p.intrinsics_k
        if z.ndim!=3 or len(ids)!=len(z) or z.shape[1:]!=first.depth_z.shape[1:] or min(z.shape[1:])<2 or z.dtype.kind!='f' or p.confidence_raw.shape!=z.shape or not np.isfinite(p.confidence_raw).all() or p.rgb.shape!=(*z.shape,3) or p.rgb.dtype!=np.uint8 or k.shape!=(len(ids),3,3) or not np.isfinite(k).all() or np.any(k[:,0,0]<=0) or np.any(k[:,1,1]<=0) or not np.allclose(k[:,2,:],[0,0,1]) or np.isinf(z).any() or np.any(z[np.isfinite(z)]<=0):raise ValueError('CHUNK_BRIDGE_ARRAYS')
        total+=z.size
        if total>24_000_000:raise ValueError('CHUNK_BRIDGE_RESOURCE_LIMIT')
        groups.append(CameraChunk(ids,p.pose_c2w))
        for j,cam in enumerate(ids):
            if cam in seen and not np.array_equal(p.rgb[j],seen[cam]):raise ValueError('CHUNK_BRIDGE_CAPTURE_MISMATCH')
            seen[cam]=p.rgb[j]
    graph=align_chunks(groups,center_tolerance=center_tolerance,rotation_tolerance_deg=rotation_tolerance_deg)
    rows=[];passed=graph.passed;h,w=first.depth_z.shape[1:]
    for i,j in combinations(range(len(chunks)),2):
        shared=sorted(set(groups[i].camera_ids)&set(groups[j].camera_ids))
        for cam in shared:
            a,b=chunks[i][1],chunks[j][1];ai=groups[i].camera_ids.index(cam);bi=groups[j].camera_ids.index(cam)
            x=a.depth_z[ai].astype(float)*graph.scales[i];y=b.depth_z[bi].astype(float)*graph.scales[j];valid=np.isfinite(x)&np.isfinite(y)&(x>0)&(y>0)
            coverage=float(valid.mean());p90=float(np.quantile(np.abs(np.log(x[valid])-np.log(y[valid])),.9)) if valid.any() else None
            dk=float(np.max(np.abs(a.intrinsics_k[ai]-b.intrinsics_k[bi])/np.array([[w,w,w],[h,h,h],[1,1,1]])))
            ok=coverage>=min_depth_overlap and p90 is not None and p90<=depth_log_p90_limit and dk<=intrinsics_relative_limit
            rows.append({'chunk_a':i,'chunk_b':j,'camera_id':cam,'depth_overlap':coverage,'depth_log_p90':p90,'intrinsics_delta_normalized':dk,'passed':bool(ok)});passed=passed and ok
    report={'passed':bool(passed),'camera_graph_passed':graph.passed,'initial_cost':graph.initial_cost,'final_cost':graph.final_cost,'max_center_error_first_chunk_units':graph.max_center_error,'max_rotation_error_deg':graph.max_rotation_error_deg,'shared_observations':rows,'units':'arbitrary_scale','scope':'internal agreement of static repeated captures; not independent physical validation'}
    if not passed:return graph,None,report
    views={}
    for i,(_,p) in enumerate(chunks):
        for j,cam in enumerate(groups[i].camera_ids):
            if cam not in views:views[cam]=(graph.transform_depth(i,p.depth_z[j]).astype(np.float32),p.confidence_raw[j],p.intrinsics_k[j],p.rgb[j])
    order=[views[cam] for cam in graph.camera_ids];z,c,k,rgb=(np.stack(a) for a in zip(*order))
    if np.isinf(z).any():raise ValueError('CHUNK_BRIDGE_DEPTH_OVERFLOW')
    return graph,replace(first,depth_z=z,confidence_raw=c,intrinsics_k=k,pose_c2w=graph.poses_c2w,rgb=rgb,ray_audit=None),report


def plan_static_chunks(count,*,chunk_size=6,overlap=4):
    """Plan bounded overlapping view windows; no broken short last window."""
    if any(type(v) is not int for v in (count,chunk_size,overlap)) or not 9<=count<=64 or not 4<=chunk_size<=8 or not 3<=overlap<chunk_size:raise ValueError('CHUNK_PLAN_LIMIT')
    starts=list(range(0,count-chunk_size+1,chunk_size-overlap))
    if starts[-1]!=count-chunk_size:starts.append(count-chunk_size)
    if len(starts)>16:raise ValueError('CHUNK_PLAN_TOO_MANY_WINDOWS')
    return [tuple(range(s,s+chunk_size)) for s in starts]


def infer_static_chunks(engine,images,*,process_res=140,chunk_size=6,overlap=4,pose_decoder='fast_ray',center_tolerance=.15,rotation_tolerance_deg=5):
    plan=plan_static_chunks(len(images),chunk_size=chunk_size,overlap=overlap)
    if type(process_res) is not int or not 112<=process_res<=504 or pose_decoder not in ('token','ray','fast_ray'):raise ValueError('CHUNK_INFERENCE_OPTIONS')
    if len(plan)*chunk_size*process_res**2>20_000_000:raise ValueError('CHUNK_INFERENCE_RESOURCE_LIMIT')
    _positive(center_tolerance);_positive(rotation_tolerance_deg)
    if rotation_tolerance_deg>45:raise ValueError('CHUNK_INFERENCE_OPTIONS')
    outputs=[]
    for group in plan:
        outputs.append((group,engine.inference([images[i] for i in group],process_res=process_res,pose_decoder=pose_decoder)))
    g,p,report=align_da3_chunks(outputs,center_tolerance=center_tolerance,rotation_tolerance_deg=rotation_tolerance_deg)
    report.update(groups=[list(x) for x in plan],camera_ids=list(g.camera_ids),scales=g.scales.tolist(),model=outputs[0][1].model_id,weights_sha256=outputs[0][1].weights_sha256,pose_decoder=pose_decoder,process_res=process_res,thresholds={'center_first_chunk_units':center_tolerance,'rotation_deg':rotation_tolerance_deg,'depth_log_p90':.2,'intrinsics_relative':.05,'min_depth_overlap':.70})
    return g,p,report


def write_chunk_result(output,prediction,report):
    """New output directory only. Audit is the commit marker, published last."""
    from pathlib import Path
    from io import BytesIO
    import json
    from .infer import save_atomic
    if type(report.get('passed')) is not bool or report['passed']!=(prediction is not None):raise ValueError('CHUNK_EXPORT_GATE_MISMATCH')
    audit=json.dumps(report,indent=2,allow_nan=False).encode();root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    if prediction is not None:
        if prediction.units!='arbitrary_scale' or prediction.provenance!='relative_estimated':raise ValueError('CHUNK_EXPORT_RELATIVE_ONLY')
        b=BytesIO();np.savez_compressed(b,depth_z=prediction.depth_z,confidence_raw=prediction.confidence_raw,intrinsics_k=prediction.intrinsics_k,pose_c2w=prediction.pose_c2w,rgb=prediction.rgb,units=np.array(prediction.units),provenance=np.array(prediction.provenance));save_atomic(root/'prediction.npz',b.getvalue())
    save_atomic(root/'audit.json',audit)


def main():
    import argparse
    from pathlib import Path
    from .da3_real import DA3RealCPU
    p=argparse.ArgumentParser(description='GeoDepth static overlapping-view chunk synchronization; relative units only')
    p.add_argument('--static-scene',action='store_true',required=True,help='Explicit attestation: all captures represent the same static scene')
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--model-id',choices=('DA3-SMALL','DA3-BASE'),default='DA3-SMALL');p.add_argument('--input',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--process-res',type=int,default=140);p.add_argument('--chunk-size',type=int,default=6);p.add_argument('--overlap',type=int,default=4);p.add_argument('--pose-decoder',choices=('token','ray','fast_ray'),default='fast_ray');p.add_argument('--center-tolerance',type=float,default=.15);p.add_argument('--rotation-tolerance-deg',type=float,default=5.)
    a=p.parse_args()
    if a.output.exists() or a.output.is_symlink():p.error('output path already exists')
    plan_static_chunks(len(a.input),chunk_size=a.chunk_size,overlap=a.overlap)
    engine=DA3RealCPU(a.model_dir,source_dir=a.source_dir,model_id=a.model_id)
    try:
        _,pred,report=infer_static_chunks(engine,a.input,process_res=a.process_res,chunk_size=a.chunk_size,overlap=a.overlap,pose_decoder=a.pose_decoder,center_tolerance=a.center_tolerance,rotation_tolerance_deg=a.rotation_tolerance_deg)
    except ValueError as exc:
        pred=None;report={'passed':False,'reason':str(exc),'scope':'static chunk inference rejected; no geometry exported'}
    write_chunk_result(a.output,pred,report)
    if pred is None:raise SystemExit(2)

if __name__=='__main__':main()
