"""5D spatiotemporal scene-field reconstruction, offline and content-agnostic.

Precisely defined five coordinates: (world_X, world_Y, world_Z, time_s,
normalized_sRGB_luma_proxy). Appearance is an image feature, NOT a fifth
physical/spatial dimension, metric radiance or new geometric degree of freedom.

The field is constructed from calibrated RGB+depth+poses. No geometry in unseen
regions or true inter-frame object correspondence is fabricated.
"""
from __future__ import annotations
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np
from scipy.spatial import cKDTree
from .geometry import validate_pose_c2w
from .infer import save_atomic

MAX_INPUT_PIXELS=24_000_000
MAX_NPZ_UNCOMPRESSED=512*1024*1024


@dataclass
class Frame5D:
    time_s:float
    depth_z:np.ndarray
    rgb:np.ndarray
    intrinsics:np.ndarray
    pose_c2w:np.ndarray
    valid:np.ndarray
    sigma_log_z:np.ndarray|None=None
    confidence_raw:np.ndarray|None=None # DA3 raw confidence, NOT a calibrated probability
    static_mask:np.ndarray|None=None # True=eligible static pixel; None=unverified static assumption


@dataclass
class Scene5D:
    xyz:np.ndarray
    time_s:np.ndarray
    luma:np.ndarray
    rgb:np.ndarray
    confidence_proxy:np.ndarray
    velocity:np.ndarray
    matches_valid:np.ndarray
    frame_index:np.ndarray
    observations:np.ndarray
    units:str
    provenance:str
    voxel_size:float
    neighbor_radius:float
    note:str='nearest-neighbor scene flow is an unverified correspondence estimate'


def _validate_frame(f:Frame5D)->None:
    depth=np.asarray(f.depth_z)
    if depth.ndim!=2 or depth.size>MAX_INPUT_PIXELS or not depth.size:
        raise ValueError('INVALID_FRAME_DEPTH_SHAPE_OR_RESOURCE_LIMIT')
    h,w=depth.shape
    if (np.asarray(f.rgb).shape!=(h,w,3) or np.asarray(f.rgb).dtype!=np.uint8
        or np.asarray(f.valid).shape!=(h,w)):
        raise ValueError('INVALID_FRAME_RGB_OR_VALID')
    k=np.asarray(f.intrinsics,dtype=np.float64)
    if (k.shape!=(3,3) or not np.isfinite(k).all() or
        min(k[0,0],k[1,1])<=0 or not np.allclose(k[2],[0,0,1],atol=1e-6)):
        raise ValueError('INVALID_INTRINSICS')
    validate_pose_c2w(f.pose_c2w)
    if not math.isfinite(float(f.time_s)):
        raise ValueError('INVALID_TIMESTAMP')
    if f.sigma_log_z is not None and np.asarray(f.sigma_log_z).shape!=depth.shape:
        raise ValueError('INVALID_SIGMA_SHAPE')
    if f.confidence_raw is not None and np.asarray(f.confidence_raw).shape!=depth.shape:
        raise ValueError('INVALID_DA3_CONFIDENCE_SHAPE')
    if f.static_mask is not None and (np.asarray(f.static_mask).shape!=depth.shape or np.asarray(f.static_mask).dtype!=bool):
        raise ValueError('INVALID_STATIC_MASK')


def _points_from_frame(f:Frame5D,*,max_points:int=200_000)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    depth=np.asarray(f.depth_z,dtype=np.float64)
    h,w=depth.shape
    valid=np.asarray(f.valid,dtype=bool)&np.isfinite(depth)&(depth>0)
    if not valid.any():raise ValueError('NO_VALID_DEPTH_IN_FRAME')
    if max_points<1:raise ValueError('INVALID_MAX_POINTS')
    stride=max(1,int(np.ceil(np.sqrt(depth.size/max_points))))
    yy,xx=np.mgrid[0:h:stride,0:w:stride]
    mask=valid[::stride,::stride]
    xx=xx[mask].astype(np.float64);yy=yy[mask].astype(np.float64)
    z=depth[::stride,::stride][mask]
    k=np.asarray(f.intrinsics,dtype=np.float64)
    ny=(yy-k[1,2])/k[1,1]
    nx=(xx-k[0,2]-k[0,1]*ny)/k[0,0]
    pts=np.column_stack((nx*z,ny*z,z))
    pose=np.asarray(f.pose_c2w,dtype=np.float64)
    pts=pts@pose[:3,:3].T+pose[:3,3]
    color=np.asarray(f.rgb)[::stride,::stride][mask]
    if f.confidence_raw is not None:
        raw=np.asarray(f.confidence_raw,dtype=np.float64)[::stride,::stride][mask]
        finite=raw[np.isfinite(raw)]
        if len(finite):
            low,hi=np.quantile(finite,[.05,.95])
            conf=np.where(np.isfinite(raw),.2+.8*np.clip((raw-low)/max(hi-low,1e-9),0,1),0.)
        else:conf=np.zeros(len(pts),dtype=np.float64)
    elif f.sigma_log_z is None:
        conf=np.ones(len(pts),dtype=np.float64)
    else:
        sig=np.asarray(f.sigma_log_z,dtype=np.float64)[::stride,::stride][mask]
        conf=np.where(np.isfinite(sig)&(sig>=0),1/(1+np.maximum(sig,0)**2),0.)
    luma=(.2126*color[:,0]+.7152*color[:,1]+.0722*color[:,2])/255.
    return pts,color.astype(np.uint8),luma.astype(np.float64),conf


def _fuse_voxels(pts:np.ndarray,rgb:np.ndarray,luma:np.ndarray,conf:np.ndarray,
                 voxel_size:float):
    ids=np.floor(pts/voxel_size).astype(np.int64)
    _,reverse=np.unique(ids,axis=0,return_inverse=True)
    count=int(reverse.max())+1
    weights=np.clip(conf,1e-5,1.0)
    summed=np.bincount(reverse,weights=weights,minlength=count)
    xyz=np.column_stack([np.bincount(reverse,weights=pts[:,i]*weights,minlength=count)/summed for i in range(3)])
    colors=np.column_stack([np.bincount(reverse,weights=rgb[:,i]*weights,minlength=count)/summed for i in range(3)])
    appearance=np.bincount(reverse,weights=luma*weights,minlength=count)/summed
    proxy=np.bincount(reverse,weights=conf*weights,minlength=count)/np.bincount(reverse,minlength=count)
    observations=np.bincount(reverse,minlength=count).astype(np.int32)
    return xyz.astype(np.float32),np.rint(colors).clip(0,255).astype(np.uint8),appearance.astype(np.float32),proxy.astype(np.float32),observations


def build_scene5d(frames:list[Frame5D],*,voxel_size:float=.025,
                  neighbor_radius:float=.1,depth_jump_rel:float=.2,
                  units:str='arbitrary_scale',provenance:str='relative_estimated',
                  max_points_per_frame:int=200_000)->Scene5D:
    if len(frames)<2:raise ValueError('TEMPORAL_RECONSTRUCTION_REQUIRES_TWO_FRAMES')
    if (not np.isfinite(voxel_size) or voxel_size<=0 or not np.isfinite(neighbor_radius)
        or neighbor_radius<=0 or depth_jump_rel<=0 or max_points_per_frame<1):
        raise ValueError('INVALID_RECONSTRUCTION_PARAMETERS')
    if units not in ('meters','arbitrary_scale'):
        raise ValueError('INVALID_UNITS')
    if provenance not in ('relative_estimated','scale_aligned','metric_estimated','metric_measured'):
        raise ValueError('INVALID_PROVENANCE')
    if units=='meters' and provenance=='relative_estimated':
        raise ValueError('METRIC_PROVENANCE_REQUIRED')
    if units=='arbitrary_scale' and provenance!='relative_estimated':
        raise ValueError('METRIC_PROVENANCE_IN_ARBITRARY_UNITS')
    for f in frames:_validate_frame(f)
    times=[float(f.time_s) for f in frames]
    if any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('TIMESTAMPS_MUST_BE_STRICTLY_INCREASING')
    allpts=[]; allrgb=[]; allluma=[]; allconf=[]; allobs=[]; allframe=[]
    for idx,f in enumerate(frames):
        pts,rgb,luma,conf=_points_from_frame(f,max_points=max_points_per_frame)
        p,c,l,q,obs=_fuse_voxels(pts,rgb,luma,conf,voxel_size)
        allpts.append(p);allrgb.append(c);allluma.append(l);allconf.append(q);allobs.append(obs)
        allframe.append(np.full(len(p),idx,dtype=np.int32))
    xyz=np.concatenate(allpts);colors=np.concatenate(allrgb)
    frame_index=np.concatenate(allframe)
    t=np.asarray(times,dtype=np.float64)[frame_index]
    vel=np.full((len(xyz),3),np.nan,dtype=np.float32)
    matches=np.zeros(len(xyz),dtype=bool)
    # Explicitly labeled approximate nearest-neighbor correspondences; not optical flow ground truth.
    for idx in range(1,len(frames)):
        prior=np.flatnonzero(frame_index==idx-1)
        now=np.flatnonzero(frame_index==idx)
        tree=cKDTree(xyz[prior])
        dist,hit=tree.query(xyz[now],k=1,distance_upper_bound=neighbor_radius,workers=1)
        reverse=cKDTree(xyz[now]).query(xyz[prior],k=1,workers=1)[1]
        good=np.isfinite(dist)&(hit<len(prior))
        candidate=np.flatnonzero(good)
        mutual=candidate[reverse[hit[candidate]]==candidate]
        if len(mutual):
            src=prior[hit[mutual]]
            dst=now[mutual]
            # Reject clear photometric mismatches; not a substitute for optical flow.
            photometric=np.linalg.norm((colors[dst].astype(float)-colors[src].astype(float))/255.,axis=1)<.5
            src,dst=src[photometric],dst[photometric]
            motion=(xyz[dst]-xyz[src])/(times[idx]-times[idx-1])
            matches[src]=True; matches[dst]=True
            vel[src]=motion; vel[dst]=motion
    return Scene5D(xyz,t,np.concatenate(allluma),colors,np.concatenate(allconf),vel,matches,
                   frame_index,np.concatenate(allobs),units,provenance,voxel_size,neighbor_radius)


def depth_surface_mesh(f:Frame5D,*,max_rel_discontinuity:float=.2):
    """Project each valid depth pixel into a true triangle surface, omitting depth jumps."""
    _validate_frame(f)
    if max_rel_discontinuity<=0:raise ValueError('INVALID_DEPTH_EDGE_THRESHOLD')
    d=np.asarray(f.depth_z,dtype=np.float64)
    if d.size>2_000_000:raise ValueError('MESH_TOO_MANY_PIXELS')
    h,w=d.shape
    ys,xs=np.mgrid[:h,:w]
    k=np.asarray(f.intrinsics,dtype=np.float64)
    y=(ys-k[1,2])/k[1,1]
    x=(xs-k[0,2]-k[0,1]*y)/k[0,0]
    pts=np.stack((x*d,y*d,d),axis=-1)
    p=np.asarray(f.pose_c2w,dtype=np.float64)
    world=pts.reshape(-1,3)@p[:3,:3].T+p[:3,3]
    mask=np.asarray(f.valid,dtype=bool)&np.isfinite(d)&(d>0)
    indices=np.arange(h*w,dtype=np.int32).reshape(h,w)
    q=np.stack((indices[:-1,:-1],indices[:-1,1:],indices[1:,:-1],indices[1:,1:]),axis=-1).reshape(-1,4)
    qd=d.reshape(-1)[q]
    ok=mask.reshape(-1)[q].all(axis=1)
    finite=np.isfinite(qd).all(axis=1)
    good=ok&finite
    if np.any(good):
        ratio=qd[good].max(axis=1)/np.maximum(qd[good].min(axis=1),1e-12)
        gi=np.flatnonzero(good)
        good[gi]=ratio<=(1+max_rel_discontinuity)
    cell=q[good]
    tri=np.concatenate((cell[:,[0,1,2]],cell[:,[1,3,2]]),axis=0).astype(np.int32)
    return world.astype(np.float32),tri,np.asarray(f.rgb,dtype=np.uint8).reshape(-1,3)


def _ply_bytes(scene:Scene5D)->bytes:
    count=len(scene.xyz)
    head=('ply\nformat binary_little_endian 1.0\n'
          'comment OCAF_GEO_5D=world_xyz,time_s,srgb_luma_proxy\n'
          f'comment OCAF_UNITS={scene.units}\n'
          f'comment OCAF_PROVENANCE={scene.provenance}\n'
          f'element vertex {count}\n'
          'property float x\nproperty float y\nproperty float z\n'
          'property double time_s\nproperty float luma_srgb\n'
          'property uchar red\nproperty uchar green\nproperty uchar blue\n'
          'property float vx\nproperty float vy\nproperty float vz\n'
          'property uchar flow_matched\nproperty int frame_index\n'
          'end_header\n').encode('ascii')
    arr=np.empty(count,dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('time_s','<f8'),
                               ('luma','<f4'),('red','u1'),('green','u1'),('blue','u1'),
                               ('vx','<f4'),('vy','<f4'),('vz','<f4'),('matched','u1'),('frame_index','<i4')])
    for i,key in enumerate(('x','y','z')):arr[key]=scene.xyz[:,i]
    arr['time_s']=scene.time_s;arr['luma']=scene.luma
    for i,key in enumerate(('red','green','blue')):arr[key]=scene.rgb[:,i]
    for i,key in enumerate(('vx','vy','vz')):arr[key]=scene.velocity[:,i]
    arr['matched']=scene.matches_valid;arr['frame_index']=scene.frame_index
    return head+arr.tobytes()


def save_scene5d(scene:Scene5D,output:Path,frames:list[Frame5D],*,mesh:bool=False,
                 tsdf:bool=False,tsdf_voxel_size:float=.1,tsdf_truncation:float=.3,
                 tsdf_max_voxels:int=262144,render_intervals:int=0,
                 fused_static_tsdf:bool=False,gaussian_splats:bool=False)->dict:
    if not (0<=render_intervals<=8):raise ValueError('INVALID_RENDER_INTERPOLATION_COUNT')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    record=BytesIO()
    five=np.column_stack((scene.xyz,scene.time_s,scene.luma)).astype(np.float64)
    np.savez_compressed(record,features_5d=five,xyz=scene.xyz,time_s=scene.time_s,
                        appearance_luma_srgb=scene.luma,rgb=scene.rgb,
                        confidence_proxy=scene.confidence_proxy,frame_index=scene.frame_index,
                        velocity_nearest_neighbor=scene.velocity,velocity_matches_valid=scene.matches_valid,
                        observations_per_voxel=scene.observations)
    save_atomic(root/'scene5d.npz',record.getvalue())
    save_atomic(root/'scene5d.ply',_ply_bytes(scene))
    if mesh:
        (root/'meshes').mkdir(parents=True,exist_ok=True)
        for i,f in enumerate(frames):
            vertices,tri,colors=depth_surface_mesh(f)
            b=BytesIO()
            for pt,col in zip(vertices,colors):
                b.write(('v '+ ' '.join(f'{x:.6f}' for x in pt)+' '
                         +' '.join(f'{x/255:.5f}' for x in col)+'\n').encode())
            for face in tri:b.write(('f '+' '.join(str(int(x)+1) for x in face)+'\n').encode())
            save_atomic(root/'meshes'/f'frame_{i:04d}.obj',b.getvalue())
    tsdf_results=[]
    if tsdf:
        from .tsdf5d import integrate_tsdf_frame,extract_surface_mesh,write_tsdf_npz,write_obj_mesh
        (root/'tsdf').mkdir(parents=True,exist_ok=True)
        for i,f in enumerate(frames):
            grid=integrate_tsdf_frame(f,voxel_size=tsdf_voxel_size,
                                      truncation=tsdf_truncation,max_voxels=tsdf_max_voxels)
            write_tsdf_npz(grid,root/'tsdf'/f'frame_{i:04d}.npz')
            vert,faces=extract_surface_mesh(grid)
            write_obj_mesh(vert,faces,root/'tsdf'/f'frame_{i:04d}.obj')
            tsdf_results.append({'frame':i,'observed_voxels':int((grid.weights>0).sum()),
                                 'mesh_faces':len(faces),'time_s':float(f.time_s)})
    fused_result=None
    if fused_static_tsdf:
        from .multiview_tsdf import integrate_multiview_tsdf
        from .tsdf5d import extract_surface_mesh,write_tsdf_npz,write_obj_mesh
        fused=integrate_multiview_tsdf(frames,voxel_size=tsdf_voxel_size,
                                       truncation=tsdf_truncation,max_voxels=tsdf_max_voxels)
        write_tsdf_npz(fused,root/'static_fused_tsdf.npz')
        v,faces=extract_surface_mesh(fused)
        write_obj_mesh(v,faces,root/'static_fused_tsdf.obj')
        fused_result={'observed_voxels':int((fused.weights>0).sum()),
                      'multiview_voxels':int((fused.weights>=2).sum()),
                      'surface_faces':len(faces),
                      'assumed_static_without_masks':any(f.static_mask is None for f in frames)}
    render_records=[]
    if render_intervals:
        from .render5d import render_scene5d
        from PIL import Image
        (root/'renders').mkdir(parents=True,exist_ok=True)
        k=frames[0].intrinsics;pose=frames[0].pose_c2w
        height,width=frames[0].depth_z.shape
        for idx in range(len(frames)-1):
            for sub in range(1,render_intervals+1):
                t=float(frames[idx].time_s+(frames[idx+1].time_s-frames[idx].time_s)*sub/(render_intervals+1))
                image,depth,valid=render_scene5d(scene,t,k,pose,width,height)
                stem=f'between_{idx:04d}_{sub:04d}'
                b=BytesIO();Image.fromarray(image).save(b,format='PNG')
                save_atomic(root/'renders'/f'{stem}.png',b.getvalue())
                b=BytesIO();np.savez_compressed(b,depth_z=depth,valid_mask=valid,
                                              camera_intrinsics=k,pose_c2w=pose,time_s=np.array(t))
                save_atomic(root/'renders'/f'{stem}.npz',b.getvalue())
                render_records.append({'time_s':t,'observed_pixels':int(valid.sum()),'name':stem})
    gaussian_info=None
    gaussian_files=[]
    if gaussian_splats:
        from .gaussian5d import gaussian_from_scene,render_gaussian5d,export_gaussian5d
        from PIL import Image
        gs=gaussian_from_scene(scene,max_points=100000)
        export_gaussian5d(gs,root/'gaussian5d.npz')
        gaussian_files.append('gaussian5d.npz')
        (root/'gaussian_renders').mkdir(exist_ok=True)
        targets=[float((frames[i].time_s+frames[i+1].time_s)/2.) for i in range(len(frames)-1)]
        gaussian_views=[]
        for i,t in enumerate(targets):
            h,w=frames[0].depth_z.shape
            img,alpha,z=render_gaussian5d(gs,t,frames[0].intrinsics,
                                          frames[0].pose_c2w,w,h)
            stem=f'gaussian_renders/between_{i:04d}'
            out=BytesIO();Image.fromarray(img).save(out,format='PNG')
            save_atomic(root/f'{stem}.png',out.getvalue())
            out=BytesIO();np.savez_compressed(out,alpha=alpha,depth_z=z,render_time_s=np.float64(t))
            save_atomic(root/f'{stem}.npz',out.getvalue())
            gaussian_files.extend([f'{stem}.png',f'{stem}.npz'])
            gaussian_views.append({'time_s':t,'raster_covered_pixels':int((alpha>0.01).sum())})
        gaussian_info={'primitive_count':len(gs.means_xyz),'method':'bounded_CPU_gaussian_surfels',
                       'trained_4dgs':False,'renders':gaussian_views}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    file_names=['scene5d.npz','scene5d.ply']
    if mesh:file_names += [f'meshes/frame_{i:04d}.obj' for i in range(len(frames))]
    if tsdf:file_names += [f'tsdf/frame_{i:04d}.{ext}' for i in range(len(frames)) for ext in ('npz','obj')]
    if render_intervals:file_names += [f'renders/{r["name"]}.{ext}' for r in render_records for ext in ('png','npz')]
    if fused_static_tsdf:file_names += ['static_fused_tsdf.npz','static_fused_tsdf.obj']
    if gaussian_splats:file_names+=gaussian_files
    report={'schema_version':'OMEGA-GeoDepth-5D-v0.3','field_definition':'(world_x, world_y, world_z, time_seconds, sRGB_luma_proxy)',
            'units':scene.units,'provenance':scene.provenance,'frames':len(frames),
            'total_points':len(scene.xyz),'matched_points':int(scene.matches_valid.sum()),
            'voxel_size':scene.voxel_size,'neighbor_radius':scene.neighbor_radius,
            'tsdf_time_slices':tsdf_results,'fused_static_tsdf':fused_result,
            'gaussian5d':gaussian_info,
            'interpolated_renders':render_records,
            'source':'input depth and poses, not a learned 5D model',
            'limitations':['appearance is luma proxy, not physical radiance',
                           'scene-flow is approximate nearest-neighbor and unverified',
                           'meshes only cover observed depth surfaces; no occluded geometry',
                           'relative inputs yield arbitrary-scale outputs',
                           'no training/optimization of dynamic 4D Gaussian splats',
                           'static multiview fusion requires known inter-frame scale and masks for moving objects'],
            'artifacts_sha256':{name:sha(root/name) for name in file_names}}
    save_atomic(root/'report.json',json.dumps(report,indent=2,ensure_ascii=False).encode('utf-8'))
    return report


def _safe_source(root:Path,value:str)->Path:
    p=Path(value)
    if p.is_absolute():raise ValueError('UNSAFE_INPUT_PATH: absolute path not allowed')
    dest=(root/p).resolve(strict=False)
    try:dest.relative_to(root.resolve())
    except ValueError as err:raise ValueError('UNSAFE_INPUT_PATH: traversal') from err
    return dest


def _read_npz(path:Path)->dict:
    if path.suffix.lower()!='.npz' or not path.is_file():raise ValueError('INVALID_OR_MISSING_NPZ')
    with zipfile.ZipFile(path,'r') as file:
        if sum(x.file_size for x in file.infolist())>MAX_NPZ_UNCOMPRESSED:
            raise ValueError('NPZ_RESOURCE_LIMIT')
    with np.load(path,allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def load_frame_manifest(path:Path)->tuple[list[Frame5D],dict]:
    root=Path(path).resolve().parent
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    records=obj.get('frames')
    if not isinstance(records,list) or not 2<=len(records)<=128:raise ValueError('INVALID_FRAME_COUNT')
    frames=[]
    for item in records:
        p=_safe_source(root,item['npz'])
        d=_read_npz(p)
        if 'depth_z' not in d or 'source_rgb' not in d or 'intrinsics_k' not in d:
            raise ValueError('FRAME_NPZ_MISSING_REQUIRED_ARRAY')
        pose=np.asarray(item['pose_c2w'],dtype=np.float64)
        frames.append(Frame5D(float(item['time_s']),d['depth_z'],d['source_rgb'],d['intrinsics_k'],pose,
                              d.get('valid_mask',np.ones_like(d['depth_z'],dtype=bool)),d.get('sigma_log_z'),
                              None,d.get('static_mask')))
    return frames,obj


def build_from_da3_sequence(path:Path,checkpoint:Path,model_id:str,*,device:str='cpu',
                            process_res:int=504,source_dir:Path|None=None,
                            verify_reprojection:bool=False,
                            max_relative_error:float=.15,runtime:str='upstream_api',
                            profile:str='fp32',allow_experimental:bool=False)->tuple[list[Frame5D],dict]:
    from .da3_adapter import DA3OfflineBackend
    path=Path(path).resolve();root=path.parent
    obj=json.loads(path.read_text(encoding='utf-8'))
    records=obj.get('frames')
    if not isinstance(records,list) or not 2<=len(records)<=128:raise ValueError('INVALID_FRAME_COUNT')
    images=[_safe_source(root,rec['image']) for rec in records]
    for im in images:
        if not im.is_file() or im.stat().st_size>128*1024*1024:raise ValueError('INVALID_DA3_IMAGE_INPUT')
    if profile!='fp32' and runtime!='native_network':
        raise ValueError('DA3_PROFILE_REQUIRES_NATIVE_RUNTIME')
    if profile in ('aq6','aq4','int8') and not allow_experimental:
        raise ValueError('DA3_EXPERIMENTAL_OPT_IN_REQUIRED')
    if profile in ('aq6','aq4','int8') and not verify_reprojection:
        raise ValueError('DA3_EXPERIMENTAL_REPROJECTION_REQUIRED')
    if runtime == 'native_network':
        if model_id != 'DA3-SMALL' or source_dir is None:
            raise ValueError('DA3_NATIVE_RUNTIME_REQUIRES_SMALL_AND_SOURCE')
        from .da3_real import DA3RealCPU
        backend = DA3RealCPU(checkpoint,source_dir=source_dir,device=device,profile=profile,allow_experimental=allow_experimental)
    elif runtime == 'upstream_api':
        backend=(DA3OfflineBackend(checkpoint,model_id,device=device,source_dir=source_dir)
                 if source_dir is not None else DA3OfflineBackend(checkpoint,model_id,device=device))
    else:
        raise ValueError('DA3_UNSUPPORTED_RUNTIME')
    out=backend.inference(images,process_res=process_res)
    if len(out.depth_z)!=len(records):raise ValueError('DA3_FRAME_COUNT_MISMATCH')
    reprojection=None
    if verify_reprojection:
        from .da3_validation import verify_multiview_reprojection
        reprojection=verify_multiview_reprojection(out,max_relative_error=max_relative_error)
        if not reprojection['passed']:
            raise ValueError('DA3_MULTIVIEW_REPROJECTION_UNVERIFIABLE: '+
                             ','.join(p['reason'] for p in reprojection['pairs'] if not p['passed']))
    frames=[Frame5D(float(rec['time_s']),out.depth_z[i],out.rgb[i],out.intrinsics_k[i],
                    out.pose_c2w[i],np.isfinite(out.depth_z[i])&(out.depth_z[i]>0),None,out.confidence_raw[i])
            for i,rec in enumerate(records)]
    meta={'units':out.units,'provenance':out.provenance,'da3_model':out.model_id,
          'weights_sha256':out.weights_sha256,'camera_poses':'estimated_by_da3'}
    if runtime=='native_network': meta['weight_profile']=backend.profile_info
    if reprojection is not None:meta['multiview_reprojection_audit']=reprojection
    return frames,meta


def main():
    import argparse
    parser=argparse.ArgumentParser(description='OMEGA GeoDepth Ω experimental 5D (XYZ,time,luma) reconstructor')
    subs=parser.add_subparsers(dest='command',required=True)
    for name in ('from-npz','from-da3'):
        p=subs.add_parser(name)
        p.add_argument('--manifest',required=True,type=Path)
        p.add_argument('--output',required=True,type=Path)
        p.add_argument('--voxel-size',type=float,default=.05)
        p.add_argument('--neighbor-radius',type=float,default=.12)
        p.add_argument('--max-points-per-frame',type=int,default=200000)
        p.add_argument('--mesh',action='store_true')
        p.add_argument('--tsdf',action='store_true',help='Reconstruct bounded TSDF volume per timestamp')
        p.add_argument('--fused-static-tsdf',action='store_true',help='Fused TSDF: requires static scene or masks; do not use for dynamic objects without masks')
        p.add_argument('--gaussian-splats',action='store_true',help='Observed 5D Gaussian surfels; NOT pretrained 4D Gaussian Splatting')
        p.add_argument('--tsdf-voxel-size',type=float,default=.1)
        p.add_argument('--tsdf-truncation',type=float,default=.3)
        p.add_argument('--tsdf-max-voxels',type=int,default=262144)
        p.add_argument('--render-intervals',type=int,default=0,help='Sparse observed-only intermediate images per pair of frames')
        if name=='from-da3':
            p.add_argument('--checkpoint',type=Path,required=True)
            p.add_argument('--model',choices=('DA3-SMALL','DA3-BASE'),default='DA3-SMALL')
            p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
            p.add_argument('--runtime',choices=('upstream_api','native_network'),default='upstream_api')
            p.add_argument('--profile',choices=('fp32','fp16','aq6','aq4','int8'),default='fp32')
            p.add_argument('--allow-experimental',action='store_true')
            p.add_argument('--process-res',type=int,default=504)
            p.add_argument('--official-source',type=Path,default=None,
                           help='Directory safely extracted from pinned official-source ZIP')
            p.add_argument('--verify-reprojection',action='store_true',
                           help='Optional relative multi-view geometry gate; may abstain on occlusions')
            p.add_argument('--max-relative-error',type=float,default=.15)
    args=parser.parse_args()
    if args.command=='from-npz':
        frames,meta=load_frame_manifest(args.manifest)
    else:
        frames,meta=build_from_da3_sequence(args.manifest,args.checkpoint,args.model,
                                            device=args.device,process_res=args.process_res,
                                            source_dir=args.official_source,
                                            verify_reprojection=args.verify_reprojection,
                                            max_relative_error=args.max_relative_error,runtime=args.runtime,profile=args.profile,allow_experimental=args.allow_experimental)
    units=meta.get('units','arbitrary_scale');provenance=meta.get('provenance','relative_estimated')
    scene=build_scene5d(frames,voxel_size=args.voxel_size,neighbor_radius=args.neighbor_radius,
                        units=units,provenance=provenance,max_points_per_frame=args.max_points_per_frame)
    report=save_scene5d(scene,args.output,frames,mesh=args.mesh,tsdf=args.tsdf,
                        tsdf_voxel_size=args.tsdf_voxel_size,tsdf_truncation=args.tsdf_truncation,
                        tsdf_max_voxels=args.tsdf_max_voxels,
                        render_intervals=args.render_intervals,
                        fused_static_tsdf=args.fused_static_tsdf,
                        gaussian_splats=args.gaussian_splats)
    report['input_metadata']=meta
    save_atomic(args.output/'report.json',json.dumps(report,indent=2,ensure_ascii=False).encode('utf-8'))
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
