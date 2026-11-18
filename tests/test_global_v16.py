import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from omega_geodepth.fast_rays import fit_ray_cameras
from omega_geodepth.chunk_graph import CameraChunk, align_chunks


def rays_fixture():
    h,w=40,56
    K=np.array([[43.,0,27.],[0,41.,19.],[0,0,1.]])
    R=Rotation.from_rotvec([.2,-.3,.1]).as_matrix(); t=np.array([1.,2.,3.])
    y,x=np.mgrid[:h,:w]; p=np.stack([x+.5,y+.5,np.ones_like(x)],-1)
    d=p@np.linalg.inv(K).T@R.T
    return np.concatenate((d,np.broadcast_to(t,d.shape)),axis=-1)[None],K,R,t


def test_fast_ray_exact_and_scaled_directions():
    rays,K,R,t=rays_fixture();rays[...,:3]*=np.random.default_rng(1).uniform(.3,3,rays.shape[:-1])[...,None]
    p,k,report=fit_ray_cameras(rays,np.ones(rays.shape[:-1]),image_shape=rays.shape[1:3])
    np.testing.assert_allclose(k[0],K,atol=1e-6);np.testing.assert_allclose(p[0,:3,:3],R,atol=1e-7);np.testing.assert_allclose(p[0,:3,3],t)
    assert report['worst_angular_p90_deg']<1e-6


def test_fast_ray_robust_outliers():
    rays,K,R,t=rays_fixture();rng=np.random.default_rng(42);a=rays.reshape(-1,6);idx=rng.choice(len(a),len(a)//12,replace=False);a[idx,:]=rng.normal(0,3,(len(idx),6))
    p,k,report=fit_ray_cameras(rays,np.ones(rays.shape[:-1]),image_shape=rays.shape[1:3]);np.testing.assert_allclose(k[0],K,atol=1.5);np.testing.assert_allclose(p[0,:3,3],t,atol=.02);assert report['worst_angular_p90_deg']<1.5


def test_fast_ray_reject_degenerate_and_reversed_field():
    rays,*_=rays_fixture();rays[...,:3]=[0,0,1]
    with pytest.raises(ValueError):fit_ray_cameras(rays,np.ones(rays.shape[:-1]),image_shape=rays.shape[1:3])
    rays,*_=rays_fixture();rays[...,:3]*=-1
    with pytest.raises(ValueError):fit_ray_cameras(rays,np.ones(rays.shape[:-1]),image_shape=rays.shape[1:3])


def test_fast_ray_reject_sparse_support():
    rays,*_=rays_fixture();c=np.zeros(rays.shape[:-1]);c[0,0:3,0:3]=1
    with pytest.raises(ValueError):fit_ray_cameras(rays,c,image_shape=rays.shape[1:3])


def chunk_fixture(noise=0.):
    rng=np.random.default_rng(22);world=np.repeat(np.eye(4)[None],9,axis=0);world[:,:3,3]=rng.uniform(-2,2,(9,3));world[:,:3,:3]=Rotation.from_rotvec(rng.normal(0,.2,(9,3))).as_matrix()
    chunks=[];truth=[]
    for i,ids in enumerate((range(0,6),range(2,8),[0,1,4,5,7,8])):
        ids=tuple(ids);s=1+.4*i;R=Rotation.from_rotvec([.1*i,.2*i,-.05*i]).as_matrix();t=np.array([i,-i,i*.2]);local=world[list(ids)].copy();local[:,:3,3]=(local[:,:3,3]-t)@R/s;local[:,:3,:3]=R.T@local[:,:3,:3];local[:,:3,3]+=rng.normal(0,noise,(len(ids),3));chunks.append(CameraChunk(ids,local));truth.append(s)
    return chunks,world,truth


def test_graph_exact_and_depth_scale():
    chunks,world,truth=chunk_fixture();result=align_chunks(chunks,center_tolerance=.02,rotation_tolerance_deg=1)
    assert result.passed and result.units=='arbitrary_scale';assert result.camera_ids==tuple(range(9));np.testing.assert_allclose(result.poses_c2w,world,atol=1e-6);np.testing.assert_allclose(result.scales,truth,atol=1e-6)
    np.testing.assert_allclose(result.transform_depth(1,np.ones((3,3))),truth[1]);assert result.final_cost<=result.initial_cost+1e-10


def test_graph_optimizes_noisy_cycle():
    chunks,_,_=chunk_fixture(.015);r=align_chunks(chunks,center_tolerance=.15,rotation_tolerance_deg=2)
    assert r.passed and r.final_cost<r.initial_cost


def test_graph_disconnected_and_inconsistent():
    chunks,_,_=chunk_fixture();bad=CameraChunk(tuple(range(10,16)),chunks[1].poses_c2w)
    with pytest.raises(ValueError):align_chunks([chunks[0],bad],center_tolerance=.1,rotation_tolerance_deg=2)
    p=chunks[2].poses_c2w.copy();p[0,:3,:3]=Rotation.from_rotvec([1,0,0]).as_matrix()@p[0,:3,:3]
    r=align_chunks([*chunks[:2],CameraChunk(chunks[2].camera_ids,p)],center_tolerance=.1,rotation_tolerance_deg=2)
    assert not r.passed


def test_graph_reject_duplicate_camera_ids():
    chunks,_,_=chunk_fixture();bad=CameraChunk((0,0,1,2,3,4),chunks[0].poses_c2w)
    with pytest.raises(ValueError):align_chunks([bad,chunks[1]],center_tolerance=.1,rotation_tolerance_deg=2)


def test_actual_fast_ray_decoder():
    from pathlib import Path
    from omega_geodepth.da3_real import DA3RealCPU
    root=Path(__file__).resolve().parents[1]
    e=DA3RealCPU(root/'models/da3',source_dir=root/'vendor/da3_src')
    im=np.random.default_rng(5).integers(0,256,(140,140,3),dtype=np.uint8)
    r=e.inference([im,np.roll(im,3,axis=1)],process_res=140,pose_decoder='fast_ray',audit_depth_rays=True)
    assert r.pose_decoder=='fast_ray' and r.units=='arbitrary_scale' and r.ray_audit['fast_fit']['worst_angular_p90_deg']<=8
    assert np.isfinite(r.depth_z).any() and not e.net.head._forward_hooks


def test_chunk_depth_and_camera_gate():
    from omega_geodepth.da3_adapter import DA3Output
    from omega_geodepth.chunk_graph import align_da3_chunks
    chunks,_,truth=chunk_fixture();outputs=[]
    for i,ch in enumerate(chunks):
        n=len(ch.camera_ids);z=np.full((n,8,8),4./truth[i],np.float32);k=np.repeat(np.array([[[8.,0,4.],[0,8.,4.],[0,0,1.]]]),n,axis=0)
        rgb=np.stack([np.full((8,8,3),v,np.uint8) for v in ch.camera_ids]);outputs.append(DA3Output(z,np.ones_like(z),k,ch.poses_c2w,rgb,'arbitrary_scale','relative_estimated','DA3-SMALL','f'*64))
    g,merged,report=align_da3_chunks(list(zip([ch.camera_ids for ch in chunks],outputs)),center_tolerance=.02,rotation_tolerance_deg=1)
    assert g.passed and merged is not None and report['passed'];assert merged.depth_z.shape==(9,8,8);np.testing.assert_allclose(merged.depth_z,4.,atol=1e-6)
    from dataclasses import replace
    broken=replace(outputs[2],depth_z=outputs[2].depth_z*2)
    _,merged,report=align_da3_chunks(list(zip([ch.camera_ids for ch in chunks],[*outputs[:2],broken])),center_tolerance=.02,rotation_tolerance_deg=1)
    assert merged is None and not report['passed']
    broken=replace(outputs[2],rgb=np.zeros_like(outputs[2].rgb))
    with pytest.raises(ValueError):align_da3_chunks(list(zip([ch.camera_ids for ch in chunks],[*outputs[:2],broken])),center_tolerance=.02,rotation_tolerance_deg=1)


def test_chunk_planner_covers_long_sequence_with_bounded_overlap():
    from omega_geodepth.chunk_graph import plan_static_chunks
    plan=plan_static_chunks(19,chunk_size=6,overlap=4)
    assert set().union(*(set(x) for x in plan))==set(range(19))
    assert all(len(x)==6 for x in plan)
    assert all(len(set(a)&set(b))>=4 for a,b in zip(plan,plan[1:]))
    with pytest.raises(ValueError):plan_static_chunks(64,chunk_size=5,overlap=4)
    with pytest.raises(ValueError):plan_static_chunks(9,chunk_size=6,overlap=2)


def test_chunked_inference_calls_engine_and_exports_only_accepted(tmp_path):
    from omega_geodepth.da3_adapter import DA3Output
    from omega_geodepth.chunk_graph import infer_static_chunks,write_chunk_result
    from omega_geodepth.quality_v1.prediction import load_prediction_npz
    class FakeEngine:
        def __init__(self):self.calls=[]
        def inference(self,images,**opts):
            self.calls.append(list(images));n=len(images);pose=np.repeat(np.eye(4)[None],n,axis=0)
            pose[:,:3,3]=[[np.sin(v),np.cos(v),v*.1] for v in images];k=np.repeat(np.array([[[8.,0,4.],[0,8.,4.],[0,0,1.]]]),n,axis=0);z=np.full((n,8,8),4.,np.float32);rgb=np.stack([np.full((8,8,3),v,np.uint8) for v in images])
            return DA3Output(z,np.ones_like(z),k,pose,rgb,'arbitrary_scale','relative_estimated','DA3-SMALL','f'*64)
    e=FakeEngine();g,p,report=infer_static_chunks(e,list(range(13)),process_res=140,center_tolerance=.01,rotation_tolerance_deg=1)
    assert p is not None and g.passed and len(e.calls)>1 and len(p.depth_z)==13
    out=tmp_path/'accepted';write_chunk_result(out,p,report);assert load_prediction_npz(out/'prediction.npz').depth.shape==(13,8,8)
    with pytest.raises(FileExistsError):write_chunk_result(out,p,report)
    write_chunk_result(tmp_path/'rejected',None,{'passed':False});assert not (tmp_path/'rejected'/'prediction.npz').exists()
    with pytest.raises(ValueError):write_chunk_result(tmp_path/'bad',p,{'passed':False})
