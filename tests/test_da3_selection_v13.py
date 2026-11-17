from dataclasses import replace
import numpy as np
import pytest
from omega_geodepth.da3_adapter import DA3Output
from omega_geodepth.da3_selection import audit_bidirectional, select_consistent


def example():
    z=np.full((2,20,24),2.,dtype=np.float32)
    k=np.repeat(np.array([[[25.,0.,11.5],[0.,25.,9.5],[0.,0.,1.]]]),2,axis=0)
    pose=np.repeat(np.eye(4)[None],2,axis=0);pose[1,0,3]=.1
    return DA3Output(z,np.ones_like(z),k,pose,np.zeros((*z.shape,3),np.uint8),'arbitrary_scale','relative_estimated','DA3-SMALL','0'*64)


def test_both_directions():
    r=audit_bidirectional(example())
    assert r['passed'] and len(r['pairs'])==2
    assert {(x['src'],x['dst']) for x in r['pairs']}=={(0,1),(1,0)}


def test_nonadjacent_pairs():
    p=example(); i=[0,1,0];p=replace(p,**{k:getattr(p,k)[i] for k in ('depth_z','confidence_raw','intrinsics_k','pose_c2w','rgb')})
    assert len(audit_bidirectional(p)['pairs'])==6


def test_reject_mislabeled_metric():
    with pytest.raises(ValueError):audit_bidirectional(replace(example(),units='meters'))


def test_select_preserves_rejected_and_abstains():
    p=example(); bad=p.pose_c2w.copy();bad[1,0,3]=100
    rejected=replace(p,pose_c2w=bad)
    assert select_consistent([('base',rejected)])['selected'] is None
    r=select_consistent([('base',rejected),('small',p)])
    assert r['selected']=='small' and not r['candidates'][0]['audit']['passed']


def test_tie_preserves_order_and_duplicate_rejected():
    p=example();assert select_consistent([('first',p),('second',p)])['selected']=='first'
    with pytest.raises(ValueError):select_consistent([('first',p),('first',p)])


def test_resolution_fallback_records_rejection():
    from omega_geodepth.da3_selection import infer_first_consistent
    p=example();bad=p.pose_c2w.copy();bad[1,0,3]=100
    class Engine:
        def inference(self, images, *, process_res):
            return replace(p,pose_c2w=bad) if process_res==280 else p
    raw,r=infer_first_consistent(Engine(),[None,None],[280,392,504])
    assert raw is p and r['selected_resolution']==392 and len(r['attempts'])==2
    assert not r['attempts'][0]['audit']['passed']
    raw,r=infer_first_consistent(Engine(),[None,None],[280])
    assert raw is None and r['selected_resolution'] is None
    with pytest.raises(ValueError):infer_first_consistent(Engine(),[None,None],[280,280])


def test_real_base_fallback():
    from pathlib import Path
    from omega_geodepth.da3_real import DA3RealCPU
    from omega_geodepth.da3_selection import infer_first_consistent
    root=Path(__file__).resolve().parents[1]
    folder=root/'models/da3_base'
    if not (folder/'model.safetensors').is_file():pytest.skip('optional Base checkpoint not installed')
    engine=DA3RealCPU(folder,source_dir=root/'vendor/da3_src',model_id='DA3-BASE')
    images=sorted((root/'examples/v08_real_da3/images').glob('*.png'))
    pred,report=infer_first_consistent(engine,images,[280,392,504])
    assert pred is not None and pred.model_id=='DA3-BASE' and pred.units=='arbitrary_scale'
    assert report['attempts'][-1]['audit']['passed'] and report['selected_resolution'] in (280,392,504)
