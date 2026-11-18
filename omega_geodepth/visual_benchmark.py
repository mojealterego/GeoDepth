"""Independent geometry audit inspired by DA3 technical report §6 (not its official benchmark).

Camera poses are C2W. Similarity is fitted exclusively on designated reference
camera centres; held-out poses and point clouds never participate in the fit.
Translation direction is evaluated in camera i coordinates, with oriented (not
sign-ambiguous) directions. AUC is the exact integral of the empirical CDF.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial import cKDTree


def _points(a, *, limit=2_000_000):
    a=np.asarray(a)
    if a.ndim!=2 or a.shape[1]!=3 or not 1<=len(a)<=limit or a.dtype.kind not in 'fiu':
        raise ValueError('expected bounded nonempty Nx3 points')
    a=a.astype(np.float64)
    if not np.isfinite(a).all() or np.max(np.abs(a))>1e12: raise ValueError('invalid coordinates')
    return a


def _poses(a):
    a=np.asarray(a)
    if a.ndim!=3 or a.shape[1:]!=(4,4) or not 2<=len(a)<=100 or a.dtype.kind not in 'fiu':
        raise ValueError('expected 2..100 C2W matrices')
    a=a.astype(np.float64)
    r=a[:,:3,:3]
    if not np.isfinite(a).all() or np.max(np.abs(a))>1e12 or not np.allclose(a[:,3,:],[0,0,0,1],atol=1e-7,rtol=0) or not np.allclose(r@r.transpose(0,2,1),np.eye(3),atol=1e-4,rtol=0) or not np.allclose(np.linalg.det(r),1,atol=1e-4,rtol=0):
        raise ValueError('invalid C2W rotation or transform')
    return a


def _positive(v):
    if not np.isfinite(v) or v<=0: raise ValueError('positive finite threshold required')
    return float(v)


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    inliers: np.ndarray

    def apply(self, points):
        return self.scale*_points(points)@self.rotation.T+self.translation

    def apply_poses(self, poses):
        p=_poses(poses).copy()
        p[:,:3,:3]=self.rotation@p[:,:3,:3]
        p[:,:3,3]=self.apply(p[:,:3,3])
        return p


def _umeyama(x,y):
    xc=x-x.mean(0); yc=y-y.mean(0)
    if np.linalg.matrix_rank(xc,tol=max(np.linalg.norm(xc),1e-15)*1e-8)<2 or np.linalg.matrix_rank(yc,tol=max(np.linalg.norm(yc),1e-15)*1e-8)<2:
        raise ValueError('degenerate camera centres')
    u,d,vt=np.linalg.svd(yc.T@xc/len(x))
    sign=np.ones(3);sign[-1]=np.sign(np.linalg.det(u@vt))
    r=(u*sign)@vt
    s=float((d@sign)/np.mean(np.sum(xc**2,axis=1)))
    _positive(s)
    t=y.mean(0)-s*(r@x.mean(0))
    return s,r,t


def fit_similarity(predicted,reference,*,inlier_threshold,seed=0,trials=512):
    x=_points(predicted,limit=100);y=_points(reference,limit=100)
    threshold=_positive(inlier_threshold)
    if x.shape!=y.shape or len(x)<3 or not isinstance(trials,int) or not 1<=trials<=4096: raise ValueError('invalid calibration correspondence count or trials')
    rng=np.random.default_rng(seed); best=None;score=(-1,-np.inf)
    # All triples for small sets; deterministic bounded sampling for larger sets.
    from itertools import combinations
    subsets=list(combinations(range(len(x)),3)) if math.comb(len(x),3)<=trials else [rng.choice(len(x),3,replace=False) for _ in range(trials)]
    for idx in subsets:
        idx=np.asarray(idx)
        try:s,r,t=_umeyama(x[idx],y[idx])
        except ValueError:continue
        residual=np.linalg.norm(s*x@r.T+t-y,axis=1); mask=residual<=threshold
        if mask.sum()<3:continue
        candidate=(int(mask.sum()),-float(np.median(residual[mask])))
        if candidate>score:best=mask;score=candidate
    if best is None or best.sum()<max(3,math.ceil(.6*len(x))):raise ValueError('no majority nondegenerate calibration consensus')
    s,r,t=_umeyama(x[best],y[best]);mask=np.linalg.norm(s*x@r.T+t-y,axis=1)<=threshold
    if mask.sum()<max(3,math.ceil(.6*len(x))):raise ValueError('refitted calibration failed')
    _umeyama(x[mask],y[mask]) # verify final consensus is nondegenerate
    return Similarity(s,r,t,mask)


def _angle(a,b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=-1),np.sum(a*b,axis=-1)))


def _rotation_angle(r):
    return np.degrees(np.arccos(np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1)))


def pose_metrics(predicted,reference):
    p=_poses(predicted);q=_poses(reference)
    if p.shape!=q.shape:raise ValueError('pose shapes differ')
    i,j=np.triu_indices(len(p),1)
    pr=p[i,:3,:3].transpose(0,2,1)@p[j,:3,:3];qr=q[i,:3,:3].transpose(0,2,1)@q[j,:3,:3]
    re=_rotation_angle(pr@qr.transpose(0,2,1))
    pt=(p[i,:3,:3].transpose(0,2,1)@(p[j,:3,3]-p[i,:3,3])[...,None])[...,0]
    qt=(q[i,:3,:3].transpose(0,2,1)@(q[j,:3,3]-q[i,:3,3])[...,None])[...,0]
    pn=np.linalg.norm(pt,axis=1);qn=np.linalg.norm(qt,axis=1)
    # Ground-truth zero baselines are undefined; predicted missing baselines are penalized.
    eligible=qn>1e-9*max(float(qn.max()),1e-15)
    if not eligible.any():raise ValueError('no reference translation baseline')
    usable=eligible&(pn>1e-9*max(float(pn.max()),1e-15));te=np.full(len(i),180.)
    te[usable]=_angle(pt[usable]/pn[usable,None],qt[usable]/qn[usable,None]);joint=np.maximum(re[eligible],te[eligible])
    return {'total_pairs':len(i),'translation_pairs':int(eligible.sum()),'predicted_nonzero_pairs':int(usable.sum()),'rotation_mean_deg':float(re.mean()),'translation_mean_deg':float(te[eligible].mean()),'auc_3':float(np.maximum(0,1-joint/3).mean()),'auc_30':float(np.maximum(0,1-joint/30).mean())}


def cloud_metrics(predicted,reference,*,threshold):
    p=_points(predicted);q=_points(reference);d=_positive(threshold)
    a=cKDTree(q).query(p,k=1,workers=1)[0];b=cKDTree(p).query(q,k=1,workers=1)[0]
    precision=float(np.mean(a<d));recall=float(np.mean(b<d))
    return {'predicted_points':len(p),'reference_points':len(q),'threshold':d,'accuracy':float(a.mean()),'completeness':float(b.mean()),'chamfer':float((a.mean()+b.mean())/2),'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0.}


def _indices(a,n,minimum):
    a=list(a)
    if len(a)<minimum or any(not isinstance(i,(int,np.integer)) or isinstance(i,(bool,np.bool_)) or not 0<=i<n for i in a) or len(set(a))!=len(a):raise ValueError('invalid view indices')
    return np.asarray(a,dtype=int)


def evaluate_scene(predicted_poses,reference_poses,predicted_eval_points,reference_eval_points,*,calibration_indices,evaluation_indices,inlier_threshold,distance_threshold,reference_units):
    p=_poses(predicted_poses);q=_poses(reference_poses)
    if p.shape!=q.shape or reference_units not in ('m','arbitrary_reference'):raise ValueError('invalid reference contract')
    c=_indices(calibration_indices,len(p),3);e=_indices(evaluation_indices,len(p),2)
    if set(c)&set(e):raise ValueError('calibration and evaluation overlap')
    sim=fit_similarity(p[c,:3,3],q[c,:3,3],inlier_threshold=inlier_threshold)
    aligned=sim.apply_poses(p)
    err=np.linalg.norm(aligned[e,:3,3]-q[e,:3,3],axis=1)
    return {'reference_units':reference_units,'alignment_kind':'reference_camera_centres_sim3_evaluation_only','calibration_indices':c.tolist(),'evaluation_indices':e.tolist(),'alignment':{'scale':sim.scale,'rotation':sim.rotation.tolist(),'translation':sim.translation.tolist(),'inlier_indices':c[sim.inliers].tolist()},'heldout_center_rmse':float(np.sqrt(np.mean(err**2))),'pose':pose_metrics(p[e],q[e]),'cloud':cloud_metrics(sim.apply(predicted_eval_points),reference_eval_points,threshold=distance_threshold),'caution':'Reference aligned evaluation, not metric inference. Caller must ensure point clouds use evaluation views only; not novel-view synthesis.'}


def select_context_views(reference_poses,*,context_count=12,target_stride=8,rotation_weight=1.):
    p=_poses(reference_poses);n=len(p)
    if not isinstance(context_count,int) or not isinstance(target_stride,int) or context_count<1 or target_stride<2 or not np.isfinite(rotation_weight) or rotation_weight<0:raise ValueError('invalid context split')
    targets=list(range(0,n,target_stride));available=np.array([i for i in range(n) if i not in targets]);
    if len(available)<context_count:raise ValueError('not enough independent context views')
    t=p[available,:3,3];r=p[available,:3,:3]
    dt=np.linalg.norm(t[:,None]-t[None,:],axis=-1);dt/=max(float(dt.max()),1e-15)
    dr=_rotation_angle(r[:,None]@r[None,:].transpose(0,1,3,2))/180
    d=dt+rotation_weight*dr;chosen=[0];nearest=d[0].copy()
    for _ in range(context_count-1):
        nearest[chosen]=-1;idx=int(np.argmax(nearest));chosen.append(idx);nearest=np.minimum(nearest,d[idx])
    return {'targets':targets,'context':available[chosen].tolist()}
