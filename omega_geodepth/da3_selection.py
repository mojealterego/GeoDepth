"""Conservative selection by all-pair relative reprojection, not metric accuracy."""
from __future__ import annotations
from dataclasses import replace
from itertools import permutations
from collections.abc import Sequence
import numpy as np
from .da3_adapter import DA3Output
from .da3_validation import verify_multiview_reprojection


def audit_bidirectional(pred: DA3Output) -> dict:
    """Check every ordered pair using unchanged v0.12 thresholds.

    Intended for a static group or synchronized cameras, never motion evidence.
    """
    if pred.units != 'arbitrary_scale' or pred.provenance != 'relative_estimated':
        raise ValueError('relative estimated DA3 required')
    n=pred.depth_z.shape[0]
    if not 2<=n<=8:
        raise ValueError('requires 2–8 static views')
    fields=('depth_z','confidence_raw','intrinsics_k','pose_c2w','rgb')
    if any(len(getattr(pred,k))!=n for k in fields):
        raise ValueError('inconsistent view count')
    pairs=[]
    for i,j in permutations(range(n),2):
        p=replace(pred,**{k:np.asarray(getattr(pred,k))[[i,j]] for k in fields})
        record=verify_multiview_reprojection(p)['pairs'][0]
        pairs.append({**record,'src':i,'dst':j})
    return {'passed':all(p['passed'] for p in pairs), 'pairs':pairs,
            'worst_coverage':min(p['coverage'] for p in pairs),
            'worst_agreement':min(p['agreement_fraction'] if p['agreement_fraction'] is not None else 0. for p in pairs),
            'units':pred.units,'interpretation':'static self-consistency only; not metric accuracy'}


def select_consistent(candidates: Sequence[tuple[str,DA3Output]]) -> dict:
    """First passing candidate in caller's priority order; keep all failures.

    Caller must supply predictions of the same ordered input images. No
    data-dependent threshold relaxation and no consensus-as-accuracy claim.
    """
    if not 1<=len(candidates)<=12:raise ValueError('candidate limit')
    ids=[name for name,_ in candidates]
    if any(not isinstance(x,str) or not x.strip() for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('invalid candidate identifiers')
    counts={p.depth_z.shape[0] for _,p in candidates}
    if len(counts)!=1:raise ValueError('different view counts')
    reports=[{'id':name,'model':pred.model_id,'audit':audit_bidirectional(pred)} for name,pred in candidates]
    return {'selected':next((x['id'] for x in reports if x['audit']['passed']),None),
            'candidates':reports,'policy':'first passing candidate in supplied priority order; abstain if none'}


def infer_first_consistent(engine, images, resolutions: Sequence[int], *, inference_options=None):
    """Lazy, bounded fallback in declared priority order; no partial exports."""
    resolutions=list(resolutions)
    if (not 1<=len(resolutions)<=4 or any(type(r) is not int or not 112<=r<=504 for r in resolutions)
        or len(set(resolutions))!=len(resolutions) or not 2<=len(images)<=8):
        raise ValueError('invalid fallback resolutions or view count')
    options=dict(inference_options or {})
    if set(options)-{"pose_decoder","audit_depth_rays"}:raise ValueError("unexpected inference options")
    attempts=[]
    for res in resolutions:
        raw=engine.inference(images,process_res=res,**options)
        audit=audit_bidirectional(raw)
        attempts.append({'resolution':res,'audit':audit})
        if audit['passed']:
            return raw,{'selected_resolution':res,'attempts':attempts,'policy':'first passing resolution; unchanged thresholds'}
        del raw
    return None,{'selected_resolution':None,'attempts':attempts,'policy':'abstain; no passing resolution'}
