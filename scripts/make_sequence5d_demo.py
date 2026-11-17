"""Generate physically consistent moving-sphere 3D+time geometry for local tests."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import argparse
import json
from io import BytesIO
import numpy as np
from PIL import Image
from omega_geodepth.infer import save_atomic


def generate(path:Path,shape:int=96):
    root=Path(path);root.mkdir(parents=True,exist_ok=True)
    h=w=shape
    u,v=np.meshgrid(np.arange(w),np.arange(h))
    focal=1.05*w
    k=np.array([[focal,0,(w-1)/2],[0,focal,(h-1)/2],[0,0,1]],dtype=np.float64)
    x=(u-k[0,2])/focal;y=(v-k[1,2])/focal
    rays=np.stack((x,y,np.ones((h,w))),axis=-1)
    a=(rays**2).sum(axis=-1)
    manifest={'units':'arbitrary_scale','provenance':'relative_estimated',
              'schema_version':'OMEGA-GeoDepth-5D-input-v1','frames':[]}
    da3={'frames':[]}
    for i,center_x in enumerate((-.45,0.,.45)):
        center=np.array([center_x,0.,2.25]);r=.52
        b=-2*(rays*center).sum(-1)
        c=float(center@center-r*r)
        disc=b*b-4*a*c
        hit=disc>0
        z=np.full((h,w),4.0,dtype=np.float32)
        front=(-b-np.sqrt(np.maximum(disc,0)))/(2*a)
        good=hit&(front>0)
        z[good]=front[good].astype(np.float32)
        rgb=np.zeros((h,w,3),dtype=np.uint8)
        rgb[...,0]=np.clip(55+30*u/w+15*np.sin(3*x),0,255).astype('uint8')
        rgb[...,1]=np.clip(90+45*v/h,0,255).astype('uint8')
        rgb[...,2]=155
        surface=rays*z[...,None]
        normal=np.maximum(-1,np.minimum(1,(surface-center)/r))
        diffuse=np.clip(.45+.55*(normal[...,2]*(-.6)+normal[...,1]*(-.8)),0,1)
        rgb[good]=np.clip(np.stack((210*diffuse+20,75*diffuse+20,62*diffuse+12),axis=-1),0,255)[good].astype('uint8')
        bfr=BytesIO();np.savez_compressed(bfr,depth_z=z,source_rgb=rgb,intrinsics_k=k,valid_mask=np.ones((h,w),bool),
                                          sigma_log_z=np.full((h,w),0.02,dtype=np.float32))
        stem=f'frame_{i:04d}'
        save_atomic(root/f'{stem}.npz',bfr.getvalue())
        image=BytesIO();Image.fromarray(rgb).save(image,format='PNG')
        save_atomic(root/f'{stem}.png',image.getvalue())
        t=round(i*.4,3)
        manifest['frames'].append({'time_s':t,'npz':f'{stem}.npz','pose_c2w':np.eye(4).tolist()})
        da3['frames'].append({'time_s':t,'image':f'{stem}.png'})
    save_atomic(root/'manifest_from_npz.json',json.dumps(manifest,indent=2).encode('utf-8'))
    save_atomic(root/'manifest_from_da3.json',json.dumps(da3,indent=2).encode('utf-8'))
    return manifest

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('examples/sequence5d'))
    args=parser.parse_args();print(json.dumps({'frames':len(generate(args.output)['frames']),'output':str(args.output)}))
