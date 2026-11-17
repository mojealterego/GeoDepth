"""Compare v0.1 and v0.2 with deterministic held-out, synthetic-only examples."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from omega_geodepth.infer import load_verified_model
from omega_geodepth.train import dataset,evaluate
from omega_geodepth.synthetic import generate_scene,log_relative_depth
from omega_geodepth.tiling import tiled_predict

def benchmark(new:Path,old:Path,count:int=128,seed:int=80000,highres_count:int=24)->dict:
    torch.set_num_threads(3)
    x,y=dataset(count,seed,64)
    results={'evaluation_domain':'procedural scenes only, no external real-world benchmarks',
             'held_out_seed_start':seed,'held_out_scene_count':count}
    for key,path in [('v0.1',old),('v0.2',new)]:
        net,meta=load_verified_model(path)
        start=time.perf_counter()
        with torch.inference_mode():result=evaluate(net,x,y)
        results[key]={'version':meta['version'],'params':meta['parameters'],
                      'benchmark_seconds':round(time.perf_counter()-start,3),**result}
        if key=='v0.2':
            tiled_mae=[];native_mae=[]
            for i in range(highres_count):
                rgb,depth=generate_scene(90_000+i,(128,128))
                truth=log_relative_depth(depth)
                a,b=tiled_predict(net,rgb,tile_size=64,overlap=24,global_side=64)
                tiled_mae.append(float(np.mean(np.abs(a-truth))))
                img=torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)[None]))
                with torch.inference_mode():direct_log,_=net(img)
                native_mae.append(float(np.mean(np.abs(direct_log[0,0].numpy()-truth))))
            results['high_res_128px_synthetic']={
                'tiled_log_depth_mae':float(np.mean(tiled_mae)),
                'direct_log_depth_mae':float(np.mean(native_mae)),
                'scene_count':highres_count,'note':'no claim that tiles are always more accurate'}
    results['change_percent_v02_vs_v01_on_synthetic_holdout']=(
        results['v0.2']['log_depth_mae']/results['v0.1']['log_depth_mae']-1)*100
    return results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--v02',type=Path,default=Path('models'))
    p.add_argument('--v01',type=Path,required=True)
    p.add_argument('--count',type=int,default=128)
    p.add_argument('--highres-count',type=int,default=24)
    p.add_argument('--output',type=Path,default=Path('reports/benchmark_v02.json'))
    a=p.parse_args()
    result=benchmark(a.v02,a.v01,a.count,highres_count=a.highres_count)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
