"""Static geometry consensus, strict offline NPZ and independent evaluation CLI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .prediction import load_prediction_npz
from .pipeline import run_pipeline,sha256_file
from .evaluation import evaluate,save_evaluation


def main():
    p=argparse.ArgumentParser(description='GeoDepth static geometry quality tools')
    subs=p.add_subparsers(dest='command',required=True)
    q=subs.add_parser('reconstruct')
    q.add_argument('--prediction',type=Path,required=True)
    q.add_argument('--output',type=Path,required=True)
    q.add_argument('--assume-static',action='store_true',required=True)
    q.add_argument('--anchors',type=Path)
    q.add_argument('--refine',action='store_true')
    q.add_argument('--quality-maps',action='store_true')
    q.add_argument('--voxel-size',type=float,default=.025)
    q.add_argument('--max-points',type=int,default=1_000_000)
    q.add_argument('--min-coverage',type=float,default=.1)
    q.add_argument('--min-consistency',type=float,default=.8)
    q.add_argument('--min-baseline-ratio',type=float,default=.005)
    q.add_argument('--relative-tolerance',type=float,default=.1)
    q.add_argument('--min-support',type=int,default=1)
    e=subs.add_parser('evaluate')
    e.add_argument('--prediction-depth',type=Path,required=True)
    e.add_argument('--ground-truth-depth',type=Path,required=True)
    e.add_argument('--alignment',choices=('none','median-scale'),default='none')
    e.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='evaluate':
        report=evaluate(a.prediction_depth,a.ground_truth_depth,alignment=a.alignment)
        save_evaluation(a.output,report)
    else:
        anchors=None
        if a.anchors:
            if a.anchors.is_symlink() or a.anchors.stat().st_size>8_000_000:raise ValueError('unsafe anchor file')
            anchors=json.loads(a.anchors.read_text())
        report=run_pipeline(load_prediction_npz(a.prediction),a.output,
                            anchors=anchors,source={'input_sha256':sha256_file(a.prediction),'static_assumption':True},
                            voxel_size=a.voxel_size,max_points=a.max_points,min_support=a.min_support,
                            min_view_coverage=a.min_coverage,min_mean_consistency=a.min_consistency,
                            min_baseline_ratio=a.min_baseline_ratio,relative_tolerance=a.relative_tolerance,
                            refine_depth=a.refine,save_quality_maps=a.quality_maps)
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':main()
