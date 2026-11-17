"""Explicit opt-in downloader for official Apache-2.0 DA3 weights.

No inference path invokes this module. Download is prohibited unless --allow-network
and --accept-license are explicitly present on the command line.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .da3_adapter import ALLOWED_DA3, register_da3_checkpoint, validate_da3_checkpoint


def main():
    parser=argparse.ArgumentParser(description='Explicit official Depth Anything 3 checkpoint provisioning')
    parser.add_argument('--model',choices=list(ALLOWED_DA3),default='DA3-SMALL')
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--revision',default='main',help='Prefer pinned commit SHA when known')
    parser.add_argument('--allow-network',action='store_true')
    parser.add_argument('--accept-license',action='store_true')
    parser.add_argument('--register-existing',action='store_true')
    args=parser.parse_args()
    if args.register_existing:
        m=register_da3_checkpoint(args.destination,args.model,args.revision)
    else:
        if not args.allow_network or not args.accept_license:
            parser.error('Explicit --allow-network and --accept-license required')
        from huggingface_hub import snapshot_download
        target=args.destination.resolve()
        target.mkdir(parents=True,exist_ok=True)
        snapshot_download(repo_id=ALLOWED_DA3[args.model],revision=args.revision,
                          local_dir=target,allow_patterns=['model.safetensors','config.json','README.md','LICENSE'])
        m=register_da3_checkpoint(target,args.model,args.revision)
    print(json.dumps({'installed':True,'model':m['model_id'],
                      'weights_sha256':m['files']['model.safetensors']},indent=2))

if __name__=='__main__':main()
