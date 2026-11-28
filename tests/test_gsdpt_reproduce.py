import torch
from scripts.reproduce_gsdpt import reproduce
from omega_geodepth.gsdpt_checkpoint import load_head


def test_trained_artifact(tmp_path):
    path=tmp_path/'head.safetensors'
    report=reproduce(path)
    assert report['holdout_after'] < report['holdout_before']
    assert report['train_after'] < report['train_before']
    assert report['reload_exact'] is True
    head=load_head(path,report['sha256'])
    assert torch.count_nonzero(head.proj.weight)>0
