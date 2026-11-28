import hashlib
import torch
import pytest
from safetensors.torch import save
from omega_geodepth.gsdpt_checkpoint import save_head, load_head


def test_roundtrip(tmp_path):
    from omega_geodepth.gsdpt import GaussianHead
    h = GaussianHead(5)
    with torch.no_grad():
        h.proj.weight.fill_(.25)
    p = tmp_path / 'head.safetensors'
    digest = save_head(h, p)
    assert digest == hashlib.sha256(p.read_bytes()).hexdigest()
    restored = load_head(p, digest)
    assert torch.equal(restored.proj.weight, h.proj.weight)
    assert torch.equal(restored.proj.bias, h.proj.bias)
    assert not restored.training


def test_bad_digest(tmp_path):
    p = tmp_path / 'head.safetensors'
    p.write_bytes(b'bad')
    with pytest.raises(ValueError, match='digest'):
        load_head(p, '0'*64)


@pytest.mark.parametrize('weight', [torch.zeros(6, 2), torch.full((6, 5), float('nan')), torch.zeros(6, 5, dtype=torch.float64)])
def test_invalid_tensor(tmp_path, weight):
    p = tmp_path / 'head.safetensors'
    data = save({'proj.weight': weight, 'proj.bias': torch.zeros(6)})
    p.write_bytes(data)
    with pytest.raises(ValueError):
        load_head(p, hashlib.sha256(data).hexdigest())


def test_refuse_symlink(tmp_path):
    p = tmp_path / 'head.safetensors'
    target = tmp_path / 'target'
    target.write_bytes(b'unchanged')
    p.symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        load_head(p, hashlib.sha256(b'unchanged').hexdigest())
    from omega_geodepth.gsdpt import GaussianHead
    with pytest.raises((ValueError, OSError)):
        save_head(GaussianHead(5), p)
    assert target.read_bytes() == b'unchanged'


def test_no_overwrite(tmp_path):
    from omega_geodepth.gsdpt import GaussianHead
    p = tmp_path / 'head.safetensors'
    digest = save_head(GaussianHead(5), p)
    with pytest.raises(FileExistsError):
        save_head(GaussianHead(5), p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == digest


def test_oversized(tmp_path):
    p = tmp_path / 'head.safetensors'
    with p.open('wb') as f:
        f.truncate(1024*1024+1)
    with pytest.raises(ValueError, match='size'):
        load_head(p, '0'*64)
