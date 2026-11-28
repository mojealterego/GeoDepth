"""Bounded, hash-pinned SafeTensors persistence for the experimental Gaussian head.

Only trusted destination directories are supported. A digest is an integrity
pin, not proof of authenticity or training provenance. No pickle is used.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
import re
import stat
import tempfile

import torch
from safetensors.torch import load, save

from .gsdpt import GaussianHead

MAX_BYTES = 1024 * 1024


def _validate(state: dict[str, torch.Tensor]) -> int:
    if set(state) != {'proj.weight', 'proj.bias'}:
        raise ValueError('invalid checkpoint keys')
    w, b = state['proj.weight'], state['proj.bias']
    if w.ndim != 2 or w.shape[0] != 6 or not 3 <= w.shape[1] <= 4096 or b.shape != (6,):
        raise ValueError('invalid checkpoint shape')
    for t in (w, b):
        if t.dtype != torch.float32 or not bool(torch.isfinite(t).all()):
            raise ValueError('invalid checkpoint tensor')
    return w.shape[1]


def save_head(head: GaussianHead, path: Path) -> str:
    """Atomically publish a new file (never overwrite) and return its SHA-256."""
    path = Path(path)
    state = {k: v.detach().to('cpu').contiguous().clone() for k, v in head.state_dict().items()}
    _validate(state)
    data = save(state)
    if len(data) > MAX_BYTES:
        raise ValueError('checkpoint size limit')
    fd, temp = tempfile.mkstemp(prefix='.gsdpt-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.link(temp, path)
    finally:
        os.unlink(temp)
    return hashlib.sha256(data).hexdigest()


def load_head(path: Path, expected_sha256: str) -> GaussianHead:
    """Read a regular file once, check the trusted digest, then strictly load."""
    if not re.fullmatch(r'[0-9a-f]{64}', expected_sha256):
        raise ValueError('invalid expected digest')
    if not hasattr(os, 'O_NOFOLLOW'):
        raise OSError('no safe file-open support on this platform')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        info = os.fstat(f.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= MAX_BYTES:
            raise ValueError('checkpoint file type or size limit')
        data = f.read(MAX_BYTES + 1)
    if len(data) != info.st_size or len(data) > MAX_BYTES:
        raise ValueError('checkpoint size changed')
    if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), expected_sha256):
        raise ValueError('checkpoint digest mismatch')
    state = load(data)
    channels = _validate(state)
    # Loading must not advance the caller's CPU random stream.
    with torch.random.fork_rng(devices=[]):
        head = GaussianHead(channels)
    head.load_state_dict(state, strict=True)
    return head.eval()
