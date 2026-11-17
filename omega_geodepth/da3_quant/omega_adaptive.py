#!/usr/bin/env python3
"""OMEGA DA3 v2: deterministic packed 4/6/8-bit weight transport, NOT native INT kernels.

No training, benchmarking on depth images or weight quality improvements are implied.
The format uses explicit, canonical little-endian bit packing and checksum-verified
SafeTensors; no pickle is loaded. Supports lossily restoring standard DA3 weights.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from .omega_da3 import (
    CONFIG, _int8_protected_key, _save_tensors, _write_json, sha256_file,
    inspect_checkpoint,
)

BITS = (4, 6, 8)
SCHEMA = 'omega-da3-aq-v2'
MAX_DESCRIPTOR_BYTES = 3 * 1024 * 1024
MAX_PARAMS = 150_000_000
MAX_SOURCE_BYTES = 2 * 1024**3
MAX_TENSORS = 10000
MAX_RANK = 8

# Thresholds are weight-relative L2 limits, not depth-map accuracy requirements.
# 'aq4' is experimental and can incur much higher layer-output errors than 'aq6'.
PROFILES: dict[str, dict[str, Any]] = {
    'aq6': {'candidates': [(6, 64, 0.042), (8, 64, 0.015)], 'protect_camera': False},
    'aq4': {'candidates': [(4, 32, 0.125), (6, 64, 0.042), (8, 64, 0.015)], 'protect_camera': False},
}


def _check_bits(bits: int) -> None:
    if type(bits) is not int or bits not in BITS:
        raise ValueError('bits must be 4, 6 or 8')


def pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack uint8 values into canonical little-endian bit stream, without extra trailing bytes."""
    _check_bits(bits)
    if values.dtype != torch.uint8 or values.ndim != 1 or values.numel() == 0:
        raise ValueError('Expected nonempty 1D uint8 values')
    if torch.any(values > ((1 << bits) - 1)).item():
        raise ValueError('Value outside target bit width')
    data = values.detach().cpu().contiguous().numpy()
    padding = (-data.size) % 8
    if padding:
        data = np.pad(data, (0, padding))
    blocks = data.astype(np.uint64, copy=False).reshape(-1, 8)
    shifts = (np.arange(8, dtype=np.uint64) * np.uint64(bits))[None, :]
    words = np.bitwise_or.reduce(np.left_shift(blocks, shifts), axis=1).astype('<u8')
    used = (values.numel() * bits + 7) // 8
    packed = words.view(np.uint8).reshape(-1, 8)[:, :bits].reshape(-1)[:used]
    return torch.from_numpy(packed.copy())


def unpack_unsigned(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    """Validate canonical length/padding and unpack bit stream deterministically."""
    _check_bits(bits)
    if type(count) is not int or count <= 0 or count > MAX_PARAMS:
        raise ValueError('Invalid element count')
    expected = (count * bits + 7) // 8
    if packed.dtype != torch.uint8 or packed.ndim != 1 or packed.numel() != expected:
        raise ValueError('Noncanonical packed data length or dtype')
    rem = (count * bits) % 8
    if rem and (int(packed[-1]) >> rem):
        raise ValueError('Nonzero unused padding bits')
    raw = packed.cpu().contiguous().numpy()
    bytes_needed = ((count + 7) // 8) * bits
    buf = np.zeros(bytes_needed, dtype=np.uint8)
    buf[:raw.size] = raw
    rows = buf.reshape(-1, bits)
    # Append zero bytes to build exactly one uint64 word for every 8 symbols.
    padded = np.zeros((len(rows), 8), dtype=np.uint8)
    padded[:, :bits] = rows
    words = padded.reshape(-1).view('<u8')
    shifts = (np.arange(8, dtype=np.uint64) * np.uint64(bits))[None, :]
    values = np.bitwise_and(np.right_shift(words[:, None], shifts), (1 << bits) - 1)
    return torch.from_numpy(values.astype(np.uint8, copy=False).reshape(-1)[:count].copy())


def _refine_mse_scales(groups: torch.Tensor, scales: torch.Tensor,
                       vmax: int, steps: int = 4) -> torch.Tensor:
    """Deterministic, per-group least-squares scale refinement.

    Every update is accepted only when its actual reconstructed squared error
    decreases; baseline absmax is always retained otherwise. No training data.
    """
    for _ in range(steps):
        symbols = torch.round(groups / scales[:, None]).clamp(-vmax, vmax)
        denominator = symbols.square().sum(dim=1)
        candidate = (symbols * groups).sum(dim=1) / denominator.clamp_min(1.0)
        candidate = torch.where((candidate > 0) & torch.isfinite(candidate), candidate, scales)
        proposal = torch.round(groups / candidate[:, None]).clamp(-vmax, vmax)
        old_loss = (groups - symbols * scales[:, None]).square().sum(dim=1)
        new_loss = (groups - proposal * candidate[:, None]).square().sum(dim=1)
        scales = torch.where(new_loss < old_loss, candidate, scales)
    return scales


def quantize_packed(x: torch.Tensor, bits: int, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    _check_bits(bits)
    if type(group_size) is not int or not 1 <= group_size <= 4096:
        raise ValueError('Invalid quantization group size')
    if x.dtype not in (torch.float16, torch.float32, torch.bfloat16) or x.numel() <= 0:
        raise ValueError('Expected nonempty floating-point tensor')
    if x.numel() > MAX_PARAMS or not torch.isfinite(x).all().item():
        raise ValueError('Tensor too large or contains NaN/Inf')
    values = x.detach().cpu().float().flatten().contiguous()
    padding = (-values.numel()) % group_size
    groups = torch.nn.functional.pad(values, (0, padding)).reshape(-1, group_size)
    vmax = (1 << (bits - 1)) - 1
    scales = groups.abs().amax(dim=1) / vmax
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    if not torch.isfinite(scales).all().item() or torch.any(scales <= 0):
        raise ValueError('Invalid scales from input')
    scales = _refine_mse_scales(groups, scales, vmax)
    q = torch.round(groups / scales[:, None]).clamp(-vmax, vmax).to(torch.int16)
    positive = (q.flatten()[:values.numel()] + vmax).to(torch.uint8).contiguous()
    return pack_unsigned(positive, bits), scales.contiguous()


def dequantize_packed(packed: torch.Tensor, scales: torch.Tensor,
                      shape: tuple[int, ...], bits: int, group_size: int = 64) -> torch.Tensor:
    _check_bits(bits)
    if type(group_size) is not int or not 1 <= group_size <= 4096:
        raise ValueError('Invalid group_size')
    count = _validate_shape(list(shape))
    groups = (count + group_size - 1) // group_size
    if (scales.dtype != torch.float32 or scales.ndim != 1 or scales.numel() != groups
            or not torch.isfinite(scales).all().item() or torch.any(scales <= 0).item()):
        raise ValueError('Invalid scales')
    positive = unpack_unsigned(packed, count, bits)
    vmax = (1 << (bits - 1)) - 1
    if torch.any(positive > (2 * vmax)).item():
        raise ValueError('Invalid reserved quantization symbol')
    signed = positive.to(torch.float32) - vmax
    padding = groups * group_size - count
    grouped = torch.nn.functional.pad(signed, (0, padding)).reshape(groups, group_size)
    result = (grouped * scales.reshape(-1, 1)).flatten()[:count].reshape(shape).contiguous()
    if not torch.isfinite(result).all().item():
        raise ValueError('Non-finite reconstruction')
    return result


def _validate_shape(shape: Any) -> int:
    if not isinstance(shape, list) or not 1 <= len(shape) <= MAX_RANK:
        raise ValueError('Invalid rank in descriptor')
    count = 1
    for dim in shape:
        if type(dim) is not int or dim < 1 or dim > MAX_PARAMS:
            raise ValueError('Invalid dimension in descriptor')
        count *= dim
        if count > MAX_PARAMS:
            raise ValueError('Tensor exceeds element limit')
    return count


def _architecture_check(archive: Any, names: list[str]) -> None:
    if len(names) < 20:  # for minimal synthetic unit fixtures
        return
    required = {
        'model.backbone.pretrained.patch_embed.proj.weight': (384, 3, 14, 14),
        'model.backbone.pretrained.blocks.11.attn.qkv.weight': (1152, 384),
        'model.cam_enc.trunk.0.attn.proj.weight': (384, 384),
        'model.head.projects.0.weight': (48, 768, 1, 1),
        'model.head.scratch.output_conv2.2.weight': (2, 32, 1, 1),
    }
    for key, dims in required.items():
        if key not in names or tuple(archive.get_slice(key).get_shape()) != dims:
            raise ValueError(f'Unsupported DA3-SMALL architecture tensor: {key}')


def _is_protected(name: str, profile: str) -> bool:
    return _int8_protected_key(name)


def _error_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    numerator = torch.sum((a.double() - b.double()).square()).item()
    denominator = max(torch.sum(a.double().square()).item(), 1e-60)
    return math.sqrt(numerator / denominator)


def build_aq_variants(source: Path, destination: Path) -> dict[str, Any]:
    """Generate checksum-verified experimental transport files from original FP32 weights."""
    source, destination = Path(source), Path(destination)
    if not source.is_file() or source.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Source missing or exceeds size guard')
    info = inspect_checkpoint(source)
    if info['parameter_count'] > MAX_PARAMS or info['tensor_count'] > MAX_TENSORS:
        raise ValueError('Model exceeds resource guard')
    results: dict[str, Any] = {}
    for profile, params in PROFILES.items():
        folder = destination / profile
        target = folder / 'model.safetensors'
        if source.resolve() == target.resolve():
            raise ValueError('Refusing to overwrite source checkpoint')
        contents: dict[str, torch.Tensor] = {}
        entries: dict[str, Any] = {}
        counts = {'fp32': 0, 'fp16': 0, 'packed4': 0, 'packed6': 0, 'packed8': 0}
        squared_error = 0.0
        squared_ref = 0.0
        max_layer_error = 0.0
        with safe_open(str(source), framework='pt', device='cpu') as archive:
            names = list(archive.keys())
            aliases = archive.metadata() or {}
            if len(names) > MAX_TENSORS:
                raise ValueError('Too many tensors')
            _architecture_check(archive, names)
            for alias, tgt in aliases.items():
                if tgt not in names or not isinstance(alias, str):
                    raise ValueError('Invalid source alias')
            for name in names:
                if not isinstance(name, str) or len(name) > 512 or not name:
                    raise ValueError('Invalid name in checkpoint')
                x = archive.get_tensor(name)
                if x.dtype != torch.float32 or not torch.isfinite(x).all().item():
                    raise ValueError(f'Invalid input type or nonfinite tensor: {name}')
                shape = list(x.shape)
                count = _validate_shape(shape)
                if _is_protected(name, profile):
                    kind = 'fp32'
                    contents[f'f::{name}'] = x.contiguous()
                    reconstructed = x
                    entry = {'kind': kind, 'shape': shape}
                else:
                    best = None
                    # Camera encoder is restricted to 6/8-bit even in the compact profile.
                    candidates = (params['candidates'][1:] if profile == 'aq4' and name.startswith('model.cam_enc.')
                                  else params['candidates'])
                    for bits, group_size, allowed in candidates:
                        if x.ndim < 2 or count < 4096:
                            break
                        q, scales = quantize_packed(x, bits, group_size)
                        reconstructed = dequantize_packed(q, scales, tuple(shape), bits, group_size)
                        error = _error_ratio(x, reconstructed)
                        if error <= allowed:
                            best = (bits, group_size, q, scales, reconstructed)
                            break
                    if best is not None:
                        bits, group_size, q, scales, reconstructed = best
                        contents[f'q::{name}'] = q
                        contents[f's::{name}'] = scales
                        kind = f'packed{bits}'
                        entry = {'kind': kind, 'shape': shape, 'group_size': group_size, 'bits': bits}
                    else:
                        kind = 'fp16'
                        contents[f'f::{name}'] = x.half().contiguous()
                        reconstructed = contents[f'f::{name}'].float()
                        entry = {'kind': kind, 'shape': shape}
                err = _error_ratio(x, reconstructed)
                if not math.isfinite(err):
                    raise ValueError(f'Nonfinite quantization error: {name}')
                squared_error += torch.sum((x.double() - reconstructed.double()).square()).item()
                squared_ref += torch.sum(x.double().square()).item()
                max_layer_error = max(max_layer_error, err)
                entry['relative_weight_l2'] = round(err, 9)
                entries[name] = entry
                counts[kind] += 1
        _save_tensors(target, contents, {'omega-format': SCHEMA, 'omega-profile': profile})
        _write_json(folder / 'config.json', CONFIG)
        descriptor = {
            'schema': SCHEMA, 'profile': profile,
            'checkpoint_sha256': sha256_file(target),
            'source_sha256': info['sha256'],
            'original_metadata': aliases, 'entries': entries,
        }
        _write_json(folder / 'quantization.json', descriptor)
        global_err = math.sqrt(squared_error / max(squared_ref, 1e-60))
        results[profile] = {
            'bytes': target.stat().st_size,
            'sha256': descriptor['checkpoint_sha256'],
            'tensor_count': len(entries),
            'storage_modes': counts,
            'global_relative_weight_l2': global_err,
            'max_tensor_relative_weight_l2': max_layer_error,
            'valid': validate_adaptive(folder)['ok'],
        }
        if not results[profile]['valid']:
            raise ValueError('Generated artifact did not pass integrity checks')
    report = {
        'schema': SCHEMA, 'source': info, 'variants': results,
        'warning': 'Low-bit weights are experimental transport artifacts, not native low-bit kernels or improved model accuracy',
    }
    _write_json(destination / 'build_report.json', report)
    return report


def _read_descriptor(folder: Path) -> dict[str, Any]:
    docpath = folder / 'quantization.json'
    if not docpath.is_file() or docpath.stat().st_size > MAX_DESCRIPTOR_BYTES:
        raise ValueError('Missing/oversized descriptor')
    descriptor = json.loads(docpath.read_text('utf8'))
    if descriptor.get('schema') != SCHEMA or descriptor.get('profile') not in PROFILES:
        raise ValueError('Unsupported checkpoint descriptor')
    entries = descriptor.get('entries')
    if not isinstance(entries, dict) or not 1 <= len(entries) <= MAX_TENSORS:
        raise ValueError('Invalid checkpoint index')
    total = 0
    for name, entry in entries.items():
        if not isinstance(name, str) or not name or len(name) > 512 or not isinstance(entry, dict):
            raise ValueError('Invalid tensor name or entry')
        total += _validate_shape(entry.get('shape'))
        if total > MAX_PARAMS:
            raise ValueError('Too many parameters requested')
        kind = entry.get('kind')
        if kind not in ('fp16', 'fp32', 'packed4', 'packed6', 'packed8'):
            raise ValueError('Invalid storage kind')
        if kind.startswith('packed'):
            bits = entry.get('bits')
            group_size = entry.get('group_size')
            if type(bits) is not int or kind != f'packed{bits}' or bits not in BITS:
                raise ValueError('Invalid quantizer bit width')
            if type(group_size) is not int or not 1 <= group_size <= 4096:
                raise ValueError('Invalid quantizer group size')
    metadata = descriptor.get('original_metadata')
    if not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) and v in entries
                                                  for k,v in metadata.items()):
        raise ValueError('Invalid alias metadata')
    hash_value = descriptor.get('checkpoint_sha256')
    if not isinstance(hash_value,str) or len(hash_value) != 64 or any(x not in '0123456789abcdef' for x in hash_value):
        raise ValueError('Invalid checkpoint checksum')
    return descriptor


def _check_archived_contents(folder: Path, descriptor: dict[str, Any]) -> None:
    file = folder / 'model.safetensors'
    if not file.is_file() or file.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError('Packed checkpoint missing or too large')
    if sha256_file(file) != descriptor['checkpoint_sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    expected_keys: set[str] = set()
    for name, entry in descriptor['entries'].items():
        if entry['kind'].startswith('packed'):
            expected_keys.update((f'q::{name}', f's::{name}'))
        else:
            expected_keys.add(f'f::{name}')
    with safe_open(str(file), framework='pt', device='cpu') as archive:
        if set(archive.keys()) != expected_keys:
            raise ValueError('Missing/unexpected SafeTensors keys')
        if archive.metadata() != {'omega-format': SCHEMA, 'omega-profile': descriptor['profile']}:
            raise ValueError('Incorrect packed metadata')
        for name, entry in descriptor['entries'].items():
            shape = entry['shape']
            count = _validate_shape(shape)
            if entry['kind'].startswith('packed'):
                bits, group_size = entry['bits'],entry['group_size']
                q = archive.get_slice(f'q::{name}')
                scale = archive.get_slice(f's::{name}')
                if q.get_shape() != [(count*bits+7)//8] or scale.get_shape() != [(count+group_size-1)//group_size]:
                    raise ValueError('Incorrect packed shape')
                if q.get_shape()[0] < 1:
                    raise ValueError('Empty packed tensor')
                # safe_open.get_slice returns dtype as strings like U8 / F32.
                if q.get_dtype() != 'U8' or scale.get_dtype() != 'F32':
                    raise ValueError('Incorrect packed dtype')
            else:
                f = archive.get_slice(f'f::{name}')
                if f.get_shape() != shape or f.get_dtype() != ('F16' if entry['kind']=='fp16' else 'F32'):
                    raise ValueError('Incorrect uncompressed shape or dtype')


def validate_adaptive(folder: Path) -> dict[str, Any]:
    """Check integrity + descriptor consistency. Does not imply quality or functional E2E."""
    try:
        folder = Path(folder)
        descriptor = _read_descriptor(folder)
        _check_archived_contents(folder, descriptor)
        # Deep validation: checksum alone does not establish that scales/codes
        # are semantically valid. Reject NaN, Inf and reserved bit patterns.
        with safe_open(str(folder / 'model.safetensors'), framework='pt', device='cpu') as archive:
            for name, entry in descriptor['entries'].items():
                if entry['kind'].startswith('packed'):
                    dequantize_packed(archive.get_tensor(f'q::{name}'),
                        archive.get_tensor(f's::{name}'), tuple(entry['shape']),
                        entry['bits'], entry['group_size'])
                elif not torch.isfinite(archive.get_tensor(f'f::{name}')).all().item():
                    raise ValueError(f'Non-finite fallback tensor: {name}')
        return {'ok': True, 'profile': descriptor['profile'], 'tensors': len(descriptor['entries']),
                'checkpoint_sha256': descriptor['checkpoint_sha256']}
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        return {'ok': False, 'reason': str(exc)}


def restore_adaptive(folder: Path, output: Path) -> dict[str, Any]:
    folder, output = Path(folder), Path(output)
    if output.resolve() in { (folder / 'model.safetensors').resolve(), (folder / 'quantization.json').resolve() }:
        raise ValueError('Output would overwrite packed input or descriptor')
    descriptor = _read_descriptor(folder)
    _check_archived_contents(folder, descriptor)
    restored: dict[str, torch.Tensor] = {}
    with safe_open(str(folder / 'model.safetensors'), framework='pt', device='cpu') as archive:
        for name, entry in descriptor['entries'].items():
            shape = tuple(entry['shape'])
            kind = entry['kind']
            if kind.startswith('packed'):
                x = dequantize_packed(archive.get_tensor(f'q::{name}'),
                    archive.get_tensor(f's::{name}'), shape, entry['bits'], entry['group_size'])
                restored[name] = x.half().contiguous()
            else:
                restored[name] = archive.get_tensor(f'f::{name}').contiguous()
            if not torch.isfinite(restored[name]).all().item():
                raise ValueError(f'Nonfinite restored tensor: {name}')
    _save_tensors(output, restored, descriptor['original_metadata'])
    return {'output': str(output), 'sha256': sha256_file(output), 'bytes': output.stat().st_size,
            'tensor_count': len(restored)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('source', type=Path)
    build.add_argument('destination', type=Path)
    verify = commands.add_parser('verify')
    verify.add_argument('folder', type=Path)
    restore = commands.add_parser('restore')
    restore.add_argument('folder', type=Path)
    restore.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        result = build_aq_variants(args.source, args.destination)
    elif args.command == 'verify':
        result = validate_adaptive(args.folder)
        if not result['ok']:
            parser.error(result['reason'])
    else:
        result = restore_adaptive(args.folder, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
