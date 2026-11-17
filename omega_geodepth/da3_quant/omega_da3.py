#!/usr/bin/env python3
"""DA3-SMALL reproducible checkpoint optimization and numerical verification.

Only trusted tensor bytes are loaded; no pickle / model code from a checkpoint is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONFIG: dict[str, Any] = {
    "model_name": "da3-small",
    "config": {
        "__object__": {"path": "depth_anything_3.model.da3", "name": "DepthAnything3Net", "args": "as_params"},
        "net": {
            "__object__": {"path": "depth_anything_3.model.dinov2.dinov2", "name": "DinoV2", "args": "as_params"},
            "name": "vits", "out_layers": [5, 7, 9, 11], "alt_start": 4,
            "qknorm_start": 4, "rope_start": 4, "cat_token": True,
        },
        "head": {
            "__object__": {"path": "depth_anything_3.model.dualdpt", "name": "DualDPT", "args": "as_params"},
            "dim_in": 768, "output_dim": 2, "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "cam_enc": {
            "__object__": {"path": "depth_anything_3.model.cam_enc", "name": "CameraEnc", "args": "as_params"},
            "dim_out": 384,
        },
        "cam_dec": {
            "__object__": {"path": "depth_anything_3.model.cam_dec", "name": "CameraDec", "args": "as_params"},
            "dim_in": 768,
        },
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    intermediate = path.with_suffix(path.suffix + ".partial")
    intermediate.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    intermediate.replace(path)


def _save_tensors(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not tensors:
        raise ValueError("No tensor data supplied")
    intermediate = path.with_suffix(".partial.safetensors")
    try:
        save_file(tensors, str(intermediate), metadata=metadata)
        # safetensors metadata insertion order can vary across identical builds.
        # Canonicalize only the JSON header (never tensor data or offsets).
        with intermediate.open("r+b") as output:
            n = int.from_bytes(output.read(8), "little")
            if n > 16 * 1024 * 1024:
                raise ValueError("SafeTensors header exceeds limit")
            header = json.loads(output.read(n).decode("utf-8"))
            normalized = json.dumps(header, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
            if len(normalized) > n:
                raise ValueError("Canonical header exceeds storage")
            output.seek(8)
            output.write(normalized + b" " * (n - len(normalized)))
            output.flush()
            os.fsync(output.fileno())
        os.replace(intermediate, path)
    finally:
        intermediate.unlink(missing_ok=True)


def inspect_checkpoint(path: Path) -> dict[str, Any]:
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as archive:
        names = list(archive.keys())
        metadata = archive.metadata() or {}
        types: dict[str, int] = {}
        params = 0
        for key in names:
            item = archive.get_slice(key)
            shape = item.get_shape()
            params += math.prod(shape)
            if len(shape) > 0:
                types[str(len(shape))] = types.get(str(len(shape)), 0) + 1
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "tensor_count": len(names),
        "parameter_count": params,
        "rank_distribution": types,
        "metadata": metadata,
    }


def _model_sensitive_key(key: str) -> bool:
    """Protect pose path, embeddings, normalization, and final heads in mixed-precision."""
    return (
        key.startswith("model.cam_")
        or key.endswith(".bias")
        or key.endswith(".gamma")
        or "norm" in key.lower()
        or "pos_embed" in key
        or "cls_token" in key
        or "output_conv2" in key
    )


def _int8_protected_key(key: str) -> bool:
    """Preserve numerically sensitive readout, normalization and calibration."""
    return (
        key.endswith(".bias") or key.endswith(".gamma")
        or "norm" in key.lower()
        or "pos_embed" in key or "cls_token" in key
        or "output_conv2" in key
        or key.startswith("model.cam_dec.")
    )


def quantize_groupwise(x: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    if group_size < 1 or group_size > 4096:
        raise ValueError("group_size must be in 1..4096")
    if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Expected floating-point tensor")
    if x.numel() == 0 or not torch.isfinite(x).all().item():
        raise ValueError("Tensor is empty or contains non-finite values")
    flat = x.detach().to(dtype=torch.float32, device="cpu").contiguous().flatten()
    groups = (flat.numel() + group_size - 1) // group_size
    padded = torch.nn.functional.pad(flat, (0, groups * group_size - flat.numel())).reshape(groups, group_size)
    scales = padded.abs().amax(dim=1) / 127.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    quantized = torch.round(padded / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.flatten()[:flat.numel()].contiguous(), scales.contiguous()


def dequantize_groupwise(
    q: torch.Tensor, scales: torch.Tensor, shape: tuple[int, ...], group_size: int = 64
) -> torch.Tensor:
    if group_size < 1 or group_size > 4096:
        raise ValueError("Invalid group size")
    count = math.prod(shape)
    if q.dtype != torch.int8 or q.numel() != count or count <= 0:
        raise ValueError("Invalid quantized data shape or dtype")
    groups = (count + group_size - 1) // group_size
    if scales.dtype != torch.float32 or scales.numel() != groups or not torch.isfinite(scales).all():
        raise ValueError("Invalid quantization scales")
    if not (scales > 0).all():
        raise ValueError("All quantization scales must be > 0")
    flat = q.reshape(-1).to(torch.float32)
    extended = torch.nn.functional.pad(flat, (0, groups * group_size - count)).reshape(groups, group_size)
    return (extended * scales.reshape(groups, 1)).flatten()[:count].reshape(shape).contiguous()


def _normalized_error(original: torch.Tensor, converted: torch.Tensor) -> float:
    a = original.to(dtype=torch.float32, device="cpu").flatten().to(torch.float64)
    b = converted.to(dtype=torch.float32, device="cpu").flatten().to(torch.float64)
    numerator = torch.linalg.vector_norm(a - b)
    denominator = torch.linalg.vector_norm(a).clamp_min(1e-30)
    return float((numerator / denominator).item())


def _check_architecture(tensors: dict[str, torch.Tensor]) -> None:
    expected_shapes = {
        "model.backbone.pretrained.patch_embed.proj.weight": (384, 3, 14, 14),
        "model.backbone.pretrained.blocks.11.attn.qkv.weight": (1152, 384),
        "model.cam_enc.trunk.0.attn.proj.weight": (384, 384),
        "model.head.projects.0.weight": (48, 768, 1, 1),
        "model.head.scratch.output_conv2.2.weight": (2, 32, 1, 1),
    }
    for key, expected in expected_shapes.items():
        if key not in tensors:
            raise ValueError(f"Checkpoint does not match DA3-SMALL, missing {key}")
        if tuple(tensors[key].shape) != expected:
            raise ValueError(f"Invalid architecture shape for {key}: {tensors[key].shape} != {expected}")


def build_variants(source: Path, destination: Path, int8_group_size: int = 64) -> dict[str, Any]:
    source, destination = Path(source), Path(destination)
    if source.stat().st_size > 8 * 1024**3:
        raise ValueError("Checkpoint exceeds the configured size guard (8 GiB)")
    if not (1 <= int8_group_size <= 4096):
        raise ValueError("Unsupported group size")
    with safe_open(str(source), framework="pt", device="cpu") as archive:
        names = list(archive.keys())
        metadata = archive.metadata() or {}
        for alias, target in metadata.items():
            if target not in names:
                raise ValueError(f"Unresolvable checkpoint alias: {alias} -> {target}")
        original = {key: archive.get_tensor(key).contiguous() for key in names}
    if len(names) > 20:
        _check_architecture(original)
    for name, tensor in original.items():
        if tensor.dtype != torch.float32:
            raise ValueError(f"Checkpoint expected FP32, found {tensor.dtype} at {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Checkpoint has non-finite values at {name}")
    source_info = inspect_checkpoint(source)
    source_info["finite"] = True
    source_info["architecture"] = "DA3-SMALL" if len(names) > 20 else "synthetic-test"

    fp16 = {name: tensor.half().contiguous() for name, tensor in original.items()}
    mixed = {name: (tensor if _model_sensitive_key(name) else tensor.half()).contiguous()
             for name, tensor in original.items()}
    q_tensors: dict[str, torch.Tensor] = {}
    q_table: dict[str, dict[str, Any]] = {}
    q_dist = {"group_int8": 0, "fp16": 0, "fp32": 0}
    for name, tensor in original.items():
        chosen = "fp16"
        if _int8_protected_key(name):
            chosen = "fp32"
        elif tensor.ndim >= 2 and tensor.numel() >= 4096:
            q, s = quantize_groupwise(tensor, int8_group_size)
            error = _normalized_error(tensor, dequantize_groupwise(q, s, tuple(tensor.shape), int8_group_size))
            if error <= 0.02:
                q_tensors[f"q::{name}"] = q
                q_tensors[f"s::{name}"] = s
                chosen = "group_int8"
        if chosen != "group_int8":
            q_tensors[f"f::{name}"] = tensor.clone() if chosen == "fp32" else tensor.half().contiguous()
        q_table[name] = {"dtype": chosen, "shape": list(tensor.shape)}
        q_dist[chosen] += 1

    variants: dict[str, dict[str, Any]] = {}
    for variant, weights, meta in [
        ("fp16", fp16, metadata),
        ("mixed", mixed, metadata),
        ("int8", q_tensors, {"omega-format": "groupwise-int8-with-fallbacks-v1"}),
    ]:
        folder = destination / variant
        target = folder / "model.safetensors"
        if source.resolve() == target.resolve():
            raise ValueError("Refusing to overwrite original source checkpoint")
        _save_tensors(target, weights, meta)
        _write_json(folder / "config.json", CONFIG)
        variants[variant] = {
            "bytes": target.stat().st_size,
            "sha256": sha256_file(target),
            "tensor_count": len(weights),
        }
    _write_json(destination / "int8" / "quantization.json", {
        "schema": "omega-da3-int8-v1", "group_size": int8_group_size,
        "source_sha256": source_info["sha256"], "original_metadata": metadata,
        "entries": q_table,
    })
    variants["int8"]["storage_modes"] = q_dist
    report = {"source": source_info, "variants": variants, "transform": "deterministic rounding without training"}
    _write_json(destination / "build_report.json", report)
    return report


def restore_int8(quantized: Path, descriptor: Path, output: Path) -> None:
    descriptor_data = json.loads(Path(descriptor).read_text(encoding="utf-8"))
    if descriptor_data.get("schema") != "omega-da3-int8-v1":
        raise ValueError("Unsupported quantization descriptor")
    if len(descriptor_data.get("entries", {})) > 10000:
        raise ValueError("Too many tensors in descriptor")
    size = descriptor_data["group_size"]
    restored: dict[str, torch.Tensor] = {}
    with safe_open(str(quantized), framework="pt", device="cpu") as archive:
        for key, entry in descriptor_data["entries"].items():
            kind = entry["dtype"]
            shape = tuple(entry["shape"])
            if kind == "group_int8":
                restored[key] = dequantize_groupwise(
                    archive.get_tensor(f"q::{key}"), archive.get_tensor(f"s::{key}"), shape, size
                ).half()
            elif kind in ("fp16", "fp32"):
                restored[key] = archive.get_tensor(f"f::{key}")
            else:
                raise ValueError(f"Invalid entry storage mode: {kind}")
            if tuple(restored[key].shape) != shape:
                raise ValueError(f"Shape mismatch after restoring {key}")
            if not torch.isfinite(restored[key]).all().item():
                raise ValueError(f"Non-finite values after restoring {key}")
    _save_tensors(Path(output), restored, descriptor_data["original_metadata"])


def validate_checkpoint(source: Path, output: Path, mode: str) -> dict[str, Any]:
    if mode not in ("fp16", "mixed", "int8"):
        raise ValueError("Unsupported validation mode")
    with safe_open(str(source), framework="pt", device="cpu") as original, safe_open(str(output), framework="pt", device="cpu") as updated:
        original_keys = list(original.keys())
        output_keys = list(updated.keys())
        if original_keys != output_keys:
            return {"ok": False, "reason": "Tensor key mismatch"}
        if (original.metadata() or {}) != (updated.metadata() or {}):
            return {"ok": False, "reason": "Shared-parameter alias metadata mismatch"}
        numerator = 0.0
        denominator = 0.0
        greatest_abs = 0.0
        worst_tensor = ["", 0.0]
        by_dtype: dict[str, int] = {}
        for key in original_keys:
            a = original.get_tensor(key)
            b = updated.get_tensor(key)
            if a.shape != b.shape or not torch.isfinite(b).all().item():
                return {"ok": False, "reason": f"Invalid shape or nonfinite tensor {key}"}
            if mode == "fp16" and b.dtype != torch.float16:
                return {"ok": False, "reason": f"Wrong dtype for fp16 tensor {key}"}
            by_dtype[str(b.dtype)] = by_dtype.get(str(b.dtype), 0) + 1
            delta = a.double() - b.double()
            numerator += float(torch.sum(delta.square()).item())
            denominator += float(torch.sum(a.double().square()).item())
            greatest_abs = max(greatest_abs, float(delta.abs().max().item()))
            err = _normalized_error(a, b)
            if err > worst_tensor[1]:
                worst_tensor = [key, err]
    relative_l2 = math.sqrt(numerator / max(denominator, 1e-60))
    limit = 0.03 if mode == "int8" else 0.005
    return {
        "ok": relative_l2 <= limit,
        "relative_l2": relative_l2,
        "max_absolute_error": greatest_abs,
        "worst_tensor": {"name": worst_tensor[0], "relative_l2": worst_tensor[1]},
        "tensors_by_dtype": by_dtype,
        "threshold": limit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd_inspect = commands.add_parser("inspect")
    cmd_inspect.add_argument("source", type=Path)
    cmd_build = commands.add_parser("build")
    cmd_build.add_argument("source", type=Path)
    cmd_build.add_argument("destination", type=Path)
    cmd_build.add_argument("--group-size", type=int, default=64)
    cmd_restore = commands.add_parser("restore-int8")
    cmd_restore.add_argument("quantized", type=Path)
    cmd_restore.add_argument("descriptor", type=Path)
    cmd_restore.add_argument("output", type=Path)
    cmd_verify = commands.add_parser("verify")
    cmd_verify.add_argument("source", type=Path)
    cmd_verify.add_argument("updated", type=Path)
    cmd_verify.add_argument("--mode", choices=["fp16", "mixed", "int8"], required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        result = inspect_checkpoint(args.source)
    elif args.command == "build":
        result = build_variants(args.source, args.destination, args.group_size)
    elif args.command == "restore-int8":
        restore_int8(args.quantized, args.descriptor, args.output)
        result = {"restored": str(args.output), "sha256": sha256_file(args.output)}
    else:
        result = validate_checkpoint(args.source, args.updated, args.mode)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.command == "verify" and not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
