"""CPU dynamic INT8 for DA3 backbone; preserve FP32 camera/depth heads."""
from __future__ import annotations
import torch
import hashlib
import struct
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as QLinear


def quantize_backbone(model: nn.Module) -> dict:
    if model.training:
        raise ValueError('INT8_REQUIRES_EVAL')
    if not isinstance(getattr(model, 'backbone', None), nn.Module):
        raise ValueError('INT8_BACKBONE_MISSING')
    if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('INT8_REQUIRES_CPU_FP32')
    engine = torch.backends.quantized.engine
    if engine not in ('x86', 'fbgemm', 'qnnpack', 'onednn'):
        raise RuntimeError('INT8_ENGINE_UNAVAILABLE')
    layers = [m for m in model.backbone.modules() if type(m) is nn.Linear]
    if not layers:
        raise ValueError('INT8_NO_LINEAR_LAYERS')
    before = sum(m.weight.numel() * m.weight.element_size() for m in layers)
    torch.ao.quantization.quantize_dynamic(model.backbone, {nn.Linear}, dtype=torch.qint8, inplace=True)
    quantized = [m for m in model.backbone.modules() if isinstance(m, QLinear)]
    if len(quantized) != len(layers):
        raise RuntimeError('INT8_INCOMPLETE_CONVERSION')
    packed = sum(m.weight().numel() * m.weight().element_size() for m in quantized)
    digest = hashlib.sha256()
    for name, module in sorted(model.backbone.named_modules()):
        if not isinstance(module, QLinear):
            continue
        weight = module.weight()
        digest.update(name.encode('utf-8') + b'\0')
        digest.update(struct.pack('<dqi', weight.q_scale(), weight.q_zero_point(), weight.ndim))
        digest.update(struct.pack('<' + 'q' * weight.ndim, *weight.shape))
        digest.update(weight.int_repr().contiguous().numpy().tobytes())
        bias = module.bias()
        if bias is not None:
            digest.update(bias.detach().contiguous().numpy().tobytes())
    return {'quantized_linear_payload_sha256': digest.hexdigest(),
            'linear_modules': len(quantized), 'original_weight_bytes': before,
            'packed_weight_bytes': packed, 'engine': engine,
            'execution_precision': 'dynamic_int8_backbone_fp32_heads',
            'notes': 'Weight payload only; excludes biases, packing overhead and activations.'}
