"""Graph-native TP expert path; one N-major packed weight copy."""
from gaudi_kernels.engine.context import context as execution_context
from pathlib import Path
import os
import torch
ROOT = Path(__file__).resolve().parent
LEGACY = 'pt2' == 'legacy'
BRIDGE = 'pt2'
torch.ops.load_library(execution_context().path('native'))

@torch.library.register_fake('native_mxfp4::rank_sum')
def _rank_sum_meta(gathered, ranks):
    return gathered.new_empty((gathered.shape[0] // ranks, gathered.shape[1]), dtype=torch.bfloat16)

@torch.library.register_fake('native_mxfp4::prep')
def _prep_meta(x, ids, blocks, splits, topk):
    tasks = ids.numel() * blocks * splits
    return (x.new_empty((tasks, x.shape[-1] // splits)), ids.new_empty((1, tasks), dtype=torch.int32))

@torch.library.register_fake('native_mxfp4::gemv')
def _gemv_meta(w, s, x, table, mapping):
    return x.new_empty((1, x.shape[0] * 512), dtype=torch.float32)

@torch.library.register_fake('native_mxfp4::gate')
def _gate_meta(partial, ids, width, h):
    tasks = ids.numel() * (h // 512)
    return (partial.new_empty((tasks, width), dtype=torch.bfloat16), ids.new_empty((1, tasks), dtype=torch.int32))

@torch.library.register_fake('native_mxfp4::combine')
def _combine_meta(partial, routing, directions, h, topk):
    return partial.new_empty((routing.numel() // topk, h))

def constants(device):
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.bfloat16)
    table = torch.stack((lut[torch.arange(256) % 16], lut[torch.arange(256) // 16]), dim=1).reshape(1, 512)
    dirs = torch.empty((2, 256), dtype=torch.uint8)
    for parity in range(2):
        for byte in range(256):
            lane = byte // 4
            dirs[parity, byte] = lane % 16 // 2 + lane // 16 % 2 * 32 + (128 if lane % 2 == parity else 0)
    return (table.to(device), dirs.to(device))

def moe(x, ids, routing, gp, gs, dp, ds, table, directions, splits=3):
    h = x.shape[-1]
    width = 256
    topk = ids.shape[-1]
    ids = ids.to(torch.int32)
    routing = routing.to(torch.float32 if '1' == '1' else torch.bfloat16)
    if not LEGACY:
        ids = ids.clone()
        routing = routing.clone()
        x = x.clone()
    (ax, gm) = torch.ops.native_mxfp4.prep(x, ids, 2 * width // 512, splits, topk)
    gp_out = torch.ops.native_mxfp4.gemv(gp, gs, ax, table, gm)
    (dx, dm) = torch.ops.native_mxfp4.gate(gp_out, ids, width, h)
    partial = torch.ops.native_mxfp4.gemv(dp, ds, dx, table, dm)
    from gaudi_kernels.serving.executor.precision_ops import combine
    return combine(partial, routing, directions, h, topk)
    return torch.ops.native_mxfp4.combine(partial, routing, directions, h, topk)
