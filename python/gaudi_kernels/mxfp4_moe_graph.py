"""Experimental functional whole-MoE in the caller's current HPU graph.

No device acquisition, private recipe/stream/replay, CPU route values or cached
expanded weights. Compiler SRAM/liveness and device quality remain required.
M1 always belongs to the existing literal historical path.
"""
import torch
from .mxfp4_moe_plan import MoePlan


def moe_historical(x, ids, routing, gp, gs, down, ds, lut, *, plan: MoePlan,
                   layout: int):
    if layout != 1:
        raise ValueError('explicit HistoricalN512=1 required; native-v2 bytes incompatible')
    tensors = (x, ids, routing, gp, gs, down, ds, lut)
    if x.device.type not in ('hpu', 'meta') or any(t.device != x.device for t in tensors):
        raise ValueError('same HPU device required; Meta supports offline construction only')
    if (x.ndim != 2 or x.shape[1] != 6144 or ids.ndim != 2
            or ids.shape[0] != x.shape[0] or routing.shape != ids.shape
            or gp.ndim != 2 or gp.shape[0] % 6144):
        raise ValueError('TP8 MoE shard shape mismatch')
    T, R, E = x.shape[0], ids.shape[1], gp.shape[0] // 6144
    expected = ((x, torch.bfloat16, (T, 6144)), (ids, torch.int32, (T, R)),
                (routing, torch.float32, (T, R)), (gp, torch.uint8, (E*6144, 256)),
                (gs, torch.uint8, (E*192, 512)), (down, torch.uint8, (E*3072, 256)),
                (ds, torch.uint8, (E*96, 512)), (lut, torch.bfloat16, (1, 512)))
    for tensor, dtype, shape in expected:
        if tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError('MoE tensor dtype/shape/contiguity contract')
    plan.require_execution_contract(T, R, E)
    buckets = plan.buckets(T, R, E)
    op = torch.ops.gaudi_moe_reference
    counts, status = op.count(ids, E)
    inverse, mapping = op.plan(ids, counts, int(plan.mode == 'bucket'), sum(b.experts for b in buckets))
    # Retain only BF16 gate fragments, in precisely the same slot order as map.
    gates = []
    for bucket in buckets:
        parts = []
        for offset in range(0, bucket.experts, plan.gp_expert_batch):
            batch = min(plan.gp_expert_batch, bucket.experts-offset)
            slot = bucket.slot_base+offset
            a = op.gather(x, ids, counts, mapping, bucket.capacity, slot, batch)
            w = op.decode(gp, gs, lut, mapping, 6144, 512, 512, slot, batch, 0, layout)
            parts.append(op.gate_rows(op.batch_mm(a, w), mapping, counts, slot))
        gates.append(parts)
    # N outer loop avoids materializing all padded down outputs. GP is outside:
    # its original weight bytes are not decoded again for each of the 12 tiles.
    output_tiles = []
    for n in range(0, 6144, plan.n_tile):
        partials = []
        for bucket, parts in zip(buckets, gates):
            for offset in range(0, bucket.experts, plan.down_expert_batch):
                batch = min(plan.down_expert_batch, bucket.experts-offset)
                first = offset // plan.gp_expert_batch
                count = (batch+plan.gp_expert_batch-1)//plan.gp_expert_batch
                chunks = parts[first:first+count]
                a = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
                w = op.decode(down, ds, lut, mapping, 256, 6144, plan.n_tile,
                              bucket.slot_base+offset, batch, n, layout)
                partials.append(op.reshape(op.batch_mm(a, w), [batch*bucket.capacity, plan.n_tile]))
        all_partial = partials[0] if len(partials) == 1 else torch.cat(partials, dim=0)
        output_tiles.append(op.combine(all_partial, ids, routing, inverse, status, E))
    return torch.cat(output_tiles, dim=1), status
