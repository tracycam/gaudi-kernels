"""Experimental fixed-row MXFP4 MoE MME graph; not installed by model policy.

Original HistoricalN512 owners are decoded only as bounded transient fragments.
Compiler SRAM placement, device FP32 quality and end-to-end gains must still be
qualified. One Python construction is NOT one device launch. Route values and
status never cross to CPU. Every public tensor has a complete-write contract.
"""
import torch
from .route_metadata import route_metadata, geometry


def moe_route_tiles(x, ids, routing, gp, gs, down, ds, lut, *, layout,
                    rows=16, capacity=None, gp_batch=2, down_batch=16,
                    n_tile=512, metadata_version=1, schedule='staged', stream_decoder='qualified_u4'):
    if layout != 1:
        raise ValueError('explicit HistoricalN512=1 layout required')
    if stream_decoder not in ('qualified_u4', 'k8'):
        raise ValueError('unknown stream decoder')
    if schedule not in ('staged', 'expert_stream'):
        raise ValueError('unknown route-tile schedule')
    if any(type(v) is not int for v in (gp_batch, down_batch, n_tile, metadata_version)):
        raise ValueError('integer static tile geometry required')
    if (gp_batch < 1 or gp_batch > 2 or down_batch < gp_batch
            or down_batch % gp_batch or n_tile < 512 or n_tile % 512
            or 6144 % n_tile or down_batch * 256 * n_tile * 2 > 16*1024*1024
            or metadata_version not in (1, 2, 3)):
        raise ValueError('bounded MME fragments and metadata version 1/2/3 required')
    if x.ndim != 2 or ids.ndim != 2 or gp.ndim != 2 or gp.shape[0] % 6144:
        raise ValueError('TP8 MoE shard geometry required')
    t, r, e = x.shape[0], ids.shape[1], gp.shape[0] // 6144
    if schedule == 'expert_stream' and (t > 8 or r > 8):
        raise ValueError('expert-stream graph currently supports at most eight tokens/top8')
    rows, capacity = geometry(t, r, e, rows, capacity)
    expected = ((x, torch.bfloat16, (t,6144)), (ids, torch.int32, (t,r)),
                (routing, torch.float32, (t,r)), (gp, torch.uint8, (e*6144,256)),
                (gs, torch.uint8, (e*192,512)), (down, torch.uint8, (e*3072,256)),
                (ds, torch.uint8, (e*96,512)), (lut, torch.bfloat16, (1,512)))
    for value, dtype, shape in expected:
        if (value.dtype != dtype or tuple(value.shape) != shape
                or value.device != x.device or value.device.type not in ('hpu','meta')
                or not value.is_contiguous() or value.requires_grad):
            raise ValueError('original tensor dtype/device/layout contract')
    make_metadata = route_metadata
    if metadata_version == 2:
        from .route_metadata_v2 import route_metadata_v2
        make_metadata = route_metadata_v2
    elif metadata_version == 3:
        from .route_metadata_v3 import route_metadata_v3
        make_metadata = route_metadata_v3
    meta = make_metadata(ids, experts=e, rows=rows, capacity=capacity)
    tile = torch.ops.gaudi_route_tiles
    core = torch.ops.gaudi_moe_reference
    # Public graph-owned logical alias; no raw address binding or CPU values.
    mapping = tile.reshape_i32(meta.tile_expert, [1, capacity])
    if schedule == 'expert_stream':
        # One gather both zero-fills unused rows and writes the MME layout.
        # Public split outputs retain producer edges; no raw tensor address
        # rebinding, CPU count readback, or independent launch in forward.
        gathered = tile.gather(x, meta.row_map, meta.status, r, rows, 0, capacity)
        decode = torch.ops.gaudi_moe_stream.decode if stream_decoder == 'k8' else core.decode
        parts = []
        for slot, a in zip(range(0, capacity, gp_batch),
                           torch.ops.gaudi_moe_stream.split(gathered, gp_batch)):
            batch = min(gp_batch, capacity-slot)
            w = decode(gp, gs, lut, mapping, 6144, 512, 512, slot, batch, 0, layout)
            gate = tile.gate(core.batch_mm(a, w), meta.valid_rows, meta.status, slot)
            # Full down-N for the same pair is only 6 MiB of transient BF16.
            # No gate from another pair is a dependency of this projection.
            w = decode(down, ds, lut, mapping, 256, 6144, 6144, slot, batch, 0, layout)
            parts.append(core.reshape(core.batch_mm(gate, w), [batch*rows, 6144]))
        partial = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return tile.combine(partial, routing, meta.inverse, meta.status), meta
    gates = []
    for slot in range(0, capacity, gp_batch):
        batch = min(gp_batch, capacity-slot)
        a = tile.gather(x, meta.row_map, meta.status, r, rows, slot, batch)
        w = core.decode(gp, gs, lut, mapping, 6144, 512, 512, slot, batch, 0, layout)
        gates.append(tile.gate(core.batch_mm(a,w), meta.valid_rows, meta.status, slot))
    # Hoist the same immutable owners outside the down N loop.
    groups = []
    for slot in range(0, capacity, down_batch):
        stop = min(slot+down_batch, capacity)
        chunks = gates[slot//gp_batch:(stop+gp_batch-1)//gp_batch]
        groups.append(chunks[0] if len(chunks)==1 else torch.cat(chunks,dim=0))
    outputs = []
    for n in range(0, 6144, n_tile):
        partials = []
        for group, slot in zip(groups, range(0,capacity,down_batch)):
            batch = min(down_batch,capacity-slot)
            w = core.decode(down, ds, lut, mapping, 256, 6144, n_tile, slot, batch, n, layout)
            partials.append(core.reshape(core.batch_mm(group,w), [batch*rows,n_tile]))
        partial = partials[0] if len(partials)==1 else torch.cat(partials,dim=0)
        outputs.append(tile.combine(partial,routing,meta.inverse,meta.status))
    return torch.cat(outputs,dim=1), meta
