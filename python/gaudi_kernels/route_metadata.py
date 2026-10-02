"""Experimental fixed-shape metadata producer; no model policy installs.

The caller loads the TPC/bridge libraries before capture. Status stays on device:
0=valid, bit0=bad ID, bit1=duplicate in token, bit2=capacity overflow,
bit3=inconsistent counts. A future consumer must consume/mask status; this
producer cannot raise a Python exception from device values during capture.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RouteMetadata:
    counts: torch.Tensor
    row_status: torch.Tensor
    prefix: torch.Tensor
    tile_expert: torch.Tensor
    tile_base: torch.Tensor
    valid_rows: torch.Tensor
    status: torch.Tensor
    row_map: torch.Tensor
    inverse: torch.Tensor
    rows: int
    capacity: int


def geometry(tokens, routes, experts=384, rows=16, capacity=None):
    if any(type(x) is not int for x in (tokens, routes, experts, rows)):
        raise ValueError('integer static geometry required')
    if not (1 <= tokens <= 513 and 1 <= routes <= 8 and routes <= experts <= 384 and 1 <= rows <= 32):
        raise ValueError('experimental route metadata domain')
    rows = min(rows, tokens)
    active = min(experts, tokens * routes)
    bound = min(active * ((tokens + rows - 1) // rows), active + (tokens * routes - active) // rows)
    if capacity is None:
        capacity = bound
    if type(capacity) is not int or not 1 <= capacity <= tokens * routes:
        raise ValueError('positive bounded capacity required')
    return rows, capacity


def route_metadata(ids, *, experts=384, rows=16, capacity=None):
    if (ids.ndim != 2 or ids.dtype != torch.int32 or
            ids.device.type not in ('hpu', 'meta') or not ids.is_contiguous()):
        raise ValueError('contiguous HPU/Meta int32 IDs[T,R] required; no narrowing')
    rows, capacity = geometry(ids.shape[0], ids.shape[1], experts, rows, capacity)
    native_ids = ids
    counts, row_status = torch.ops.gaudi_route_metadata.count(native_ids, experts)
    prefix, tile_expert, tile_base, valid, status = torch.ops.gaudi_route_metadata.prefix(
        counts, row_status, ids.shape[1], rows, capacity)
    row_map, inverse = torch.ops.gaudi_route_metadata.scatter(native_ids, prefix, status, rows, capacity)
    return RouteMetadata(counts, row_status, prefix, tile_expert, tile_base, valid,
                         status, row_map, inverse, rows, capacity)
