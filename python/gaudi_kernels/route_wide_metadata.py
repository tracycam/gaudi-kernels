"""Experimental four-TPC route-balanced metadata; no production install."""
from dataclasses import dataclass
import torch
from .route_metadata import RouteMetadata
@dataclass(frozen=True)
class RouteMetadataV3(RouteMetadata):
    chunk_offsets:torch.Tensor

def route_wide_metadata(ids,*,experts=384,rows=64,capacity=None):
    if ids.ndim!=2 or ids.dtype!=torch.int32 or ids.device.type not in ('hpu','meta') or not ids.is_contiguous():
        raise ValueError('contiguous HPU/Meta int32 IDs required, no narrowing')
    rows,capacity=geometry(ids.shape[0],ids.shape[1],experts,rows,capacity)
    op=torch.ops.gaudi_route_metadata_wide;flat=op.flatten(ids)
    counts,flags,offsets=op.count(ids,flat,experts)
    prefix,expert,base,valid,status=op.prefix(counts,flags,ids.shape[1],rows,capacity)
    inverse=op.inverse(ids,flat,prefix,status,offsets,rows,capacity)
    mapping=op.row_map(inverse,valid,status,rows)
    return RouteMetadataV3(counts,flags,prefix,expert,base,valid,status,mapping,inverse,rows,capacity,offsets)


def geometry(tokens,routes,experts=384,rows=64,capacity=None):
    """Explicit wide domain; never coerces a small row count into C64/C128."""
    if any(type(v)is not int for v in (tokens,routes,experts,rows)):
        raise ValueError('integer static wide geometry required')
    if not(rows in (64,128) and rows<=tokens<=513 and 1<=routes<=8 and routes<=experts<=384):
        raise ValueError('explicit C64/C128 with C<=T<=513 required')
    active=min(experts,tokens*routes)
    bound=min(active*((tokens+rows-1)//rows),active+(tokens*routes-active)//rows)
    if capacity is None:capacity=bound
    if type(capacity)is not int or not 1<=capacity<=tokens*routes:
        raise ValueError('positive bounded capacity required')
    return rows,capacity
