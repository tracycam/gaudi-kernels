"""Experimental four-TPC route-balanced metadata; no production install."""
from dataclasses import dataclass
import torch
from .route_metadata import RouteMetadata,geometry
@dataclass(frozen=True)
class RouteMetadataV3(RouteMetadata):
    chunk_offsets:torch.Tensor

def route_metadata_v3(ids,*,experts=384,rows=16,capacity=None):
    if ids.ndim!=2 or ids.dtype!=torch.int32 or ids.device.type not in ('hpu','meta') or not ids.is_contiguous():
        raise ValueError('contiguous HPU/Meta int32 IDs required, no narrowing')
    rows,capacity=geometry(ids.shape[0],ids.shape[1],experts,rows,capacity)
    op=torch.ops.gaudi_route_metadata_v3;flat=op.flatten(ids)
    counts,flags,offsets=op.count(ids,flat,experts)
    prefix,expert,base,valid,status=op.prefix(counts,flags,ids.shape[1],rows,capacity)
    inverse=op.inverse(ids,flat,prefix,status,offsets,rows,capacity)
    mapping=op.row_map(inverse,valid,status,rows)
    return RouteMetadataV3(counts,flags,prefix,expert,base,valid,status,mapping,inverse,rows,capacity,offsets)
