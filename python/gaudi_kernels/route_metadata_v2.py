"""Independent experimental vector-count metadata namespace; v1 unchanged."""
from dataclasses import dataclass
import torch
from .route_metadata import RouteMetadata, geometry


@dataclass(frozen=True)
class RouteMetadataV2(RouteMetadata):
    chunk_counts: torch.Tensor


def route_metadata_v2(ids, *, experts=384, rows=16, capacity=None):
    if (ids.ndim!=2 or ids.dtype!=torch.int32 or
            ids.device.type not in ('hpu','meta') or not ids.is_contiguous()):
        raise ValueError('contiguous HPU/Meta int32 IDs required; no narrowing')
    rows,capacity=geometry(ids.shape[0],ids.shape[1],experts,rows,capacity)
    flat=torch.ops.gaudi_route_metadata_v2.flatten(ids)
    counts,flags,chunks=torch.ops.gaudi_route_metadata_v2.count(ids,flat,experts)
    prefix,tile_expert,tile_base,valid,status=torch.ops.gaudi_route_metadata_v2.prefix(
        counts,flags,ids.shape[1],rows,capacity)
    row_map,inverse=torch.ops.gaudi_route_metadata_v2.scatter(ids,prefix,status,chunks,rows,capacity)
    return RouteMetadataV2(counts,flags,prefix,tile_expert,tile_base,valid,status,
                           row_map,inverse,rows,capacity,chunks)
