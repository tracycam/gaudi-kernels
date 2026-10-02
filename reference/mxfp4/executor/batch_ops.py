"""Phase-independent token-row MoE. All routing stays on HPU; no data .item()."""
from pathlib import Path
import os
import torch

ROOT=Path(__file__).resolve().parent.parent
torch.ops.load_library(str(ROOT/'ops-build/unified_batch_ops.so'))

@torch.library.register_fake('unified_batch::gp')
def _gp(w,s,x,table,ids):
    return x.new_empty((1,ids.numel()*3*512),dtype=torch.float32)

@torch.library.register_fake('unified_batch::gate')
def _gate(partial,ids):
    return partial.new_empty((ids.numel(),256),dtype=torch.bfloat16)

@torch.library.register_fake('unified_batch::down')
def _down(w,s,x,table,ids):
    return x.new_empty((1,ids.numel()*6144),dtype=torch.float32)

@torch.library.register_fake('unified_batch::gate_broadcast')
def _broadcast(partial,ids):
    n=ids.numel()*12
    return partial.new_empty((n,256),dtype=torch.bfloat16),ids.new_empty((1,n))

@torch.library.register_fake('unified_batch::sorted_prep')
def _sorted(x,ids,permutation):
    n=ids.numel()*3
    return x.new_empty((n,2048)),ids.new_empty((1,n)),ids.new_empty(ids.shape)

@torch.library.register_fake('unified_batch::combine')
def _combine(partial,routing,directions,inverse):
    return partial.new_empty((routing.shape[0],6144))

@torch.library.register_fake('unified_batch::masked_gemv')
def _masked(w,s,x,table,mapping):
    return x.new_empty((1,x.shape[0]*512),dtype=torch.float32)

def moe(x,ids,routing,gp,gs,dp,ds,table,directions,*,mode='broadcast',route_mask=None,debug=False):
    if x.ndim!=2 or x.shape[1]!=6144 or ids.ndim!=2 or ids.shape!=routing.shape or ids.shape[0]!=x.shape[0]:
        raise ValueError('Expected x[T,6144] and ids/routing[T,topk]')
    if x.dtype!=torch.bfloat16 or ids.shape[1]<1 or ids.shape[1]>384:
        raise ValueError('Expected BF16 activation and 1 <= top-k <= 384')
    if x.shape[0]==0:
        if debug:raise ValueError('Empty batches have no intermediate debug tensors')
        return x.new_empty((0,6144),dtype=torch.float32)
    if route_mask is not None:
        if route_mask.shape!=ids.shape or route_mask.dtype!=torch.bool:
            raise ValueError('route_mask must be bool[T,topk]')
        # Explicit mask supports ragged per-token k with fixed graph capacity.
        # Unused routes have a safe weight address; no host readback of counts.
        if mode=='compact':mode='broadcast'
        ids=torch.where(route_mask,ids,torch.full_like(ids,-1))
        routing=torch.where(route_mask,routing,torch.zeros_like(routing))
    # Existing Lazy CustomOp bridge requires materialization for view dependencies.
    # No host inspection of expert IDs/counts, irrespective of source phase.
    precise=os.getenv('UNIFIED_PRECISION_ROUTER')=='1'
    if precise and mode=='sorted':raise ValueError('FP32 routing is certified for compact/broadcast only')
    x=x.clone();ids=ids.to(torch.int32).clone();routing=routing.to(torch.float32 if precise else torch.bfloat16).clone()
    if mode=='compact':
        gp_out=torch.ops.unified_batch.gp(gp,gs,x,table,ids)
        gate=torch.ops.unified_batch.gate(gp_out,ids)
        partial=torch.ops.unified_batch.down(dp,ds,gate,table,ids)
    else:
        if mode not in ('broadcast','sorted'):
            raise ValueError('Expected compact, broadcast or sorted')
        if mode=='sorted':
            values,permutation=torch.sort(ids.reshape(-1))
            sorted_ids=values.reshape(ids.shape).clone()
            permutation=permutation.to(torch.int32).reshape(ids.shape).clone()
            ax,mapping,inverse=torch.ops.unified_batch.sorted_prep(x,sorted_ids,permutation)
            selected_ids=sorted_ids
        else:
            ax,mapping=torch.ops.native_mxfp4.prep(x,ids,1,3,ids.shape[1])
            selected_ids=ids
        gemv=torch.ops.unified_batch.masked_gemv if route_mask is not None else torch.ops.native_mxfp4.gemv
        gp_out=gemv(gp,gs,ax,table,mapping)
        gate,down_map=torch.ops.unified_batch.gate_broadcast(gp_out,selected_ids)
        partial=gemv(dp,ds,gate,table,down_map)
    if precise:
        from precision_ops import combine
        out=combine(partial,routing,directions,6144,ids.shape[1])
    else:
        out=(torch.ops.unified_batch.combine(partial,routing,directions,inverse) if mode=='sorted'
             else torch.ops.native_mxfp4.combine(partial,routing,directions,6144,ids.shape[1]))
    return (out,gp_out,gate,partial) if debug else out
