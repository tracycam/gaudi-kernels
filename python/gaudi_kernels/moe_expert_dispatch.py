"""Experimental device expert-M partition, original-width W4A16 weights.

The static plan allocates capacities; actual routing/counts never leave HPU.
Empty MME slots still execute in this Synapse graph and their cost is measured.
No production policy selects this module without an explicit installed plan.
"""
import torch
from .moe_expert_plan import ExpertPlan


def expert_moe(x, ids, routing, gp, gs, down, ds, lut, directions, *, plan, tpc, debug=False, decoder="historical", empty_mode=0):
    if type(plan) is not ExpertPlan or x.ndim != 2 or ids.ndim != 2:
        raise ValueError('explicit expert plan and token/route matrices required')
    t,r=ids.shape
    if gp.ndim!=2 or gp.shape[0]%6144:
        raise ValueError('HistoricalN512 GP weights required')
    e=gp.shape[0]//6144
    expected=((x,torch.bfloat16,(t,6144)),(ids,torch.int32,(t,r)),
              (routing,torch.float32,(t,r)),(gp,torch.uint8,(e*6144,256)),
              (gs,torch.uint8,(e*192,512)),(down,torch.uint8,(e*3072,256)),
              (ds,torch.uint8,(e*96,512)),(lut,torch.bfloat16,(1,512)))
    for value,dtype,shape in expected:
        if (value.dtype!=dtype or tuple(value.shape)!=shape or value.device!=x.device
                or value.device.type not in ('hpu','meta') or not value.is_contiguous()
                or value.requires_grad):
            raise ValueError('expert-M weight/activation/layout contract')
    if decoder not in ('historical','k8') or type(empty_mode) is not int or empty_mode not in (0,1,2):
        raise ValueError('explicit decoder and empty slot policy required')
    if decoder=='historical' and empty_mode:
        raise ValueError('historical decoder always writes zero')
    cost=plan.costs(t,r,e)
    if not cost['fits_budget']:
        raise ValueError('expert-M conservative workspace bound exceeds budget')
    if not cost['buckets']:
        result=tpc(x,ids,routing,gp,gs,down,ds,lut,directions)
        return (result,None,()) if debug else result
    old=torch.ops.gaudi_route_metadata_v3
    op=torch.ops.gaudi_expert_partition
    core=torch.ops.gaudi_moe_reference
    flat=old.flatten(ids)
    counts,flags,offsets=old.count(ids,flat,e)
    # GP and down see the same selected expert set. The device histogram is
    # shared by all buckets; no .item(), CPU copy, per-expert host branch or sync.
    buckets=[]
    for b in cost['buckets']:
        prefix,experts,base,valid,status=op.prefix(counts,flags,r,b.rows,b.slots,b.lower,b.upper,int(b.fallback_overflow))
        inverse=op.inverse(ids,flat,prefix,status,offsets,b.rows,b.slots)
        mapping=op.row_map(inverse,valid,status,b.rows)
        buckets.append((b,experts,valid,status,inverse,mapping))
    if plan.min_mme_rows==1 and not plan.max_slots_per_bucket:
        result=torch.zeros_like(x,dtype=torch.float32)
    else:
        # Selection can deliberately cap capacity. Every unselected route,
        # including experts overflowing a bucket, must remain owned by TPC.
        selected=buckets[0][4]>=0
        for item in buckets[1:]:selected=selected|(item[4]>=0)
        small=(~selected.reshape(ids.shape))&(buckets[0][3]==0)
        result=tpc(x,ids,routing,gp,gs,down,ds,lut,directions,route_mask=small)
    def decode(w,s,k,n,nt,slot,batch,nb,ids_view):
        if decoder=='historical':return core.decode(w,s,lut,ids_view,k,n,nt,slot,batch,nb,1)
        return op.decode(w,s,lut,ids_view,k,n,nt,slot,batch,nb,empty_mode)
    debug_metadata=[]
    for b,experts,valid,status,inverse,mapping in buckets:
        # One expert appears in exactly one slot, in exactly one bucket.
        # Unlike fixed-row route tiles, M is not split into repeated weight
        # decode tasks. Capacity padding belongs only to this expert's bucket.
        ids_view=torch.ops.gaudi_route_tiles.reshape_i32(experts,[1,b.slots])
        gates=[]
        for slot in range(0,b.slots,2):
            batch=min(2,b.slots-slot)
            a=op.gather(x,mapping,status,r,b.rows,slot,batch)
            w=decode(gp,gs,6144,512,512,slot,batch,0,ids_view)
            gates.append(op.gate(core.batch_mm(a,w),valid,status,slot))
        groups=[]
        for slot in range(0,b.slots,16):
            stop=min(slot+16,b.slots)
            chunks=gates[slot//2:(stop+1)//2]
            groups.append(chunks[0] if len(chunks)==1 else torch.cat(chunks,dim=0))
        outputs=[]
        for n in range(0,6144,plan.down_n_tile):
            partials=[]
            for a,slot in zip(groups,range(0,b.slots,16)):
                batch=min(16,b.slots-slot)
                w=decode(down,ds,256,6144,plan.down_n_tile,slot,batch,n,ids_view)
                partials.append(core.reshape(core.batch_mm(a,w),[batch*b.rows,plan.down_n_tile]))
            partial=partials[0] if len(partials)==1 else torch.cat(partials,dim=0)
            outputs.append(op.combine(partial,routing,inverse,status))
        result=result+torch.cat(outputs,dim=1)
        if debug:
            debug_metadata.append(dict(bucket=b,experts=experts.clone(),valid=valid.clone(),
                                       status=status.clone(),inverse=inverse.clone(),mapping=mapping.clone()))
    return (result,counts.clone(),tuple(debug_metadata)) if debug else result
