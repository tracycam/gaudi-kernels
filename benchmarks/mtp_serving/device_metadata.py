"""Device tensor producer for fixed-capacity target verification metadata.

Scheduler supplies/reserves the physical page table at admission. Decode
cursors and query lengths stay on device. The returned lists contain masked
holes; binding them to the deployed attention consumer still needs its gate.
No claim that this prototype is the current vLLM/native serving implementation.
"""
import torch


def query_metadata(page_table, lengths, query_lengths, capacity, block_size, window, pad_page, kv_num_pages):
    """Produce per-query full/SWA lists and isolated padding slots.

    page_table[B,P] and cursors/query_lengths[B] are integer device tensors.
    Pages <0 are unallocated. Invalid active requests report fault[B] and use
    only padding storage; the surrounding cycle must suppress commit on fault.
    """
    b,p = page_table.shape
    assert lengths.shape == query_lengths.shape == (b,)
    # BF16 usage preserves integer counts exactly for these page sizes.
    assert capacity > 0 and 0 < block_size <= 256 and window > 0 and 0 <= pad_page < kv_num_pages
    device=page_table.device
    columns=torch.arange(capacity,device=device,dtype=lengths.dtype)
    positions=lengths[:,None]+columns
    logical=positions//block_size
    active=columns<query_lengths[:,None]
    fault=(lengths<0)|(query_lengths<0)|(query_lengths>capacity)
    fault=fault|((logical>=p)&active).any(1)
    requested=(torch.arange(p,device=device)[None,:]<((lengths+query_lengths+block_size-1)//block_size)[:,None])
    bad_page=(page_table<0)|(page_table>=kv_num_pages)|(page_table==pad_page)
    fault=fault|(bad_page&requested&(query_lengths>0)[:,None]).any(1)
    valid=active&~fault[:,None]
    physical=page_table.gather(1,logical.clamp(0,p-1).long())
    scratch=pad_page*block_size+torch.arange(b*capacity,device=device).reshape(b,capacity)%block_size
    slots=torch.where(valid,physical*block_size+positions%block_size,scratch).to(torch.int32)
    groups=torch.arange(b*capacity,device=device,dtype=torch.int32).reshape(b,capacity,1)

    def form(ids,logical_pages):
        included=valid[:,:,None]&(logical_pages<=logical[:,:,None])&(logical_pages>=0)
        page_ids=torch.where(included,ids,pad_page).to(torch.int32)
        block_groups=torch.where(included,groups,-1).to(torch.int32)
        usage=torch.where(logical_pages==logical[:,:,None],positions[:,:,None]%block_size+1,block_size)
        usage=torch.where(included,usage,1).to(torch.bfloat16)
        return page_ids.flatten(),block_groups.flatten(),usage.flatten()

    all_pages=torch.arange(p,device=device,dtype=lengths.dtype)[None,None,:]
    full=form(page_table[:,None,:].expand(b,capacity,p),all_pages)
    window_capacity=(window+block_size-2)//block_size+1
    left=(positions-window+1).clamp_min(0)//block_size
    window_pages=left[:,:,None]+torch.arange(window_capacity,device=device,dtype=lengths.dtype)
    window_ids=page_table.gather(1,window_pages.clamp(0,p-1).reshape(b,-1).long()).reshape(b,capacity,-1)
    swa=form(window_ids,window_pages)
    return dict(positions=torch.where(valid,positions,0).to(torch.int32).reshape(-1,1),
                slot_mapping=slots.reshape(-1,1),block_list=full[0],block_groups=full[1],block_usage=full[2],
                window_block_list=swa[0],window_block_groups=swa[1],window_block_usage=swa[2],fault=fault)
