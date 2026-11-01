# SPDX-License-Identifier: Apache-2.0
"""Device-owned committed DFlash KV ring, independent of ephemeral draft KV.

Call commit only with target-verified features projected by Draft.project_context.
Counts include the accepted prefix under the caller's EOS/output-budget policy.
Prefill chunks must be <= window, chronological, with unique accepted positions;
large initial prefill may project just its trailing window. Request reordering
belongs to the serving scheduler and is not implied by this fixed-batch object.
"""
import torch
from torch import nn


class CommittedContext(nn.Module):
    def __init__(self,*,layers,batch,window,kv_heads,dim,device,dtype=torch.bfloat16):
        super().__init__()
        assert min(layers,batch,window,kv_heads,dim)>0
        self.layers,self.batch,self.window=layers,batch,window
        # The extra slot is a write-only sink for rejected/padded rows. It never
        # becomes visible context, even when every proposal is rejected.
        for layer in range(layers):
            for kind in ('key','value'):
                self.register_buffer(f'{kind}_{layer}',torch.zeros(batch,window+1,kv_heads,dim,device=device,dtype=dtype))
        self.register_buffer('positions',torch.full((batch,window+1),-1,device=device,dtype=torch.int32))

    def reset(self,rows):
        assert rows.shape==(self.batch,)and rows.dtype==torch.bool
        self.positions.masked_fill_(rows[:,None],-1)

    def commit(self,projected,positions,counts):
        assert positions.ndim==2 and positions.shape[0]==self.batch and 0<positions.shape[1]<=self.window
        assert counts.shape==(self.batch,)and len(projected)==self.layers
        assert positions.dtype in (torch.int32,torch.int64)and counts.dtype in (torch.int32,torch.int64)
        t=positions.shape[1]
        accepted=(torch.arange(t,device=positions.device)[None,:]<counts[:,None])&(positions>=0)
        slots=torch.where(accepted,positions%self.window,self.window).long()
        for layer,pair in enumerate(projected):
            assert len(pair)==2
            for kind,source in zip(('key','value'),pair):
                target=getattr(self,f'{kind}_{layer}')
                assert source.shape==(self.batch,t,*target.shape[2:])and source.dtype==target.dtype
                index=slots[:,:,None,None].expand_as(source)
                target.scatter_(1,index,torch.where(accepted[:,:,None,None],source,0))
        self.positions.scatter_(1,slots,torch.where(accepted,positions,-1).int())

    def state(self):
        positions=self.positions[:,:self.window]
        pairs=tuple((getattr(self,f'key_{i}')[:,:self.window],getattr(self,f'value_{i}')[:,:self.window])for i in range(self.layers))
        return pairs,positions,positions>=0
