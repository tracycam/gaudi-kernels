"""Functional consumer/dataflow specification, CPU only.

Projection and literal gate implementations are injected. This file validates
route flow and bounded lifetimes; it does not certify any numerical callback.
No shared mutable scatter cache, opaque address, device readback or kernel load.
"""
from dataclasses import dataclass
import numpy as np
from planner import validate


@dataclass(frozen=True)
class Interface:
    gp_batch: int = 2
    down_batch: int = 16
    input_width: int = 6144
    gate_width: int = 256
    output_width: int = 6144
    n_tile: int = 512

    def check(self):
        dimensions=(self.gp_batch,self.down_batch,self.input_width,self.gate_width,self.output_width,self.n_tile)
        if any(type(value) is not int or value<1 for value in dimensions):
            raise ValueError('positive integer dimensions and batches required')
        if (self.down_batch<self.gp_batch or self.down_batch%self.gp_batch
                or self.output_width%self.n_tile):
            raise ValueError('invalid functional fragment geometry')


def metadata_arrays(plan):
    """Proposed device metadata outputs; returned CPU arrays are the oracle."""
    c=plan.capacity
    return dict(counts=np.array(plan.counts,np.int32),
        tile_expert=np.array(plan.tile_expert,np.int32),
        tile_ordinal_base=np.array(plan.tile_ordinal_base,np.int32),
        tile_valid_rows=np.array(plan.tile_valid_rows,np.int32),
        row_to_route=np.array(plan.row_to_route,np.int32).reshape(c.max_tiles,c.rows),
        inverse=np.array(plan.inverse,np.int32).reshape(c.tokens,c.routes))


def execute(x_bf16_bits, ids, routing, plan, *, gp, literal_gate, down, combine, interface=Interface()):
    """Compose GP→literal gate→down→ordered combine with immutable intermediates.

    gp(A_bf16_bits[batch,C,K],expert_ids) -> FP32[batch,C,2I]
    literal_gate(P_fp32,valid_rows) -> BF16_BITS[batch,C,I]
    down(G_bf16_bits[batch,C,I],expert_ids,n0,width) -> FP32[batch,C,width]
    combine(P_ordered[T,R,width],routing[T,R]) -> FP32[T,width], ascending-slot FMA.
    Real implementations must preserve original weight ownership, FP32 MME and
    exact stated BF16 gate boundaries. Arbitrary callbacks are NOT certification.
    """
    interface.check();validate(ids,plan);cap=plan.capacity;meta=metadata_arrays(plan)
    if x_bf16_bits.dtype!=np.uint16 or x_bf16_bits.shape!=(cap.tokens,interface.input_width):raise ValueError('original BF16 bit payload required')
    if routing.dtype!=np.float32 or routing.shape!=(cap.tokens,cap.routes):raise ValueError('FP32 ordered routing required')
    counters=dict(gp_fragments=0,down_fragments=0,gate_concats=0,down_partial_concats=0,
                  gp_active_weight_tiles=0,down_active_weight_tile_width=0,
                  max_local_gather_bytes=0,max_local_gp_partial_bytes=0,max_down_tile_payload_bytes=0)
    gates=[]
    for start in range(0,cap.max_tiles,interface.gp_batch):
        stop=min(start+interface.gp_batch,cap.max_tiles);batch=stop-start
        a=np.zeros((batch,cap.rows,interface.input_width),np.uint16)
        for local,tile in enumerate(range(start,stop)):
            for row,route in enumerate(meta['row_to_route'][tile]):
                if route>=0:a[local,row]=x_bf16_bits[int(route)//cap.routes]
        experts=meta['tile_expert'][start:stop]
        p=gp(a,experts)
        if p.dtype!=np.float32 or p.shape!=(batch,cap.rows,2*interface.gate_width):raise ValueError('GP must return FP32 logical rows')
        g=literal_gate(p,meta['tile_valid_rows'][start:stop])
        if g.dtype!=np.uint16 or g.shape!=(batch,cap.rows,interface.gate_width):raise ValueError('gate must return the declared BF16 boundary')
        for local,count in enumerate(meta['tile_valid_rows'][start:stop]):assert not np.any(g[local,int(count):]),'padding must not carry NaNs or stale values'
        gates.append(g);counters['gp_fragments']+=1;counters['gp_active_weight_tiles']+=int(np.count_nonzero(experts>=0))
        counters['max_local_gather_bytes']=max(counters['max_local_gather_bytes'],a.nbytes)
        counters['max_local_gp_partial_bytes']=max(counters['max_local_gp_partial_bytes'],p.nbytes)
        del a,p
    # Hoist immutable gate coalescing outside N: share these same owners across
    # all down tiles. No partial-scatter alias contract is required.
    down_groups=[]
    for start in range(0,cap.max_tiles,interface.down_batch):
        stop=min(start+interface.down_batch,cap.max_tiles);first=start//interface.gp_batch
        parts=gates[first:(stop+interface.gp_batch-1)//interface.gp_batch]
        group=parts[0] if len(parts)==1 else np.concatenate(parts,axis=0)
        counters['gate_concats']+=int(len(parts)>1);down_groups.append(group)
    output_tiles=[]
    for n0 in range(0,interface.output_width,interface.n_tile):
        partials=[]
        for index,start in enumerate(range(0,cap.max_tiles,interface.down_batch)):
            stop=min(start+interface.down_batch,cap.max_tiles);experts=meta['tile_expert'][start:stop]
            p=down(down_groups[index],experts,n0,interface.n_tile)
            if p.dtype!=np.float32 or p.shape!=(stop-start,cap.rows,interface.n_tile):raise ValueError('down must return FP32 logical rows')
            partials.append(p.reshape(-1,interface.n_tile));counters['down_fragments']+=1
            counters['down_active_weight_tile_width']+=int(np.count_nonzero(experts>=0))*interface.n_tile
        full=partials[0] if len(partials)==1 else np.concatenate(partials,axis=0)
        counters['down_partial_concats']+=int(len(partials)>1)
        counters['max_down_tile_payload_bytes']=max(counters['max_down_tile_payload_bytes'],full.nbytes)
        # CPU gather is the specification for the existing ordered combine's
        # inverse-address loads; no corresponding T*R copy is needed on device.
        ordered=full[meta['inverse']]
        y=combine(ordered,routing)
        if y.dtype!=np.float32 or y.shape!=(cap.tokens,interface.n_tile):raise ValueError('ordered FP32 combine output required')
        output_tiles.append(y)
        del p,partials,full,ordered
    counters.update(retained_gate_source_bytes=sum(g.nbytes for g in gates),
                    retained_coalesced_gate_bytes=sum(g.nbytes for g in down_groups),
                    declared_device_status='CPU_DATAFLOW_ONLY_NOT_A_PRECISION_CERTIFICATE')
    return np.concatenate(output_tiles,axis=1),counters
