"""CPU specification for a future fixed-capacity device route-tile graph.

No Torch, device, model integration, host readback or runtime dispatch. Device
count/prefix/stable gather kernels and compiled lifetimes are still required.
"""
from dataclasses import dataclass
from math import ceil

GP_N, GP_K, DOWN_N, DOWN_K = 512, 6144, 6144, 256
PER_EXPERT_BYTES = (GP_N*GP_K + DOWN_N*DOWN_K)*17//32


@dataclass(frozen=True)
class Capacity:
    tokens: int
    routes: int
    experts: int = 384
    requested_rows: int = 16

    def __post_init__(self):
        if any(type(x) is not int for x in (self.tokens,self.routes,self.experts,self.requested_rows)):
            raise ValueError('integer shape metadata required')
        if not (1 <= self.tokens <= 65536 and 1 <= self.routes <= self.experts and 1 <= self.requested_rows <= 1024):
            raise ValueError('unsupported static route geometry')

    @property
    def rows(self):return min(self.tokens,self.requested_rows)

    @property
    def max_tiles(self):
        # For a active experts and L routes: each first tile costs >=1 route,
        # each further tile >=C routes. Also each expert has <=T routes because
        # top-k IDs are unique per token. The largest bound occurs at a=A.
        length=self.tokens*self.routes;active=min(self.experts,length)
        return min(active*ceil(self.tokens/self.rows),active+(length-active)//self.rows)

    def costs(self, active_tiles=None, active_experts=None, *, n_tile=512):
        if type(n_tile) is not int or n_tile<512 or n_tile%512 or DOWN_N%n_tile or 16*n_tile*DOWN_K*2>16*1024**2:
            raise ValueError("aligned N tile and <=16MiB decoded down fragment required")
        c,b=self.rows,self.max_tiles;rows=c*b;gp_frag=(b+1)//2;down_frag=(b+15)//16;down_tiles=DOWN_N//n_tile
        gate=rows*DOWN_K*2;partial=rows*n_tile*4
        local_gp_rows=c*min(b,2);local_down_rows=c*min(b,16)
        desired=(2*gate+2*partial+local_gp_rows*(GP_K*2+GP_N*4)
                 +local_down_rows*DOWN_K*2+2*self.tokens*DOWN_N*4)
        total=self.tokens*self.routes
        metadata=16*self.experts+12*b+4*rows+4*total+8
        all_functional=(rows*GP_K*2+rows*GP_N*4+(1+down_tiles)*gate
                        +2*rows*DOWN_N*4+2*self.tokens*DOWN_N*4+metadata)
        return dict(tokens=self.tokens,routes=self.routes,experts=self.experts,tile_rows=c,
            max_tiles=b,down_n_tile=n_tile,static_row_slots=rows,valid_routes=total,logical_rows_per_route=rows/total,
            actual_tiles=active_tiles,empty_tiles=None if active_tiles is None else b-active_tiles,
            empty_tile_fraction=None if active_tiles is None else (b-active_tiles)/b,
            actual_tile_padding_rows=None if active_tiles is None else active_tiles*c-total,
            gp_mme_nodes=gp_frag,down_mme_nodes=down_tiles*down_frag,mme_nodes=gp_frag+down_tiles*down_frag,
            mme_node_count_kind='static construction estimate; not compiled physical nodes or host launches',
            # Assume the existing public functional split/concat design, not a
            # shared partial-scatter cache with unproved aliasing.
            retained_gate_bytes=gate,padded_down_tile_bytes=partial,concat_sources_plus_destination_bytes=2*(gate+partial),
            desired_serial_fragment_activation_bound_bytes=desired,
            metadata_payload_bytes=metadata,
            all_functional_intermediates_payload_bytes=all_functional,
            all_functional_with_hoisted_gate_concat_bytes=all_functional-(down_tiles-1)*gate,
            all_declared_decoded_payload_bytes=b*(GP_N*GP_K+DOWN_N*DOWN_K)*2,
            payload_bound_scope='serial bound assumes verified lifetimes; all-functional bound sums distinct declared payloads, excluding original inputs, allocator alignment and compiler workspace',
            gp_decoded_fragment_bytes=min(2,b)*GP_N*GP_K*2,
            down_decoded_fragment_bytes=min(16,b)*n_tile*DOWN_K*2,
            all_decoded_store_bytes_including_empty=b*(GP_N*GP_K+DOWN_N*DOWN_K)*2,
            active_decoded_store_bytes=None if active_tiles is None else active_tiles*(GP_N*GP_K+DOWN_N*DOWN_K)*2,
            empty_zero_store_bytes=None if active_tiles is None else (b-active_tiles)*(GP_N*GP_K+DOWN_N*DOWN_K)*2,
            requested_original_bytes=None if active_tiles is None else active_tiles*PER_EXPERT_BYTES,
            unique_original_bytes=None if active_experts is None else active_experts*PER_EXPERT_BYTES,
            per_route_tpc_requested_original_bytes=total*PER_EXPERT_BYTES,
            resident_original_bytes=self.experts*PER_EXPERT_BYTES,
            count_scan_ID_reads_existing_algorithm=self.experts*total,
            # This is a scenario, NOT measured geometry for the new graph.
            illustrative_128_minrow_slots=b*((c+127)//128)*128,
            graph_lifetime_proved=False,physical_HBM_measured=False,device_implemented=False)


@dataclass
class Assignment:
    capacity: Capacity
    counts: list
    tile_expert: list
    tile_ordinal_base: list
    tile_valid_rows: list
    row_to_route: list
    inverse: list


def assign(ids, capacity):
    t,r,e=capacity.tokens,capacity.routes,capacity.experts
    if len(ids)!=t or any(len(row)!=r for row in ids):raise ValueError('ID shape mismatch')
    if any(any(type(x) is not int or not 0<=x<e for x in row) or len(set(row))!=r for row in ids):
        raise ValueError('valid, unique ordered IDs within each token required')
    queues=[[] for _ in range(e)]
    for token,row in enumerate(ids):
        for slot,expert in enumerate(row):queues[expert].append(token*r+slot)
    c,b=capacity.rows,capacity.max_tiles
    tile_expert=[-1]*b;tile_base=[0]*b;valid=[0]*b;row_to_route=[-1]*(b*c);inverse=[-1]*(t*r);tile=0
    for expert,queue in enumerate(queues):
        for start in range(0,len(queue),c):
            assert tile<b,'proved static capacity exceeded'
            group=queue[start:start+c];tile_expert[tile]=expert;tile_base[tile]=start;valid[tile]=len(group)
            for row,route in enumerate(group):row_to_route[tile*c+row]=route;inverse[route]=tile*c+row
            tile+=1
    result=Assignment(capacity,[len(q) for q in queues],tile_expert,tile_base,valid,row_to_route,inverse)
    validate(ids,result)
    return result


def validate(ids, plan):
    cap=plan.capacity;c,b=cap.rows,cap.max_tiles;flat=[x for row in ids for x in row];length=cap.tokens*cap.routes
    assert len(plan.counts)==cap.experts and sum(plan.counts)==length
    assert len(plan.tile_expert)==len(plan.tile_ordinal_base)==len(plan.tile_valid_rows)==b
    assert len(plan.row_to_route)==b*c and len(plan.inverse)==length
    seen=[];by_expert=[[] for _ in range(cap.experts)]
    for tile,expert in enumerate(plan.tile_expert):
        valid=plan.tile_valid_rows[tile];routes=plan.row_to_route[tile*c:(tile+1)*c]
        if expert==-1:assert valid==0 and routes==[-1]*c;continue
        assert 0<=expert<cap.experts and 1<=valid<=c
        assert routes[valid:]==[-1]*(c-valid)
        assert plan.tile_ordinal_base[tile]==len(by_expert[expert])
        for row,route in enumerate(routes[:valid]):
            assert 0<=route<length and flat[route]==expert
            assert plan.inverse[route]==tile*c+row
            seen.append(route);by_expert[expert].append(route)
    assert sorted(seen)==list(range(length)),'route dropped or duplicated'
    for expert,routes in enumerate(by_expert):
        assert routes==[route for route,value in enumerate(flat) if value==expert]
        assert len(routes)==plan.counts[expert]
    # Consumers iterate original token/slot order, never tile/expert order.
    for token in range(cap.tokens):
        restored=[plan.row_to_route[plan.inverse[token*cap.routes+slot]] for slot in range(cap.routes)]
        assert restored==list(range(token*cap.routes,(token+1)*cap.routes))


def fixture(tokens,routes,experts,kind):
    if kind=='uniform':return [[(token*routes+slot)%experts for slot in range(routes)] for token in range(tokens)]
    if kind=='hot':return [[experts-1-slot for slot in range(routes)] for _ in range(tokens)]
    if kind=='skew':
        cut=(3*tokens)//4
        return [[experts-1-slot for slot in range(routes)] if token<cut else
                [((token-cut)*routes+slot)%experts for slot in range(routes)] for token in range(tokens)]
    raise ValueError('unknown fixture')
