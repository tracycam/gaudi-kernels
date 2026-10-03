"""Static capacities for device-selected expert-M buckets; no route readback."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertBucket:
    lower: int
    upper: int
    rows: int
    slots: int


@dataclass(frozen=True)
class ExpertPlan:
    min_mme_rows: int = 64
    down_n_tile: int = 2048
    workspace_budget: int = 1024 * 1024**2

    def buckets(self, tokens, routes, experts):
        if any(type(v) is not int for v in
               (tokens,routes,experts,self.min_mme_rows,self.down_n_tile,self.workspace_budget)):
            raise ValueError('integer expert-plan geometry required')
        if not (0 <= tokens <= 513 and 1 <= routes <= min(8,experts)
                and 1 <= experts <= 384 and self.min_mme_rows >= 1
                and self.down_n_tile in (512,1024,2048) and self.workspace_budget > 0):
            raise ValueError('unqualified expert-plan geometry')
        result = []
        lower = self.min_mme_rows
        cap = 1 << (lower-1).bit_length()
        while lower <= tokens:
            upper = min(cap,tokens)
            result.append(ExpertBucket(lower,upper,upper,min(experts,tokens*routes//lower)))
            lower,cap = upper+1,cap*2
        return tuple(result)

    def costs(self, tokens, routes, experts):
        buckets = self.buckets(tokens,routes,experts)
        padded = sum(b.slots*b.rows for b in buckets)
        # Conservative sum of all padded gates, down tile sources+concat and
        # per-bucket output. This is a planning bound, not allocator telemetry.
        temporary = sum(2*b.slots*b.rows*256*2 +
                        2*b.slots*b.rows*self.down_n_tile*4 +
                        2*min(2,b.slots)*b.rows*(6144*2+512*4) +
                        tokens*6144*4 for b in buckets) + tokens*6144*8
        return dict(buckets=buckets, padded_mme_rows=padded,
                    slots=sum(b.slots for b in buckets),
                    temporary_bound_bytes=temporary,
                    fits_budget=temporary<=self.workspace_budget)


def reference_partition(ids, experts, bucket):
    """CPU test oracle only. Runtime decisions must stay in the device kernels."""
    t=len(ids)
    if not t or not ids[0] or any(len(row)!=len(ids[0]) for row in ids):
        raise ValueError('nonempty rectangular IDs required')
    r=len(ids[0]); counts=[0]*experts; flags=[0]*t; occurrences=[[] for _ in counts]
    for token,row in enumerate(ids):
        for slot,e in enumerate(row):
            if type(e) is not int:raise ValueError('integer IDs required')
            if not 0<=e<experts:flags[token]|=1
            else:counts[e]+=1;occurrences[e].append(token*r+slot)
            if e in row[:slot]:flags[token]|=2
    bad=0
    for f in flags:bad|=f
    if sum(counts)!=t*r or any(n>t for n in counts):bad|=8
    selected=[e for e,n in enumerate(counts) if bucket.lower<=n<=bucket.upper]
    if len(selected)>bucket.slots:bad|=4
    prefix=[0]*(experts+1); inverse=[-1]*(t*r)
    mapping=[-1]*(bucket.slots*bucket.rows); chosen=[-1]*bucket.slots;valid=[0]*bucket.slots
    cursor=0
    for e in range(experts):
        prefix[e]=cursor
        if not bad and e in selected:
            chosen[cursor]=e;valid[cursor]=counts[e]
            for j,q in enumerate(occurrences[e]):
                target=cursor*bucket.rows+j
                inverse[q]=target;mapping[target]=q
            cursor+=1
    prefix[experts]=cursor
    return dict(counts=counts,row_status=flags,prefix=prefix,tile_expert=chosen,
                tile_base=[0]*bucket.slots,valid_rows=valid,status=[bad],inverse=inverse,row_map=mapping)
