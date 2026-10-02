"""Original production N512 expert byte intervals; no weight allocation."""
from collections import Counter

E, R, H, I = 384, 8, 6144, 256
SHAPES = {'gp': (E*6144,256), 'gs': (E*192,512),
          'dp': (E*3072,256), 'ds': (E*96,512)}


def routes(tokens, state):
    if tokens not in (2,8):raise ValueError('bounded T2/T8 fixture only')
    if state == 'hot':return [list(range(376,384)) for _ in range(tokens)]
    start = {'sparse':0,'rotated':320,'zero':128}[state]
    return [list(range(start+row*R,start+(row+1)*R)) for row in range(tokens)]


def accounting(ids):
    if not ids or any(len(row)!=R or len(set(row))!=R or any(type(e)is not int or not 0<=e<E for e in row) for row in ids):
        raise ValueError('fixture routes must be valid, unique top8 per token')
    intervals=[]
    for token,row in enumerate(ids):
        for slot,e in enumerate(row):
            for split in range(3):
                for owner,begin,length in [('gp',(e*6144+split*2048)*256,2048*256),
                                           ('gs',(e*192+split*64)*512,64*512)]:
                    intervals.append(dict(owner=owner,token=token,slot=slot,expert=e,tile=split,begin=begin,end=begin+length))
            for tile in range(12):
                for owner,begin,length in [('dp',(e*3072+tile*256)*256,256*256),
                                           ('ds',(e*96+tile*8)*512,8*512)]:
                    intervals.append(dict(owner=owner,token=token,slot=slot,expert=e,tile=tile,begin=begin,end=begin+length))
    union=0
    for owner,shape in SHAPES.items():
        selected=sorted((x['begin'],x['end']) for x in intervals if x['owner']==owner)
        end=-1
        for lo,hi in selected:
            assert 0<=lo<hi<=shape[0]*shape[1]
            union+=max(0,hi-max(lo,end));end=max(end,hi)
    total=sum(x['end']-x['begin'] for x in intervals)
    tokens=len(ids);slots=tokens*R;counts=Counter(e for row in ids for e in row)
    per_expert=2_506_752
    assert total==slots*per_expert and union==len(counts)*per_expert
    return {'tokens':tokens,'topk':R,'experts':E,'active_experts':len(counts),
            'routes_per_expert':dict(sorted(counts.items())), 'resident_weight_scale_bytes':sum(a*b for a,b in SHAPES.values()),
            'requested_weight_scale_bytes_each_mode':total,'unique_selected_weight_scale_bytes':union,
            'gp_tasks_each_mode':slots*3,'down_tasks_each_mode':slots*12,
            'broadcast_gp_activation_buffer_bytes':slots*3*2048*2,
            'broadcast_down_activation_buffer_bytes':slots*12*256*2,
            'compact_source_activation_bytes':tokens*6144*2,'compact_gate_bytes':slots*256*2,
            'gp_fp32_partial_bytes_each_mode':slots*1536*4,
            'down_fp32_partial_bytes_each_mode':slots*6144*4,'intervals':intervals,
            'scope':'ISA-requested logical byte intervals and functional tensor sizes; duplicate expert reads counted per route; not physical HBM/cache transactions'}
