"""Recheck full saved paired outputs; report logical work and actual graph size."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
case=a.archive/'results'/a.case;manifest=json.loads((a.archive/'remote-sha256.json').read_text());d=json.loads((case/'result.json').read_text());ex=json.loads((case/'exit.json').read_text())
assert ex['runner_exit_code']==0 and ex['postflight']['returncode']==0 and d['status']=='PASS_SAME_INPUT_BROADCAST_VS_ROUTED_MME'
fixture=a.archive/'fixtures/t513-e384';ts=sorted({r['T']for r in d['checks']});numeric=[];words=0
for t in ts:
 for state in ('uniform','hot','skew','zero','restored'):
  path=fixture/('uniform'if state=='restored'else state)
  ref=np.load(path/'expected.npy')[:t];ab=np.load(path/'absolute.npy')[:t];u=2**-24;bound=(264*u/(1-264*u))*ab+4*np.abs(np.spacing(ref)).astype(np.float64)
  pair={}
  for mode in ('broadcast','routed_mme'):
   f=case/f't{t}-{state}-{mode}.pt';m=manifest['files'][str(f.relative_to(a.archive))];assert hashlib.sha256(f.read_bytes()).hexdigest()==m['sha256']
   # Only our generated, sealed protocol4 outputs; local weights_only lacks149.
   x=torch.load(f,weights_only=False);y=x['y'];assert y.dtype==torch.float32 and tuple(y.shape)==(t,6144)
   error=np.abs(y.numpy().astype(np.float64)-ref)
   assert np.isfinite(y.numpy()).all()and np.all(error<=bound)
   assert torch.equal(x['consumer'].view(torch.uint8),(y+.03125).view(torch.uint8))
   words+=2*y.numel();pair[mode]=y
  numeric.append(dict(T=t,state=state,cross_mode_word_differences=int((pair['broadcast'].view(torch.int32)!=pair['routed_mme'].view(torch.int32)).sum()),both_pass_FP32_bound=True))
rows=[]
for t in ts:
 for state in ('uniform','hot','skew'):
  counts=collections.Counter(np.load(fixture/state/'ids.npy')[:t].flatten().tolist())
  c=min(t,32);active=sum(math.ceil(n/c)for n in counts.values());capacity=min(t*8,384)+(t*8-min(t*8,384))//c
  medians={mode:statistics.median(r['event_us']for r in d['timing']if r['T']==t and r['state']==state and r['mode']==mode)for mode in ('broadcast','routed_mme')}
  rows.append(dict(T=t,state=state,actual_route_counts_histogram=dict(collections.Counter(counts.values())),row_capacity=c,tile_capacity=capacity,active_tiles=active,empty_tiles=capacity-active,padded_rows_over_real_rows=capacity*c/(t*8),event_us=medians,
    logical_MAC_flops=2*t*8*(6144*512+256*6144),broadcast_logical_route_weight_requests_bytes=t*8*(6144*512+256*6144)*17//32,
    routed_logical_tile_weight_requests_bytes=active*(6144*512+256*6144)*17//32,
    note='Logical requests are not physical HBM bytes; hot subsets may reside in cache. Padded rows are not MME physical tensor-core operation counts.'))
q=case/'post_graph.json';graphs=[]
for f in [q]if q.is_file()else q.rglob('*.json'):
 for g in json.loads(f.read_text())['graphs']:
  cnt=collections.Counter(n['guid']for n in g['nodes'])
  if not(cnt['nm_gemv']or cnt['gk_mxfp4_graph_decode_historical']):continue
  graphs.append(dict(name=g['name'],variant='broadcast'if cnt['nm_gemv']else'routed_mme',physical_nodes=sum(not n['is_logical']for n in g['nodes']),guid_counts=dict(cnt),workspace_bytes=g['workspace_size']))
report=dict(status='PASS_NUMERICS_PERFORMANCE_NOT_PROMOTED',source_commit=manifest['source_commit'],raw_output_words_rechecked=words,rows=rows,numerics=numeric,graphs=graphs,
    samples=d['timing'],route_scope='Prefix rows of frozen T513 states: skew is entirely hot at T<=384; full T512/513 are skewed. Counts above are authoritative.',
    scope='Complete TP-local same-input public graphs. Original uint8 owners shared, BF16/FP32 contract retained. Not model TPS, pure engine cycles or physical HBM utilization.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items()if k not in ('samples','graphs','numerics')},indent=2))
