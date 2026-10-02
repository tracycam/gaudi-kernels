"""Local audit of all optimization attempts; isolate diagnostic from product results."""
import argparse,collections,hashlib,importlib.util,json,math,re,statistics,sys
from pathlib import Path
import numpy as np
from analyze import correctness, digest, ports
from isa_identity import text_section


def placement(case):
 p=case/'post_graph.json';p=next(p.glob('*.json'))if p.is_dir()else p
 g=json.loads(p.read_text())['graphs'][0];ts={t['name']:t for t in g['tensors']};ns=[n for n in g['nodes']if not n['is_logical']]
 ds=[n for n in ns if n['guid']in ('gk_grouped_mxfp4_decode','gk_mixed_m_fill')];coverage=collections.defaultdict(list);buffers=[]
 for n in ds:
  idx=ts[n['input_tensors'][3]];base=ts[idx.get('alias_of',idx['name'])];w=ts[n['output_tensors'][0]]
  assert w['dtype']=='bf16' and not w['persistent']
  for source in n['input_tensors'][:2]:assert ts[source]['dtype']=='uint8' and ts[source]['allocation']=='DRAM'
  start=(idx['offset']-base['offset'])//4;count=idx['max_shape'][0];assert w['max_shape'][2]==count
  family=n['input_tensors'][0];coverage[family].extend(range(start,start+count))
  buffers.append(dict(node=n['name'],family=family,allocation=w['allocation'],shape=w['max_shape'],offset=w['offset'],bytes=2*math.prod(w['max_shape'])))
 for family,seen in coverage.items():assert sorted(seen)==list(range(len(seen))),family
 for n in ns:
  if n['engine']=='MME':assert ts[n['output_tensors'][0]]['dtype']=='float32'
 spans=ports.merge([(b['offset'],b['offset']+b['bytes'])for b in buffers if b['allocation']=='SRAM']);size=sum(b-a for a,b in spans);assert size<=48*1024*1024
 return dict(graph_sha256=digest(p),all_expanded_weights_sram=all(b['allocation']=='SRAM'for b in buffers),
             logical_expert_coverage={k:len(v)for k,v in coverage.items()},sram_union_bytes=size,buffers=buffers,
             physical_nodes=dict(collections.Counter(n['guid']for n in ns)),
             mme_strategies=sorted(set(n.get('mme_node_strategy','')for n in ns if n['engine']=='MME')))


def chain_check(case,p):
 e,k,h,t=p['experts'],p['K'],p['H'],p['tokens'];counts=np.fromfile(case/'counts.bin',dtype='<i4');rows=np.fromfile(case/'token_rows.bin',dtype='<i4').reshape(e,3)
 routes=[int(np.sum([rows[j,m]==i for j in range(e)for m in range(counts[j])]))for i in range(t)];assert routes==[8]*t
 x=np.fromfile(case/'activation.bin',dtype='<u2').reshape(e,3,k)
 for i in range(e):assert np.all(x[i,counts[i]:]==0x7fc1)
 keys=('gp_packed','gp_scales','down_packed','down_scales','activation','counts','token_rows','expert_ids')
 signature=hashlib.sha256(''.join(digest(case/(s+'.bin'))for s in keys).encode()).hexdigest();out=[]
 for b in range(p['banks']):
  record=dict(bank=b,input_signature=signature,outputs={})
  for name,n,dt in [('gp_result',2*h,'<f4'),('gate_result',h,'<u2'),('down_result',k,'<f4')]:
   path=case/f'{name}_bank{b}.bin';a=np.fromfile(path,dtype=dt).reshape(e,3,n)
   if dt=='<u2':assert np.all((a&0x7f80)!=0x7f80)
   else:assert np.isfinite(a).all()
   for i in range(e):assert np.all(a[i,counts[i]:]==0)
   record['outputs'][name]=digest(path)
  y=case/f'combined_bank{b}.bin';assert y.read_bytes()==(case/f'combine_oracle_bank{b}.bin').read_bytes();record['outputs']['combined']=digest(y);out.append(record)
 return out


p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();records=[];isa=[]
for root in a.roots:
 for folder in ('decoder','increment','pipeline'):
  obj=root/folder/'decode.o';asm=obj.with_suffix('.s')
  if not obj.exists():continue
  source=asm.read_text().split('\t.type\ttpc_compiler')[0];local=[s.strip()for s in source.splitlines()if re.search(r'\b(?:ld|st)_l(?:_v)?\b',s)and'mmio'not in s];assert not local
  isa.append(dict(root=str(root),variant=folder,text_sha256=hashlib.sha256(text_section(obj)).hexdigest(),
                  max_vector_register=max(int(x)+(kind=='D')for kind,x in re.findall(r'%(V|D)(\d+)',source)),
                  whole_function_add=source.count('add.i32'),whole_function_set_indx=source.count('set_indx'),non_mmio_local=local))
 for case in sorted((root/'results').glob('*')):
  status=json.loads((case/'exit.json').read_text());assert status['module']==5 and status['source_identity_verified']
  r=dict(root=str(root),case=case.name,source_commit=status['git_commit'],returncode=status['runner_exit_code']);records.append(r)
  if status['runner_exit_code']:r['accepted']=False;continue
  rows=[json.loads(s)for s in (case/'run.log').read_text().splitlines()if s.startswith('{"stage":')];plan=next(x for x in rows if x['stage']=='plan');r['plan']=plan
  if 'mme_nodes'in plan:
   r.update(ports.audit(case));r['diagnostic']=True;r['accepted']=True;continue
  r['placement']=placement(case);r['compile_only']=any(x['stage']=='compile_only'for x in rows)
  if not r['placement']['all_expanded_weights_sram']:
   assert r['compile_only'];r['accepted']=False;r['reason']='expanded DRAM';continue
  if r['compile_only']:continue
  checks=[x for x in rows if x['stage']=='correctness'];assert checks
  assert all(all(v==0 for k,v in check.items()if k=='bad'or k.endswith('_bad'))for check in checks)
  r['correctness']=checks;r['accepted']=True;r['samples_us']=[x['event_us']for x in rows if x['stage']=='timing'];r['event_median_us']=statistics.median(r['samples_us'])
  if 'policy'in plan:
   runtime=next(x for x in rows if x['stage']=='runtime_inputs');r['banks']=correctness(case,plan,runtime)
   r['diagnostic']=next(x for x in rows if x['stage']=='diagnostic')['synthetic_producer'];r['upstream']=next(x for x in rows if x['stage']=='upstream')['mode']
   r['timing_repeats']=next((x['repeats']for x in rows if x['stage']=='timing_config'),20)
   if r['upstream']!='none':
    token=np.fromfile(case/'tokens.bin',dtype='<u2').reshape(runtime['activation_banks'],16,plan['K']);mapping=np.fromfile(case/'token_rows.bin',dtype='<i4').reshape(plan['experts'],3)
    raw=(case/'activation.bin').read_bytes()
    for b,check in enumerate(r['banks']):
     ab=b if runtime['changing_counts']else 0;x=np.frombuffer(raw[ab*runtime['activation_stride']:(ab+1)*runtime['activation_stride']],dtype='<u2').reshape(plan['experts'],3,plan['K'])
     for e,count in enumerate(check['counts']):assert np.array_equal(x[e,:count],token[ab,mapping[e,:count]])
  else:
   r['chain_banks']=chain_check(case,plan);r['diagnostic']=False
   r['upstream']=next((x['mode']for x in rows if x['stage']=='upstream'),'none')
   r['measured_scope']=('token gather + 'if r['upstream']!='none'else'already-routed ')+ 'gate/up + SiLU + down + weighted combine; routing selection excluded'
  traces=list((case/'trace').glob('*_7.json'))
  if traces:
   tr=ports.trace(traces[0],None);r['profile']=tr
   if 'policy'not in plan:
    gp=[n for n in tr['nodes']if n['engine']=='MME'and n['name'].startswith('gp_')];down=[n for n in tr['nodes']if n['engine']=='TPC'and n['name'].startswith('down_')and'decode'in n['name']]
    assert gp and down;last=max(n['end_us']for n in gp);tr['last_gp_mme_end_us']=last;tr['first_down_decode_start_us']=min(n['start_us']for n in down);tr['down_decode_finished_before_last_gp_mme']=sum(n['end_us']<=last for n in down)
# Exact comparisons for identical original inputs, separately for synthetic producers.
seen={};chains={}
for r in records:
 if not r.get('accepted'):continue
 for b in r.get('banks',[]):
  key=(r['diagnostic'],b['input_signature']);value=b['canonical_output_sha256'];assert key not in seen or seen[key]==value;seen[key]=value
 for b in r.get('chain_banks',[]):
  key=(b['input_signature'],b['bank']);assert key not in chains or chains[key]==b['outputs'];chains[key]=b['outputs']
abba=[]
for root in a.roots:
 groups=collections.defaultdict(list)
 for r in records:
  m=re.fullmatch(r'abba-(.+)-([0-3])',r['case'])
  if m and r['root']==str(root):groups[m[1]].append((int(m[2]),r))
 for name,arms in groups.items():
  arms=[r for i,r in sorted(arms)];assert len(arms)==4 and all(r.get('accepted')and'profile'not in r for r in arms)
  before=statistics.mean(arms[i]['event_median_us']for i in (0,3));after=statistics.mean(arms[i]['event_median_us']for i in (1,2))
  abba.append(dict(root=str(root),comparison=name,arms=[r['case']for r in arms],baseline_us=before,candidate_us=after,saved_us=before-after,latency_reduction=1-after/before,diagnostic=any(r['diagnostic']for r in arms)))
endpoint=[]
for root in a.roots:
 arms=sorted((r for r in records if r['root']==str(root)and r['case'].startswith('mme-slope-')),key=lambda r:r['case'])
 if arms:
  assert len(arms)==4;short=statistics.mean(arms[i]['event_median_us']for i in (0,3));long=statistics.mean(arms[i]['event_median_us']for i in (1,2));delta=arms[1]['plan']['mme_read_bytes']-arms[0]['plan']['mme_read_bytes'];endpoint.append(dict(root=str(root),short_us=short,long_us=long,increment_bytes=delta,incremental_TBps=delta/(long-short)/1e6,scope='Same N512/K6144/M3 batch2; repeated SRAM-resident operand, setup cost canceled by node-count delta. Not original-weight performance.'))
out=dict(scope='All attempted optimizations; projection and routed synthetic MoE chain, not full model TPS',ABBA=abba,same_shape_mme=endpoint,isa=isa,records=records,
         exact_output_comparisons_pass=True,physical_hbm_bus_bytes_measured=False)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(ABBA=abba,same_shape_mme=endpoint),indent=2))
