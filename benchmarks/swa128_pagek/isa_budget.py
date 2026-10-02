"""Static scheduled packet budget; never treats issue packets as device latency."""
import argparse,collections,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def parse(path):
 blocks=collections.defaultdict(list);label=0;bundle=None
 for raw in path.read_text().splitlines():
  match=re.match(r'(?:\.LBB0_|// %bb\.)(\d+)',raw.strip())
  if match:label=int(match[1]);continue
  s=raw.split('//')[0].strip()
  if not s or s.startswith('.')or s.startswith('$'):continue
  if s=='{':bundle=[];continue
  if s=='}':blocks[label].append(bundle);bundle=None;continue
  if bundle is not None:bundle.append(s)
  else:blocks[label].append([s])
 return blocks
def budget(blocks,ids):
 packets=[p for i in ids for p in blocks[i]];ops=collections.Counter(s.split()[0]for p in packets for s in p)
 return {'packets':len(packets),'nop_packets':sum(all(s.lower()=='nop'for s in p)for p in packets),'opcodes':dict(ops)}
old=parse(a.old);new=parse(a.new);groups={0:[9,10,11,12,16],1:[9,10,14,16],2:[9,13,16],3:[9,10,11,15,16]}
r={'old_assembly_sha256':hashlib.sha256(a.old.read_bytes()).hexdigest(),'new_assembly_sha256':hashlib.sha256(a.new.read_bytes()).hexdigest(),'old_first_page_token':budget(old,[6,7]),'old_second_page_token':budget(old,[9,10]),'candidate_by_q_group':{g:budget(new,ids)for g,ids in groups.items()},'candidate_group_iterations':{0:64,1:64,2:32,3:32},'scope':'static scheduled assembly packets including compiler NOPs, excluding loop setup/query load/softmax/AV; no memory stalls or device time inferred'}
r['old_aligned128_packets_per_head']=128*r['old_first_page_token']['packets'];r['candidate_one_page_packets_per_head']=sum(r['candidate_by_q_group'][g]['packets']*n for g,n in r['candidate_group_iterations'].items());r['candidate_two_partial_pages_packets_per_head']=2*r['candidate_one_page_packets_per_head']
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
