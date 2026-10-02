"""Static four-slot and architectural vector-register audit (not a cycle model)."""
import argparse, collections, hashlib, json, re
from pathlib import Path

def parse(path):
 blocks=collections.OrderedDict();current=None
 for line in path.read_text().splitlines():
  label=re.fullmatch(r'[0-9a-f]+ (main|\.LBB\w+):',line.strip())
  if label:current=label[1];blocks[current]=[];continue
  if current:
   m=re.match(r'\s*[0-9a-f]+:\s+(.*)',line)
   if m:
    body=m[1].split('//',1)[0].strip();slots=[x.strip() for x in body.split(';')]
    blocks[current].append(slots)
   elif line.startswith('.section') or line.startswith('TPC compiler'):break
 return blocks

def regs(text):
 result=set(re.findall(r'\bV\d+\b',text))
 for index in re.findall(r'\bD(\d+)\b',text):result|={f'V{int(index)}',f'V{int(index)+1}'}
 return result

def use_def(slots):
 use=set();define=set()
 for ins in slots:
  rr=regs(ins)
  if not rr:continue
  # These audited programs have one vector destination, except conversion D input.
  if ins.startswith(('st_', 'event')):use|=rr;continue
  m=re.search(r'\bV\d+\b',ins);dst=m[0];define.add(dst)
  use|=regs(ins[m.end():])
  if ins.startswith('mac.') or re.search(r'\bSP\d+\b',ins):use.add(dst)
 return use,define

def backwards(packets,end):
 live=set(end);trace=[sorted(live)]
 for slots in reversed(packets):
  use,define=use_def(slots);live=(live-define)|use;trace.append(sorted(live))
 return live,list(reversed(trace))

def audit(path):
 blocks=parse(path);all_packets=[p for b in blocks.values() for p in b];loop=blocks['.LBB0_4']
 suffix=[];active=False
 for label,packets in blocks.items():
  if label=='.LBB0_5':active=True
  if active:suffix+=packets
 exit_live,_=backwards(suffix,set());live=set(exit_live)
 for _ in range(64):
  begin,trace=backwards(loop,live|exit_live)
  if begin<=live:break
  live|=begin
 else:raise RuntimeError('liveness did not converge')
 _,trace=backwards(loop,live|exit_live)
 counts=collections.Counter(ins.split()[0] for p in loop for ins in p if ins!='nop')
 all_regs=regs(' '.join(' '.join(p) for p in all_packets))
 return dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
  inner_loop_packets=len(loop),inner_loop_full_nops=sum(all(x=='nop' for x in p) for p in loop),
  occupied_slots=dict(zip(['LSU','SPU','VPU','ST'],[sum(len(p)==4 and p[i]!='nop' for p in loop) for i in range(4)])),
  opcode_counts=dict(sorted(counts.items())),
  vector_registers=sorted(all_regs,key=lambda x:int(x[1:])),vector_register_count=len(all_regs),highest_vector_register=max(map(lambda x:int(x[1:]),all_regs)),
  vector_local_spill_instructions=[ins for p in all_packets for ins in p if ins.startswith(('ld_l_v','st_l_v'))],
  conservative_loop_live_in=sorted(live),conservative_loop_live_peak=max(map(len,trace)),loop_liveness_trace=trace,
  liveness_scope='architectural vector values at packet boundaries; not execution-pipeline occupancy',
  hardware_cycle_estimate=None)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 result={name:audit(a.build/(name+'.dis')) for name in ['neumaier_baseline','neumaier_lookahead','neumaier_handschedule']}
 a.out.write_text(json.dumps(dict(device_tested=False,variants=result),indent=2)+'\n')
 print(json.dumps({k:{x:v[x] for x in ['inner_loop_packets','inner_loop_full_nops','occupied_slots','vector_register_count','conservative_loop_live_peak']} for k,v in result.items()},indent=2))
