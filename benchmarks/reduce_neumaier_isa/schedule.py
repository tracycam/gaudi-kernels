"""Reschedule only product-error register transfers, with conservative RAW gaps.

The two next-group FMA results remain in V20/V21 across the loop edge. Copy them
into V6/V9 during free LSU slots at bundles12/13 of the next iteration, rather
than waiting at the previous tail. All FP32 arithmetic instructions keep order.
"""
import argparse,json,re,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('disassembly',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
lines=[];body=False;inside=False
for line in a.disassembly.read_text().splitlines():
 label=re.fullmatch(r'([0-9a-f]+) (main|\.LBB\w+):',line.strip())
 if label:
  lines.append(label[2]+':');body=True;continue
 if body:
  match=re.match(r'\s*[0-9a-f]+:\s+(.*)',line)
  if match:lines.append(match[1].split('//',1)[0].strip())
  elif line.startswith('Disassembly of section') or line.startswith('TPC compiler'):break
start=lines.index('.LBB0_4:');end=lines.index('.LBB0_5:');head=lines[:start];inner=lines[start+1:end];tail=lines[end:]
def replace_once(sequence,old,new):
 hits=[i for i,s in enumerate(sequence) if old in s];assert len(hits)==1,(old,hits)
 i=hits[0];sequence[i]=sequence[i].replace(old,new)
for old,new in [('xor.f32  V6, V8','xor.f32  V20, V8'),('xor.f32  V9, V7','xor.f32  V21, V7'),('mac.f32  V6, V10','mac.f32  V20, V10'),('mac.f32  V9, V12','mac.f32  V21, V12')]:replace_once(head,old,new)
for old,new in [('xor.f32  V11, V15','xor.f32  V20, V15'),('xor.f32  V14, V16','xor.f32  V21, V16'),('mac.f32  V11, V12','mac.f32  V20, V12'),('mac.f32  V14, V13','mac.f32  V21, V13')]:replace_once(inner,old,new)
assert len(inner)==38,len(inner)
for index,expected,new in [(12,'add.f32  V18, V7, V18','mov V6, V20'),(13,'mul.f32  V10, V10, V11','mov V9, V21')]:
 slots=[s.strip() for s in inner[index].split(';')];assert slots[0]=='nop' and expected in slots[2],(index,slots);slots[0]=new;inner[index]='; '.join(slots)
assert all(all(s.strip()=='nop' for s in line.split(';')) for line in inner[32:36])
assert 'mov  V6, V11' in inner[36] and 'mov  V9, V14' in inner[37]
inner=inner[:32]
# The removed tail also separated the last correction update from the epilogue.
# Restore its RAW gap only after the loop, not once per group.
tail=tail[:1]+['nop; nop; nop; nop']*3+tail[1:]
a.out.write_text('.text\n.globl main\n'+'\n'.join(head+['.LBB0_4:']+inner+tail)+'\n')
report=dict(input_sha256=hashlib.sha256(a.disassembly.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(a.out.read_bytes()).hexdigest(),old_loop_bundles=38,new_loop_bundles=32,extra_registers=['V20','V21'],removed_tail_full_nops=4,moved_copy_instructions=2,old_FMA_to_copy_min_bundles=6,new_loop_carried_FMA_to_copy_min_bundles=14,exit_only_nops=3,final_correction_to_epilogue_min_bundles=6,arithmetic_order_changed=False)
a.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
