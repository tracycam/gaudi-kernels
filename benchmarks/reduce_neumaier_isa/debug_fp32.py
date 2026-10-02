"""Instrument only the final stores of an ISA clone to expose pre-round FP32.

No reduction instruction is changed. Debug GUIDs are never production choices.
"""
import argparse,re,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
if a.source.suffix=='.dis':
 lines=['.text','.globl main'];body=False
 for line in a.source.read_text().splitlines():
  label=re.fullmatch(r'[0-9a-f]+ (main|\.LBB\w+):',line.strip())
  if label:lines.append(label[1]+':');body=True;continue
  if body:
   hit=re.match(r'\s*[0-9a-f]+:\s+(.*)',line)
   if hit:lines.append(hit[1].split('//',1)[0].strip())
   elif line.startswith('Disassembly of section') or line.startswith('.section') or line.startswith('TPC compiler'):break
else:lines=a.source.read_text().splitlines()
converts=[i for i,l in enumerate(lines) if 'convert.f32 all_lanes target_type=bf16 rhne V0, D0' in l]
assert len(converts)==1
index=converts[0];slots=lines[index].split(';');assert slots[2].strip().startswith('convert.f32');slots[2]='nop';lines[index]=';'.join(slots)
stores=[i for i,l in enumerate(lines) if 'st_tnsr pack 0x4' in l];assert len(stores)==1
index=stores[0];reg=re.search(r'st_tnsr pack 0x4, (I\d+), V0',lines[index])[1]
lines[index:index+1]=[f'nop; nop; nop; st_tnsr 0x4, {reg}, V0',f'nop; add.i32 b00001 {reg}, 0x40, {reg}; nop; nop',f'nop; nop; nop; st_tnsr 0x4, {reg}, V1']
a.out.write_text('\n'.join(lines)+'\n')
a.out.with_suffix('.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),pre_round_output=True,reduction_instruction_changes=0),indent=2)+'\n')
