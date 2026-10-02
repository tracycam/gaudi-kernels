"""Two causal diagnostics at the original 274 packets; not speed candidates."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False);source=(a.original/'gp.s').read_text();prefix,body=source.split('.LBB0_3:\n');body,suffix=body.split('.LBB0_6:');original=[[s.strip() for s in l.split(';')] for l in body.splitlines()];assert len(original)==274
for name in ('late_rename','first_prefetch','first_prefetch_safe214','first_prefetch_spacing208'):
 rows=[r.copy() for r in original]
 if name=='late_rename':
  rows[227][0]=rows[227][0].replace('V34','V38')
  rows[234][0]=rows[234][0].replace('V36','V34')
  rows[234][2]='convert.u8 all_lanes target_type=uint16 rhne D16, V38'
  rows[241][2]='convert.u8 all_lanes target_type=uint16 rhne D22, V34'
  for packet,dst in ((242,16),(243,17),(249,22),(250,23)):rows[packet][2]=f'shl.i16 V{dst}, V{dst}, 0x7'
  for packet,dst,src in ((251,28,16),(252,30,17),(258,32,22),(259,34,23)):rows[packet][2]=f'convert.bf16 all_lanes target_type=fp32 rhne D{dst}, V{src}'
 else:
  position=214 if name=='first_prefetch_safe214' else 208 if name=='first_prefetch_spacing208' else 199
  assert rows[position][0]=='nop';rows[position][0]=rows[227][0].replace('V34','V38');rows[227][0]='nop'
  # Crucial control: preserve I7 update at original227, not the failed199.
  assert rows[227][1]=='add.i32 b00001 I7, 0x100, I7'
  rows[234][2]=rows[234][2].replace(', V34',', V38')
 assert [r[2] for r in rows if r[2].startswith('mac.')]==[r[2] for r in original if r[2].startswith('mac.')]
 d=a.output/name;d.mkdir();(d/'candidate.s').write_text(prefix+'.LBB0_3:\n'+'\n'.join('; '.join(r) for r in rows)+'\n.LBB0_6:'+suffix);(d/'candidate.o').write_bytes((a.original/'gp.o').read_bytes())
 cmds=[['tpc-clang','-mcpu=gaudi2','-c','candidate.s','-o','assembled.o'],['objcopy','--dump-section','.text=candidate.text','assembled.o'],['objcopy','--update-section','.text=candidate.text','candidate.o'],['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','candidate.o','candidate_x86.o']]
 root=Path(__file__).resolve().parents[2];cmds.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/lib/habanatools/include',str(root/'csrc/host/moe_gp_scale_tail_glue.cpp'),'candidate_x86.o','-o','libgaudi_gp_scale_tail_tpc.so'])
 with (d/'build.log').open('w') as log:
  for cmd in cmds:subprocess.run(cmd,cwd=d,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
 (d/'diagnostic.json').write_text(json.dumps(dict(variant=name,packets=274,mac_order_unchanged=True,device_qualified=False,performance_candidate=False,commands=cmds),indent=2)+'\n')
