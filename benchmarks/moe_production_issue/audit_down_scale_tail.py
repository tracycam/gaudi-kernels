"""Offline current N512 down identity and scale-tail schedule proposal; no device."""
import argparse,hashlib,io,json,re,subprocess,sys
from collections import Counter
from pathlib import Path
from elftools.elf.elffile import ELFFile
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from moe_activation_fold_core import embedded_elf,fold
from scale_tail import transform
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
lib=a.canonical/'artifacts/builds/production-integration/model-runs/production-route-m512-preflight-u/source/production-runtime/libraries/libgaudi_down_activation_tpc.so'
asm=a.canonical/'artifacts/builds/moe-activation-folded/final-a/device-a/builds/qualified-down/direct_down_broadcast.s'
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(lib.read_bytes())=='9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a'
elf=embedded_elf(lib.read_bytes(),'direct_down');assert sha(elf)=='b930eb299830f036be389bb4a9d7f0597fa9ee24e51c2880e4e308689c6d3b75'
(out/'production.o').write_bytes(elf);(out/'production.s').write_bytes(asm.read_bytes())
command=['tpc-clang','-mcpu=gaudi2','-c','production.s','-o','reassembled.o']
with(out/'compile.log').open('w')as log:subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
def text(b):return ELFFile(io.BytesIO(b)).get_section_by_name('.text').data()
assert text(elf)==text((out/'reassembled.o').read_bytes())
def split(source):
 prefix,body=source.split('.LBB0_3:\n');body,suffix=body.split('.LBB0_6:')
 return prefix,[[v.strip()for v in line.split(';')]for line in body.splitlines()],suffix
def counts(source):
 _,rows,_=split(source)
 return dict(packets=len(rows),slots=[dict(Counter(row[i].split()[0]for row in rows if row[i]!='nop'))for i in range(4)],full_nop_packets=sum(r==['nop']*4 for r in rows),active_vpu=sum(r[2]!='nop'for r in rows),vector_register_max=max(map(int,re.findall(r'\bV(\d+)\b',source))),VLM_spills=re.findall(r'\b(?:ld_l_v|st_l_v)\b',source))
source=asm.read_text();folded,fold_plan=fold(source);prefix,rows,suffix=split(folded);assert len(rows)==269
# The existing qualified GP retimer has an otherwise identical K32 suffix,
# shifted by five packets from folded down. Padding is an indexing adapter only;
# the five synthetic NOPs are removed from the resulting schedule immediately.
padded=prefix+'.LBB0_3:\n'+5*'nop; nop; nop; nop\n'+'\n'.join('; '.join(r)for r in rows)+'\n.LBB0_6:'+suffix
retimed,gp_plan=transform(padded,load_safe=True);pr,rr,su=split(retimed);assert rr[:5]==[['nop']*4]*5
proposed=pr+'.LBB0_3:\n'+'\n'.join('; '.join(r)for r in rr[5:])+'\n.LBB0_6:'+su
_,new,_=split(proposed)
assert [r[2]for r in rows if r[2].startswith('mac.')]==[r[2]for r in new if r[2].startswith('mac.')]
assert new[13][0]=='ld_tnsr UNPCK_8_TO_16 unpack V38, 0x1, I7'
assert rows[4][1]=='add.i32 b00010 I7, S3, I7'
assert new[203][0]=='ld_tnsr UNPCK_8_TO_16 unpack V34, 0x1, I7'
(out/'folded.s').write_text(folded);(out/'proposed-uncompiled.s').write_text(proposed)
checks=[dict(register=r['register'],last_access=r['last_access']-5,first_overwrite=r['first_overwrite']-5)for r in gp_plan['overwritten_register_last_use']]
edges=[dict(producer=r['producer']-5,consumer=r['consumer']-5,gap=r['gap'])for r in gp_plan['dependency_packet_gaps']]
report=dict(status='OFFLINE_SCHEDULE_OPPORTUNITY_NOT_QUALIFIED',current_library=str(lib),library_sha256=sha(lib.read_bytes()),elf_sha256=sha(elf),text_sha256=sha(text(elf)),actual_text_matches_assembly=True,compile_command=command,
 current=counts(source),folded=counts(folded),proposed=counts(proposed),MAC_sequence_unchanged=True,scale_loads_per_K32_unchanged=2,
 first_scale_load=dict(old_folded_packet=222,proposed_packet=13,I7_ready_packet=4,previous_tensor_load_packet=11,next_tensor_load_packet=17),second_scale_load=dict(old_folded_packet=229,proposed_packet=203),last_use_checks=checks,dependency_packet_gaps=edges,
 tasks=96,K32_per_task=8,balanced_tasks_per_24_cores=4,nominal_packets_per_core=dict(current=289*32,folded=269*32,proposed=242*32),
 device_tested=False,candidate_assembled=False,candidate_simulated=False,scope='Current production .text verified; proposal checks instruction slots/register reuse against existing GP retimer, not device latency or legality. Existing folded-only down had no useful M1 speed gain; GP SIM-passing schedules have failed hardware.')
(out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k]for k in('status','current','folded','proposed','nominal_packets_per_core')}))
