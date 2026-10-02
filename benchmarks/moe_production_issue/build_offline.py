"""Extract actual frozen production ELFs and build one isolated scale-tail candidate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--conservative',action='store_true');p.add_argument('--war-safe',action='store_true');p.add_argument('--load-safe',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'tools'))
from moe_activation_fold_core import embedded_elf
from scale_tail import transform
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
libs=a.canonical/'artifacts/builds/production-integration/model-runs/production-isa-70-c/source/production-runtime/libraries'
assembled=a.canonical/'artifacts/builds/moe-activation-folded/final-a/device-a/builds'
targets=[('gp','libgaudi_moe_activation_folded_tpc.so','folded_gp',assembled/'folded-a/folded_gp.s'),
         ('down','libgaudi_down_activation_tpc.so','direct_down',assembled/'qualified-down/direct_down_broadcast.s')]
report=dict(device_acquired=False,status='BUILDING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),targets={},commands=[])
def save():(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
def run(cmd):
    report['commands'].append(cmd);save()
    with (out/'commands.log').open('a') as log:
        log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
def counts(s):
    prefix,body=s.split('.LBB0_3:\n');body,suffix=body.split('.LBB0_6:')
    pre,task=prefix.split('.LBB0_2:')
    def packets(x):return [l for l in x.splitlines() if ';' in l or l.startswith('loop ')]
    rows=[[v.strip() for v in line.split(';')] for line in body.splitlines()]
    return dict(kernel_prologue_packets=len(packets(pre)),task_prologue_packets=len(packets(task)),
                task_epilogue_packets=len(packets(suffix.split('.LBB0_7:')[0])),
                kernel_epilogue_packets=len(packets(suffix.split('.LBB0_7:')[1])),
                body_packets=len(rows),body_slot_instructions=[dict(Counter(r[i].split()[0] for r in rows if r[i]!='nop')) for i in range(4)],
                body_full_nops=sum(r==['nop']*4 for r in rows),max_vector_register=max(map(int,re.findall(r'\bV(\d+)\b',s))),
                vector_local_load_store_instructions=re.findall(r'\b(?:ld_l_v|st_l_v)\b',s))
try:
    for name,lib,stem,asm in targets:
        (out/(name+'.o')).write_bytes(embedded_elf((libs/lib).read_bytes(),stem));(out/(name+'.s')).write_bytes(asm.read_bytes())
        run(['tpc-clang','-mcpu=gaudi2','-c',name+'.s','-o',name+'-assembled.o'])
        run(['objcopy','--dump-section','.text='+name+'.text',name+'.o'])
        run(['objcopy','--dump-section','.text='+name+'-assembled.text',name+'-assembled.o'])
        assert (out/(name+'.text')).read_bytes()==(out/(name+'-assembled.text')).read_bytes()
        with (out/(name+'.objdump')).open('w') as f:subprocess.run(['tpc-llvm-objdump','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','-d',str(out/(name+'.o'))],stdout=f,check=True,timeout=30)
        report['targets'][name]=dict(frozen_library=str(libs/lib),library_sha256=sha(libs/lib),elf_sha256=sha(out/(name+'.o')),text_sha256=sha(out/(name+'.text')),assembly_sha256=sha(asm),actual_text_equals_assembly=True,counts=counts(asm.read_text()))
    candidate,plan=transform((out/'gp.s').read_text(),a.conservative,a.war_safe,a.load_safe);(out/'candidate.s').write_text(candidate);(out/'candidate.o').write_bytes((out/'gp.o').read_bytes());report['candidate']=plan|dict(counts=counts(candidate))
    run(['tpc-clang','-mcpu=gaudi2','-c','candidate.s','-o','candidate-assembled.o'])
    run(['objcopy','--dump-section','.text=candidate.text','candidate-assembled.o'])
    run(['objcopy','--update-section','.text=candidate.text','candidate.o'])
    with (out/'candidate.objdump').open('w') as f:subprocess.run(['tpc-llvm-objdump','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','-d',str(out/'candidate.o')],stdout=f,check=True,timeout=30)
    report['candidate'].update(elf_sha256=sha(out/'candidate.o'),text_sha256=sha(out/'candidate.text'))
    report['status']='BUILT_NOT_SIM_OR_DEVICE_QUALIFIED'
except BaseException as error:report.update(status='FAIL',error=repr(error));raise
finally:save()
print(json.dumps(dict(status=report['status'],targets=report['targets'],candidate=report.get('candidate'))))
