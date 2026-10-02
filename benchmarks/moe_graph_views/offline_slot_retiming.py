"""Offline-only retiming candidate; never builds/loads a runtime library.

Checks actual ELF .text against the supplied scheduled assembly, folds the
conservative activation prefill into proven-empty slots, and optionally tests
two activation buffers. The already device-qualified default generator stays
unchanged. Native ISA output comparison is not a device latency qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

p=argparse.ArgumentParser()
p.add_argument('--assembly',required=True,type=Path);p.add_argument('--elf',required=True,type=Path)
p.add_argument('--simulator',required=True,type=Path);p.add_argument('--reference',required=True,type=Path)
p.add_argument('--output',required=True,type=Path);p.add_argument('--two-buffers',action='store_true')
a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
report={'scope':'OFFLINE_ONLY_NOT_DEVICE_QUALIFIED','device_used':False,'TPC_RUNNER':0,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'commands':[],'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [a.assembly,a.elf,a.simulator,a.reference,Path(__file__)]}}
def save():(out/'result.json').write_text(json.dumps(report,indent=2)+'\n')
def run(command):
    report['commands'].append(command);save()
    with (out/'commands.log').open('a') as log:
        log.write(json.dumps(command)+'\n');log.flush()
        subprocess.run(command,cwd=out,env=dict(os.environ,TPC_RUNNER='0'),timeout=45,stdout=log,stderr=subprocess.STDOUT,check=True)
try:
    original=a.assembly.read_text();prefix,body=original.split('.LBB0_3:\n');body,suffix=body.split('.LBB0_6:')
    rows=[[v.strip() for v in line.split(';')] for line in body.splitlines()]
    before=len(rows);assert before in [294,289]
    load=next(i for i,row in enumerate(rows) if row[0]=='ld_tnsr V39, 0x2, I6')
    begin=load-6
    assert all(rows[i]==['nop']*4 for i in range(begin,load))
    replication=rows[load+8][2];assert replication.startswith('mov_dg.all')
    assert all(rows[i]==['nop']*4 for i in [*range(load+1,load+8),*range(load+9,begin+20)])
    del rows[begin:begin+20]
    index_write=next(i for i,row in enumerate(rows) if row[3]=='set_indx I6, b00001, S7')
    new_load=index_write+6;new_replication=new_load+6
    assert rows[new_load][0]=='nop' and rows[new_replication][2]=='nop'
    rows[new_load][0]='ld_tnsr V39, 0x2, I6';rows[new_replication][2]=replication
    first_macs=[i for i,row in enumerate(rows) if row[2].startswith('mac.bf16 acc_fp32 D14,')]
    shuffle=[i for i,row in enumerate(rows) if row[2].startswith('shuffle.u8')]
    assert len(first_macs)==len(shuffle)==32 and shuffle[0]-new_replication>=6
    k=-1
    for i,row in enumerate(rows):
        if i in first_macs:k+=1
        if a.two_buffers and row[2].startswith('mac.bf16'):
            row[2],n=re.subn(r'V(?:36|37|38)$',f'V{36+k%2}',row[2]);assert n==1
    if a.two_buffers:
        for k,i in enumerate(shuffle):
            rows[i][2],n=re.subn(r'^shuffle.u8 V(?:36|37|38),',f'shuffle.u8 V{36+k%2},',rows[i][2]);assert n==1
    liveness=[];buffers=2 if a.two_buffers else 3
    for k,(sh,mac) in enumerate(zip(shuffle,first_macs)):
        assert mac-sh>=6
        next_write=shuffle[k+buffers] if k+buffers<32 else None
        if next_write is not None:assert next_write>mac+3
        liveness.append({'k':k,'buffer':36+k%buffers,'shuffle':sh,'first_mac':mac,'last_mac':mac+3,'next_same_buffer_write':next_write})
    candidate=prefix+'.LBB0_3:\n'+'\n'.join('; '.join(row) for row in rows)+'\n.LBB0_6:'+suffix
    (out/'original.s').write_text(original);(out/'candidate.s').write_text(candidate)
    shutil.copyfile(a.elf,out/'original.o');shutil.copyfile(a.elf,out/'candidate.o')
    run(['tpc-clang','-mcpu=gaudi2','-c','original.s','-o','original_slots.o'])
    run(['objcopy','--dump-section','.text=original.text','original.o'])
    run(['objcopy','--dump-section','.text=assembled_original.text','original_slots.o'])
    assert (out/'original.text').read_bytes()==(out/'assembled_original.text').read_bytes()
    run(['tpc-clang','-mcpu=gaudi2','-c','candidate.s','-o','candidate_slots.o'])
    run(['objcopy','--dump-section','.text=candidate.text','candidate_slots.o'])
    run(['objcopy','--update-section','.text=candidate.text','candidate.o'])
    run([str(a.simulator.resolve()),str(out/'candidate.o'),str(out/'candidate.bin')])
    reference=a.reference.read_bytes();actual=(out/'candidate.bin').read_bytes();assert len(reference)==len(actual)
    mismatches=sum(reference[i:i+4]!=actual[i:i+4] for i in range(0,len(actual),4))
    report.update(status='SIM_PASS_NOT_DEVICE_QUALIFIED' if not mismatches else 'SIM_FAIL',
        checked_fp32_words=len(actual)//4,mismatches=mismatches,original_k32_packets=before,candidate_k32_packets=len(rows),
        max_vector_register=max(map(int,re.findall(r'\bV(\d+)\b',candidate))),
        two_activation_buffers=a.two_buffers,original_text_matches_supplied_assembly=True,
        prefill_moves={'I6_final_set':index_write,'tensor_load':new_load,'DG_replicate':new_replication,'first_shuffle':shuffle[0],'first_mac':first_macs[0]},
        activation_lifetimes=liveness)
    with (out/'candidate.objdump').open('w') as log:subprocess.run(['tpc-llvm-objdump','--triple=tpc','-d',str(out/'candidate.o')],stdout=log,check=True)
    assert mismatches==0
except Exception as error:report.update(status='FAIL',error=repr(error));raise
finally:save()
print(json.dumps({k:report[k] for k in ['status','checked_fp32_words','candidate_k32_packets','two_activation_buffers']}))
