"""Unscheduled correctness scaffold; no speed claim or production registration."""
import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--period',type=int,choices=(4,6));a=p.parse_args()
root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=a.canonical/'artifacts/builds/moe-production-issue/final-a/offline/offline-a/down.s'
text=source.read_text();head=text.split('loop S3, S4, 1, <, .LBB0_7')[0]
NOP='nop; nop; nop; nop\n';lines=[head,'loop S3, S4, 1, <, .task_end\n',NOP*2]
def raw(s,delay=7):lines.append(s+'\n'+NOP*delay)
def ins(s,delay=7):raw('nop; '+s+'; nop; nop',delay)
for v in range(4,8):raw(f'mov.f32 V{v}, 0x0; nop; nop; nop',0)
lines.append(NOP*8+'loop 0, 8, 1, <, .route_end\n'+NOP*2)
raw('set_indx I0, b11111, 0x0; nop; nop; nop');ins('set_indx I0, b00001, S33')
raw('nop; nop; nop; gen_addr AD0, 0x4, I0');raw('ld_g S10, AD0; nop; nop; nop',12)
raw('nop; nop; nop; gen_addr AD0, 0x5, I0');raw('ld_g S26, AD0; nop; nop; nop',12)
ins('shr.u32 S13, S32, 1');ins('and.i32 S14, S32, 1');ins('mul.i32 S12, S10, 12');ins('add.i32 S12, S12, S13')
ins('shl.i32 S1, S12, 9');ins('add.i32 S1, S1, S14');ins('shl.i32 S3, S12, 4');ins('add.i32 S3, S3, S14');ins('mov.i32 S4, S1')
for coord in ('I5','I6','I7'):raw(f'set_indx {coord}, b11111, 0x0; nop; nop; nop')
ins('set_indx I6, b00010, S33')
for v in range(4):raw(f'mov.f32 V{v}, 0x0; nop; nop; nop',0)
if a.period:
    ins('set_indx I5, b00010, S1');ins('set_indx I7, b00010, S3')
lines.append(NOP*8+'loop 0, 8, 1, <, .group_end\n'+NOP*2)
if not a.period:ins('shl.i32 S6, S34, 1')
body=(a.plan/'n256-unscheduled-body.sfrag').read_text().replace('S33','S34')
body=body.replace('add.i32 b00010 I5, 0x1, I5','add.i32 b00010 I5, 0x2, I5').replace('add.i32 S4, S4, 0x20','add.i32 S4, S4, 0x40').replace('set_indx I7, b00010, S34','set_indx I7, b00010, S6')
if a.period:
    from retime import schedule
    body,plan=schedule(a.period);(out/'schedule.json').write_text(json.dumps(plan,indent=2)+'\n')
lines += [body,'.group_end:\n',NOP*8]
for v in range(4):raw(f'nop; nop; mac.f32 V{v+4}, V{v}, S26; nop',1)
lines += ['.route_end:\n',NOP*8]
raw('set_indx I0, b11111, 0x0; nop; nop; nop');raw('ld_tnsr V8, 0x6, I0; nop; nop; nop');ins('set_indx I0, b00010, 1');raw('ld_tnsr V9, 0x6, I0; nop; nop; nop')
raw('set_indx I4, b11111, 0x0; nop; nop; nop');ins('shl.i32 S4, S32, 8');ins('set_indx I4, b00001, S4')
for even,odd in [(4,5),(6,7)]:
    for dest,src,groups in [(16,even,(0,0,1,1)),(17,odd,(0,0,1,1)),(18,even,(2,2,3,3)),(19,odd,(2,2,3,3))]:
        raw(f'nop; nop; mov_dg.all sdg0={groups[0]} sdg1={groups[1]} sdg2={groups[2]} sdg3={groups[3]} weg0=3 weg1=3 weg2=3 weg3=3 V{dest}, V{src}, 0xffffffff; nop',1)
    lines.append(NOP*8)
    for dest in range(20,24):raw(f'mov.f32 V{dest}, 0x0; nop; nop; nop',0)
    lines.append(NOP*8)
    for dest,src,direction in [(20,16,8),(21,17,9),(22,18,8),(23,19,9)]:raw(f'nop; nop; shuffle.f32 V{dest}, V{src}, V{direction}; nop',1)
    lines.append(NOP*8)
    raw('nop; nop; add.f32 V24, V20, V21; nop');raw('nop; nop; add.f32 V25, V22, V23; nop')
    raw('nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x7, I4, V24',1)
    raw('nop; add.i32 b00001 I4, 0x40, I4; nop; st_tnsr 0x7, I4, V25',1)
lines += ['.task_end:\n',NOP*8,'nop; halt; halt; nop\n',NOP*8]
(out/'candidate.s').write_text(''.join(lines))
commands=[['tpc-clang','-O2','-mcpu=gaudi2','-c',str(root/'csrc/tpc/moe_down_combine_issue/descriptor.c'),'-o','candidate.o'],['tpc-clang','-mcpu=gaudi2','-c','candidate.s','-o','assembled.o'],['objcopy','--dump-section','.text=candidate.text','assembled.o'],['objcopy','--update-section','.text=candidate.text','candidate.o']]
report=dict(status='BUILDING',device_used=False,scope='Offline schedule hypothesis; no device qualification or production registration',period=a.period,commands=commands)
try:
    with (out/'build.log').open('w') as log:
        for command in commands:log.write(json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=45)
    with (out/'candidate.objdump').open('w') as f:subprocess.run(['tpc-llvm-objdump','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','-d',str(out/'candidate.o')],check=True,stdout=f,timeout=45)
    report.update(status='COMPILED_NOT_SIMULATED',elf_sha256=hashlib.sha256((out/'candidate.o').read_bytes()).hexdigest(),text_sha256=hashlib.sha256((out/'candidate.text').read_bytes()).hexdigest(),max_vector_register=max(map(int,re.findall(r'\bV(\d+)\b',''.join(lines)))),V40_or_private_VLM=False)
except BaseException as error:report.update(status='FAIL',error=repr(error));raise
finally:(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
