"""Build the mixed-row harness and device preparation. No device acquisition."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--decoder',type=Path,required=True);a=p.parse_args()
src=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
commands=[]
decode=(a.decoder/'decode.c').resolve().read_text()
old='void main(tensor packed,tensor scales,tensor table,tensor ids,tensor output)'
assert decode.count(old)==1
decode=decode.replace(old,'void main(tensor packed,tensor scales,tensor table,tensor ids,tensor activation,tensor counts,tensor output,tensor prepared_a)')
old='int source=s_i32_ld_g(gen_addr(ep,ids));'
assert decode.count(old)==1
decode=decode.replace(old,old+'int valid=s_i32_ld_g(gen_addr(ep,counts));')
old='        int5 sp={0,g,nb,source,0};'
assert decode.count(old)==1
decode=decode.replace(old,'''        // One owner per K128 activation segment; weight requests unchanged.
        if(nb==0 && (g&3)==0) {
          for(int m=0;m<3;++m) {
            int5 p={g*32,m,expert,0,0};
            bfloat128 a=v_bf16_ld_tnsr_b(p,activation,0,(bfloat128)0,m<valid);
            v_bf16_st_tnsr(p,prepared_a,a);
          }
        }
'''+old)
(out/'fused.c').write_text(decode)
for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-mcpu=gaudi2',flag,str(src/'prepare.c'),'-o','prepare'+suffix])
for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-mcpu=gaudi2',flag,str(src/'bounded.c'),'-o','bounded'+suffix])
for name in ('lean','fill','gather'):
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-mcpu=gaudi2',flag,str(src/(name+'.c')),'-o',name+suffix])
 for cmd in [['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'],['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',name+'.o']]:commands.append(cmd)
for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2','-DGK_MXFP4_DECODE_FAST_SCALE=2',flag,'fused.c','-o','fused'+suffix])
commands += [['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','prepare.o','prepare_x86.o'],
 ['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','bounded.o','bounded_x86.o'],
 ['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','fused.o','fused_x86.o'],
 ['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','prepare.o'],
 ['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','bounded.o'],
 ['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn','fused.o'],
 ['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/include/habanalabs',str(src/'glue.cpp'),'prepare_x86.o','bounded_x86.o','fused_x86.o','lean_x86.o','fill_x86.o','gather_x86.o','-o','libmixed.so'],
 ['g++','-O3','-std=c++17','-I/usr/include/habanalabs',str(src/'probe.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','probe']]
record=dict(commands=commands,state='building')
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:
   log.write(json.dumps(cmd)+'\n');log.flush()
   if cmd[0]=='tpc-llvm-objdump':
    with (out/(cmd[-1]+'.dis')).open('w') as dis:subprocess.run(cmd,cwd=out,stdout=dis,stderr=log,check=True)
   else:subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
 record['state']='built'
finally:
 record['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='build.json'}
 (out/'build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(state=record['state'],output=str(out))))
