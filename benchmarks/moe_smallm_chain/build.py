"""Bounded default-off full MoE TPC experiment; build does not acquire a card."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
from generate import generate
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--max-cap',type=int,choices=(4,8),default=4);p.add_argument('--dense',action='store_true');p.add_argument('--stripe',action='store_true');p.add_argument('--wide-m1',action='store_true');p.add_argument('--wide-window',type=int,choices=(4,8,16),default=4);a=p.parse_args()
assert not a.wide_m1 or (a.dense and a.max_cap==4)
root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
names=('metadata','gp4','down4')+(('gp8','down8')if a.max_cap==8 else())+('gate','combine')
for n in names:
    text=generate(root,int(n[-1]),n.startswith('down'),a.dense,a.stripe,a.wide_m1,a.wide_window) if n[-1]in'48' else (root/'benchmarks/moe_smallm_chain'/f'{n}.c').read_text()
    (out/(n+'.c')).write_text(text)
commands=[]
defs=(['-DGK_SMALLM_CAP8=1']if a.max_cap==8 else[])+(['-DGK_SMALLM_DENSE=1']if a.dense else[])+(['-DGK_SMALLM_STRIPE=1']if a.stripe else[])
if a.wide_m1:defs.append('-DGK_SMALLM_WIDE_M1=1')
for n in names:
    for flag,suffix in (('-c','.o'),('-S','.s')):commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*defs,flag,n+'.c','-o',n+suffix])
    commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',n+'.o',n+'_x86.o'])
    commands.append(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',n+'.o'])
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I/usr/include/habanalabs',*defs,str(root/'benchmarks/moe_smallm_chain/glue.cpp'),*[n+'_x86.o'for n in names],'-o','libgaudi_smallm_tpc.so'])
meta=dict(state='building',max_cap=a.max_cap,dense=a.dense,stripe=a.stripe,wide_m1=a.wide_m1,wide_window=a.wide_window,commands=commands,device_verified=False,production_changed=False)
try:
    with(out/'build.log').open('w')as log:
        for cmd in commands:
            log.write(json.dumps(cmd)+'\n');log.flush()
            if cmd[0]=='tpc-llvm-objdump':
                with(out/(cmd[-1]+'.dis')).open('w')as f:subprocess.run(cmd,cwd=out,stdout=f,stderr=log,check=True)
            else:subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
    import habana_frameworks.torch as ht
    from torch.utils.cpp_extension import load
    h=Path(ht.__file__).parent
    load(name='gaudi_smallm',sources=[str(root/'benchmarks/moe_smallm_chain/torch.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2',*defs],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
    meta['state']='built_not_device_verified'
finally:
    meta['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()and p.name!='build.json'}
    (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
