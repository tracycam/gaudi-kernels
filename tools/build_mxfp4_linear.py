"""Build the reusable public-Synapse core and benchmark; never acquire a device."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--include-dir',default='/usr/include/habanalabs')
p.add_argument('--tpc-compiler',default='tpc-clang')
p.add_argument('--optimized-decode',action='store_true',help='Previously verified four-chain decoder and normal-domain BF16 scale multiply')
a=p.parse_args(); root=Path(__file__).resolve().parents[1]; out=a.output_dir.resolve()
out.mkdir(parents=True,exist_ok=False)
commands=[]; objects=[]
for name in ['decode','gemv','bias','reduce']:
    src=root/f'csrc/tpc/mxfp4_linear/{name}.c'; target='mxfp4_'+name
    for flag,suffix in [('-c','.o'),('-S','.s')]:
        defines=['-DGK_MXFP4_DECODE_UNROLL=4','-DGK_MXFP4_DECODE_FAST_SCALE=2'] if name=='decode' and a.optimized_decode else []
        commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',*defines,flag,str(src),'-o',target+suffix])
    commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',target+'.o',target+'_x86.o'])
    objects.append(target+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,
                 str(root/'csrc/host/mxfp4_linear_glue.cpp'),*objects,'-o','libgaudi_mxfp4_linear_tpc.so'])
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,
                 str(root/'csrc/ops/mxfp4_linear.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','libgaudi_mxfp4_linear.so'])
commands.append(['g++','-O3','-std=c++17','-I'+a.include_dir,str(root/'benchmarks/mxfp4_linear/probe.cpp'),
                 '-L.','-Wl,-rpath,$ORIGIN','-lgaudi_mxfp4_linear','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','mxfp4_probe'])
sources=[*root.glob('csrc/tpc/mxfp4_linear/*'),root/'csrc/host/mxfp4_linear_glue.cpp',
         *root.glob('csrc/ops/mxfp4_linear.*'),*root.glob('benchmarks/mxfp4_linear/*'),Path(__file__)]
meta={'source_sha256':{str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources if s.is_file()},
      'commands':commands,'state':'building','scope':'public Synapse builder; no Torch OpBackend integration'}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
with (out/'build.log').open('w') as log:
    try:
        for command in commands:
            log.write(json.dumps(command)+'\n'); log.flush()
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    except (OSError,subprocess.CalledProcessError) as error:
        meta.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='built_not_validated',artifacts={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in out.iterdir() if s.is_file() and s.name!='build.json'})
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps({'state':meta['state'],'output':str(out)}))
