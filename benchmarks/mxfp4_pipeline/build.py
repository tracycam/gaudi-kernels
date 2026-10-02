"""Build the reusable public-Synapse core and benchmark; never acquire a device."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--include-dir',default='/usr/include/habanalabs')
p.add_argument('--tpc-compiler',default='tpc-clang')
p.add_argument('--data-dependency',action='store_true')
p.add_argument('--decode-unroll',type=int,choices=[1,2,4,8],default=1)
p.add_argument('--smallm-rows',type=int,choices=[1,2,4])
p.add_argument('--smallm-unroll',type=int,choices=[1,4],default=1)
p.add_argument('--smallm-fold-scale',action='store_true')
p.add_argument('--smallm-vector',action='store_true')
p.add_argument('--smallm-straight',action='store_true')
p.add_argument('--smallm-window',type=int,choices=[8,16,32],default=32)
scale=p.add_mutually_exclusive_group()
scale.add_argument('--fast-scale',action='store_true',help='Exact integer scale shortcut for validated E8M0 2..252')
scale.add_argument('--bf16-scale',action='store_true',help='Exact normal-domain BF16 multiply for validated E8M0 2..252')
a=p.parse_args(); root=Path(__file__).resolve().parents[2]; out=a.output_dir.resolve()
if a.smallm_vector and (not a.smallm_rows or a.smallm_unroll!=4):p.error('vector fetch requires smallm rows and unroll 4')
if a.smallm_straight and not a.smallm_vector:p.error('straight K32 requires vector fetch')
out.mkdir(parents=True,exist_ok=False)
commands=[]; objects=[]
for name in ['decode','gemv','bias','reduce']:
    src=root/f'csrc/tpc/mxfp4_linear/{name}.c'; target='mxfp4_'+name
    if name=='decode' and a.data_dependency:src=root/'benchmarks/mxfp4_pipeline/ordered_decode.c'
    if name=='gemv' and a.smallm_rows:src=root/'benchmarks/mxfp4_smallm/reuse.c'
    for flag,suffix in [('-c','.o'),('-S','.s')]:
        defines=([f'-DGK_MXFP4_DECODE_UNROLL={a.decode_unroll}'] if name=='decode' and a.decode_unroll!=1 else [])
        if name=='decode' and a.fast_scale:defines.append('-DGK_MXFP4_DECODE_FAST_SCALE=1')
        if name=='decode' and a.bf16_scale:defines.append('-DGK_MXFP4_DECODE_FAST_SCALE=2')
        if name=='gemv' and a.smallm_rows:
            defines += [f'-DGK_SMALLM_ROWS={a.smallm_rows}',f'-DGK_SMALLM_UNROLL={a.smallm_unroll}']
            if a.smallm_fold_scale:defines.append('-DGK_SMALLM_FOLD_SCALE=1')
            if a.smallm_vector:defines.append('-DGK_SMALLM_VECTOR=1')
            if a.smallm_straight:defines.append(f'-DGK_SMALLM_STRAIGHT={a.smallm_window}')
        commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',*defines,flag,str(src),'-o',target+suffix])
    commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',target+'.o',target+'_x86.o'])
    objects.append(target+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,
                 *([f'-DGK_SMALLM_ROWS={a.smallm_rows}'] if a.smallm_rows else []),
                 *(['-DGK_SMALLM_VECTOR=1'] if a.smallm_vector else []),
                 str(root/('benchmarks/mxfp4_pipeline/ordered_glue.cpp' if a.data_dependency else 'csrc/host/mxfp4_linear_glue.cpp')),*objects,'-o','libgaudi_mxfp4_linear_tpc.so'])
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.include_dir,
                 *(['-DGK_DATA_DEPENDENCY'] if a.data_dependency else []),str(root/'benchmarks/mxfp4_pipeline/append.cpp'),'-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','libgaudi_mxfp4_linear.so'])
commands.append(['g++','-O3','-std=c++17','-I'+a.include_dir,str(root/'benchmarks/mxfp4_linear/probe.cpp'),
                 '-L.','-Wl,-rpath,$ORIGIN','-lgaudi_mxfp4_linear','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','mxfp4_probe'])
sources=[*root.glob('csrc/tpc/mxfp4_linear/*'),root/'csrc/host/mxfp4_linear_glue.cpp',
         *root.glob('csrc/ops/mxfp4_linear.*'),*root.glob('benchmarks/mxfp4_linear/*'),*root.glob('benchmarks/mxfp4_pipeline/*')]
sources += list(root.glob('benchmarks/mxfp4_smallm/*'))
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
