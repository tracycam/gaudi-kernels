"""Offline Gaudi2 TPC compile; never acquires a device or invents a host ABI."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path)
p.add_argument('--tpc-compiler',default='/usr/bin/tpc-clang')
p.add_argument('--public-include-dir',type=Path,help='Matching public Synapse header snapshot for host syntax/planner compile')
p.add_argument('--tpc-include-dir',type=Path,help='Runtime-matching modern TPC SDK; omitted when unavailable')
p.add_argument('--native',action='store_true',help='Link public Synapse core and explicit-module device probe; does not run it')
a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve()
out.mkdir(parents=True,exist_ok=False)
commands=[]
for name in ['decode_native','decode_rows','decode_rows256']:
    for flag,suffix in [('-c','.o'),('-S','.s')]:
        commands.append([a.tpc_compiler,'-O2','-ffp-contract=off','-mcpu=gaudi2',flag,
                         str(root/f'csrc/tpc/mxfp4_compact/{name}.c'),'-o',name+suffix])
sources=[*root.glob('csrc/tpc/mxfp4_compact/*'),root/'python/gaudi_kernels/mxfp4_compact.py',
         *root.glob('benchmarks/mxfp4_compact/*'),*root.glob('csrc/ops/mxfp4_compact.*'),
         root/'csrc/host/mxfp4_compact_glue.cpp',Path(__file__)]
if a.public_include_dir:
    inc='-I'+str(a.public_include_dir.resolve())
    commands.append(['g++','-O2','-std=c++17','-fPIC','-ffunction-sections','-fdata-sections',inc,
                     '-c',str(root/'csrc/ops/mxfp4_compact.cpp'),'-o','mxfp4_compact_host.o'])
    commands.append(['g++','-O2','-std=c++17','-ffunction-sections','-fdata-sections',inc,
                     str(root/'benchmarks/mxfp4_compact/planner_probe.cpp'),'mxfp4_compact_host.o',
                     '-Wl,--gc-sections','-o','planner_probe'])
if a.tpc_include_dir:
    if not (a.tpc_include_dir/'tpc_kernel_lib_interface.h').is_file():
        raise ValueError('modern tpc_kernel_lib_interface.h required; old gcapi is not interchangeable')
    for name in ['decode_native','decode_rows','decode_rows256']:
        commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o'])
    commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
                     '-I'+str(a.tpc_include_dir.resolve()),str(root/'csrc/host/mxfp4_compact_glue.cpp'),
                     'decode_native_x86.o','decode_rows_x86.o','decode_rows256_x86.o','-o','libgaudi_mxfp4_compact_tpc.so'])
    commands.append(['g++','-O2','-std=c++17','-I'+str(a.tpc_include_dir.resolve()),
                     str(root/'benchmarks/mxfp4_compact/glue_audit.cpp'),'-L.','-Wl,-rpath,$ORIGIN',
                     '-lgaudi_mxfp4_compact_tpc','-o','glue_audit'])
if a.native:
    if not a.public_include_dir or not a.tpc_include_dir:raise ValueError('native build requires both public and TPC SDK headers')
    commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs',
                     '-I'+str(a.public_include_dir.resolve()),str(root/'csrc/ops/mxfp4_compact.cpp'),
                     '-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','libgaudi_mxfp4_compact.so'])
    commands.append(['g++','-O3','-std=c++17','-I'+str(a.public_include_dir.resolve()),
                     str(root/'benchmarks/mxfp4_compact/device_probe.cpp'),'-L.','-Wl,-rpath,$ORIGIN',
                     '-lgaudi_mxfp4_compact','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs',
                     '-lSynapse','-o','device_probe'])
commit=(subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() if (root/'.git').exists()
        else json.loads((root/'source-identity.json').read_text())['git_commit'])
meta={'state':'building','source_commit':commit,
      'source_sha256':{str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources if s.is_file()},
      'commands':commands,'device_executed':False,
      'scope':'TPC ELF/ISA plus optional public host syntax/pure planner; no native graph validation.',
      'modern_tpc_glue_built':bool(a.tpc_include_dir),
      'native_probe_built':a.native,
      'tpc_header_sha256':({str((a.tpc_include_dir/'tpc_kernel_lib_interface.h').resolve()):
          hashlib.sha256((a.tpc_include_dir/'tpc_kernel_lib_interface.h').read_bytes()).hexdigest()}
          if a.tpc_include_dir else {}),
      'public_header_sha256':{str(s.resolve()):hashlib.sha256(s.read_bytes()).hexdigest()
          for s in (a.public_include_dir.glob('synapse*') if a.public_include_dir else []) if s.is_file()}}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
with (out/'build.log').open('w') as log:
    version=subprocess.run([a.tpc_compiler,'--version'],stdout=log,stderr=subprocess.STDOUT)
    try:
        for command in commands:
            log.write(json.dumps(command)+'\n');log.flush()
            subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    except (OSError,subprocess.CalledProcessError) as error:
        meta.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
native=(out/'decode_native.s').read_text();rows=(out/'decode_rows.s').read_text();wide=(out/'decode_rows256.s').read_text()
assert 'ld_tnsr partial UNPCK_8_TO_16 unpack' in rows
assert 'st_tnsr partial' in rows and 'ld_g ' in rows
assert 'ld_tnsr partial UNPCK_8_TO_16 unpack' in wide and 'shuffle' in wide and 'st_tnsr partial' in wide
assert 'sdg0=0 sdg1=0 sdg2=0 sdg3=0 weg0=3 weg1=3 weg2=3 weg3=3' in wide
assert 'convert.u16 all_lanes target_type=uint32' in native
assert 'pack.u16' in native and 'convert.u32 all_lanes target_type=uint16' in rows
for name,asm in [('decode_native',native),('decode_rows',rows),('decode_rows256',wide)]:
    (out/(name+'-memory-lanes.txt')).write_text('\n'.join(line for line in asm.splitlines()
      if any(s in line for s in ['ld_tnsr','ld_g ','st_tnsr','convert.','pack.','mov_dg','lookup']))+'\n')
try:
  if a.public_include_dir:
    with (out/'planner-audit.log').open('w') as log:
        subprocess.run([sys.executable,str(root/'benchmarks/mxfp4_compact/planner_audit.py'),
                        '--probe',str(out/'planner_probe'),'--output',str(out/'planner-audit.json')],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
  if a.tpc_include_dir:
    with (out/'glue-audit.json').open('w') as log:
        subprocess.run([str(out/'glue_audit')],stdout=log,stderr=subprocess.STDOUT,check=True)
except (OSError,subprocess.CalledProcessError) as error:
    meta.update(state='failed_offline_audit',error=str(error))
    (out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='compiled_isa_checked_device_unvalidated',artifacts={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in out.iterdir() if s.is_file() and s.name!='build.json'})
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps({'state':meta['state'],'output':str(out)}))
