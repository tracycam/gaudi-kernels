"""Compile locally; only link the native executable if Synapse runtime exists."""
import argparse,hashlib,json,subprocess,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--reuse-tpc',type=Path);p.add_argument('--abi-include',default='/usr/lib/habanatools/include');p.add_argument('--api-include',required=True);p.add_argument('--hccl-include',required=True);p.add_argument('--runtime',default='/usr/lib/habanalabs');p.add_argument('--draft',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[2];src=Path(__file__).resolve().parent;out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
identity=root/'source-identity.json'
if a.draft:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
else:
 d=json.loads(identity.read_text());commit=d['git_commit']
 for name,want in d['files_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
 (out/'source-identity.json').write_bytes(identity.read_bytes())
commands=[['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',flag,str(src/'sum_f32.c'),'-o','sum_f32'+ext] for flag,ext in [('-c','.o'),('-S','.s')]]
commands += [['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',flag,str(src/'gate_f32.c'),'-o','gate_f32'+ext] for flag,ext in [('-c','.o'),('-S','.s')]]
commands += [['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','gate_f32.o','gate_f32_x86.o']]
commands+=[['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64','sum_f32.o','sum_f32_x86.o'],['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.abi_include,str(src/'glue.cpp'),'sum_f32_x86.o','gate_f32_x86.o','-o','libtp8_fp32_tpc.so'],['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-I'+a.api_include,'-I'+a.hccl_include,'-c',str(src/'native.cpp'),'-o','native.o']]
if a.reuse_tpc:
 old=a.reuse_tpc.resolve();assert old.is_file()
 for name in ['libtp8_fp32_tpc.so','sum_f32.o','sum_f32.s','gate_f32.o','gate_f32.s']:
  shutil.copy2(old.parent/name,out/name)
 commands=[commands[-1]]  # native object only; reuse the qualified device library/ELFs
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wall','-Wextra','-Werror','-I'+a.api_include,'-I'+a.hccl_include,str(src/'api_probe.cpp'),'-ldl','-pthread','-o','libtp8_api_probe.so'])
if (Path(a.runtime)/'libSynapse.so').exists():commands.append(['g++','native.o','-L'+a.runtime,'-Wl,-rpath,'+a.runtime,'-lSynapse','-pthread','-ldl','-o','native'])
meta=dict(source_commit=commit,draft=a.draft,commands=commands,state='building',device_validated=False,reused_tpc_sha256=hashlib.sha256(a.reuse_tpc.read_bytes()).hexdigest() if a.reuse_tpc else None);(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for c in commands:log.write(json.dumps(c)+'\n');log.flush();subprocess.run(c,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as e:meta.update(state='failed',error=str(e));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='compiled',native_linked=(out/'native').exists(),artifacts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='build.json'});(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
