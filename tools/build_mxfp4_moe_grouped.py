import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--abi-include',default='/usr/lib/habanatools/include');p.add_argument('--api-include');a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);commands=[];objects=[]
for name in ['group_prepare','group_decode','group_combine','group_gate','group_legacy_gate','group_decode_legacy']:
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*(['-DGK_HISTORICAL_N512=1'] if name=='group_decode_legacy' else []),flag,str(root/'csrc/tpc/mxfp4_moe/legacy/compact_gate.c') if name=='group_legacy_gate' else str(root/'csrc/tpc/mxfp4_moe'/ ('group_decode.c' if name=='group_decode_legacy' else name+'.c')),'-o',name+suffix])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.abi_include,str(root/'csrc/host/mxfp4_moe_grouped_glue.cpp'),*objects,'-o','libmxfp4_moe_grouped_tpc.so'])
if a.api_include:commands.append(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-ffunction-sections','-I'+a.api_include,'-c',str(root/'csrc/ops/mxfp4_moe_grouped.cpp'),'-o','graph.o'])
if a.api_include:commands.append(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-I'+a.api_include,'-c',str(root/'benchmarks/mxfp4_moe_grouped/probe.cpp'),'-o','probe.o'])
if a.api_include and Path("/usr/lib/habanalabs/libSynapse.so").exists():commands.append(['g++','-O2','-std=c++17','-I'+a.api_include,str(root/'benchmarks/mxfp4_moe_grouped/probe.cpp'),'graph.o','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','probe'])
identity=root/'source-identity.json'
commit=json.loads(identity.read_text())['git_commit'] if identity.is_file() else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if identity.is_file():
 for name,want in json.loads(identity.read_text())['files_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:raise RuntimeError('source identity mismatch '+name)
 (out/'source-identity.json').write_bytes(identity.read_bytes())
meta={'source_commit':commit,'commands':commands,'device_validated':False,'state':'building'}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as error:meta.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='offline_compiled_only',artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'});(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['state'])
