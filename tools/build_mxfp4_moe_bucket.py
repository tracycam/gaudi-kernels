"""Build the experimental bucket kernels/core; runtime acquisition stays outside."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--abi-include',default='/usr/lib/habanatools/include');p.add_argument('--api-include');p.add_argument('--draft',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
names=['bucket_count','bucket_gather','bucket_decode','bucket_combine','bucket_gate','bucket_legacy_gate','bucket_decode_legacy'];commands=[];objects=[];sources=[]
for name in names:
 source=root/'csrc/tpc/mxfp4_moe'/('legacy/compact_gate.c' if name=='bucket_legacy_gate' else 'bucket_decode.c' if name=='bucket_decode_legacy' else name+'.c');sources.append(source)
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*(['-DGK_HISTORICAL_N512=1'] if name=='bucket_decode_legacy' else []),flag,str(source),'-o',name+suffix])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
glue=root/'csrc/host/mxfp4_moe_bucket_glue.cpp';sources.append(glue)
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.abi_include,str(glue),*objects,'-o','libmxfp4_moe_bucket_tpc.so'])
if a.api_include:
 for path,name in [('csrc/ops/mxfp4_moe_bucket.cpp','graph'),('benchmarks/mxfp4_moe_bucket/probe.cpp','probe')]:
  source=root/path
  if source.exists():sources.append(source);commands.append(['g++','-O2','-std=c++17','-Wall','-Wextra','-Werror','-ffunction-sections','-I'+a.api_include,'-c',str(source),'-o',name+'.o'])
 if (root/'benchmarks/mxfp4_moe_bucket/probe.cpp').exists() and Path('/usr/lib/habanalabs/libSynapse.so').exists():commands.append(['g++','probe.o','graph.o','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o','probe'])
identity=root/'source-identity.json';commit=None
if a.draft:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if not a.draft:
 data=json.loads(identity.read_text());commit=data['git_commit']
 for source in sources:
  if str(source.relative_to(root)) not in data['files_sha256']:raise RuntimeError('build source absent from identity: '+str(source))
 for name,want in data['files_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:raise RuntimeError('source identity mismatch '+name)
 (out/'source-identity.json').write_bytes(identity.read_bytes())
meta={'source_commit':commit,'draft_uncommitted':a.draft,'commands':commands,'device_validated':False,'state':'building','build_source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as error:
 meta.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='offline_compiled_only',artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'});(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['state'])
