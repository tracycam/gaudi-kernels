"""Build the experimental current-graph MoE kernels; runtime acquisition stays outside."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--abi-include',default='/usr/lib/habanatools/include');p.add_argument('--api-include');p.add_argument('--draft',action='store_true');p.add_argument('--hoist-addresses',action='store_true');p.add_argument('--simplify-scale-bits',action='store_true');p.add_argument('--decode-unroll',type=int,choices=(1,2,4,8),default=1);a=p.parse_args()
if a.decode_unroll != 1 and not a.hoist_addresses:p.error('--decode-unroll requires --hoist-addresses')
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
names=['graph_count','graph_plan','graph_gather','graph_decode','graph_gate_rows','graph_combine','graph_literal_gate']; source_map={'graph_count':'bucket_count.c','graph_decode':'bucket_decode.c','graph_combine':'bucket_combine.c','graph_literal_gate':'legacy/compact_gate.c'};commands=[];objects=[];sources=[]
for name in names:
 source=root/'csrc/tpc/mxfp4_moe'/source_map.get(name,name+'.c');sources.append(source)
 for flag,suffix in [('-c','.o'),('-S','.s')]:commands.append(['tpc-clang','-O2','-ffp-contract=off','-mcpu=gaudi2',*(['-DGK_HISTORICAL_N512=1'] if name=='graph_decode' else []),*(['-DGK_ROUTE_DECODE_HOIST=1'] if a.hoist_addresses and name=='graph_decode' else []),*(['-DGK_ROUTE_DECODE_SCALE_BITS=1'] if a.simplify_scale_bits and name=='graph_decode' else []),*(['-DGK_ROUTE_DECODE_UNROLL='+str(a.decode_unroll)] if name=='graph_decode' and a.decode_unroll!=1 else []),flag,str(source),'-o',name+suffix])
 commands.append(['objcopy','-I','binary','-O','elf64-x86-64','-B','i386:x86-64',name+'.o',name+'_x86.o']);objects.append(name+'_x86.o')
glue=root/'csrc/host/mxfp4_moe_graph_glue.cpp';sources.append(glue)
commands.append(['g++','-O2','-std=c++17','-shared','-fPIC','-Wl,-z,defs','-I'+a.abi_include,str(glue),*objects,'-o','libmxfp4_moe_graph_tpc.so'])
identity=root/'source-identity.json';commit=None
if a.draft:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
if not a.draft:
 data=json.loads(identity.read_text());commit=data['git_commit']
 for source in sources:
  if str(source.relative_to(root)) not in data['files_sha256']:raise RuntimeError('build source absent from identity: '+str(source))
 for name,want in data['files_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:raise RuntimeError('source identity mismatch '+name)
 (out/'source-identity.json').write_bytes(identity.read_bytes())
meta={'source_commit':commit,'draft_uncommitted':a.draft,'commands':commands,'device_validated':False,'hoist_addresses':a.hoist_addresses,'simplify_scale_bits':a.simplify_scale_bits,'decode_unroll':a.decode_unroll,'scale_domain':'E2M1 only, E8M0 [2,252]; general exact reader unchanged','state':'building','build_source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for cmd in commands:log.write(json.dumps(cmd)+'\n');log.flush();subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as error:
 meta.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='offline_compiled_only',artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'});(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(meta['state'])
