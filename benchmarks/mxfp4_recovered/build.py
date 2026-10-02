"""Build real native Synapse runner and both comparison kernel libraries."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
commands=[['python3',str(root/'tools/build_mxfp4_exact.py'),'--output-dir',str(out/'exact'),'--abi-include','/usr/include/habanalabs','--api-include','/usr/include/habanalabs'],['python3',str(root/'tools/build_mxfp4_overhead.py'),'--output-dir',str(out/'history')],['g++','-O2','-std=c++17','-I/usr/include/habanalabs',str(root/'benchmarks/mxfp4_recovered/probe.cpp'),str(out/'exact/mxfp4_exact_graph.o'),'-L'+str(out/'history'),'-lmxfp4_overhead','-Wl,-rpath,$ORIGIN/history','-L/usr/lib/habanalabs','-Wl,-rpath,/usr/lib/habanalabs','-lSynapse','-o',str(out/'probe')]]
identity=json.loads((root/'source-identity.json').read_text());meta={'git_commit':identity['git_commit'],'commands':commands,'source_identity_sha256':hashlib.sha256((root/'source-identity.json').read_bytes()).hexdigest(),'state':'building'};(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 with (out/'build.log').open('w') as log:
  for c in commands:log.write(json.dumps(c)+'\n');log.flush();subprocess.run(c,cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
except Exception as e:meta.update(state='failed',error=str(e));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='built_not_device_validated',artifacts={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='build.json'});(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps({'state':meta['state']}))
