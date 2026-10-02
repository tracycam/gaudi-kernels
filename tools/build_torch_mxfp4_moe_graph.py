"""Public registration build; explicit --offline-meta links a rejecting stub."""
import argparse, hashlib, json, os, subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--offline-meta',action='store_true');p.add_argument('--draft',action='store_true');p.add_argument('--bridge-include');p.add_argument('--api-include');a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('MAX_JOBS','1');os.environ.setdefault('PT_HPU_LAZY_MODE','1')
import torch
from torch.utils.cpp_extension import load
sources=[root/'csrc/torch/mxfp4_moe_graph.cpp'];flags=[]
if a.offline_meta:
 if not a.bridge_include or not a.api_include:raise ValueError('offline header paths required')
 sources.append(root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp');includes=[a.bridge_include,a.api_include]
else:
 import habana_frameworks.torch as ht
 h=Path(ht.__file__).parent;includes=[str(h/'include'),'/usr/include/habanalabs'];flags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin']
identity=root/'source-identity.json'
if a.draft:
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
else:
 data=json.loads(identity.read_text());commit=data['git_commit']
 for source in sources:
  if str(source.relative_to(root)) not in data['files_sha256']:raise RuntimeError('source not in identity '+str(source))
 for name,want in data['files_sha256'].items():
  if hashlib.sha256((root/name).read_bytes()).hexdigest()!=want:raise RuntimeError('source changed '+name)
 (out/'source-identity.json').write_bytes(identity.read_bytes())
meta=dict(source_commit=commit,draft_uncommitted=a.draft,torch=torch.__version__,offline_meta_only=a.offline_meta,state='building',sources_sha256={str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:
 load(name='gaudi_mxfp4_moe_graph',sources=list(map(str,sources)),extra_include_paths=includes,extra_cflags=['-O2'],extra_ldflags=flags,build_directory=str(out),is_python_module=False,verbose=True)
except Exception as e:
 meta.update(state='failed',error=str(e));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='built',binary_sha256=hashlib.sha256((out/'gaudi_mxfp4_moe_graph.so').read_bytes()).hexdigest());(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
