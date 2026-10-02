import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--offline-meta',action='store_true');p.add_argument('--bridge-include');p.add_argument('--draft',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','1')
import torch
from torch.utils.cpp_extension import load
sources=[Path(__file__).resolve().parent/'torch_sum.cpp'];ld=[]
if a.offline_meta:
 assert a.bridge_include;sources.append(root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp');includes=[a.bridge_include]
else:
 import habana_frameworks.torch as ht
 h=Path(ht.__file__).parent;includes=[str(h/'include')];ld=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin']
if a.draft:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
else:
 data=json.loads((root/'source-identity.json').read_text());commit=data['git_commit']
 for name,want in data['files_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
 (out/'source-identity.json').write_text(json.dumps(data,indent=2)+'\n')
meta=dict(source_commit=commit,draft=a.draft,offline_meta_only=a.offline_meta,torch=torch.__version__,state='building',sources_sha256={str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:load(name='tp8_fp32_sum',sources=list(map(str,sources)),extra_include_paths=includes,extra_cflags=['-O2'],extra_ldflags=ld,build_directory=str(out),is_python_module=False,verbose=True)
except Exception as e:meta.update(state='failed',error=str(e));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='built',library_sha256=hashlib.sha256((out/'tp8_fp32_sum.so').read_bytes()).hexdigest());(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
if a.offline_meta:
 x=torch.empty(8,6144,device='meta');y=torch.ops.tp8_fp32_probe.sum(x);assert y.shape==(1,6144) and y.dtype==torch.float32
 try:torch.ops.tp8_fp32_probe.sum(x.to(torch.bfloat16))
 except RuntimeError:pass
 else:raise AssertionError('BF16 rejection absent')
 (out/'meta-check.json').write_text(json.dumps(dict(all_pass=True,device_validated=False,original_FP32_only=True))+'\n')
