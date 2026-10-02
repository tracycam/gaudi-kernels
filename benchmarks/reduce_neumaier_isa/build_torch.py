import argparse,hashlib,json,os,subprocess,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--offline-meta',action='store_true');p.add_argument('--bridge-include');p.add_argument('--draft',action='store_true');p.add_argument('--identity',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','1')
import torch
from torch.utils.cpp_extension import load
sources=[root/'csrc/torch/reduce_neumaier_isa.cpp'];ld=[]
if a.offline_meta:
 assert a.bridge_include;sources.append(root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp');includes=[a.bridge_include]
else:
 import habana_frameworks.torch as ht
 h=Path(ht.__file__).parent;includes=[str(h/'include')];ld=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin']
if a.draft:commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
else:
 data=json.loads((a.identity or root/'source-identity.json').read_text());commit=data['git_commit']
 for name,want in data['files_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==want,name
 (out/'source-identity.json').write_text(json.dumps(data,indent=2)+'\n')
meta=dict(source_commit=commit,draft=a.draft,offline_meta_only=a.offline_meta,torch=torch.__version__,state='building',sources_sha256={str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
for source in sources:
 dst=out/'source'/source.relative_to(root);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
meta['public_bridge_header_sha256']=hashlib.sha256((Path(includes[0])/'hpu_custom_op_pt2.h').read_bytes()).hexdigest()
(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
try:load(name='gk_reduce_neumaier_isa_torch',sources=list(map(str,sources)),extra_include_paths=includes,extra_cflags=['-O2'],extra_ldflags=ld,build_directory=str(out),is_python_module=False,verbose=True)
except Exception as e:meta.update(state='failed',error=str(e));(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n');raise
meta.update(state='built',library_sha256=hashlib.sha256((out/'gk_reduce_neumaier_isa_torch.so').read_bytes()).hexdigest());(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
if a.offline_meta:
 p=torch.empty(3,2,129,device='meta');sa=torch.empty(3,2,1,device='meta');sw=torch.empty(2,3,device='meta');bias=torch.empty(129,device='meta')
 for name in ['baseline','lookahead','handschedule','baseline_debug','lookahead_debug','handschedule_debug']:
  y=getattr(torch.ops.gk_reduce_isa,name)(p,sa,sw,bias);assert y.shape==(2,129) and y.dtype==(torch.float32 if name.endswith('_debug') else torch.bfloat16)
  for bad in [(p.half(),sa,sw,bias),(p,sa[:,:,:0],sw,bias),(p,sa,sw[:,:2],bias),(p,sa,sw,bias[:128])]:
   try:getattr(torch.ops.gk_reduce_isa,name)(*bad)
   except RuntimeError:pass
   else:raise AssertionError('invalid shape or dtype was accepted')
 (out/'meta-check.json').write_text(json.dumps(dict(all_pass=True,device_validated=False,original_FP32_inputs_BF16_output=True))+'\n')
