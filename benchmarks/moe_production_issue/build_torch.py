"""Target CPU build/Meta only; no device acquire."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['PT_HPU_LAZY_MODE']='1';os.environ['MAX_JOBS']='2'
import torch
import habana_frameworks.torch as ht
from torch.utils.cpp_extension import load
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2];h=Path(ht.__file__).parent
load(name='gaudi_gp_scale_tail',sources=[str(root/'csrc/torch/moe_gp_scale_tail.cpp')],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(a.output),is_python_module=False,verbose=True)
for m in (1,2,8):
 x=torch.empty((m,6144),dtype=torch.bfloat16,device='meta');ids=torch.empty((m,8),dtype=torch.int32,device='meta');dummy=torch.empty((1,),device='meta');y=torch.ops.gaudi_gp_scale_tail.gp(dummy,dummy,x,dummy,ids);assert tuple(y.shape)==(1,m*8*1536) and y.dtype==torch.float32
(a.output/'build.json').write_text(json.dumps(dict(device_acquired=False,meta_pass=True,torch=torch.__version__,library_sha256=hashlib.sha256((a.output/'gaudi_gp_scale_tail.so').read_bytes()).hexdigest()),indent=2)+'\n')
