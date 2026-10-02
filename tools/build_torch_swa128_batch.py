"""Build public batch SWA registration without acquiring a device."""
import argparse,hashlib,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
import torch
import habana_frameworks.torch as ht
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parents[1];src=root/'csrc/torch/swa128_batch.cpp';h=Path(ht.__file__).parent
meta=dict(torch=torch.__version__,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),state='building')
try:
 load(name='gaudi_swa128_batch_torch',sources=[str(src)],extra_include_paths=[str(h/'include'),'/usr/include/habanalabs'],extra_cflags=['-O2'],
      extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 meta.update(state='built',binary_sha256=hashlib.sha256((out/'gaudi_swa128_batch_torch.so').read_bytes()).hexdigest())
except Exception as error:
 meta.update(state='failed',error=str(error));raise
finally:(out/'build.json').write_text(json.dumps(meta,indent=2)+'\n')
