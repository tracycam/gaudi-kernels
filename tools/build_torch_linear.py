"""Build public CustomOp registration against installed Torch/Habana; no device use."""
import argparse
import hashlib
import json
import os
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.utils.git_info as git_info
from torch.utils.cpp_extension import load
root=Path(__file__).resolve().parents[1];source=root/'csrc/torch/linear.cpp';h=Path(ht.__file__).parent
metadata={'torch':torch.__version__,'bridge_commit':git_info.get_commit_hash(),
          'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'state':'building'}
sources=[source,root/'csrc/torch/mxfp4.cpp']
metadata['sources_sha256']={str(s.relative_to(root)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}
(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
try:
    load(name='gaudi_kernels_torch',sources=[str(s) for s in sources],extra_include_paths=[str(h/'include'),'/usr/include/habanalabs'],
         extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],
         build_directory=str(out),is_python_module=False,verbose=True)
except Exception as error:
    metadata.update(state='failed',error=str(error));(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n');raise
metadata.update(state='built',binary_sha256=hashlib.sha256((out/'gaudi_kernels_torch.so').read_bytes()).hexdigest())
(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
