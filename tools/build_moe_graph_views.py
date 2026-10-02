"""Build the graph-owned MoE reshape bridge; no device acquire or launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
os.environ.setdefault('PT_HPU_LAZY_MODE','1');os.environ.setdefault('MAX_JOBS','2')
source=root/'csrc/torch/moe_graph_views.cpp'
identity=root/'source-identity.json'
commit=(json.loads(identity.read_text())['git_commit'] if identity.exists() and not (root/'.git').exists()
        else subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip())
metadata={'commit':commit,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'state':'building','device_acquired':False}
def save():(out/'build.json').write_text(json.dumps(metadata,indent=2)+'\n')
save()
try:
    import torch
    import habana_frameworks.torch as ht
    from torch.utils.cpp_extension import load
    h=Path(ht.__file__).parent;metadata['torch']=torch.__version__
    load(name='gaudi_moe_graph_views',sources=[str(source)],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],
         extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],
         build_directory=str(out),is_python_module=False,verbose=True)
    metadata.update(state='built_not_device_validated',binary_sha256=hashlib.sha256((out/'gaudi_moe_graph_views.so').read_bytes()).hexdigest())
except Exception as error:
    metadata.update(state='failed',error=str(error));raise
finally:save()
