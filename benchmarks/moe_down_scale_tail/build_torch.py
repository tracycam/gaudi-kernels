"""Public runtime bridge CPU build only, from committed/exported source identity."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
name='csrc/torch/moe_down_scale_tail.cpp';source=root/name
if(root/'.git').exists():
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+name],cwd=root)
else:
 identity=json.loads((root/'source-identity.json').read_text());commit=identity['git_commit'];assert hashlib.sha256(source.read_bytes()).hexdigest()==identity['files_sha256'][name]
if os.getenv('PT_HPU_LAZY_MODE','1')!='1':raise ValueError('public lazy bridge requires PT_HPU_LAZY_MODE=1')
os.environ['PT_HPU_LAZY_MODE']='1';os.environ.setdefault('MAX_JOBS','2')
import habana_frameworks.torch as ht
from torch.utils.cpp_extension import load
h=Path(ht.__file__).parent;(out/'source.cpp').write_bytes(source.read_bytes())
load(name='gaudi_down_scale_tail',sources=[str(source)],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
(out/'build.json').write_text(json.dumps(dict(source_commit=commit,device_acquired=False,TPC_compiler_invoked=False,library_sha256=hashlib.sha256((out/'gaudi_down_scale_tail.so').read_bytes()).hexdigest()),indent=2)+'\n')
