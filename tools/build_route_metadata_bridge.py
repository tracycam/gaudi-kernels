"""Build the metadata v3 public bridge only; preserve all qualified TPC bytes."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
name='csrc/torch/moe_route_metadata_v3.cpp';source=root/name
if(root/'source-identity.json').exists():
 identity=json.loads((root/'source-identity.json').read_text());commit=identity['git_commit'];assert hashlib.sha256(source.read_bytes()).hexdigest()==identity['files_sha256'][name]
else:
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip();assert source.read_bytes()==subprocess.check_output(['git','show',commit+':'+name],cwd=root)
os.environ['PT_HPU_LAZY_MODE']='1';os.environ.setdefault('MAX_JOBS','2')
import habana_frameworks.torch as ht
from torch.utils.cpp_extension import load
h=Path(ht.__file__).parent
report=dict(source_commit=commit,TPC_compiler_invoked=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),compiler=subprocess.check_output(['c++','--version'],text=True),flags=['-O2'])
(out/'source.cpp').write_bytes(source.read_bytes())
try:
 load(name='gaudi_route_metadata_v3',sources=[str(source)],extra_include_paths=[str(h/'include')],extra_cflags=['-O2'],extra_ldflags=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin'],build_directory=str(out),is_python_module=False,verbose=True)
 report['status']='CPU_BUILT_NOT_DEVICE_QUALIFIED'
finally:
 report['files_sha256']={str(x.relative_to(out)):hashlib.sha256(x.read_bytes()).hexdigest()for x in out.rglob('*')if x.is_file()};(out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
