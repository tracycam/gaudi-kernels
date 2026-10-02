"""Local stub Meta or target-runtime CPU-only public bridge build; no acquire."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ['PT_HPU_LAZY_MODE']='1';os.environ['MAX_JOBS']='2'
import torch
from torch.utils.cpp_extension import load
from meta_gate import run
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--offline-meta',action='store_true');p.add_argument('--bridge-include',type=Path);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
sources=[str(root/'csrc/torch/moe_down_combine_issue.cpp')]
if a.offline_meta:
    assert a.bridge_include
    sources.append(str(root/'benchmarks/mxfp4_moe_graph/meta_bridge_stub.cpp'))
    includes=[str(a.bridge_include)];links=[]
else:
    import habana_frameworks.torch as ht
    h=Path(ht.__file__).parent;includes=[str(h/'include')];links=[f'-L{h}/lib',f'-Wl,-rpath,{h}/lib','-lhabana_pytorch_plugin']
report=dict(device_used=False,offline_Meta_stub=a.offline_meta,torch=torch.__version__,source_hashes={str(Path(f).relative_to(root)):hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},state='BUILDING')
try:
    load(name='gaudi_down_combine_isa',sources=sources,extra_include_paths=includes,extra_cflags=['-O2'],extra_ldflags=links,build_directory=str(a.output),is_python_module=False,verbose=True)
    report.update(state='BUILT_CPU_META_PASS',meta=run(torch),library_sha256=hashlib.sha256((a.output/'gaudi_down_combine_isa.so').read_bytes()).hexdigest())
except BaseException as error:report.update(state='FAIL',error=repr(error));raise
finally:(a.output/'build.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='meta'}))
