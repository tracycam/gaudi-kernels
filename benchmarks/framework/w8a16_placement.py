"""Test whether hidden public CustomOp outputs stay transient inside HPU Graph.

Numerical pass alone does NOT qualify this path. Compiled weight placement must
also pass. Keeping a decoded PyTorch tensor outside capture would invalidate it.
"""
import json
import os
from pathlib import Path
import sys
out=Path(os.environ['PROBE_OUT']);(out/'graphs').mkdir()
os.environ.update(ENABLE_EXPERIMENTAL_FLAGS='true',GRAPH_VISUALIZATION='1',
    GRAPH_VISUALIZATION_DIR=str(out/'graphs'),SRAM_SLICER_GRAPH_VISUALIZATION='1',
    DUMP_POST_GRAPHS=str(out/'post_graph.json'))
import torch
import habana_frameworks.torch
import habana_frameworks.torch.core as hc
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
from gaudi_kernels.torch_linear import load_extension
load_extension(root/'artifacts/builds/torch-c/gaudi_kernels_torch.so')
torch.manual_seed(27);torch.set_num_threads(4)


def sync():hc.mark_step();torch.hpu.synchronize()


def linear(x,w,ws,b):
    decoded=torch.ops.gaudi_kernels._fp8_decode(w)
    partial=torch.ops.gaudi_kernels._bf16_mm_f32(x,decoded)
    return torch.ops.gaudi_kernels._fp8_epilogue16(partial,ws,b)


records=[]
for m,n,k in [(1,129,257),(64,1024,2048),(1,8192,8192)]:
    xc=torch.randn(m,k).bfloat16();wc=(torch.randn(n,k)*16).clamp(-240,240).to(torch.float8_e4m3fn)
    ws=torch.ones(n).to('hpu');bc=torch.randn(n)*.1
    x,w,b=xc.to('hpu'),wc.to('hpu'),bc.to('hpu');sync()
    graph=torch.hpu.HPUGraph();stream=torch.hpu.Stream()
    with torch.hpu.graph(graph,stream=stream):y=linear(x,w,ws,b)
    sync();graph.replay();sync();yc=y.cpu()
    ref=(xc.double()@wc.double().T+bc.double()).bfloat16()
    rel=float((yc.double()-ref.double()).norm()/ref.double().norm())
    rec={'M':m,'N':n,'K':k,'relative_l2_vs_once_rounded_fp64':rel,'full_output_elements':m*n,
         'numerical_pass':bool(torch.isfinite(yc).all()) and rel<.001,'placement_pass':'requires_postgraph_audit'}
    records.append(rec);print(json.dumps(rec),flush=True);assert rec['numerical_pass']
(out/'result.json').write_text(json.dumps(records,indent=2)+'\n')
