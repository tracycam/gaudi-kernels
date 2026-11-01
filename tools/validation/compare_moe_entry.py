"""Compare old/new MoE entry operator sequences using metadata-only tensors.

Kernel callbacks are spies. This proves dispatch/cast/materialization ordering
for the fixtures, not device numerics, DMA traffic or model performance.
"""
import argparse
import ast
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def function(source, name, namespace):
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<MoE entry audit>','exec'),namespace)
    return namespace[name]


def trace(source, *, rows, mode, gp, down, masked, debug):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    calls=[]
    def describe(value):
        if isinstance(value,torch.Tensor):return {'shape':list(value.shape),'dtype':str(value.dtype)}
        if isinstance(value,(tuple,list)):return [describe(v) for v in value]
        if isinstance(value,dict):return {k:describe(v) for k,v in value.items()}
        return str(value)
    class Observe(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            calls.append([str(func),describe(args),describe(kwargs or {})])
            return func(*args,**(kwargs or {}))
    def empty(shape,dtype=torch.float32):return torch.empty(shape,dtype=dtype,device='meta')
    def kernel(name,fn):
        def call(*args):
            calls.append([name,describe(args)])
            return fn(*args)
        return call
    def prep(x,ids,blocks,splits,r):
        tasks=ids.numel()*blocks*splits
        return empty((tasks,x.shape[-1]//splits),torch.bfloat16),empty((1,tasks),torch.int32)
    def projection(w,s,x,lut,ids):return empty((1,ids.numel()*3*512))
    def gate(p,ids):return empty((ids.numel(),256),torch.bfloat16)
    def down_op(w,s,x,lut,ids):return empty((1,ids.numel()*6144))
    def broadcast(p,ids):return empty((ids.numel()*12,256),torch.bfloat16),empty((1,ids.numel()*12),torch.int32)
    native=SimpleNamespace(prep=kernel('prep',prep),gemv=kernel('gemv',lambda w,s,x,lut,ids:empty((1,x.shape[0]*512))))
    batch=SimpleNamespace(gp=kernel('gp',projection),gate=kernel('gate',gate),
                          down=kernel('down',down_op),gate_broadcast=kernel('gate_broadcast',broadcast),
                          masked_gemv=kernel('masked_gemv',lambda w,s,x,lut,ids:empty((1,x.shape[0]*512))))
    facade=SimpleNamespace(**{key:getattr(torch,key) for key in ('bfloat16','float32','int32','bool','where','full_like','zeros_like')})
    facade.ops=SimpleNamespace(native_mxfp4=native,unified_batch=batch,
        gaudi_activation_folded=SimpleNamespace(gp=kernel('folded_gp',projection)),
        gaudi_down_activation=SimpleNamespace(broadcast=kernel('vector_down',down_op)))
    precision=ModuleType('gaudi_kernels.serving.executor.precision_ops')
    precision.combine=kernel('fp32_combine',lambda partial,routing,directions,n,r:empty((routing.shape[0],n)))
    scale=ModuleType('gaudi_kernels.serving.executor.gp_scale_tail_runtime')
    scale.operator=lambda:kernel('scale_tail_gp',projection)
    namespace={'torch':facade,'_GP_POLICY':gp,'_DOWN_POLICY':down,
               '_GP_COUNTS':{gp:0},'_DOWN_COUNTS':{down:0}}
    operation=function(source,'moe',namespace)
    args=(empty((rows,6144),torch.bfloat16),empty((rows,8),torch.int64),
          empty((rows,8),torch.bfloat16),*[empty((1,256),torch.uint8) for _ in range(4)],
          empty((1,512),torch.bfloat16),empty((2,256),torch.uint8))
    mask=empty((rows,8),torch.bool) if masked else None
    with patch.dict('sys.modules',{precision.__name__:precision,scale.__name__:scale}),Observe():
        try:
            output=operation(*args,mode=mode,route_mask=mask,debug=debug)
            result={'output':describe(output)}
        except ValueError as error:
            # Rejected sorted mode and empty debug retain their fail-fast contract.
            result={'error':str(error)}
    return {'calls':calls,'result':result,'gp_counts':namespace['_GP_COUNTS'],
            'down_counts':namespace['_DOWN_COUNTS']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--before',type=Path,required=True);p.add_argument('--after',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checks=[]
    variants=[('compact',gp,down,False,False)
              for gp in ('legacy','vector_fetch_folded','vector_fetch_scale_tail')
              for down in ('legacy','vector_fetch')]
    variants += [('broadcast','legacy','legacy',False,True),
                 ('compact','vector_fetch_scale_tail','vector_fetch',True,True),
                 ('sorted','legacy','legacy',False,False),
                 ('compact','legacy','legacy',False,True)]
    for rows in (0,1,2,8,512,2048,4096):
        for mode,gp,down,masked,debug in variants:
            case=dict(rows=rows,mode=mode,gp=gp,down=down,masked=masked,debug=debug)
            old=trace(a.before.read_text(),**case);new=trace(a.after.read_text(),**case)
            check={'case':case,'equal':old==new}
            if old!=new:check.update(before=old,after=new)
            checks.append(check)
    report={'pass':all(c['equal'] for c in checks),'cases':len(checks),'checks':checks,
            'scope':'metadata-only operator/cast/materialization order; not hardware performance'}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
    if not report['pass']:raise SystemExit(1)


if __name__=='__main__':main()
