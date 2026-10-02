"""Specialize the existing full MoE gate; save exact executed source, no default changes."""
import os
from pathlib import Path

root=Path(__file__).resolve().parents[2]
source=(root/'benchmarks/moe_graph_views/device_moe_gate.py').read_text()
source=source.replace("a = p.parse_args()", "p.add_argument('--scale-tail-library',type=Path,required=True)\np.add_argument('--control-folded-library',type=Path,required=True)\np.add_argument('--fixed-down-library',type=Path,required=True)\na = p.parse_args()")
insert='''
for path in (a.scale_tail_library,a.control_folded_library,a.fixed_down_library):torch.ops.load_library(str(path.resolve(strict=True)))
def scale_tail_gp(*args):
    return (torch.ops.gaudi_gp_scale_tail.gp if capture_variant=='scale_tail' else torch.ops.gaudi_activation_folded.gp)(*args)
source=(out/'batch_ops-executed.py').read_text()
for old,new in [('gp_out=torch.ops.unified_batch.gp(gp,gs,x,table,ids)','gp_out=_scale_tail_gp(gp,gs,x,table,ids)'),('partial=torch.ops.unified_batch.down(dp,ds,gate,table,ids)','partial=torch.ops.gaudi_down_activation.broadcast(dp,ds,gate,table,ids)')]:
    assert source.count(old)==1;source=source.replace(old,new)
batch.__dict__['_scale_tail_gp']=scale_tail_gp
function=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='moe'];assert len(function)==1
exec(compile(ast.Module(body=function,type_ignores=[]),str(fixture/'executor/batch_ops.py'),'exec'),batch.__dict__)
(out/'batch_ops-scale-tail-executed.py').write_text(source)
'''
assert source.count('def sync():')==1
source=source.replace('def sync():',insert+'\ndef sync():')
fp32_gate='''
    result['direct_gp_fp32_checks']=[]
    direct_records=[]
    for count in ((1,) if a.quick else (1,2,8)):
        ax=(torch.randn(count,6144)*.1).bfloat16()
        ix=(torch.arange(count*8,dtype=torch.int32).reshape(count,8)%experts)
        axh,ixh=ax.to('hpu'),ix.to('hpu');sync()
        old=torch.ops.gaudi_activation_folded.gp(gp,gs,axh,table,ixh);sync();oldcpu=old.cpu()
        new=torch.ops.gaudi_gp_scale_tail.gp(gp,gs,axh,table,ixh);sync();newcpu=new.cpu()
        valid=same(oldcpu,newcpu) and bool(torch.isfinite(oldcpu).all())
        result['direct_gp_fp32_checks'].append(dict(rows=count,checked_values=oldcpu.numel()*2,bitwise_equal=valid))
        direct_records.append(dict(x=ax,ids=ix,old=oldcpu,candidate=newcpu))
        torch.save(direct_records,out/'direct-gp-fp32.pt');save();assert valid
'''
needle='    with torch.inference_mode():'
assert source.count(needle)==1
source=source.replace(needle,fp32_gate+'\n'+needle)
needle='    with torch.inference_mode():'
assert source.count(needle)==1
source=source.replace(needle,"    cases=[('compact',1,False),('compact',2,True),('compact',8,True)]\n    if a.quick:cases=cases[:1]\n    variants=[('folded_control','0','0'),('scale_tail','0','0')]\n    result['scope']='GP scale-tail versus frozen production folded GP; production down_vec fixed, original nonlinear/combine order'\n"+needle)
out=Path(os.environ['PROBE_OUT']);(out/'full-gate-executed.py').write_text(source)
exec(compile(source,str(root/'benchmarks/moe_graph_views/device_moe_gate.py'),'exec'),globals())
