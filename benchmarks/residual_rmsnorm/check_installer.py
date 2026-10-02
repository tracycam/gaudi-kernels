"""CPU-only structure audit; fake HPU metadata is never device evidence."""
import ast,json,sys,types
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'python'))
class FakeTensor:
 def __init__(self,shape,*,name='tensor',dtype=torch.bfloat16,contiguous=True,grad=False):
  self.shape=tuple(shape);self.name=name;self.dtype=dtype;self.device=types.SimpleNamespace(type='hpu',index=0);self.requires_grad=grad;self.contiguous=contiguous;self.views=[];self.freezes=0
 @property
 def ndim(self):return len(self.shape)
 def numel(self):
  result=1
  for n in self.shape:result*=n
  return result
 def is_contiguous(self):return self.contiguous
 def requires_grad_(self,value):self.requires_grad=value;self.freezes+=1;return self
 def __add__(self,other):
  assert self.shape==other.shape;return FakeTensor(self.shape,name="vendor_residual")
 def float(self):return FakeTensor(self.shape,name=self.name+'_f32',dtype=torch.float32)
 def bfloat16(self):return FakeTensor(self.shape,name=self.name+'_bf16',dtype=torch.bfloat16)
 def reshape(self,shape):return self.view(shape)
 def view(self,*shape):
  if len(shape)==1 and isinstance(shape[0],(tuple,list)):shape=shape[0]
  shape=list(shape)
  if -1 in shape:shape[shape.index(-1)]=self.numel()//(-torch.tensor(shape).prod().item())
  self.views.append(tuple(shape));return FakeTensor(shape,name=self.name)
class HPURMSNorm:
 def __init__(self,h=129):self.weight=FakeTensor((h,),name='gamma',grad=True);self.variance_epsilon=1e-6;self.variance_size_override=None;self._forward_method=self.forward_oot
 def forward_oot(self,x,residual=None):return ('vendor',x,residual)
 def forward(self,x,residual=None):return self._forward_method(x,residual)
class HPUGemmaRMSNorm(HPURMSNorm):
 def forward_oot(self,x,residual=None):return ('gemma',x,residual)
for name in ['vllm_gaudi','vllm_gaudi.ops','vllm_gaudi.ops.hpu_layernorm']:
 sys.modules[name]=types.ModuleType(name);sys.modules[name].__path__=[]
sys.modules['vllm_gaudi.ops.hpu_layernorm'].HPURMSNorm=HPURMSNorm
sys.modules['vllm_gaudi.ops.hpu_layernorm'].HPUGemmaRMSNorm=HPUGemmaRMSNorm
sys.modules['vllm_gaudi.extension']=types.ModuleType('vllm_gaudi.extension')
sys.modules['vllm_gaudi.extension.kernels']=types.ModuleType('vllm_gaudi.extension.kernels')
class VendorNorm:
 @staticmethod
 def apply(x,w,e):
  assert x.dtype==w.dtype
  return FakeTensor(x.shape,name='vendor_f32' if x.dtype==torch.float32 else 'vendor',dtype=x.dtype)
sys.modules['vllm_gaudi.extension.kernels'].rms_norm=lambda:VendorNorm
original=HPURMSNorm.forward_oot;old=HPURMSNorm();gemma=HPUGemmaRMSNorm()
from gaudi_kernels import vllm_norm as adapter
assert HPURMSNorm.forward_oot is original and adapter.get_norm_policy()=='vendor'
model=types.SimpleNamespace(modules=lambda:iter([old,gemma]))
report=adapter.install_vllm_norm(model=model)
assert report['prepared']=={'matched':1,'gamma_frozen':1,'dispatch_rebound':1,'non_oot_dispatch_retained':0}
assert not old.weight.requires_grad and gemma.weight.requires_grad
new=HPURMSNorm();assert new.weight.freezes==1 and not new.weight.requires_grad
calls=[]
def kernel(x,r,w,eps):calls.append((x,r,w,eps));return FakeTensor(r.shape,name='residual'),FakeTensor(x.shape,name='norm')
adapter.residual_rmsnorm_bf16=kernel
adapter.pure_rmsnorm_bf16=lambda x,w,e:FakeTensor(x.shape,name='pure_fp32_bf16')
adapter.reshape_bf16_graph=lambda x,shape:FakeTensor(shape,name=x.name)
x=FakeTensor((2,129),name='x');r=FakeTensor((2,129),name='r')
assert old.forward(x,r)[0].name=='vendor';adapter.set_norm_policy('fp32')
y,z=old.forward(x,r);assert y.name=='norm' and z.name=='residual' and calls[-1][0] is x and calls[-1][1] is r and not x.views and not r.views
for xs,rs in [((2,3,129),(2,3,129)),((2,3,129),(6,129)),((6,129),(2,3,129))]:
 xx=FakeTensor(xs);rr=FakeTensor(rs);yy,zz=new.forward(xx,rr);assert yy.shape==xs and zz.shape==rs and calls[-1][0].shape==(6,129) and calls[-1][1].shape==(6,129)
for shape in [(2,129),(1,2,129)]:
 pure=FakeTensor(shape);before=len(calls);value=new.forward(pure,None)
 assert value.name=='pure_fp32_bf16' and value.shape==shape and value.dtype==torch.bfloat16 and len(calls)==before and not pure.views
for xx,rr in [(FakeTensor((2,129),dtype=torch.float32),None),(FakeTensor((2,129),dtype=torch.float32),r),(FakeTensor((2,129),contiguous=False),r),(FakeTensor((2,129),grad=True),r),(FakeTensor((2,8193)),FakeTensor((2,8193)))]:
 before=len(calls);value=new.forward(xx,rr)[0];assert (value=='vendor' if rr is None else value.name=='vendor') and len(calls)==before
adapter.set_norm_policy('vendor')
for xs,rs in [((2,129),(2,129)),((2,3,129),(2,3,129)),((2,3,129),(6,129))]:
 xx=FakeTensor(xs);rr=FakeTensor(rs);yy,zz=new.forward(xx,rr);assert yy.shape==xs and zz.shape==rs and not xx.views and not rr.views
assert adapter._vendor_shape(x,x.shape) is x
adapter.set_norm_policy('fp32')
assert gemma.forward(x,r)[0]=='gemma'
new.weight=FakeTensor((129,),grad=True)
try:new.forward(x,r)
except RuntimeError:pass
else:raise AssertionError('replaced grad gamma silently dispatched')
prep=adapter.prepare_vllm_norm(types.SimpleNamespace(modules=lambda:iter([new])));assert prep['gamma_frozen']==1;new.forward(x,r);assert new.weight.freezes==1
assert adapter.set_norm_policy('vendor')['captured_graph_invalidation_required'];assert new.forward(x,r)[0].name=='vendor'
try:adapter.set_norm_policy('auto')
except ValueError:pass
else:raise AssertionError('implicit policy accepted')
source=(root/'python/gaudi_kernels/vllm_norm.py').read_text();tree=ast.parse(source);forward=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_forward')
for n in ast.walk(forward):
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute):assert n.func.attr not in ['detach','clone','contiguous','to','cpu','item','requires_grad_']
assert 'quant' not in ast.unparse(forward)
result={'status':'PASS_CPU_STRUCTURE_ONLY','device_verified':False,'checks':['default_vendor_no_import_patch','resident_bound_dispatch_rebound','constructor_gamma_frozen_once','Gemma_excluded','vendor_same_shape_preserves_producer_identity','vendor_different_shape_graph_owned','2D_identity_no_views','3D_and_mixed_shapes_restored','pure_fp32_input_gamma_and_final_bf16','pure_no_synthetic_add_or_reshape','unsupported_inputs_vendor','replaced_gamma_requires_prepare','policy_switch_requires_external_graph_invalidation','no_hotpath_copy_detach_freeze_or_quant']}
print(json.dumps(result,indent=2))
