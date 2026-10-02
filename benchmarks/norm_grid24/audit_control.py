"""Protocol fault injections; CPU stand-ins do not qualify device arithmetic."""
import argparse,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'python'))
from gaudi_kernels import norm_grid24_audit as module,norm_grid24_binding as binding
from gaudi_kernels import block_fp8_fp32_contract as contract,fp32_artifact_binding as implementation
records=[]
paths={'GK_NORM_GRID24_TPC_LIBRARY':a.archive/'remote/tpc-a/libgaudi_norm_grid24_tpc.so',
 'GK_NORM_GRID24_TORCH_LIBRARY':a.archive/'remote/builds/torch-a/gaudi_norm_grid24_torch.so',
 'GK_NORM_TPC_LIBRARY':a.archive/'remote/libraries/libgaudi_residual_rmsnorm_tpc.so',
 'GK_NORM_TORCH_LIBRARY':a.archive/'remote/libraries/gaudi_residual_rmsnorm_torch.so'}
with patch.dict(os.environ,{k:str(v.resolve())for k,v in paths.items()}):
 real=binding.verify(require_loaded=False);assert real['guid']=='gk_norm_block128_grid24_v1'
 with tempfile.TemporaryDirectory()as temp:
  bad=Path(temp)/'unknown.so';bad.write_bytes(paths['GK_NORM_GRID24_TPC_LIBRARY'].read_bytes()+b'changed')
  with patch.dict(os.environ,GK_NORM_GRID24_TPC_LIBRARY=str(bad)):
   try:binding.verify(require_loaded=False)
   except ValueError:records.append(dict(case='new_bytes_cannot_self_qualify',passed=True))
   else:raise AssertionError('unqualified producer accepted')
x=torch.ones(1,6144,dtype=torch.bfloat16);r=torch.zeros_like(x);g=torch.ones(6144,dtype=torch.bfloat16)
q=torch.ones(48,1,128,dtype=torch.float8_e4m3fn);sa=torch.ones(48,1,1)
y=torch.ones(1,3392,dtype=torch.bfloat16);fault=None;events=[]
def old_norm(x,r,g,e):events.append('qualified_old_norm');return x+r,x.clone()
def old_quant(yy):events.append('old_quant');return q.clone(),sa.clone()
def producer(*args):
 events.append('actual_grid24');rr=x+r;qq=q.clone();ss=sa.clone()
 if fault=='q':qq.view(torch.uint8)[0,0,0]^=1
 if fault=='scale':ss[0,0,0]=torch.nextafter(ss[0,0,0],torch.tensor(float('inf')))
 if fault=='residual':rr[0,0]+=1
 return rr,qq,ss
def consumer(qq,ss,prepared,*,bias,reduce_op,audit_tensors):
 events.append('actual_grid_consumer');audit_tensors.update(q_native=qq,activation_scales=ss,partial=torch.ones(48,1,3392),output=y.clone());return y.clone()
state=SimpleNamespace(audit_enabled=True,audit_frame={'positions':[128],'input_ids':[1]},audit_contract='fp32_arithmetic_v1',audit_records=[],reduction_policy='neumaier_isa_fp32',policy='decode_a8_bf16_fp32',audit_tag='fixture')
layer=SimpleNamespace(prefix='model.layers.1.self_attn.qkv_proj',bias=None,_gk_oracle_weight=torch.ones(1),_gk_oracle_scales=torch.ones(1),_gk_oracle_layout='fixture',_gk_block_fp8=SimpleNamespace(rounded_half_values=0,checkpoint_weight_sha256='a'*64,checkpoint_scales_sha256='b'*64))
with patch.object(module,'residual_rmsnorm_bf16',old_norm),patch.object(torch.ops.gaudi_block_fp8,'quant',old_quant,create=True),patch.object(torch.ops.gaudi_norm_grid24,'block128',producer,create=True),patch.object(module,'linear_block_fp8_quantized',consumer),patch.object(contract,'build_reference',return_value={}),patch.object(contract,'audit',return_value={'passed':True,'numerical_passed':True}),patch.object(implementation,'verify_artifacts',return_value=None),patch.object(binding,'verify',return_value=real):
 for fault in [None,'q','scale','residual']:
  events.clear();module.record(state,layer,x,r,g,1e-6,x+r,y,None);row=state.audit_records[-1]
  assert row['fp32_contract']['passed']==(fault is None)
  assert row['norm_grid24_producer']['passed']==(fault is None)
  assert events==['qualified_old_norm','old_quant','actual_grid24','actual_grid_consumer']
  records.append(dict(case='audit_'+str(fault),classification=row['fp32_contract'].get('classification','PASS'),producer_pass=row['norm_grid24_producer']['passed']))
 state.audit_enabled=False;before=len(state.audit_records);events.clear();module.record(state,layer,x,r,g,1e-6,x+r,y,None);assert not events and len(state.audit_records)==before
records.append(dict(case='ordinary_path_no_witness_or_readback',passed=True))
(a.output/'result.json').write_text(json.dumps(dict(status='PASS_CPU_PROTOCOL_ONLY',records=records,scope='strict producer-relation and pin failure injection; arithmetic backend intentionally stubbed'),indent=2)+'\n');print(json.dumps(records))
