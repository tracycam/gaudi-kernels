#!/usr/bin/env python3
"""Read-only production-path, byte and saved-performance audit. No device APIs."""
import argparse,hashlib,json,subprocess
from pathlib import Path
from planner import Capacity,PER_EXPERT_BYTES
p=argparse.ArgumentParser();p.add_argument('--canonical',type=Path,required=True);p.add_argument('--executor',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[2];identities={}
def read(relative):
 path=a.canonical/relative;raw=path.read_bytes();identities[relative]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
identity={}
for name in ('python/gaudi_kernels/production_integration.py','python/gaudi_kernels/block_fp8.py','python/gaudi_kernels/mxfp4_moe_graph.py','python/gaudi_kernels/mxfp4_moe_plan.py','csrc/ops/mxfp4_moe_grouped.cpp','csrc/ops/mxfp4_moe_bucket.cpp'):
 identity[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.executor,text=True).strip()
for name in ('executor/batch_ops.py','executor/native_model.py','executor/moe_dispatch_runtime.py','executor/native_service_test.py','executor/native_backend.py','executor/precision_runtime.py'):
 identity['executor:'+name]=hashlib.sha256(subprocess.check_output(['git','show',commit+':'+name],cwd=a.executor)).hexdigest()
rows=[]
for m in (1,2,8,32,128,512,513):
 g,n,k,r=48,3392,6144,8
 rows.append(dict(M=m,production_qkv='explicit block A8 only for selected mixed policy' if m==1 else 'BF16 activation / FP32 scale decode / BF16 MME / FP32 output',
  production_moe_baseline='compact TPC' if m<=1 else 'broadcast per-route TPC',
  production_moe_compact8='compact TPC' if m<=8 else 'broadcast per-route TPC',
  qkv=dict(original_weight_bytes=n*k,prepared_scale_bytes=27*g*4,activation_bf16_bytes=2*m*k,decoded_bf16_weight_bytes=2*n*k,
   a16_mme_fp32_output_bytes=4*m*n,final_bf16_bytes=2*m*n,a8_mme_input_q_bytes=g*m*128,a8_scales_bytes=g*m*4,
   a8_fp32_block_partial_bytes=4*g*m*n,a8_to_a16_partial_ratio=g),
  moe=dict(resident_original_bytes=384*PER_EXPERT_BYTES,per_route_requested_original_bytes=m*r*PER_EXPERT_BYTES,
   max_unique_original_bytes=min(384,m*r)*PER_EXPERT_BYTES,hot8_unique_original_bytes=r*PER_EXPERT_BYTES,
   uniform_average_rows_per_expert=m*r/384,gp_tasks=3*m*r,down_tasks=12*m*r,
   nominal_gp_task_waves_24tpc=(3*m*r+23)//24,nominal_down_task_waves_24tpc=(12*m*r+23)//24,
   original_activation_bytes=2*m*6144,broadcast_gp_activation_bytes=m*r*3*2048*2,
   broadcast_down_activation_bytes=m*r*12*256*2,compact_gate_bytes=m*r*256*2,
   gp_partial_bytes=m*r*3*512*4,down_partial_bytes=m*r*6144*4),
  interpretation='logical tensors/ISA requested intervals, not physical HBM or measured occupancy'))
block=read('artifacts/builds/block-fp8-framework/device-raw/block-qualified-b/audit.json')
perf=[dict(case=x['case'],event_us=x['event_us'],wall_us=x['wall_us'],effective_TFLOPS=x['effective_TFLOPS'],placement=x['placement']) for x in block['records'] if x['case']['name'].startswith('qkv-a16')]
smokes=[]
for m in (1,2,512):
 d=read(f'artifacts/builds/block-fp8-3d/device-raw/production-3d-c/m{m}-bf16_fp32/result.json');smokes.append({k:d[k] for k in ('name','status','input_shape','checked_outputs','ten_replay_wall_us','timing_scope')})
model=read('artifacts/builds/production-integration/model-runs/production-fp32-70-d/result.json')
captures=[]
for run in model['runs']:
 captures.append(dict(run=run['name'],policy=run['policy'],ranks=[dict(rank=s['rank'],dispatch=s.get('moe_dispatch')) for s in run.get('production_state',[])]))
util={}
for label in ('gateup','down','gateup-m257'):
 d=read('evidence/mxfp4-coverage/audit-'+label+'.json');util[label]=d['mme']
report={'scope':'offline production/static-byte/saved-result audit; no new performance or qualification',
 'kernel_source_base':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'executor_source_commit':commit,'source_sha256':identity,'record_sha256':identities,
 'dispatch_rows':rows,'historical_qkv_a16_bf16_scale':perf,'current_qkv_fp32_scale_capture_smoke':smokes,
 'mxfp4_single_expert_compiler_estimates':util,'model70d':{'status':model['status'],'candidate_accepted':model['candidate_accepted'],'config':model['config'],'captures':captures,'batch_or_32k_completed':False},
 'critical_limits':['historical QKV performance uses BF16 scale math, not current FP32 production math','MXFP4 M is token rows, grouped-MME performance depends on per-expert M_e','No production grouped-MME threshold exists','70-d ended before batch/32K; do not backfill acceptance']}
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'rows':len(rows),'historical_qkv_records':len(perf),'model_status':model['status'],'source_files':len(identity),'raw_records':len(identities)}))
