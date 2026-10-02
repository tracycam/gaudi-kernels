"""Device diagnostic using only public Torch current-graph ops; no native pointers.

Run only under the coordinator's module7 lock/runner. Exporting intermediates
changes lifetimes/placement, so this is not the original serving recipe or TPS.
"""
import argparse,hashlib,json,os,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--torch-library',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
import torch
import habana_frameworks.torch.core as hc
assert os.environ.get('HLS_MODULE_ID')=='7' and os.environ.get('GAUDI_KERNELS_MODULE_ID')=='7'
a.out.mkdir(parents=True,exist_ok=False)
# The production tensor library was frozen in this exact model case.
assert hashlib.sha256(a.torch_library.read_bytes()).hexdigest()=='3c9a092b4461c0990200cb88fb04aab62afe031b4e3b0ce488d0177aac6c6e0a'
torch.ops.load_library(str(a.torch_library.resolve()))
index=json.loads((a.checkpoint/'model.safetensors.index.json').read_text())['weight_map'];prefix='model.layers.1.self_attn.qkv_proj.'
def read(name,begin,end):
 with (a.checkpoint/index[name]).open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];assert size<128*1024*1024
  entry=json.loads(f.read(size))[name];rows,cols=entry['shape']
  dtype,itemsize=(torch.float8_e4m3fn,1) if entry['dtype']=='F8_E4M3' else (torch.float32,4)
  assert entry['dtype'] in ('F8_E4M3','F32') and 0<=begin<end<=rows
  assert entry['data_offsets'][1]-entry['data_offsets'][0]==rows*cols*itemsize
  f.seek(8+size+entry['data_offsets'][0]+begin*cols*itemsize)
  data=bytearray(f.read((end-begin)*cols*itemsize));assert len(data)==(end-begin)*cols*itemsize
  return torch.frombuffer(data,dtype=dtype).reshape(end-begin,cols).clone()
w=read(prefix+'weight',6*3392,7*3392);s=read(prefix+'weight_scale_inv',6*27,7*27)
def sha(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
assert sha(w)=='f97b94d5737e9ba1277499fcb138fc331c3be89047afe1fb38847fd2b7c292a8'
assert sha(s)=='4c2a3096850d91c0a2fee0df1bd4d12ef60a8d31a3d962f35886f9d4e82ded02'
assert tuple(w.shape)==(3392,6144) and tuple(s.shape)==(27,48)
source=w.view(torch.uint8);mag=(source&127).to(torch.int16)
half=torch.where(mag>=16,mag-8,(mag>>1)+((mag&1)&((mag>>1)&1))).to(torch.uint8)
packed=((source&128)|half).reshape(3392,48,128).permute(1,0,2).contiguous().view(torch.float8_e4m3fn)
raw=bytearray((a.input_dir/'input-bf16.bin').read_bytes());x=torch.frombuffer(raw,dtype=torch.bfloat16).reshape(1,6144).clone()
witness=torch.load(a.input_dir/'first-qkv-rank6-layer1.pt',map_location='cpu',weights_only=False)
assert torch.equal(x,witness['input_bf16'])
x=x.to('hpu');pw=packed.to('hpu');sc=(s*2).to('hpu');bias=torch.zeros(3392,dtype=torch.float32,device='hpu');hc.mark_step();torch.hpu.synchronize()
# Full production QKV dimensions; expose outputs through ordinary tensor owners.
def apply():
 q,sa=torch.ops.gaudi_block_fp8.quant(x)
 partial=torch.ops.gaudi_block_fp8.batch_mm(q,pw)
 y=torch.ops.gaudi_block_fp8.reduce(partial,sa,sc,bias)
 return q,sa,partial,y
reports=[];retained=[]
for mode in ['eager','graph']:
 if mode=='eager':values=apply();hc.mark_step();torch.hpu.synchronize()
 else:
  graph=torch.hpu.HPUGraph();graph.capture_begin();values=apply();hc.mark_step();graph.capture_end();graph.replay();torch.hpu.synchronize();retained.append(graph)
 retained.append(values)
 host=[v.detach().cpu() for v in values]
 for name,t in zip(['quant-fp8','activation-scales-f32','partial-f32','output-bf16'],host):
  (a.out/f'{mode}-{name}.bin').write_bytes(t.contiguous().view(torch.uint8).numpy().tobytes())
 q,sa,part,y=host;expected=witness['actual_qkv'];different=int((y.view(torch.int16)!=expected.view(torch.int16)).sum())
 reports.append(dict(mode=mode,q_shape=list(q.shape),partial_shape=list(part.shape),output_shape=list(y.shape),all_finite=bool(torch.isfinite(part).all() and torch.isfinite(y).all()),same_as_captured_production_output=different==0,different_bf16_outputs=different,first_element_bits=int(y[0,3312].view(torch.int16))&65535))
metadata=dict(device_diagnostic=True,source_checkpoint_original_shard_hash=sha(w),source_scales_hash=sha(s),library_sha256=hashlib.sha256(a.torch_library.read_bytes()).hexdigest(),rows=reports,scope='Public eager/current-HPUGraph diagnostic with retained quant/scale/FP32 partial/output tensors. Their persistence can change scheduling/placement; no assertion of serving MME tree identity, no performance claim.')
(a.out/'summary.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps(metadata,indent=2));assert all(r['all_finite'] for r in reports)
