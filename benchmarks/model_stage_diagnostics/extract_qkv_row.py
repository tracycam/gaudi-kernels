"""Read-only checkpoint extraction. Standard library only: no HPU/runtime imports.

One original TP shard is hashed to bind it to production loading; only one
weight row and its group-aware FP32 scale row are returned as base64 JSON.
"""
import argparse,base64,hashlib,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--layer',type=int,required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--local-row',type=int,required=True);p.add_argument('--include-shard',action='store_true');a=p.parse_args()
config_bytes=(a.checkpoint/'config.json').read_bytes();config=json.loads(config_bytes);config=config.get('text_config',config)
swa=config['hybrid_layer_pattern'][a.layer]==1
heads=config['swa_num_attention_heads' if swa else 'num_attention_heads'];kv=config['swa_num_key_value_heads' if swa else 'num_key_value_heads'];head_dim=config['swa_head_dim' if swa else 'head_dim'];v_dim=config['swa_v_head_dim' if swa else 'v_head_dim'];k=config['hidden_size']
assert kv==8 and heads%kv==0 and 0<=a.rank<8
n=(heads//kv+1)*head_dim+v_dim;g=(k+127)//128;nb=(n+127)//128;assert 0<=a.local_row<n
index_bytes=(a.checkpoint/'model.safetensors.index.json').read_bytes();index=json.loads(index_bytes)['weight_map'];prefix=f'model.layers.{a.layer}.self_attn.qkv_proj.'
def read_rows(name,first,count,expected_shape,itemsize):
 path=a.checkpoint/index[name]
 with path.open('rb') as f:
  hsize=struct.unpack('<Q',f.read(8))[0];assert hsize<128*1024*1024
  header=json.loads(f.read(hsize));entry=header[name];assert entry['shape']==expected_shape,entry
  assert entry['data_offsets'][1]-entry['data_offsets'][0]==expected_shape[0]*expected_shape[1]*itemsize
  f.seek(8+hsize+entry['data_offsets'][0]+first*expected_shape[1]*itemsize);data=f.read(count*expected_shape[1]*itemsize)
  assert len(data)==count*expected_shape[1]*itemsize
 return data,dict(file=str(path),tensor=name,header=entry,first_row=first,rows=count)
w,wm=read_rows(prefix+'weight',a.rank*n,n,[8*n,k],1);s,sm=read_rows(prefix+'weight_scale_inv',a.rank*nb,nb,[8*nb,g],4)
assert wm['header']['dtype']=='F8_E4M3' and sm['header']['dtype']=='F32',(wm,sm)
row=w[a.local_row*k:(a.local_row+1)*k];local_scale=a.local_row//128;scale=s[local_scale*g*4:(local_scale+1)*g*4]
print(json.dumps(dict(checkpoint=str(a.checkpoint),config_sha256=hashlib.sha256(config_bytes).hexdigest(),index_sha256=hashlib.sha256(index_bytes).hexdigest(),layer=a.layer,rank=a.rank,local_row=a.local_row,global_weight_row=a.rank*n+a.local_row,global_scale_row=a.rank*nb+local_scale,group_rows=n,scale_rows_per_group=nb,k=k,weight_shard_sha256=hashlib.sha256(w).hexdigest(),scale_shard_sha256=hashlib.sha256(s).hexdigest(),weight_row_b64=base64.b64encode(row).decode(),scales_row_b64=base64.b64encode(scale).decode(),weight_read=wm,scale_read=sm,device_accessed=False,full_weight_shard_b64=base64.b64encode(w).decode() if a.include_shard else None,full_scale_shard_b64=base64.b64encode(s).decode() if a.include_shard else None),indent=2))
