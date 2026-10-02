"""Locate real-input GP/gate differences using independent FP32 order witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);root=a.archive;case=root/'results/checkpoint-gp-stages-a'
manifest=json.loads((root/'remote-sha256.json').read_text());digests={}
def checked(path):
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==manifest['files'][str(path.relative_to(root))]['sha256']
    digests[str(path.relative_to(root))]=digest;return path
def load(path):return torch.load(checked(path),map_location='cpu',weights_only=False)
result=json.loads(checked(case/'result.json').read_text())
assert result['status']=='DIAGNOSTIC_PREFILL_STAGES_WITH_COMPLETE_OUTPUT_RELATION'
data=load(root/'fixtures/inputs-rank0.pt');old=load(case/'broadcast-stages.pt');new=load(case/'routed_pure-stages.pt')
for mode,values in [('broadcast',old),('routed_pure',new)]:
    ref=load(root/'references'/(mode+'.pt'))
    for key in ('y','consumer'):assert torch.equal(values[key].view(torch.uint8),ref[key].view(torch.uint8))
assert new['status'].tolist()==[0]
n=torch.arange(512);physical=(n//128)*128+(n%2)*64+(n%128)//2
splits=old['gp'].reshape(4096,3,512).index_select(-1,physical)
old_gp=torch.zeros(4096,512)
for s in range(3):old_gp=old_gp+splits[:,s]
new_gp=new['gp'].reshape(-1,512).index_select(0,new['inverse'].long())
old_gate=old['gate'].reshape(4096,12,256)
assert torch.equal(old_gate,old_gate[:,0:1].expand_as(old_gate))
old_gate=old_gate[:,0].contiguous()
new_gate=new['gate'].reshape(-1,256).index_select(0,new['inverse'].long())
def metrics(x,y):
    delta=(x.float()-y.float()).abs();dt=torch.int32 if x.dtype==torch.float32 else torch.int16
    return dict(words=x.numel(),different_words=int((x.contiguous().view(dt)!=y.contiguous().view(dt)).sum()),
                max_abs=float(delta.max()),relative_l2=float(delta.norm()/x.float().norm()))
rounded_old,rounded_new=old_gp.bfloat16(),new_gp.bfloat16()
gate_changed=(old_gate.view(torch.int16)!=new_gate.view(torch.int16))
changed_inputs=(rounded_old[:,:256].view(torch.int16)!=rounded_new[:,:256].view(torch.int16))|(rounded_old[:,256:].view(torch.int16)!=rounded_new[:,256:].view(torch.int16))

# Explicit witnesses include largest raw deltas, first BF16-boundary changes and
# fixed-seed samples. They are not a full independent arithmetic qualification.
flat=(old_gp-new_gp).abs().reshape(-1)
points=[divmod(int(i),512)for i in flat.topk(16).indices]
points += [(int(r),int(c))for r,c in (rounded_old.view(torch.int16)!=rounded_new.view(torch.int16)).nonzero()[:16]]
rng=np.random.default_rng(280928);points += [divmod(int(i),512)for i in rng.integers(0,4096*512,32)]
points=list(dict.fromkeys(points));rows=np.array([r for r,c in points]);cols=np.array([c for r,c in points])
expert=data['ids'].numpy().reshape(-1)[rows];token=rows//8
packed=data['gp'].numpy();scales=data['gs'].numpy();x=data['x'].float().numpy()[token].T
k=np.arange(6144)[:,None]
raw=packed[expert[None,:]*6144+k,((cols//256)*128+cols%128)[None,:]]
codes=(raw>>((cols%256>=128).astype('u1')*4)[None,:])&15
lut=np.array([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype='f4')
q=lut[codes];e=scales[expert[None,:]*192+np.arange(192)[:,None],cols[None,:]]
assert e.min()>0 and e.max()<255
scale=np.ldexp(np.ones(e.shape,dtype='f4'),e.astype('i4')-127)
partial=np.zeros((3,len(points)),dtype='f4')
for s in range(3):
    for g in range(s*64,(s+1)*64):
        part=np.zeros(len(points),dtype='f4')
        for kk in range(g*32,(g+1)*32):
            # BF16 x E2M1 products are exact in normal FP32. Scaling by a power
            # of two is also exact in this retained checkpoint range.
            part=np.add(part,np.multiply(q[kk],x[kk],dtype='f4'),dtype='f4')
        partial[s]=np.add(partial[s],np.multiply(part,scale[g],dtype='f4'),dtype='f4')
expected=splits[rows,:,cols].numpy().T.copy()
partial_equal=np.equal(partial.view('u4'),expected.view('u4'))
scaled_q=np.multiply(q,np.repeat(scale,32,axis=0),dtype='f4')
products=np.multiply(scaled_q,x,dtype='f4')
nonzero=products[products!=0];assert np.isfinite(products).all() and (not len(nonzero)or np.abs(nonzero).min()>=np.finfo('f4').tiny)
serial=np.zeros(len(points),dtype='f4')
for product in products:serial=np.add(serial,product,dtype='f4')
tree=np.zeros((8192,len(points)),dtype='f4');tree[:6144]=products
while tree.shape[0]>1:tree=np.add(tree[::2],tree[1::2],dtype='f4')
balanced=tree[0];absolute_sum=np.abs(products).sum(axis=0,dtype='f4')
witnesses=[]
for i,(row,col) in enumerate(points):
    av,bv=float(old_gp[row,col]),float(new_gp[row,col]);den=float(absolute_sum[i])
    witnesses.append(dict(route_row=row,token=row//8,slot=row%8,expert=int(expert[i]),column=col,
        old=float(av),mme=float(bv),cpu_serial_fp32=float(serial[i]),cpu_balanced_fp32=float(balanced[i]),
        old_split_words_equal=int(partial_equal[:,i].sum()),old_bf16_bits=int(rounded_old[row,col].view(torch.int16))&65535,
        mme_bf16_bits=int(rounded_new[row,col].view(torch.int16))&65535,
        mme_vs_balanced_over_abs_product_sum=abs(bv-float(balanced[i]))/den if den else None,
        old_vs_balanced_over_abs_product_sum=abs(av-float(balanced[i]))/den if den else None))
report=dict(status='ROOT_RECHECKED_PREFILL_STAGE_DIAGNOSTIC',tensor_sha256=digests,
    complete_plain_output_relation=True,gp=metrics(old_gp,new_gp),gp_after_bf16=metrics(rounded_old,rounded_new),gate=metrics(old_gate,new_gate),
    changed_gate_values_without_changed_rounded_gp=int((gate_changed&~changed_inputs).sum()),
    independent_old_split_words=int(partial_equal.size),independent_old_split_words_equal=int(partial_equal.sum()),
    witnesses=witnesses,model_quality_qualified=False,performance_qualified=False,
    scope='One real rank/layer input, selected scalar FP32 arithmetic witnesses only. All summations use FP32, no FP64 acceptance. Export +0 canonicalizes signed zero; this witness does not certify signed-zero transport. Full model/KV trajectory remains outside this operator test.')
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k not in ('tensor_sha256','witnesses')}))
