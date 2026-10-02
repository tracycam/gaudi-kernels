"""Metadata/control contracts only; does not access HPU data or acquire a card."""
def run(torch):
    ops=torch.ops.gaudi_down_combine_isa;records=[]
    def fixture(experts=24):
        empty=lambda shape,dtype:torch.empty(shape,dtype=dtype,device='meta')
        dp=empty((experts*3072,256),torch.uint8);ds=empty((experts*96,512),torch.uint8)
        gate=empty((8,256),torch.bfloat16);lut=empty((1,512),torch.bfloat16);ids=empty((1,8),torch.int32)
        routing=empty((1,8),torch.float32);directions=empty((2,256),torch.uint8)
        return dp,ds,gate,lut,ids,routing,directions
    for e in (8,24,384):
        dp,ds,*rest=fixture(e);w=ops.weight_view(dp,False);s=ops.weight_view(ds,True)
        assert w.shape==(e*6144,128) and s.shape==(e*192,256)
        assert w.numel()==dp.numel() and s.numel()==ds.numel()
        for name in ('p4','p6'):
            y=getattr(ops,name)(w,s,*rest,1,True);assert y.shape==(1,6144) and y.dtype==torch.float32
        y=ops.old_body(dp,ds,*rest[:3]);assert y.shape==(1,49152) and y.dtype==torch.float32
        records.append(dict(experts=e,pass_all_Meta=True))
    dp,ds,*rest=fixture();args=[ops.weight_view(dp,False),ops.weight_view(ds,True),*rest,1,True]
    def reject(name,call):
        try:call()
        except (RuntimeError,ValueError):records.append(dict(case=name,rejected=True));return
        raise AssertionError('accepted bad contract: '+name)
    for name in ('p4','p6'):
        for index,value,label in [(0,dp,'old_width_without_view'),(2,rest[0].float(),'activation_fp32'),
                (2,torch.empty(16,256,dtype=torch.bfloat16,device='meta'),'M2'),
                (2,torch.empty(8,257,dtype=torch.bfloat16,device='meta'),'K257'),
                (4,torch.empty(1,7,dtype=torch.int32,device='meta'),'top7'),
                (5,rest[3].bfloat16(),'routing_bf16'),(7,2,'wrong_layout'),(8,False,'missing_certificate'),
                (0,torch.empty(24*6144,256,dtype=torch.uint8,device='meta')[:,::2],'noncontiguous')]:
            changed=list(args);changed[index]=value;reject(name+'_'+label,lambda changed=changed:getattr(ops,name)(*changed))
    import os
    previous=os.environ.get('PT_ENABLE_INT64_SUPPORT')
    long_args=list(args);long_args[4]=rest[2].long()
    try:
        os.environ['PT_ENABLE_INT64_SUPPORT']='0'
        for name in ('p4','p6'):
            assert getattr(ops,name)(*long_args).shape==(1,6144)
            records.append(dict(case=name+'_logical_Long_explicit_physical_I32',pass_all_Meta=True))
        os.environ['PT_ENABLE_INT64_SUPPORT']='1'
        for name in ('p4','p6'):reject(name+'_true_INT64_rejected',lambda name=name:getattr(ops,name)(*long_args))
        os.environ.pop('PT_ENABLE_INT64_SUPPORT')
        reject('Long_without_explicit_I32_mode',lambda:ops.p6(*long_args))
    finally:
        if previous is None:os.environ.pop('PT_ENABLE_INT64_SUPPORT',None)
        else:os.environ['PT_ENABLE_INT64_SUPPORT']=previous
    reject('view_wrong_width',lambda:ops.weight_view(torch.empty(24*3072,128,dtype=torch.uint8,device='meta'),False))
    reject('view_no_experts',lambda:ops.weight_view(torch.empty(3072,256,dtype=torch.uint8,device='meta'),False))
    reject('old_M2',lambda:ops.old_body(dp,ds,torch.empty(16,256,dtype=torch.bfloat16,device='meta'),rest[1],torch.empty(2,8,dtype=torch.int32,device='meta')))
    return dict(device_used=False,real_Meta=True,all_pass=True,records=records,schemas={name:str(getattr(ops,name).default._schema) for name in ('p4','p6','old_body','weight_view')},scope='Shape/domain tags only; a true certified flag is a caller contract, not runtime value validation')
