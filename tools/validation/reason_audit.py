"""Metadata-only capture admission reasons; enabled only for diagnostic isolation."""
_INSTALLED=False

def install():
    global _INSTALLED
    if _INSTALLED:return
    _INSTALLED=True
    from gaudi_kernels import vllm_norm_grid24 as grid, vllm_qkv_postprocess as post
    for module,name in [(grid,'_reason'),(post,'_fallback_reason')]:
        original=getattr(module,name);counts=[0]
        def traced(*args,_original=original,_counts=counts,_name=name,**kwargs):
            result=_original(*args,**kwargs)
            if _counts[0]<12:
                import json
                shapes=[list(a.shape) for a in args if hasattr(a,'shape')]
                print('CANONICAL_ADMISSION '+json.dumps({'stage':_name,'reason':result,'shapes':shapes}),flush=True)
                _counts[0]+=1
            return result
        setattr(module,name,traced)
