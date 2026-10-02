"""CPU control flow only: no fake implementation identity qualifies hardware."""
import sys
import types
import numpy as np
import pytest
import torch
from gaudi_kernels import block_fp8 as block
from gaudi_kernels import production_integration as pi
from gaudi_kernels import fp32_artifact_binding as binding
from gaudi_kernels.block_fp8_fp32_contract import build_reference, epilogue_certificate


@pytest.mark.parametrize('different_debug_recipe', [False, True])
def test_plain_completed_first_and_returned_when_debug_recipe_changes(monkeypatch, different_debug_recipe):
    x=torch.ones(1,128,dtype=torch.bfloat16)
    w=torch.ones(2,128).to(torch.float8_e4m3fn); scales=torch.ones(1,1)
    ref=build_reference(x,w,scales); prepared=block.prepare_block_fp8(w,scales)
    plain=epilogue_certificate(ref['partial_balanced'].numpy()[:,0,:],ref['factors'],ref['bias'].numpy(),'sequential')
    actual={k:ref[k] for k in ('q_native','activation_scales','prepared_weight','prepared_scales','bias')}
    actual.update(partial=ref['partial_balanced'],output=plain.clone())
    events=[]
    def fake_linear(x, prepared, **kw):
        if kw.get('audit_tensors') is None:
            events.append('plain_linear'); return plain
        events.append('staged_linear')
        kw['audit_tensors'].update(actual)
        if different_debug_recipe:kw['audit_tensors']['output']=plain+1
        return kw['audit_tensors']['output']
    monkeypatch.setattr(block,'linear_block_fp8',fake_linear)
    original_cpu=torch.Tensor.cpu
    def cpu(tensor,*args,**kwargs):
        if tensor.data_ptr()==plain.data_ptr():events.append('plain_readback')
        return original_cpu(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'cpu',cpu)
    # Explicit fake plugin/operator symbols only for the Python flow test.
    class Upstream:pass
    class Method(Upstream):pass
    plugin=types.ModuleType('vllm_gaudi.ops.hpu_fp8')
    plugin.OrigFp8LinearMethod=Upstream;plugin.Fp8LinearMethod=Method
    plugin.fp8=types.SimpleNamespace(Fp8LinearMethod=Method)
    monkeypatch.setitem(sys.modules,plugin.__name__,plugin)
    monkeypatch.setattr(torch.ops,'gaudi_block_fp8',types.SimpleNamespace(**{
        name:lambda *a:None for name in ('quant','batch_mm','reduce','decode','decode_fast','mm','finish','reshape')}))
    monkeypatch.setattr(pi,'_active',None)
    installation=pi.install_block_fp8(plugin,expected_qkv_shape=None,policy='decode_a8_bf16_fp32')
    installation.choose=lambda layer,rows:('per_block_fp8','fp32')
    monkeypatch.setattr(binding,'verify_artifacts',lambda reduction:binding._VerifiedImplementation(
        reduction,True,dict(scope='TEST DOUBLE ONLY; not real qualification')))
    layer=types.SimpleNamespace(prefix='model.layers.0.qkv_proj',_gk_block_fp8_selected=True,
        _gk_block_fp8=prepared,_gk_oracle_weight=w,_gk_oracle_scales=scales,_gk_oracle_layout={'test':True})
    pi.configure_same_input_audit(True,contract='fp32_arithmetic_v1',tag='test')
    pi.set_same_input_audit_frame(dict(input_ids=[7],positions=[129]))
    output=installation.method_class().apply(layer,x)
    assert output is plain
    assert events.index('plain_linear')<events.index('plain_readback')<events.index('staged_linear')
    record=installation.audit_records[0]
    assert record['plain_vs_staged']['all_bits_equal'] is (not different_debug_recipe)
    assert record['fp32_contract']['passed'] is (not different_debug_recipe)
    if different_debug_recipe:
        assert record['fp32_contract']['classification']=='DECLARED_FP32_EPILOGUE_MISMATCH'
        # An already observed arithmetic error must not be hidden by the
        # additional plain/debug relation failure.
