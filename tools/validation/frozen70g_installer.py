"""Explicit model-quality diagnostics, disabled during performance measurements.

Capture complete logits after a normal bridge invocation. A fixed prompt plus
the reference continuation gives both policies identical teacher-forced input.
This diagnostic adds synchronization/readback and must never count as TPS.
"""
import os
from pathlib import Path


def install_runner(module):
    original = module.HPUModelRunner._execute_model_generic
    original_use_graphs = module.HPUModelRunner._use_graphs

    def use_graphs(self, *args, **kwargs):
        if (getattr(self, '_production_cpu_reference', False)
                or getattr(self, '_production_same_input_audit', False)
                or getattr(self, '_production_layer_diagnostics', None) is not None):
            return False
        return original_use_graphs(self, *args, **kwargs)

    def execute(self, *args, **kwargs):
        token_ids = args[0] if args else kwargs['token_ids']
        positions = args[1] if len(args)>1 else kwargs['position_ids']
        metadata = args[2] if len(args)>2 else kwargs['attn_metadata']
        stages = getattr(self, '_production_layer_diagnostics', None)
        staged_fp32 = (getattr(self, '_production_same_input_audit', False)
                       and os.getenv('GK_PRODUCTION_QUALITY_CONTRACT', 'legacy') == 'fp32_arithmetic_v1')
        if staged_fp32:
            from gaudi_kernels.production_integration import set_same_input_audit_frame
            frame = None
            if token_ids.numel() == 1 and not metadata.is_prompt:
                frame = {'input_ids': token_ids.detach().cpu().reshape(-1).tolist(),
                         'positions': positions.detach().cpu().reshape(-1).tolist()}
            set_same_input_audit_frame(frame)
        if stages is not None:stages.enabled=token_ids.numel()==1 and not metadata.is_prompt
        try:
            outputs = original(self, *args, **kwargs)
            if stages is not None and stages.enabled:
                stages.flush(self._production_stage_path,token_ids,positions)
        finally:
            if stages is not None:stages.enabled=False;stages.stack.clear()
            if staged_fp32:set_same_input_audit_frame(None)
        destination = getattr(self, '_production_quality_destination', None)
        if destination is not None and self.is_driver_worker:
            import torch
            # Only the final invocation is retained: chunked prefill may emit
            # several intermediate rows before the requested last-token logits.
            logits = outputs[-1].detach().cpu().float()
            torch.save({'logits': logits, 'input_ids': token_ids.detach().cpu(),
                        'positions': positions.detach().cpu()}, destination)
        return outputs

    module.HPUModelRunner._execute_model_generic = execute
    module.HPUModelRunner._use_graphs = use_graphs


def configure_capture(worker, tag=None, audit_mode=None):
    import torch
    import habana_frameworks.torch.core as hc
    hc.mark_step()
    torch.hpu.synchronize()
    if os.getenv('NATIVE_BACKEND_ACTIVE', '0') != '0':
        raise ValueError('Quality capture requires the bridge; disable native replay first')
    if audit_mode not in (None, 'plain', 'staged'):
        raise ValueError('Unknown quality capture mode')
    path = None
    if tag is not None:
        if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in tag):
            raise ValueError('Invalid quality capture tag')
        root = Path(os.environ['E1_WORK_ROOT']) / os.environ['E1_CASE'] / 'quality-logits'
        root.mkdir(exist_ok=True)
        path = root / (tag + '.pt')
        if path.exists():
            raise FileExistsError(path)
    worker.model_runner._production_quality_destination = path
    stage_path=None
    stages=getattr(worker.model_runner,'_production_layer_diagnostics',None)
    if stages is not None:stages.close();worker.model_runner._production_layer_diagnostics=None
    if audit_mode != 'plain' and tag is not None and tag.endswith('-teacher-0') and os.getenv('GK_PRODUCTION_LAYER_DIAGNOSTICS')=='1':
        from layer_diagnostics import LayerDiagnostics
        stage_path=path.with_name(path.stem+f'-stages-rank{worker.rank}.pt')
        if stage_path.exists():raise FileExistsError(stage_path)
        worker.model_runner._production_layer_diagnostics=LayerDiagnostics(worker.model_runner.model,
            capture_rotary=os.getenv('GK_QKV_POST_ENABLED')!='1')
        worker.model_runner._production_stage_path=stage_path
    audit = []
    if os.getenv('GK_BLOCK_FP8_AUDIT_SAME_INPUT') == '1':
        from gaudi_kernels.production_integration import configure_same_input_audit
        audit = configure_same_input_audit(tag is not None and audit_mode != 'plain',
            contract=os.getenv('GK_PRODUCTION_QUALITY_CONTRACT', 'legacy'), tag=tag)
        worker.model_runner._production_same_input_audit = tag is not None and audit_mode != 'plain'
    from gaudi_kernels.production_integration import snapshot as block_snapshot
    grid_audit_layers=[]
    if os.getenv('GK_NORM_GRID24_ENABLED')=='1':
        from gaudi_kernels.vllm_norm_grid24 import snapshot as grid_snapshot
        grid_state=grid_snapshot()
        if grid_state['policy']=='grid24':grid_audit_layers=grid_state['expected_audit_layers']
    return {'rank': worker.rank, 'path': str(path) if path else None, 'stage_path':str(stage_path) if stage_path else None,
            'same_input_qkv_audit': audit, 'capture_mode': audit_mode, 'selected_qkv_layers': list(block_snapshot().get('selected_layers', {})),
            'expected_grid24_qkv_layers':grid_audit_layers}


def configure_policy(worker, policy):
    import torch
    import habana_frameworks.torch.core as hc
    from gaudi_kernels.production_integration import set_policy, set_reduction_policy
    hc.mark_step()
    torch.hpu.synchronize()
    runner = worker.model_runner
    if os.getenv('NATIVE_BACKEND_ACTIVE', '0') != '0':
        raise ValueError('Drain/disable native replay before changing arithmetic policy')
    if hasattr(runner, '_native_reset'):
        runner._native_reset()
    clear = getattr(runner.model, 'clear_cache', None)
    if not callable(clear):
        raise RuntimeError('No graph cache invalidator; refusing stale-policy graph replay')
    clear()
    block_policy, *suffixes = policy.split('+')
    if len(set(suffixes)) != len(suffixes) or set(suffixes) - {'qkv_post_full','moe_sum_bf16','moe_compact8','norm_fp32','norm_qkv_grid24','swa_fp32_fast','swa_fp32_quad','swa_fp32_av_hoist','gp_vec','gp_fold','gp_scale_tail','down_vec','qkv_neumaier','qkv_neumaier_isa','f32_ag','qkv_post','router_post_scalar','router_post_vector'}:
        raise ValueError('Unknown production policy suffix')
    router_suffixes=set(suffixes)&{'router_post_scalar','router_post_vector'}
    if len(router_suffixes)>1:raise ValueError('Select one router post-TopK implementation')
    router_policy=next(iter(router_suffixes)).removeprefix('router_post_') if router_suffixes else 'vendor'
    if router_suffixes:
        from router_post_runtime import validate_policy
        validate_policy(router_policy)
    if 'gp_vec' in suffixes and 'gp_fold' in suffixes:
        raise ValueError('Select one GP implementation')
    if 'gp_scale_tail' in suffixes:
        if 'gp_fold' not in suffixes:
            raise ValueError('GP scale-tail requires the gp_fold baseline condition')
        from gp_scale_tail_runtime import validate_policy
        validate_policy()
    swa_suffixes=set(suffixes)&{'swa_fp32_fast','swa_fp32_quad','swa_fp32_av_hoist'}
    if len(swa_suffixes)>1:raise ValueError('Select one SWA attention implementation')
    swa_selected='fp32_av_hoist' if 'swa_fp32_av_hoist' in suffixes else 'fp32_quad' if 'swa_fp32_quad' in suffixes else 'fp32_fast' if swa_suffixes else 'vendor'
    if swa_selected=='fp32_quad':
        from swa_quad_runtime import validate_policy
        validate_policy(swa_selected)
    if swa_selected=='fp32_av_hoist':
        from swa_av_runtime import validate_policy
        validate_policy(swa_selected)
    if swa_suffixes and os.getenv('GK_SWA_ENABLED') != '1':
        raise ValueError('Requested SWA policy without loading its extension')
    if 'gp_vec' in suffixes and os.getenv('GK_MXFP4_GP_ENABLED') != '1':
        raise ValueError('Requested vector-fetch GP without loading its extension')
    if 'gp_fold' in suffixes and os.getenv('GK_MXFP4_FOLDED_ENABLED') != '1':
        raise ValueError('Requested folded GP without loading its extension')
    if 'down_vec' in suffixes and os.getenv('GK_MXFP4_DOWN_ENABLED') != '1':
        raise ValueError('Requested vector-fetch down without loading its extension')
    if {'qkv_neumaier', 'qkv_neumaier_isa'} <= set(suffixes):
        raise ValueError('Select one QKV Neumaier implementation')
    if 'qkv_neumaier_isa' in suffixes:
        from qkv_neumaier_isa_runtime import validate_policy
        validate_policy()
    if 'qkv_neumaier' in suffixes and os.getenv('GK_BLOCK_REDUCE_ENABLED') != '1':
        raise ValueError('Requested compensated QKV reduction without loading its extension')
    if 'f32_ag' in suffixes and os.getenv('GK_FP32_GRAPH_AG_ENABLED') != '1':
        raise ValueError('Requested FP32 graph AG without explicit experiment enablement')
    from qkv_post_full_runtime import attention_scope as qkv_attention_scope
    post_attention_scope=qkv_attention_scope(suffixes)
    if 'qkv_post' in suffixes and os.getenv('GK_QKV_POST_ENABLED') != '1':
        raise ValueError('Requested QKV postprocess without loading its extension')
    if 'qkv_post' in suffixes and not swa_suffixes:
        raise ValueError('QKV postprocess requires an explicitly selected FP32 SWA consumer')
    from moe_sum_bf16_runtime import validate_policy as validate_moe_sum
    moe_sum_policy='moe_final_bf16' if 'moe_sum_bf16' in suffixes else 'baseline'
    validate_moe_sum(moe_sum_policy)
    if moe_sum_policy!='baseline' and 'f32_ag' not in suffixes:
        raise ValueError('moe_sum_bf16 requires the explicit FP32 AllGather policy')
    if 'norm_qkv_grid24' in suffixes:
        if not {'norm_fp32','qkv_post'}.issubset(suffixes):
            raise ValueError('grid24 requires explicit FP32 norm and fused QKV postprocess')
        from norm_grid24_runtime import validate_policy
        validate_policy(True)
    from moe_dispatch_runtime import validate_policy as validate_moe_dispatch
    moe_dispatch_policy='compact8' if 'moe_compact8' in suffixes else 'baseline'
    validate_moe_dispatch(moe_dispatch_policy)
    if moe_dispatch_policy=='compact8' and not {'gp_fold','down_vec'}.issubset(suffixes):
        raise ValueError('moe_compact8 is paired with the qualified gp_fold/down_vec kernels')
    set_policy(block_policy)
    reduction = ('neumaier_isa_fp32' if 'qkv_neumaier_isa' in suffixes
                 else 'neumaier_fp32' if 'qkv_neumaier' in suffixes else 'sequential')
    result = set_reduction_policy(reduction)
    reducer_isa = None
    if os.getenv('GK_BLOCK_REDUCE_ISA_ENABLED') == '1':
        from qkv_neumaier_isa_runtime import record_selection
        reducer_isa = record_selection('qkv_neumaier_isa' in suffixes)
    if not hasattr(worker, '_production_base_reduce_policy'):
        worker._production_base_reduce_policy = os.getenv('UNIFIED_PRECISION_REDUCE','off')
    os.environ['UNIFIED_PRECISION_REDUCE'] = ('auto_fp32' if 'f32_ag' in suffixes
                                             else worker._production_base_reduce_policy)
    from precision_runtime import COUNTS
    COUNTS['f32_gather_f32_sum'] = 0
    runner._production_cpu_reference = block_policy.startswith('cpu_')
    norm = None
    if os.getenv('GK_NORM_ENABLED') == '1':
        from gaudi_kernels.vllm_norm import prepare_vllm_norm, set_norm_policy
        norm = {'prepared': prepare_vllm_norm(runner.model),
                'policy': set_norm_policy('fp32' if 'norm_fp32' in suffixes else 'vendor')}
        if not norm['prepared']['matched']:
            raise RuntimeError('Requested norm integration did not find model norm layers')
    elif 'norm_fp32' in suffixes:
        raise ValueError('Requested norm policy without loading its extension')
    swa = None
    if os.getenv('GK_SWA_ENABLED') == '1':
        from gaudi_kernels.vllm_swa_install import set_swa_policy, prepare_vllm_swa, snapshot_swa_counters
        selected = swa_selected
        snapshot_swa_counters(reset=True)
        set_swa_policy(selected)
        swa = prepare_vllm_swa(runner.model)
        if os.getenv('GK_SWA_QUAD_ENABLED')=='1':
            from swa_quad_runtime import snapshot as quad_snapshot
            swa['quad_libraries']=quad_snapshot()
        if os.getenv('GK_SWA_AV_ENABLED')=='1':
            from swa_av_runtime import snapshot as av_snapshot
            swa['av_libraries']=av_snapshot()
        if selected != 'vendor' and not swa['matched']:
            raise RuntimeError('Requested SWA integration found no matching attention layers')
    post = None
    if os.getenv('GK_QKV_POST_ENABLED') == '1':
        from gaudi_kernels.vllm_qkv_postprocess import install_vllm_qkv_postprocess, snapshot_qkv_postprocess
        selected='fused' if 'qkv_post' in suffixes else 'vendor'
        snapshot_qkv_postprocess(reset=True)
        post=(install_vllm_qkv_postprocess(selected,model=runner.model,attention_scope=post_attention_scope)
              if 'qkv_post_full' in suffixes else install_vllm_qkv_postprocess(selected,model=runner.model))
        if selected=='fused' and not post['prepared']['matched']:
            raise RuntimeError('Requested QKV postprocess found no matching layers')
        if 'qkv_post_full' in suffixes and not post['prepared'].get('matched_by_attention_kind',{}).get('full'):
            raise RuntimeError('Requested full-attention postprocess found no matching full layers')
    grid24=None
    if os.getenv('GK_NORM_GRID24_ENABLED')=='1':
        from gaudi_kernels.vllm_norm_grid24 import install as install_grid24, snapshot as grid24_snapshot
        grid24_snapshot(reset=True)
        grid24=install_grid24('grid24' if 'norm_qkv_grid24' in suffixes else 'separate',model=runner.model)
        if 'norm_qkv_grid24' in suffixes and not grid24['prepared']['matched']:
            raise RuntimeError('grid24 found no eligible SWA input-norm/QKV pair')
    gp = None
    if os.getenv('GK_MXFP4_GP_ENABLED') == '1' or os.getenv('GK_MXFP4_FOLDED_ENABLED') == '1':
        from batch_ops import set_gp_policy, gp_snapshot
        set_gp_policy('vector_fetch_scale_tail' if 'gp_scale_tail' in suffixes else 'vector_fetch_folded' if 'gp_fold' in suffixes else 'vector_fetch' if 'gp_vec' in suffixes else 'legacy')
        gp = gp_snapshot()
    down = None
    if os.getenv('GK_MXFP4_DOWN_ENABLED') == '1':
        from batch_ops import set_down_policy, down_snapshot
        set_down_policy('vector_fetch' if 'down_vec' in suffixes else 'legacy')
        down = down_snapshot()
    router_post = None
    if os.getenv('GK_ROUTER_POST_ENABLED') == '1':
        from router_post_runtime import set_policy as set_router_post_policy
        router_post = set_router_post_policy(router_policy)
    from moe_dispatch_runtime import set_policy as set_moe_dispatch
    moe_dispatch=set_moe_dispatch(moe_dispatch_policy)
    from moe_sum_bf16_runtime import set_policy as set_moe_sum
    moe_sum=set_moe_sum(moe_sum_policy)
    runner._production_selected_policy=policy
    return {'rank': worker.rank, 'policy': policy, 'details': result, 'norm': norm, 'swa':swa, 'gp':gp, 'down':down,
            'collective_policy':os.environ['UNIFIED_PRECISION_REDUCE'],'qkv_postprocess':post,'router_post':router_post,
            'qkv_neumaier_isa':reducer_isa,'moe_dispatch':moe_dispatch,'moe_sum_bf16':moe_sum,'norm_grid24':grid24}


def compare_logits(reference, candidate, *, contract='legacy'):
    import torch
    if contract not in ('legacy', 'fp32_arithmetic_v1'):
        raise ValueError('Unknown quality arithmetic contract')
    if reference.shape != candidate.shape:
        raise ValueError('Teacher logits shape mismatch')
    reference, candidate = reference.double(), candidate.double()
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all())
    if not finite:
        return {'pass': False, 'finite': False}
    lp, lq = reference.log_softmax(-1), candidate.log_softmax(-1)
    kl = float((lp.exp() * (lp - lq)).sum(-1).max())
    relative_l2 = float((reference-candidate).norm() / reference.norm().clamp_min(1e-30))
    return {'pass': kl <= .01 and (contract == 'fp32_arithmetic_v1' or relative_l2 <= .02), 'finite': True,
            'max_kl_ref_to_candidate': kl, 'relative_l2': relative_l2,
            'top1_match_fraction': float((reference.argmax(-1)==candidate.argmax(-1)).double().mean()),
            'gate': {'max_kl': .01, 'relative_l2': .02 if contract == 'legacy' else 'diagnostic'},
            'scope': 'full-vocabulary teacher-forced logits; bounded regression gate, not broad quality evaluation',
            'contract': contract}


def snapshot(worker):
    from gaudi_kernels.production_integration import snapshot as block_snapshot
    result = {'rank': worker.rank, 'block_fp8': block_snapshot()}
    if os.getenv('GK_NORM_GRID24_ENABLED')=='1':
        from gaudi_kernels.vllm_norm_grid24 import snapshot as grid24_snapshot
        result['norm_grid24']=grid24_snapshot()
    from moe_dispatch_runtime import snapshot as moe_dispatch_snapshot
    result['moe_dispatch']=moe_dispatch_snapshot()
    from moe_sum_bf16_runtime import snapshot as moe_sum_snapshot
    result['moe_sum_bf16']=moe_sum_snapshot()
    if os.getenv('GK_BLOCK_REDUCE_ISA_ENABLED') == '1':
        from qkv_neumaier_isa_runtime import snapshot as isa_snapshot
        result['qkv_neumaier_isa'] = isa_snapshot()
    from precision_runtime import COUNTS
    result['collective']={'policy':os.getenv('UNIFIED_PRECISION_REDUCE','off'),
                          'fp32_ag_capture_calls':COUNTS.get('f32_gather_f32_sum',0),
                          'scope':'Python captures; native command list must independently prove HCCL calls'}
    if os.getenv('GK_SWA_ENABLED') == '1':
        from gaudi_kernels.vllm_swa_install import snapshot_swa_counters
        result['swa'] = snapshot_swa_counters()
        if os.getenv('GK_SWA_QUAD_ENABLED')=='1':
            from swa_quad_runtime import snapshot as quad_snapshot
            result['swa']['quad_libraries']=quad_snapshot()
        if os.getenv('GK_SWA_AV_ENABLED')=='1':
            from swa_av_runtime import snapshot as av_snapshot
            result['swa']['av_libraries']=av_snapshot()
    if os.getenv('GK_QKV_POST_ENABLED') == '1':
        from gaudi_kernels.vllm_qkv_postprocess import snapshot_qkv_postprocess
        result['qkv_postprocess']=snapshot_qkv_postprocess()
    if os.getenv('GK_MXFP4_GP_ENABLED') == '1' or os.getenv('GK_MXFP4_FOLDED_ENABLED') == '1':
        from batch_ops import gp_snapshot
        result['gp'] = gp_snapshot()
    if os.getenv('GK_MXFP4_DOWN_ENABLED') == '1':
        from batch_ops import down_snapshot
        result['down'] = down_snapshot()
    if os.getenv('GK_ROUTER_POST_ENABLED') == '1':
        from router_post_runtime import snapshot as router_post_snapshot
        result['router_post'] = router_post_snapshot()
    return result
