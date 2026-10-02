"""Resident bridge/native/native/bridge serving A/B with actual token returns."""
import argparse,json,os,statistics,time,traceback
from gaudi_kernels.engine.context import context as execution_context
from pathlib import Path
from tools.validation.executor.serving_gate import performance_qualified

def numeric_suffix(policy):
    # GP/down/final-sum candidates must separately pass bitwise arithmetic gates. They
    # share an independent old-GP/down CPU-QKV reference rather than rerunning
    # identical references for each instruction schedule.
    suffixes=set(policy.split('+')[1:])
    # Quad is bit-exact to the original fast SWA; retain the independent old
    # attention schedule in its CPU-QKV reference instead of self-referencing.
    if {'swa_fp32_quad','swa_fp32_av_hoist'}&suffixes:suffixes.add('swa_fp32_fast')
    return ''.join('+'+s for s in ('norm_fp32','swa_fp32_fast','f32_ag') if s in suffixes)


def install_step_timer(engine, steps, counts, *, clock=time.perf_counter_ns):
    """Own the engine callable separately from mutable model/audit locals."""
    engine_step=engine.step
    def timed_step():
        begin=clock();outputs=engine_step();end=clock();delta=0;per_request=[]
        for output in outputs:
            if not output.outputs:continue
            count=len(output.outputs[0].token_ids);increment=max(0,count-counts.get(output.request_id,0));delta+=increment;counts[output.request_id]=count
            if increment:per_request.append(increment)
        steps.append(dict(begin_ns=begin,end_ns=end,ms=(end-begin)/1e6,new_tokens=delta,emitted=max(counts.values(),default=0),
                          emitted_min=min(counts.values(),default=0),known_requests=len(counts),per_request_new=per_request))
        return outputs
    engine.step=timed_step


from tools.validation.legacy_selection import typed_policy

def runtime_policy(label):
    document = typed_policy(label)
    document['engine']['runtime'] = execution_context().selection.engine.to_dict()['runtime']
    return document

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--layers',type=int,default=70);p.add_argument('--tokens',type=int,default=160)
    p.add_argument('--context',type=int,default=4096);p.add_argument('--native',action='store_true')
    p.add_argument('--short',action='store_true')
    p.add_argument('--boundary-hashes',action='store_true')
    p.add_argument('--setup-reference',action='store_true',help='Diagnostic isolation of the historical installer')
    p.add_argument('--production-policies', default='',
                   help='Comma-separated resident block-FP8 policies; requires quality capture hook')
    p.add_argument('--batch-sizes', default='', help='Optional resident serving batches, e.g. 1,2,8')
    p.add_argument('--batch-tokens',type=int,default=160,
                   help='Long enough to observe all requests decoding after staggered prefill')
    p.add_argument('--batch-policy-abba-control',default='',
                   help='Optional resident batch A/B/B/A control; requires batch sizes and a distinct final production policy')
    p.add_argument('--batch-policy-abba-expect-bitwise',action='store_true',
                   help='Require every batch warmup/measured output to match the control policy, for arithmetic-preserving candidates')
    p.add_argument('--cpu-oracle',action='store_true',
                   help='M1 independent CPU references selected by quality contract; diagnostic only, requires host weights')
    p.add_argument('--quality-contract',choices=('legacy','fp32_arithmetic_v1'),default='legacy',
                   help='Explicit new-run precision contract; never reclassifies historical result files')
    p.add_argument('--quality-reference-policy',default='',
                   help='v1 explicit resident A16 reference, e.g. bf16_fp32+norm_fp32')
    p.add_argument('--quality-candidate-policies',default='',
                   help='v1 comma-separated candidate subset; default is final resident policy; other policies are performance controls')
    p.add_argument('--fp32-ocp-model-reference',action='store_true',
                   help='Optional original-OCP FP32 model diagnostic in v1; primary matched reference is native-half FP32')
    p.add_argument('--require-fp32-ocp-quality',action='store_true',
                   help='Require finite/KL<=.01 versus original-OCP FP32 M1 QKV reference; needs v1 and --fp32-ocp-model-reference')
    p.add_argument('--profile-native',action='store_true',help='Three device-traced replay steps in a separate diagnostic run')
    p.add_argument('--native-policy-abba',action='store_true',
                   help='Append resident native A8 baseline/final candidate/final candidate/baseline runs')
    p.add_argument('--native-abba-control',default='decode_a8_bf16_fp32',
                   help='Explicit resident control for warmed ABBA, also present in production-policies')
    p.add_argument('--integration-preflight',action='store_true',
                   help='Two-layer interface diagnostic; can never qualify model quality or performance')
    p.add_argument('--preflight-max-seqs',type=int,choices=(1,8),default=1,
                   help='Exercise capacity padding for a multi-request server in a B1 preflight')
    p.add_argument('--lifecycle',action='store_true');p.add_argument('--long-context',type=int,default=0)
    p.add_argument('--single-rpc',action='store_true');p.add_argument('--reuse-pages',action='store_true');a=p.parse_args()
    if a.layers!=execution_context().startup.layers:raise ValueError('CLI layers differ from the serving manifest')
    batch_sizes=[int(x) for x in a.batch_sizes.split(',')] if a.batch_sizes else []
    if any(x<1 or x>64 for x in batch_sizes):raise ValueError('Bounded batch sizes must be in [1,64]')
    if not 32<=a.batch_tokens<=512:raise ValueError('batch token budget must be 32..512')
    if a.integration_preflight and (a.layers!=2 or batch_sizes or not 8<=a.tokens<=16):
        raise ValueError('Integration preflight requires exactly two layers, 8..16 tokens and no batch benchmark')
    if execution_context().native.boundary_positions and not a.integration_preflight:
        raise ValueError('Boundary tensor readbacks are preflight diagnostics, never serving TPS')
    if a.preflight_max_seqs!=1 and not a.integration_preflight:
        raise ValueError('Explicit preflight capacity applies only to integration diagnostics')
    policies=a.production_policies.split(',') if a.production_policies else [None]
    from tools.validation.executor.batch_policy_abba import validate_batch_abba
    validate_batch_abba(a.batch_policy_abba_control,policies,batch_sizes,a.batch_policy_abba_expect_bitwise)
    if a.require_fp32_ocp_quality:
        from tools.validation.executor.fp32_quality_contract import validate_ocp_requirement
        validate_ocp_requirement(a.quality_contract,a.fp32_ocp_model_reference,a.require_fp32_ocp_quality)
    quality_plan=None
    if a.quality_contract=='fp32_arithmetic_v1':
        from tools.validation.executor.fp32_quality_contract import plan
        quality_plan=plan(policies,a.quality_reference_policy,a.quality_candidate_policies,
                          require_fp32_ocp_quality=a.require_fp32_ocp_quality)
        if (not a.production_policies or not a.cpu_oracle
                or not execution_context().startup.keep_cpu_oracle
                or not execution_context().startup.same_input_audit):
            raise ValueError('v1 requires production policies, --cpu-oracle, original host weights and complete same-input audit')
        from gaudi_kernels.block_fp8_fp32_contract import require_fast_backend
        cpu_backend=require_fast_backend()
    elif a.quality_reference_policy or a.quality_candidate_policies or a.fp32_ocp_model_reference:
        raise ValueError('Explicit quality roles/OCP FP32 reference require fp32_arithmetic_v1')
    if a.quality_contract!=execution_context().selection.engine.precision_contract:
        raise ValueError('Quality contract differs from explicit engine configuration')
    from vllm import LLM,SamplingParams
    from vllm.sampling_params import RequestOutputKind
    opts=dict(model='/data/models/MiMo-V2.6-Pro-RL',trust_remote_code=True,tensor_parallel_size=8,
              enable_expert_parallel=False,dtype='bfloat16',enforce_eager=False,async_scheduling=False,
              max_model_len=max(8192,a.context+a.tokens+128,a.long_context+a.tokens+128),block_size=128,max_num_seqs=max([a.preflight_max_seqs]+batch_sizes),
              gpu_memory_utilization=.3 if a.integration_preflight else .95,max_num_batched_tokens=512,disable_log_stats=True,
              enable_prefix_caching=False,seed=0,limit_mm_per_prompt={'audio':0,'image':0,'video':0})
    from gaudi_kernels.serving.host_placement import vllm_kwargs
    opts.update(vllm_kwargs())
    if a.layers!=70:
        cfg=json.loads((Path(opts['model'])/'config.json').read_text())
        opts['hf_overrides']={'num_hidden_layers':a.layers,'hybrid_layer_pattern':cfg['hybrid_layer_pattern'][:a.layers],
                              'moe_layer_freq':cfg['moe_layer_freq'][:a.layers]}
    if a.integration_preflight:
        # Two layers leave most model memory free; automatic pooling would
        # allocate tens of millions of KV slots despite a 128-token fixture.
        opts['num_gpu_blocks_override']=256
    result=dict(single_rpc=a.single_rpc,reuse_pages=a.reuse_pages,status='RUNNING',candidate_accepted=False,config=opts,runs=[],model_scope='full_model' if a.layers==70 else 'truncated_model',
                scope='actual vLLM streaming decode, synchronous B1 greedy; original storage and FP32 intermediate policy')
    result['quality_contract']=a.quality_contract
    if quality_plan is not None:result.update(quality_plan=quality_plan,cpu_fp32_backend=cpu_backend)
    def save():
        # Progress readers must see a complete version while staged audits add
        # large records. Never expose a truncated JSON during an in-place write.
        temporary=a.out/'result.json.tmp'
        temporary.write_text(json.dumps(result,indent=2)+'\n')
        temporary.replace(a.out/'result.json')
    save();start=time.monotonic();llm=LLM(**opts);result['init_s']=time.monotonic()-start;save()
    tok=llm.get_tokenizer()
    prefix=tok.encode('Record: the verification code for the copper lantern is 7391.\n',add_special_tokens=False)
    filler=tok.encode('This document describes reproducible experiments and laboratory observations.\n',add_special_tokens=False)
    suffix=tok.encode('\nQuestion: What is the verification code for the copper lantern? State the code, explain the evidence, and describe how to verify it.\nAnswer:',add_special_tokens=False)
    n=a.context-len(prefix)-len(suffix);assert n>0
    prompt={'prompt_token_ids':prefix+(filler*((n+len(filler)-1)//len(filler)))[:n]+suffix}
    steps=[];counts={}
    install_step_timer(llm.llm_engine,steps,counts)
    def add_request(prompt,params,lora_request=None,priority=0):
        params.output_kind=RequestOutputKind.CUMULATIVE
        return llm.llm_engine.add_request(str(next(llm.request_counter)),prompt,params,lora_request=lora_request,priority=priority)
    llm._add_request=add_request
    baseline=None
    if any(x and x.startswith('cpu_') for x in policies):
        raise ValueError('CPU oracle policies are diagnostics and cannot enter TPS measurements')
    if a.cpu_oracle and (not a.production_policies or not execution_context().startup.keep_cpu_oracle):
        raise ValueError('CPU reference requires block-FP8 installation and explicit load-time host weight retention')
    if a.production_policies:
        if (a.layers!=70 and not a.integration_preflight) or not execution_context().has('block_fp8'):
            raise ValueError('Production policies require 70 layers and explicit quality hook')
        if a.quality_contract=='legacy' and policies[0]!='bf16_fp32':
            raise ValueError('The first resident quality reference must be bf16_fp32')
    reference_ids=None
    current_policy=None
    policy_baselines={}
    try:
        variants=[('bridge',False),('native',True)] if a.short else [('bridge',False),('native',True),('native-repeat',True),('bridge-repeat',False)]
        schedule=[(policy,name,active) for policy in policies for name,active in variants]
        if a.native_policy_abba:
            if a.native_abba_control not in policies or policies[-1]==a.native_abba_control:
                raise ValueError('Native policy ABBA needs the requested resident control and a distinct final candidate')
            schedule += [(policy,'native-policy-abba-'+str(i),True) for i,policy in enumerate(
                (a.native_abba_control,policies[-1],policies[-1],a.native_abba_control))]
        for policy,name,active in schedule:
            if policy is not None and policy!=current_policy:
                llm.collective_rpc('native_configure',args=(False,))
                changed=llm.collective_rpc('production_policy_reference' if a.setup_reference else 'production_policy',args=(runtime_policy(policy),))
                result.setdefault('policy_changes',[]).append(changed)
                if len(changed)!=8 or any(x['details']['selected_layer_count']!=a.layers for x in changed):
                    raise AssertionError('Block-FP8 did not load every requested QKV layer on all eight ranks')
                baseline=policy_baselines.get(policy);current_policy=policy
            if name.startswith('native-policy-abba-'):
                arm = int(name.rsplit('-', 1)[-1])
                document = typed_policy(policies[-1])
                llm.collective_rpc('native_configure',args=(False,))
                method = 'production_policy_reference' if arm in (0,3) else 'production_policy'
                changed = llm.collective_rpc(method,args=(document,))
                result.setdefault('migration_abba',[]).append({'arm':arm, 'installer':method,
                    'resolved_selection':document, 'worker_records':changed})
                # A policy switch invalidates Lazy/native graph caches. Warm
                # the same complete token/page sequence before EVERY arm,
                # including repeated B, rather than letting only B2 inherit
                # a warm page-boundary graph from B1. Keep the cold work as
                # separate evidence, outside the measured native window.
                warm_controls=llm.collective_rpc('native_configure',args=(False,))
                steps.clear();counts.clear();warm_start=time.monotonic()
                warm_output=llm.generate([prompt],SamplingParams(
                    temperature=0,max_tokens=a.tokens,ignore_eos=True),use_tqdm=False)[0].outputs[0]
                warm_ids=list(warm_output.token_ids)
                warm=dict(arm=name,policy=policy,controls=warm_controls,
                          wall_s=time.monotonic()-warm_start,ids=warm_ids,text=warm_output.text,
                          steps=list(steps),token_count_pass=len(warm_ids)==a.tokens,
                          bitwise_token_match=warm_ids==policy_baselines[policy],
                          quality_pass='7391' in warm_output.text,
                          scope='bridge preconditioning; cold costs retained outside native ABBA timing')
                result.setdefault('native_policy_preconditioning',[]).append(warm);save()
                if not (warm['token_count_pass'] and warm['bitwise_token_match']):
                    raise AssertionError('ABBA preconditioning changed the resident output: '+name)
                if a.layers==70 and not warm['quality_pass']:
                    raise AssertionError('ABBA preconditioning failed full-model retrieval: '+name)
            if policy is not None:name=policy+'-'+name
            controls=llm.collective_rpc('native_configure',args=(active,active and a.single_rpc,active and a.reuse_pages));steps.clear();counts.clear()
            begin=time.monotonic()
            output=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=a.tokens,ignore_eos=True),use_tqdm=False)[0].outputs[0]
            ids=list(output.token_ids)
            if baseline is None:baseline=ids;policy_baselines[policy]=ids
            if reference_ids is None:reference_ids=ids
            raw=llm.collective_rpc('native_summary')
            placement=llm.collective_rpc('host_placement_snapshot')
            run=dict(name=name,policy=policy,active=active,controls=controls,wall_s=time.monotonic()-begin,ids=ids,text=output.text,
                     token_count_pass=len(ids)==a.tokens,bitwise_token_match=ids==baseline,quality_pass='7391' in output.text,
                     steps=list(steps),ranks=raw,host_placement=placement)
            run['native_used_all_ranks']=len(raw)==8 and all(len(x['native_steps'])>0 for x in raw)
            if policy is not None:
                run['production_state']=llm.collective_rpc('production_snapshot')
                branch='per_block_fp8/fp32' if policy.startswith('decode_a8') else 'bf16/fp32'
                if any(x['block_fp8']['python_apply_branch_counts'].get(branch,0)==0 for x in run['production_state']):
                    raise AssertionError('Requested production arithmetic branch was never captured')
                swa_key='fp32_av_hoist' if 'swa_fp32_av_hoist' in policy.split('+')[1:] else 'fp32_quad' if 'swa_fp32_quad' in policy.split('+')[1:] else 'fp32_fast' if 'swa_fp32_fast' in policy.split('+')[1:] else None
                if swa_key and any(x.get('swa',{}).get('policy')!=swa_key or x.get('swa',{}).get('by_policy',{}).get(swa_key,{}).get('eligible_custom_calls',0)==0
                                   for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested SWA kernel was never captured on every rank')
                if '+gp_vec' in policy and any(x.get('gp',{}).get('python_capture_calls',{}).get('vector_fetch',0)==0
                                               for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested vector-fetch GP was never captured on every rank')
                gp_counter='vector_fetch_scale_tail' if 'gp_scale_tail' in policy.split('+')[1:] else 'vector_fetch_folded'
                if '+gp_fold' in policy and any(x.get('gp',{}).get('policy')!=gp_counter or x.get('gp',{}).get('python_capture_calls',{}).get(gp_counter,0)==0
                                                for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested folded GP was never captured on every rank')
                if gp_counter=='vector_fetch_scale_tail' and any(
                        not x.get('gp',{}).get('scale_tail',{}).get('prepared')
                        or not x.get('gp',{}).get('scale_tail',{}).get('selected')
                        or x.get('gp',{}).get('scale_tail',{}).get('operator')!='gaudi_gp_scale_tail::gp'
                        for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('GP scale-tail namespace or prepared libraries missing on a rank')
                suffixes=set(policy.split('+')[1:])
                from tools.validation.executor.moe_sum_bf16_runtime_gates import capture_gate as moe_sum_gate
                run['moe_sum_bf16_capture_gate']=moe_sum_gate(policy,run['production_state'])
                if not run['moe_sum_bf16_capture_gate']['pass']:
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested MoE final sum branch was not captured on every rank')
                for suffix,reduction in [('qkv_neumaier','neumaier_fp32'),('qkv_neumaier_isa','neumaier_isa_fp32')]:
                    if suffix in suffixes and any(x['block_fp8'].get('reduction_capture_counts',{}).get(reduction,0)==0
                                                  for x in run['production_state']):
                        result['failed_production_dispatch']=run;save()
                        raise AssertionError('Requested QKV reducer was never captured on every rank: '+suffix)
                if 'qkv_neumaier_isa' in suffixes and any(
                        x['block_fp8'].get('reduction_operator')!='gk_reduce_isa::handschedule'
                        or not x.get('qkv_neumaier_isa',{}).get('prepared')
                        or not x.get('qkv_neumaier_isa',{}).get('selected') for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('ISA QKV reducer namespace or artifact preparation was not recorded on every rank')
                if '+f32_ag' in policy and any(x.get('collective',{}).get('fp32_ag_capture_calls',0)==0
                                              for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested FP32 graph AG was never captured on every rank')
                if '+qkv_post' in policy and any(x.get('qkv_postprocess',{}).get('by_policy',{}).get('fused',{}).get('eligible_custom_calls',0)==0
                                                for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested QKV postprocess was never captured on every rank')
                from tools.validation.executor.qkv_post_full_runtime_gates import capture_gate as full_post_gate
                run['qkv_post_full_capture_gate']=full_post_gate(
                    policies[-1] if 'native-policy-abba-' in name else policy, run['production_state'])
                if not run['qkv_post_full_capture_gate']['pass']:
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested full-attention QKV postprocess was not captured on every admitted layer/rank')
                if '+norm_qkv_grid24' in policy and policy.startswith('decode_a8') and any(
                        x.get('norm_grid24',{}).get('policy')!='grid24'
                        or x.get('norm_grid24',{}).get('counts',{}).get('eligible_custom_calls',0)==0
                        or not x.get('norm_grid24',{}).get('producer_binding') for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Actual pinned grid24 producer was never captured on every rank')
                if '+down_vec' in policy and any(x.get('down',{}).get('python_capture_calls',{}).get('vector_fetch',0)==0
                                                 for x in run['production_state']):
                    result['failed_production_dispatch']=run;save()
                    raise AssertionError('Requested vector-fetch down was never captured on every rank')
                for router_variant in ('scalar','vector'):
                    if '+router_post_'+router_variant in policy and any(
                            x.get('router_post',{}).get('applied_capture_calls',{}).get(router_variant,0)==0
                            for x in run['production_state']):
                        result['failed_production_dispatch']=run;save()
                        raise AssertionError('Requested router post-TopK candidate was never captured on every rank')
            run['all_capture_gates_pass']=all(c['status']=='NUMERICAL_REPLAY_PASS' for x in raw for c in x['captures'])
            run['performance_qualified']=performance_qualified(a.layers,run)
            run['execution_gate_pass']=run['performance_qualified']
            if a.production_policies:
                run['performance_qualified']=False
                run['quality_state']='PENDING_FULL_LOGITS_AND_CHAT'
            steady=[s for s in steps if s['new_tokens']>0 and s['emitted']>=min(32,a.tokens//2) and s['emitted']<a.tokens]
            if steady:
                duration=(steady[-1]['end_ns']-steady[0]['begin_ns'])/1e9
                run['serving_timing' if run['performance_qualified'] else 'diagnostic_timing']=dict(steady_steps=len(steady),tps=sum(s['new_tokens'] for s in steady)/duration,
                                          p50_itl_ms=statistics.median(s['ms'] for s in steady),
                                          scope='coordinator wall, includes capture/boundary fallback within the window; no outlier removal')
            result['runs'].append(run);save()
            print('NATIVE_SERVICE_PHASE',json.dumps({k:v for k,v in run.items() if k not in ('steps','ranks','ids','text')}),flush=True)
            if not (run['token_count_pass'] and run['bitwise_token_match'] and run['all_capture_gates_pass']):raise AssertionError('Native/bridge execution gate failed: '+name)
            if active and a.single_rpc and not all(x['single_rpc_steps'] for x in raw):raise AssertionError('Single RPC path not used')
            if active and not run['native_used_all_ranks']:raise AssertionError('Native backend was not used on every rank')
            if a.layers==70 and not run['quality_pass']:raise AssertionError('Full-model code retrieval failed: '+name)
        if execution_context().native.boundary_positions:
            requested=set(execution_context().native.boundary_positions)
            result['boundary_tensor_checks']=[]
            for policy in policies:
                bridge=next(r for r in result['runs'] if r['policy']==policy and not r['active'])
                native=next(r for r in result['runs'] if r['policy']==policy and r['active'])
                for rank,(left,right) in enumerate(zip(bridge['ranks'],native['ranks'])):
                    x={r['position']:r for r in left['boundary_audit']}
                    y={r['position']:r for r in right['boundary_audit']}
                    complete=set(x)==set(y)==requested
                    equal=complete and all(x[pos]['tensors']==y[pos]['tensors'] for pos in requested)
                    result['boundary_tensor_checks'].append(dict(policy=policy,rank=rank,
                        positions=sorted(requested),complete=complete,all_tensor_bits_match=equal,
                        native_modes={str(pos):row['mode'] for pos,row in y.items()}))
                    save()
                    if not equal:raise AssertionError('Boundary hidden/selected/logit comparison failed')
        result['arithmetic_preserving_gp_pairs']=[]
        for policy,ids in policy_baselines.items():
            if policy:
                from tools.validation.executor.qkv_post_full_runtime_gates import arithmetic_baseline
                for suffix in ('+gp_vec','+gp_fold','+down_vec','+qkv_post','+qkv_post_full'):
                    old=arithmetic_baseline(policy,suffix)
                    if old is None:continue
                    if old in policy_baselines:
                        equal=ids==policy_baselines[old]
                        result['arithmetic_preserving_gp_pairs'].append({'kernel':suffix[1:],'baseline':old,'candidate':policy,'tokens_bitwise_match':equal})
                        save()
                        if not equal:raise AssertionError('Arithmetic-preserving MoE kernel changed model tokens: '+policy)
        if a.profile_native:
            if policies[-1] is not None:
                llm.collective_rpc('native_configure',args=(False,))
                llm.collective_rpc('production_policy',args=(runtime_policy(policies[-1]),))
            llm.collective_rpc('native_configure',args=(True,a.single_rpc,a.reuse_pages))
            armed=llm.collective_rpc('native_profile',args=(3,))
            result['device_profile']={'armed':armed,'policy':policies[-1],
                                      'scope':'separate run; excluded from serving timing'};save()
            if all(x['armed'] for x in armed):
                generated=llm.generate([prompt],SamplingParams(temperature=0,max_tokens=32,ignore_eos=True),use_tqdm=False)[0].outputs[0]
                result['device_profile']['ranks']=llm.collective_rpc('native_summary')
                result['device_profile']['ids']=list(generated.token_ids)
            llm.collective_rpc('native_configure',args=(False,));save()
        if a.production_policies and (not a.integration_preflight or a.cpu_oracle):
            import torch
            from gaudi_kernels.serving.executor.production_quality import compare_logits
            llm.collective_rpc('native_configure',args=(False,))
            quality_reference=policies[0] if quality_plan is None else quality_plan['reference_policy']
            if quality_plan is not None:reference_ids=policy_baselines[quality_reference]
            result['teacher_forced']={'status':'RUNNING','reference_policy':quality_reference,
                                     'prompt_ids':prompt['prompt_token_ids'],'reference_ids':reference_ids,'rows':[]}
            references={};captures={}
            checked_policies=policies if quality_plan is None else quality_plan['teacher_policies']
            suffixes=list(dict.fromkeys(numeric_suffix(p) for p in checked_policies if p.startswith('decode_a8')))
            oracle_bases=('cpu_ocp_a8_bf16_fp32','cpu_native_a8_bf16_fp32') if quality_plan is None else (
                ('cpu_native_a8_fp32_v1','cpu_ocp_a8_fp32_v1') if a.fp32_ocp_model_reference else ('cpu_native_a8_fp32_v1',))
            oracle_policies=[base+suffix for suffix in (suffixes or ['']) for base in oracle_bases] if a.cpu_oracle else []
            quality_policies=checked_policies+oracle_policies
            positions=(0,) if a.integration_preflight else (0,16,48)
            for policy in quality_policies:
                llm.collective_rpc('production_policy',args=(runtime_policy(policy),))
                for position in positions:
                    tag=policy.replace('_','-').replace('+','-')+'-teacher-'+str(position)
                    paths=llm.collective_rpc('production_quality_capture',args=(tag,) if quality_plan is None else (tag,'plain'))
                    forced=prompt['prompt_token_ids']+reference_ids[:position]
                    forced_next=reference_ids[position]
                    output=llm.generate([{'prompt_token_ids':forced}],SamplingParams(
                        temperature=0,max_tokens=2,ignore_eos=True,allowed_token_ids=[forced_next]),use_tqdm=False)[0].outputs[0]
                    capture_finished=llm.collective_rpc('production_quality_capture',args=(None,))
                    path=next(x['path'] for x in paths if x['rank']==0)
                    captured=torch.load(path,weights_only=True,map_location='cpu')
                    captures[policy,position]=captured
                    if policy==quality_reference:references[position]=captured
                    ref=references[position]
                    aligned=(torch.equal(ref['input_ids'],captured['input_ids']) and
                             torch.equal(ref['positions'],captured['positions']))
                    check=compare_logits(ref['logits'],captured['logits'],contract=a.quality_contract) if aligned else {'pass':False}
                    row={'policy':policy,'position':position,'ids':list(output.token_ids),
                         'forced_decode_token':forced_next,'same_decode_input':aligned,'logits_path':path,'check':check,
                         'stage_paths':[x['stage_path'] for x in paths if x.get('stage_path')],
                         'same_input_qkv_audit':capture_finished,
                         'reference_scope':'A16 control; A8 comparisons include intentional activation quantization'}
                    if quality_plan is not None:
                        from tools.validation.executor.fp32_quality_contract import role
                        row.update(tag=tag,quality_role=role(policy,quality_plan),
                                   decode_position=int(captured['positions'].item()),
                                   reference_scope='explicit A16 reference; reference-vs-reference rows are diagnostic, not candidate gates')
                    if captured['input_ids'].numel()!=1 or int(captured['input_ids'].item())!=forced_next:
                        raise AssertionError('Teacher forcing did not provide the specified decode token')
                    if quality_plan is not None and policy in quality_plan['candidate_policies']:
                        # Keep the graph-enabled teacher output as the primary model
                        # measurement. A second, graph-disabled pass collects staged
                        # evidence and must match the completed primary logits in full.
                        from gaudi_kernels.block_fp8_fp32_contract import recipe_relation
                        audit_tag=tag+'-staged'
                        llm.collective_rpc('production_policy',args=(runtime_policy(policy),))
                        audit_paths=llm.collective_rpc('production_quality_capture',args=(audit_tag,'staged'))
                        llm.generate([{'prompt_token_ids':forced}],SamplingParams(
                            temperature=0,max_tokens=2,ignore_eos=True,allowed_token_ids=[forced_next]),use_tqdm=False)
                        audited=llm.collective_rpc('production_quality_capture',args=(None,))
                        audit_path=next(x['path'] for x in audit_paths if x['rank']==0)
                        diagnostic=torch.load(audit_path,weights_only=True,map_location='cpu')
                        relation=recipe_relation(captured['logits'],diagnostic['logits'])
                        relation['same_decode_input']=(torch.equal(captured['input_ids'],diagnostic['input_ids'])
                            and torch.equal(captured['positions'],diagnostic['positions']))
                        relation['scope']='complete graph-enabled teacher versus graph-disabled staged pass; no TPS'
                        row.update(audit_tag=audit_tag,same_input_qkv_audit=audited,
                                   audit_logits_path=audit_path,plain_vs_audit_logits=relation)
                    result['teacher_forced']['rows'].append(row);save()
                    if not check.get('finite',True):raise AssertionError('Nonfinite full model logits: '+tag)
            gate_rows=result['teacher_forced']['rows'] if quality_plan is None else [
                x for x in result['teacher_forced']['rows'] if x['quality_role']=='candidate']
            result['teacher_forced']['status']='PASS' if gate_rows and all(x['check']['pass'] for x in gate_rows) else 'FAIL'
            if a.cpu_oracle:
                result['same_contract_reference']=[]
                for reference_policy in oracle_policies:
                    for policy in checked_policies:
                        if not policy.startswith('decode_a8') or numeric_suffix(reference_policy)!=numeric_suffix(policy):continue
                        for position in positions:
                            ref=captures[reference_policy,position];cand=captures[policy,position]
                            aligned=torch.equal(ref['input_ids'],cand['input_ids']) and torch.equal(ref['positions'],cand['positions'])
                            result['same_contract_reference'].append({'reference_policy':reference_policy,'policy':policy,
                                'position':position,'same_decode_input':aligned,
                                'check':compare_logits(ref['logits'],cand['logits'],contract=a.quality_contract) if aligned else {'pass':False},
                                'required_for_acceptance':quality_plan is None or reference_policy.startswith('cpu_native_a8_fp32_v1')
                                    or (a.require_fp32_ocp_quality and reference_policy.startswith('cpu_ocp_a8_fp32_v1')),
                                'scope':('independent balanced FP32 M1 QKV with original OCP bytes' if reference_policy.startswith('cpu_ocp_')
                                    else 'independent balanced FP32 M1 QKV with native_half_rne_v1')
                                    + '; shared A16/native-half prefill and KV arithmetic; not GPU tree emulation' if quality_plan is not None
                                    else 'same block A8 quantization; FP64 MAC accuracy reference, not GPU bitwise emulation'})
            if quality_plan is not None and a.fp32_ocp_model_reference:
                result['native_half_adaptation']=[]
                for suffix in suffixes:
                    original_policy='cpu_ocp_a8_fp32_v1'+suffix
                    native_policy='cpu_native_a8_fp32_v1'+suffix
                    for position in positions:
                        ocp_capture=captures[original_policy,position];native_capture=captures[native_policy,position]
                        aligned=torch.equal(ocp_capture['input_ids'],native_capture['input_ids']) and torch.equal(ocp_capture['positions'],native_capture['positions'])
                        result['native_half_adaptation'].append(dict(original_policy=original_policy,native_policy=native_policy,
                            position=position,same_decode_input=aligned,
                            check=compare_logits(ocp_capture['logits'],native_capture['logits'],contract='fp32_arithmetic_v1') if aligned else {'pass':False},
                            role='adaptation diagnostic; candidate-vs-original OCP gate is reported separately',
                            scope='same balanced FP32 algorithm, native-half versus original-OCP M1 QKV weights/activations; shared A16/native-half prefill/KV; whole-model propagation, not FP32-order error'))
            save()
            # CPU readback is excluded from serving and normal chat. Restore a
            # device policy and invalidate all diagnostic graphs first.
            llm.collective_rpc('production_policy',args=(runtime_policy(policies[-1]),))
            quality_tasks=[
                ('arithmetic','Compute 17 + 25. Output only the integer, with no explanation.','42'),
                ('retrieval','Record: the sealed envelope contains code 4826. Output only that code.','4826'),
                ('chinese','请把“苹果、香蕉、苹果、梨”中重复的水果名称输出一次，只输出水果名称。','苹果')]
            result['chat_quality']=[]
            for policy in ([] if a.integration_preflight else checked_policies):
                llm.collective_rpc('production_policy',args=(runtime_policy(policy),))
                for task,text,expected in quality_tasks:
                    rendered=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,
                                                    add_generation_prompt=True,enable_thinking=False)
                    answer=llm.generate([rendered],SamplingParams(temperature=0,max_tokens=128),use_tqdm=False)[0].outputs[0]
                    normalized=answer.text.strip().strip('.。!！\n ')
                    row={'policy':policy,'task':task,'expected':expected,'text':answer.text,'token_ids':list(answer.token_ids),
                         'pass':normalized==expected,'scope':'small deterministic chat sanity gate'}
                    if quality_plan is not None:
                        row.update(finish_reason=answer.finish_reason,
                                   normal_eos_pass=answer.finish_reason=='stop' and bool(answer.token_ids)
                                       and answer.token_ids[-1]==tok.eos_token_id)
                    result['chat_quality'].append(row);save()
        if batch_sizes and a.batch_policy_abba_control:
            from tools.validation.executor.batch_policy_abba import run_batch_policy_abba
            from tools.validation.executor.moe_dispatch_gates import batch_capture_gate
            capture_gate = batch_capture_gate if any('moe_compact8' in p.split('+')[1:] for p in policies) else None
            def make_batch_prompts(batch):
                prompts=[];codes=[]
                for index in range(batch):
                    code=str(7301+index*17);codes.append(code)
                    lead=tok.encode('The verification code for record '+str(index)+' is '+code+'.\n',add_special_tokens=False)
                    tail=tok.encode('\nQuestion: State the verification code for record '+str(index)+'.\nAnswer:',add_special_tokens=False)
                    count=512-len(lead)-len(tail)
                    if count<=0:raise ValueError('Batch retrieval prompt exceeds 512-token fixture')
                    prompts.append({'prompt_token_ids':lead+(filler*((count+len(filler)-1)//len(filler)))[:count]+tail})
                return prompts,codes
            def generate_batch(prompts,token_limit):
                return llm.generate(prompts,SamplingParams(temperature=0,max_tokens=token_limit,ignore_eos=True),use_tqdm=False)
            run_batch_policy_abba(rpc=llm.collective_rpc,generate=generate_batch,make_prompts=make_batch_prompts,
                policies=policies,control=a.batch_policy_abba_control,batches=batch_sizes,
                token_limit=a.batch_tokens,layers=a.layers,expect_bitwise=a.batch_policy_abba_expect_bitwise,
                steps=steps,counts=counts,result=result,save=save,capture_gate=capture_gate)
        elif batch_sizes:
            llm.collective_rpc('native_configure',args=(False,))
            batch_policy=policies[-1]
            if batch_policy is not None:llm.collective_rpc('production_policy',args=(runtime_policy(batch_policy),))
            result['batch_runs']=[]
            for batch in batch_sizes:
                prompts=[];codes=[]
                for index in range(batch):
                    code=str(7301+index*17);codes.append(code)
                    lead=tok.encode('The verification code for record '+str(index)+' is '+code+'.\n',add_special_tokens=False)
                    tail=tok.encode('\nQuestion: State the verification code for record '+str(index)+'.\nAnswer:',add_special_tokens=False)
                    count=512-len(lead)-len(tail)
                    prompts.append({'prompt_token_ids':lead+(filler*((count+len(filler)-1)//len(filler)))[:count]+tail})
                previous=None
                for trial in ('warmup','measured'):
                    steps.clear();counts.clear();start=time.monotonic()
                    outputs=llm.generate(prompts,SamplingParams(temperature=0,max_tokens=a.batch_tokens,ignore_eos=True),use_tqdm=False)
                    elapsed=time.monotonic()-start
                    texts=[x.outputs[0].text for x in outputs];ids=[list(x.outputs[0].token_ids) for x in outputs]
                    quality=all(code in text for code,text in zip(codes,texts))
                    steady=[s for s in steps if s['new_tokens']>0 and 32<=s['emitted']<a.batch_tokens]
                    timing=None
                    if steady:
                        seconds=(steady[-1]['end_ns']-steady[0]['begin_ns'])/1e9
                        timing={'aggregate_serving_tps':sum(s['new_tokens'] for s in steady)/seconds,
                                'median_step_ms':statistics.median(s['ms'] for s in steady),
                                'scope':'coordinator window; may include staggered prefill and partial batches'}
                    from tools.validation.executor.batch_timing import full_batch_window
                    full_batch=full_batch_window(steps,batch,a.batch_tokens)
                    row={'batch':batch,'trial':trial,'policy':batch_policy,'prompt_tokens_each':512,
                         'generated_tokens_each':a.batch_tokens,'wall_s':elapsed,'texts':texts,'ids':ids,
                         'quality_pass':quality,'repeat_match':previous is None or ids==previous,
                         'serving_timing':timing if quality else None,'steps':list(steps),
                         'full_batch_decode_timing':full_batch if quality else None,
                         'scope':'vLLM coordinator wall; original expert broadcast remains the large-M control'}
                    if batch_policy and 'moe_compact8' in batch_policy.split('+')[1:]:
                        states=llm.collective_rpc('production_snapshot')
                        row['production_state']=states
                        used=(len(states)==8 and all(
                            x.get('moe_dispatch',{}).get('policy')=='compact8'
                            and x.get('moe_dispatch',{}).get('compact_max_rows')==8
                            and x.get('moe_dispatch',{}).get('python_capture_calls_by_rows',{}).get(str(batch),{}).get(
                                'compact' if batch<=8 else 'broadcast',0)>0 for x in states))
                        row['moe_dispatch_captured_all_ranks']=used
                        row['scope']='bridge mature-batch timing; explicit compact8 MoE capture dispatch with M>8 broadcast fallback'
                        if not used:
                            result['batch_runs'].append(row);save()
                            raise AssertionError('Requested MoE batch dispatch was never captured on every rank')
                    result['batch_runs'].append(row);save()
                    previous=ids
        if a.boundary_hashes:
            from tools.validation.boundary_gate import run as boundary_gate
            result['boundary_hashes']=boundary_gate(llm,prompt,reference_ids,typed_policy(policies[-1]),a.layers)
            save()
        if a.lifecycle:
            from tools.validation.executor.native_lifecycle_test import check_lifecycle
            result['lifecycle']=check_lifecycle(llm,prompt,baseline,a.single_rpc,a.reuse_pages)
            save()
        if a.long_context:
            from tools.validation.executor.native_long_context_test import check_long_context
            result['long_context']={'status':'RUNNING','runs':[]}
            check_long_context(llm,a.long_context,a.tokens,a.layers,a.single_rpc,a.reuse_pages,
                               result['long_context'],steps,counts,save)
        result['status']='PASS' if a.layers==70 else 'DIAGNOSTIC'
        result['candidate_accepted']=a.layers==70 and all(x['quality_pass'] and x['bitwise_token_match'] for x in result['runs'])
        if quality_plan is not None:
            from tools.validation.executor.fp32_quality_contract import classify
            result['fp32_quality']=classify(result,quality_plan,positions,a.layers)
            result['candidate_accepted'] &= result['fp32_quality']['passed']
        elif a.production_policies and not a.integration_preflight:
            result['candidate_accepted'] &= result['teacher_forced']['status']=='PASS'
            result['candidate_accepted'] &= all(x['pass'] for x in result['chat_quality'])
            result['candidate_accepted'] &= all(x['check']['pass'] for x in result.get('same_contract_reference',[]))
        if batch_sizes:
            result['candidate_accepted'] &= all(x['quality_pass'] and x['repeat_match'] for x in result['batch_runs'])
        if a.layers==70 and not result['candidate_accepted']:
            result['status']='REJECTED'
        for run in result['runs']:
            run['model_candidate_accepted']=result['candidate_accepted']
            if a.production_policies:
                run['quality_state']='PASS' if result['candidate_accepted'] else 'REJECTED'
                run['performance_qualified']=run['execution_gate_pass'] and result['candidate_accepted']
                if run['performance_qualified'] and 'diagnostic_timing' in run:
                    run['serving_timing']=run.pop('diagnostic_timing')
        for run in result.get('long_context',{}).get('runs',[]):
            run['model_candidate_accepted']=result['candidate_accepted']
            if a.production_policies and not result['candidate_accepted']:
                run['performance_qualified']=False
                if 'serving_timing' in run:run['diagnostic_timing']=run.pop('serving_timing')
        for run in result.get('batch_runs',[]):
            run['model_candidate_accepted']=result['candidate_accepted']
            run['performance_qualified']=bool(result['candidate_accepted'] and run['quality_pass']
                and run['repeat_match'] and (run.get('full_batch_decode_timing') or {}).get('qualified'))
            run['quality_scope']='full_batch_decode_timing.qualified validates the timing window only; model acceptance is separate'
        if a.batch_policy_abba_control:
            from tools.validation.executor.batch_policy_abba import finalize_batch_abba
            finalize_batch_abba(result,a.layers)
        result['acceptance_scope']='bounded full-model B1 greedy execution/return path; not general serving readiness'
        save()
    except BaseException:
        result.update(status='FAIL',candidate_accepted=False,error=traceback.format_exc());save();raise
    finally:
        try:
            result['cleanup']=llm.collective_rpc('native_configure',args=(False,))
        except BaseException:
            # A dead worker may reject the drain RPC. Preserve the original
            # diagnostic failure and record cleanup separately; the outer
            # owned-process runner still performs bounded group cleanup.
            result['cleanup_error']=traceback.format_exc()
            if result.get('status')!='FAIL':
                result.update(status='FAIL',candidate_accepted=False)
                raise
        finally:
            save()

if __name__=='__main__':main()
