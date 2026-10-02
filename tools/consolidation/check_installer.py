#!/usr/bin/env python3
"""Execute old/new installer logic against CPU spies, before model loading.

Spies substitute device/library setters only. The real frozen installer runs
unchanged and its ordered calls are compared with the generated installer for
all closed benchmark configurations. This catches variable shadowing and lost
branches without pretending to validate device arithmetic.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'));sys.path.insert(0,str(ROOT))
from tools.validation.legacy_selection import qualified_selections


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check(snapshot):
    calls = []
    evidence = []

    def spy(name, result=None):
        def called(*args, **kwargs):
            calls.append((name, args, kwargs))
            return result.copy() if isinstance(result, dict) else result
        return called

    modules = {}

    def module(name, methods):
        result = ModuleType(name)
        for key, value in methods.items():
            setattr(result, key, value)
        modules[name] = result
        return result

    module('torch', {'hpu': SimpleNamespace(synchronize=spy('synchronize'))})
    module('habana_frameworks', {})
    module('habana_frameworks.torch', {})
    module('habana_frameworks.torch.core', {'mark_step': spy('mark_step')})
    module('gaudi_kernels.production_integration', {'set_policy': spy('block_policy'),
            'set_reduction_policy': spy('block_reduce', {'ok': True})})
    for name in ('gp_scale_tail_runtime', 'swa_av_runtime', 'swa_quad_runtime',
                 'qkv_neumaier_isa_runtime', 'norm_grid24_runtime'):
        module(name, {'prepare': spy(name + '.prepare'), 'validate_policy': spy(name + '.validate'),
                      'record_selection': spy(name + '.select', {}), 'snapshot': spy(name + '.snapshot', {})})
    module('qkv_post_full_runtime', {'attention_scope': lambda suffixes: 'swa128_or_full' if 'qkv_post_full' in suffixes else 'swa128'})
    for name in ('moe_sum_bf16_runtime', 'moe_dispatch_runtime'):
        module(name, {'validate_policy': spy(name + '.validate'), 'set_policy': spy(name + '.select', {})})
    module('precision_runtime', {'COUNTS': {}})
    module('gaudi_kernels.vllm_norm', {'prepare_vllm_norm': spy('norm.prepare', {'matched': 70}),
                                    'set_norm_policy': spy('norm.select', {})})
    module('gaudi_kernels.vllm_swa_install', {'prepare_vllm_swa': spy('swa.prepare', {'matched': 60}),
        'set_swa_policy': spy('swa.select'), 'snapshot_swa_counters': spy('swa.snapshot', {})})
    module('gaudi_kernels.vllm_qkv_postprocess', {
        'install_vllm_qkv_postprocess': spy('qkv.post', {'prepared': {'matched': 70,
                                                 'matched_by_attention_kind': {'full': 10}}}),
        'snapshot_qkv_postprocess': spy('qkv.snapshot', {})})
    module('gaudi_kernels.vllm_norm_grid24', {
        'install': spy('grid24.select', {'prepared': {'matched': 60}}), 'snapshot': spy('grid24.snapshot', {})})
    module('batch_ops', {'set_gp_policy': spy('gp.select'), 'gp_snapshot': spy('gp.snapshot', {}),
                         'set_down_policy': spy('down.select'), 'down_snapshot': spy('down.snapshot', {})})
    module('router_post_runtime', {'validate_policy': spy('router.validate'),
                                  'set_policy': spy('router.select', {})})
    flags = {'NATIVE_BACKEND_ACTIVE': '0', 'UNIFIED_PRECISION_REDUCE': 'auto',
             'GK_CONSOLIDATION_INVENTORY': '0'}
    for name in ('GK_BLOCK_FP8_ENABLED', 'GK_SWA_ENABLED', 'GK_MXFP4_FOLDED_ENABLED',
                 'GK_MXFP4_DOWN_ENABLED', 'GK_FP32_GRAPH_AG_ENABLED', 'GK_QKV_POST_ENABLED',
                 'GK_BLOCK_REDUCE_ISA_ENABLED', 'GK_NORM_ENABLED', 'GK_SWA_AV_ENABLED',
                 'GK_NORM_GRID24_ENABLED', 'GK_ROUTER_POST_ENABLED'):
        flags[name] = '1'
    flags['GK_SWA_QUAD_ENABLED'] = '0'
    flags['GK_MXFP4_GP_ENABLED'] = '0'
    # Both functions are repository/asset source. Nothing is patched or generated.
    from gaudi_kernels.engine import context as runtime_context
    stub=SimpleNamespace(startup=SimpleNamespace(inventory=False), native=SimpleNamespace(active=False),
                         collective_mode='auto', has=lambda name:True)
    modules['moe_dispatch_runtime'].validate_rows=modules['moe_dispatch_runtime'].validate_policy
    def rows_select(rows):
        name='compact8' if rows==tuple(range(1,9)) else 'baseline'
        calls.append(('moe_dispatch_runtime.select',(name,),{}));return {}
    modules['moe_dispatch_runtime'].set_rows=rows_select
    for name, value in list(modules.items()):
        if '.' not in name or name in ('habana_frameworks.torch.core',):
            if name.endswith('_runtime') or name=='batch_ops':
                modules['gaudi_kernels.serving.executor.'+name]=value
    with patch.dict(sys.modules, modules), patch.dict(os.environ, flags, clear=True), patch.object(runtime_context,'context',return_value=stub):
        old = import_file('historical_quality_spy', snapshot / 'production_quality.py')
        new = import_file('canonical_quality_spy', ROOT / 'python/gaudi_kernels/serving/executor/production_quality.py')
        checked = 0
        for label, document in qualified_selections().items():
            # Diagnostic CPU labels may use a different suffix order;
            # use the typed selection's canonical old label for equivalence.
            from tools.validation.legacy_selection import PolicySelection
            selected = PolicySelection.from_dict(document)
            observed = []
            for function, argument in ((old.configure_policy, selected.label),
                                       (new.configure_policy, document)):
                calls.clear()
                os.environ['UNIFIED_PRECISION_REDUCE'] = 'auto'
                runner = SimpleNamespace(model='model-owner', _native_reset=spy('native.reset'))
                runner.model = SimpleNamespace(clear_cache=spy('cache.clear'))
                worker = SimpleNamespace(rank=0, model_runner=runner)
                result = function(worker, argument)
                # Model owner object identities differ; compare call
                # sequences after substituting that single named owner.
                def normalize(value):
                    if value is runner.model:
                        return 'model-owner'
                    if isinstance(value, (list, tuple)):
                        return tuple(normalize(v) for v in value)
                    if isinstance(value, dict):
                        return {k: normalize(v) for k, v in value.items()}
                    return value
                setters={'mark_step','synchronize','native.reset','cache.clear','block_policy','block_reduce',
                    'qkv_neumaier_isa_runtime.select','norm.select','swa.select','qkv.post','grid24.select',
                    'gp.select','down.select','router.select','moe_dispatch_runtime.select','moe_sum_bf16_runtime.select'}
                normalized=[]
                for name,args_,kwargs_ in calls:
                    if name not in setters:continue
                    if name=='qkv.post':
                        kwargs_={**kwargs_};kwargs_.setdefault('attention_scope','swa128')
                    normalized.append((name,normalize(args_),normalize(kwargs_)))
                observed.append((normalized, selected.engine.arithmetic_fingerprint, result['collective_policy']))
                if function is new.configure_policy:
                    assert runner._production_config == selected.engine
                    assert result['resolved_config'] == document
            if observed[0] != observed[1]:
                raise AssertionError('Installer setter calls changed: ' + label + '\n' + str(observed))
            evidence.append({'historical_label': label, 'resolved_selection': document,
                'calls_match': True, 'ordered_setter_calls': observed[0][0],
                'collective_policy': observed[0][2]})
            checked += 1
        # Exercise the actual migrated batch client through every arm and
        # its final restore RPC. No device/framework calls are simulated
        # as quality evidence: these spies check serialization only.
        batch = import_file('batch_rpc_serialization_spy', ROOT / 'tools/validation/executor/batch_policy_abba.py')
        table = qualified_selections()
        device = [k for k, v in table.items() if v['reference'] == 'device']
        control = next(k for k in device if k.endswith('qkv_post_full') and 'moe_compact8' not in k)
        candidate = next(k for k in device if k.endswith('qkv_post_full') and 'moe_compact8' in k)
        rpc_documents = []
        steps, counts, result = [], [], {}

        def rpc(method, args=()):
            if method == 'production_policy':
                PolicySelection.from_dict(args[0])
                rpc_documents.append(args[0])
                return [{'details': {'selected_layer_count': 70}} for _ in range(8)]
            if method == 'native_summary':
                return [{'native_steps': []} for _ in range(8)]
            if method == 'native_configure':
                return [{'active': False} for _ in range(8)]
            return [{} for _ in range(8)]

        def prompts(rows):
            return ([{'prompt_token_ids': [1]*512} for _ in range(rows)], ['7391']*rows)

        def generate(values, token_limit):
            for i in range(token_limit):
                steps.append({'known_requests': len(values), 'emitted_min': i+1, 'emitted': i+1,
                    'per_request_new': [1]*len(values), 'new_tokens': len(values),
                    'begin_ns': i*10000000, 'end_ns': (i+1)*10000000, 'ms': 10.0})
            return [SimpleNamespace(outputs=[SimpleNamespace(text='7391', token_ids=list(range(token_limit)))])
                    for _ in values]

        batch.run_batch_policy_abba(rpc=rpc, generate=generate, make_prompts=prompts,
            policies=[control,candidate], control=control, batches=[1,2,8], token_limit=160,
            layers=70, expect_bitwise=True, steps=steps, counts=counts, result=result, save=lambda:None)
        if len(rpc_documents) != 13:
            raise AssertionError('Incomplete batch protocol coverage')
    return {'configurations_checked': checked, 'pass': True, 'configurations': evidence,
            'batch_rpc_documents_checked': len(rpc_documents),
            'historical_source_sha256': hashlib.sha256((snapshot / 'production_quality.py').read_bytes()).hexdigest(),
            'scope': 'Real frozen installer logic with CPU spies for setters; not device arithmetic or model quality'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot-source', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = check(a.snapshot_source.resolve())
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2) + '\n')
    print('Equivalent installer CPU configurations:', result['configurations_checked'])
