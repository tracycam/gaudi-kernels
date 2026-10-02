"""CPU/ABBA-only compatibility bridge around the untouched frozen installer.

Neither historical tags nor its environment adapter are imported by serving.
The legacy setter targets are the same qualified operators as the candidate.
"""
import hashlib
import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from gaudi_kernels.engine.context import context
from tools.validation.legacy_selection import PolicySelection

PIN = '5f5748c739a788fc95bad2da2cc669afdcfcfb375befeec1853870c71c6fc2ff'


def configure_reference(worker, document):
    selected = PolicySelection.from_dict(document)
    runtime = context()
    if runtime.native.active: raise ValueError('Disable replay before reference setup')
    path = Path(__file__).with_name('frozen70g_installer.py')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PIN: raise ValueError('Frozen installer SHA mismatch')
    from tools.validation import frozen70g_installer
    aliases = {}
    for name in ('gp_scale_tail_runtime', 'swa_av_runtime', 'qkv_neumaier_isa_runtime',
                 'norm_grid24_runtime', 'moe_sum_bf16_runtime', 'precision_runtime',
                 'batch_ops', 'router_post_runtime'):
        aliases[name] = importlib.import_module('gaudi_kernels.serving.executor.' + name)
    aliases['qkv_post_full_runtime'] = importlib.import_module('tools.validation.executor.qkv_post_full_runtime_gates')
    from gaudi_kernels.serving.executor import moe_dispatch_runtime as dispatch
    row_sets = {'baseline': (1,), 'compact8': tuple(range(1, 9))}
    aliases['moe_dispatch_runtime'] = SimpleNamespace(
        validate_policy=lambda value: dispatch.validate_rows(row_sets[value]),
        set_policy=lambda value: dispatch.set_rows(row_sets[value]))
    flags = {'NATIVE_BACKEND_ACTIVE': '0', 'UNIFIED_PRECISION_REDUCE': 'auto'}
    for name in ('GK_SWA_ENABLED', 'GK_MXFP4_FOLDED_ENABLED', 'GK_MXFP4_DOWN_ENABLED',
                 'GK_BLOCK_REDUCE_ISA_ENABLED', 'GK_FP32_GRAPH_AG_ENABLED', 'GK_QKV_POST_ENABLED',
                 'GK_NORM_ENABLED', 'GK_SWA_AV_ENABLED', 'GK_NORM_GRID24_ENABLED', 'GK_ROUTER_POST_ENABLED'):
        flags[name] = '1'
    with patch.dict(sys.modules, aliases), patch.dict(os.environ, flags):
        result = frozen70g_installer.configure_policy(worker, selected.label)
        runtime.collective_mode = result['collective_policy']
    runtime.selection = selected
    worker.model_runner._production_config = selected.engine
    result['resolved_config'] = selected.to_dict()
    result['migration_only_legacy_installer_reference'] = True
    return result
