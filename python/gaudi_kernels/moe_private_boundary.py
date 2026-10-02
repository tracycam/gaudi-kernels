"""Explicit, pinned private-ABI MoE boundary experiment. No import-time action.

The original executor descriptors and all numeric casts stay in place. This
module only replaces the audited three clone sites, optionally followed by the
separate precision-combine clone. Existing captured graphs must be discarded
before changing flags. No production qualification is implied by installation.
"""
import hashlib
import os
from pathlib import Path
import re

_pin = None


def install(extension_path, expected_sha256):
    global _pin
    if os.environ.get('PT_HPU_LAZY_MODE') != '1':
        raise RuntimeError('private MoE boundary requires explicit Lazy mode 1')
    if os.environ.get('PT_HPU_LAZY_ACC_PAR_MODE') != '0':
        raise RuntimeError('private MoE boundary is qualified only with accumulation parallel mode 0')
    if os.environ.get('PT_ENABLE_INT64_SUPPORT', 'false').lower() not in ('0', 'false'):
        raise RuntimeError('private MoE boundary requires the audited int32 index contract')
    path = Path(extension_path).resolve(strict=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise RuntimeError('private helper binary fingerprint mismatch')
    import torch
    if hasattr(torch.ops.gaudi_view_safe, 'runtime_pin') and str(path) not in torch.ops.loaded_libraries:
        raise RuntimeError('private helper namespace was loaded from another library')
    torch.ops.load_library(str(path))
    runtime = torch.ops.gaudi_view_safe.runtime_pin()
    match = re.fullmatch(r'private_abi_lazy_only sha256=([0-9a-f]{64}) build_id=([0-9a-f]+) path=(.+)', runtime)
    if not match or hashlib.sha256(Path(match[3]).read_bytes()).hexdigest() != match[1]:
        raise RuntimeError('pinned bridge file fingerprint changed; loaded build-ID alone is insufficient')
    _pin = dict(extension=str(path), binary_sha256=expected_sha256, runtime=runtime,
                runtime_file_sha256_verified=match[1])
    return dict(_pin)


def normalize_inputs(tensors):
    if _pin is None:
        raise RuntimeError('install the exact pinned private helper before capture')
    import torch
    return torch.ops.gaudi_view_safe.normalize_list(tensors)


def prepare(x, ids, routing, precise, mode):
    if mode not in ('compact', 'broadcast'):
        raise ValueError('private clone candidate only admits compact/broadcast')
    import torch
    # Explicit conversions remain even when metadata suggests they are no-ops.
    return normalize_inputs([x, ids.to(torch.int32),
                             routing.to(torch.float32 if precise else torch.bfloat16)])


def rewrite_batch_source(source, namespace):
    old = '    x=x.clone();ids=ids.to(torch.int32).clone();routing=routing.to(torch.float32 if precise else torch.bfloat16).clone()'
    if source.splitlines().count(old) != 1:
        raise ValueError('pinned executor three-clone line changed')
    new = """    if os.getenv('GK_MOE_PRIVATE_VIEWS','0')=='1':
        x,ids,routing=_gk_private_prepare(x,ids,routing,precise,mode)
    else:
""" + '    ' + old
    result = source.replace(old, new)
    compile(result, '<moe-private-boundary>', 'exec')
    namespace['_gk_private_prepare'] = prepare
    return result


def rewrite_precision_source(source, namespace):
    old = '    return torch.ops.precision_fix.combine(p,r.float().clone(),d,h)'
    if '_gk_private_normalize' in source or source.splitlines().count(old) != 1:
        raise ValueError('pinned precision-combine clone line changed')
    new = """    if os.getenv('GK_MOE_PRIVATE_COMBINE','0')=='1':
        route=_gk_private_normalize([r.float()])[0]
        return torch.ops.precision_fix.combine(p,route,d,h)
""" + old
    result = source.replace(old, new)
    # Current precision_ops has no os import; bind it without altering unrelated
    # source imports or re-executing an already loaded module's original file.
    namespace.update(os=os, _gk_private_normalize=normalize_inputs)
    compile(result, '<moe-private-combine>', 'exec')
    return result


def snapshot():
    return {'pin': _pin, 'remove_initial_three': os.getenv('GK_MOE_PRIVATE_VIEWS') == '1',
            'remove_precision_combine': os.getenv('GK_MOE_PRIVATE_COMBINE') == '1',
            'scope': 'explicit private-ABI experiment; not default production dispatch'}
