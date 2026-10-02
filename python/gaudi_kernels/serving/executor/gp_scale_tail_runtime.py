"""Default-off, byte-qualified compact GP schedule. Import does no Torch/device work."""
from gaudi_kernels.engine.context import context as execution_context
import hashlib
import json
import os
from pathlib import Path
import shutil
QUALIFIED_SHA256 = {'tpc': 'f45f038808d6ec6ba14ac99445fe7bdce35a523c6f63271a3f37624e57130e07', 'torch': 'c73a48e86e32fb5b531cc97d5c205bc35f28c654820773cd82814702688db410'}
PREFIX = 'GK_MXFP4_SCALE_TAIL_'
NAMESPACE = 'gaudi_gp_scale_tail'
OPERATOR = 'gp'
SCHEMA = 'gaudi_gp_scale_tail::gp(Tensor w, Tensor scale, Tensor x, Tensor table, Tensor ids) -> Tensor'
_LIBRARIES = {}
_OP = None
_SELECTED = False
_LOADS = 0
_META_CHECKS = 0

def verify_libraries():
    libraries = execution_context().libraries('scale_tail', QUALIFIED_SHA256)
    registered = {str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH', '').split(':') if p}
    if libraries['tpc']['path'] not in registered:
        raise ValueError('Qualified TPC artifact missing from GC_KERNEL_PATH')
    return libraries

def prepare():
    global _OP, _LIBRARIES, _LOADS, _META_CHECKS
    if not execution_context().has('scale_tail'):
        return snapshot()
    libraries = verify_libraries()
    if _OP is not None:
        if libraries != _LIBRARIES:
            raise ValueError('GP scale-tail paths changed after preparation')
        _verify_operator()
        return snapshot()
    import torch
    namespace = getattr(torch.ops, NAMESPACE)
    loaded = {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
    if hasattr(namespace, OPERATOR) and libraries['torch']['path'] not in loaded:
        raise ValueError('GP scale-tail namespace was registered by another library')
    torch.ops.load_library(libraries['torch']['path'])
    operator = getattr(namespace, OPERATOR).default
    if str(operator._schema) != SCHEMA:
        raise ValueError('GP scale-tail public schema changed')
    if not torch._C._dispatch_has_kernel_for_dispatch_key(NAMESPACE + '::' + OPERATOR, 'Meta'):
        raise ValueError('GP scale-tail Meta registration is missing')
    for rows in (1, 2, 8):
        w = torch.empty(1, dtype=torch.uint8, device='meta')
        scale = torch.empty(1, dtype=torch.uint8, device='meta')
        x = torch.empty(rows, 6144, dtype=torch.bfloat16, device='meta')
        table = torch.empty(512, dtype=torch.bfloat16, device='meta')
        ids = torch.empty(rows, 8, dtype=torch.int32, device='meta')
        output = operator(w, scale, x, table, ids)
        if tuple(output.shape) != (1, rows * 8 * 1536) or output.dtype != torch.float32 or output.device.type != 'meta':
            raise ValueError('GP scale-tail returned an invalid Meta output')
    (_OP, _LIBRARIES) = (operator, libraries)
    _LOADS += 1
    _META_CHECKS += 3
    return snapshot()

def _verify_operator():
    import torch
    if getattr(getattr(torch.ops, NAMESPACE), OPERATOR).default is not _OP:
        raise ValueError('GP scale-tail namespace changed after preparation')

def validate_policy():
    if _OP is None:
        raise ValueError('GP scale-tail must be prepared before policy selection')
    if verify_libraries() != _LIBRARIES:
        raise ValueError('GP scale-tail paths changed after preparation')
    _verify_operator()

def record_selection(enabled):
    global _SELECTED
    if enabled:
        validate_policy()
    _SELECTED = bool(enabled)

def operator():
    if _OP is None or not _SELECTED:
        raise ValueError('GP scale-tail is not selected')
    return _OP

def snapshot():
    return dict(prepared=_OP is not None, selected=_SELECTED, operator=NAMESPACE + '::' + OPERATOR, libraries={k: dict(v) for (k, v) in _LIBRARIES.items()}, qualified_sha256=dict(QUALIFIED_SHA256), library_load_count=_LOADS, meta_check_count=_META_CHECKS, capture_counter='gp.python_capture_calls.vector_fetch_scale_tail', count_scope='Python compact dispatches; native cached replays are counted separately', model_quality_qualified=False, arithmetic_order_changed=False, qualification='Corrected load18 ELF only; M1/M2/M8 top8 full-path bits passed; earlier load199 candidates failed', numerical_limits='Retains the original folded GP contract; does not add full-range exact arithmetic')

def freeze_artifacts(destination):
    if not execution_context().has('scale_tail'):
        return {}
    destination = Path(destination)
    libraries = verify_libraries()
    folder = destination / 'production-runtime/gp-scale-tail'
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for (kind, data) in libraries.items():
        target = folder / (kind + '.so')
        shutil.copy2(data['path'], target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != data['sha256']:
            raise ValueError('GP scale-tail library changed while freezing')
        manifest[str(target.relative_to(destination))] = actual
    metadata = folder / 'qualification.json'
    metadata.write_text(json.dumps(dict(libraries=libraries, operator=NAMESPACE + '::' + OPERATOR, qualified_sha256=QUALIFIED_SHA256, model_quality_qualified=False, original_fp32_mac_order=True, kernel_guid='gk_moe_gp_scale_tail_v1', source_commit='a611991186e5f5d563d68150d5b4aeed5600cc7a', elf_sha256='fe7cd09c6bb2d20b02a6addc901916a44cc30a9cd03be3e9704116048ec72460', text_sha256='83ca326c53fce16fe744b15e0711ece4368f3b62748e41d98a9467c53ed3e1f9'), indent=2) + '\n')
    manifest[str(metadata.relative_to(destination))] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    return manifest
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-libraries', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(dict(libraries=verify_libraries(), device_accessed=False), sort_keys=True))
