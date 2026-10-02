"""Default-off exact-old reducer experiment. No import-time Torch or device work."""
from gaudi_kernels.engine.context import context as execution_context
import hashlib
import json
import os
from pathlib import Path
import shutil
QUALIFIED_SHA256 = {'tpc': 'b60f5fbc597b7314653802d3ecf618d9306349f04b62054242ec8232bee5305a', 'torch': '62e4f3eaae0ad25c5d370bf007ffe9fe7dea7f69bba5b58a18b0c64735d6ecaa'}
NAMESPACE = 'gk_reduce_isa'
OPERATOR = 'handschedule'
SCHEMA = 'gk_reduce_isa::handschedule(Tensor partial, Tensor a, Tensor w, Tensor bias) -> Tensor'
PREFIX = 'GK_BLOCK_REDUCE_ISA_'
_LIBRARIES = {}
_OP = None
_SELECTED = False
_LOADS = 0
_META_CHECKS = 0

def verify_libraries():
    libraries = execution_context().libraries('reduce_isa', QUALIFIED_SHA256)
    registered = {str(Path(p).resolve()) for p in os.getenv('GC_KERNEL_PATH', '').split(':') if p}
    if libraries['tpc']['path'] not in registered:
        raise ValueError('Qualified TPC artifact missing from GC_KERNEL_PATH')
    return libraries

def prepare():
    """Load/check the public namespace and Meta contract; never select a policy."""
    global _LIBRARIES, _OP, _LOADS, _META_CHECKS
    if not execution_context().has('reduce_isa'):
        return snapshot()
    libraries = verify_libraries()
    if _OP is not None:
        if libraries != _LIBRARIES:
            raise ValueError('ISA reducer paths changed after preparation')
        _verify_operator_identity()
        return snapshot()
    import torch
    namespace = getattr(torch.ops, NAMESPACE)
    loaded = {str(Path(p).resolve()) for p in torch.ops.loaded_libraries}
    if hasattr(namespace, OPERATOR) and libraries['torch']['path'] not in loaded:
        raise ValueError('ISA namespace was registered by another library path')
    torch.ops.load_library(libraries['torch']['path'])
    operator = getattr(namespace, OPERATOR).default
    if str(operator._schema) != SCHEMA:
        raise ValueError('ISA reducer schema does not match the qualified public bridge')
    if not torch._C._dispatch_has_kernel_for_dispatch_key(NAMESPACE + '::' + OPERATOR, 'Meta'):
        raise ValueError('ISA reducer lacks its required Meta registration')
    partial = torch.empty(48, 1, 3392, dtype=torch.float32, device='meta')
    activation = torch.empty(48, 1, 1, dtype=torch.float32, device='meta')
    weight = torch.empty(27, 48, dtype=torch.float32, device='meta')
    bias = torch.empty(3392, dtype=torch.float32, device='meta')
    output = operator(partial, activation, weight, bias)
    if tuple(output.shape) != (1, 3392) or output.dtype != torch.bfloat16 or output.device.type != 'meta':
        raise ValueError('ISA reducer returned an invalid Meta output')
    (_LIBRARIES, _OP) = (libraries, operator)
    _LOADS += 1
    _META_CHECKS += 1
    return snapshot()

def _verify_operator_identity():
    import torch
    if getattr(getattr(torch.ops, NAMESPACE), OPERATOR).default is not _OP:
        raise ValueError('ISA reducer namespace changed after preparation')

def validate_policy():
    if _OP is None:
        raise ValueError('ISA reducer must be prepared before policy selection')
    if verify_libraries() != _LIBRARIES:
        raise ValueError('ISA reducer paths changed after preparation')
    _verify_operator_identity()

def record_selection(enabled):
    global _SELECTED
    _SELECTED = bool(enabled)
    return snapshot()

def snapshot():
    return {'prepared': _OP is not None, 'selected': _SELECTED, 'operator': NAMESPACE + '::' + OPERATOR, 'libraries': {k: dict(v) for (k, v) in _LIBRARIES.items()}, 'qualified_sha256': dict(QUALIFIED_SHA256), 'library_load_count': _LOADS, 'meta_check_count': _META_CHECKS, 'capture_counter': 'block_fp8.reduction_capture_counts.neumaier_isa_fp32', 'count_scope': 'Python captures/eager calls; native cached replays require independent counters', 'sram_placement_qualified': False, 'model_quality_qualified': False, 'qualification': 'Exact-old faster reducer only; existing DRAM partials and unresolved full-model quality remain'}

def freeze_artifacts(destination):
    """Freeze libraries under unique names and preserve their original paths."""
    if not execution_context().has('reduce_isa'):
        return {}
    destination = Path(destination)
    libraries = verify_libraries()
    folder = destination / 'production-runtime/qkv-neumaier-isa'
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for (kind, data) in libraries.items():
        target = folder / (kind + '.so')
        shutil.copy2(data['path'], target)
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != data['sha256']:
            raise ValueError('ISA library changed while freezing: ' + kind)
        manifest[str(target.relative_to(destination))] = actual
    metadata = folder / 'qualification.json'
    metadata.write_text(json.dumps({'libraries': libraries, 'operator': NAMESPACE + '::' + OPERATOR, 'qualified_sha256': QUALIFIED_SHA256, 'sram_placement_qualified': False, 'model_quality_qualified': False, 'algorithm': 'exact-old Neumaier FP32 operation order'}, indent=2) + '\n')
    manifest[str(metadata.relative_to(destination))] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    return manifest
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-libraries', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps({'libraries': verify_libraries(), 'device_accessed': False}, sort_keys=True))
