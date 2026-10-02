"""Opt-in MoE view/clone experiment, currently offline-qualified only.

The source rewriter operates on the executor's ALREADY modified source, so it
does not undo precise routing or output-return rewrites. No import-time patch,
library load, acquisition or device-value inspection is performed.
"""
import hashlib
import math
import os
from pathlib import Path

_source_ready = False
_rewrites = []


def graph_reshape(tensor, shape):
    """Keep dtype, and skip identity views at their production site."""
    import torch
    shape = tuple(shape)
    if tuple(tensor.shape) == shape:
        return tensor
    if (not shape or len(shape) > 4 or any(not isinstance(n, int) or not 0 < n <= 2147483647 for n in shape)
            or math.prod(shape) != tensor.numel() or not tensor.is_contiguous()):
        raise ValueError('graph reshape requires nonempty contiguous static shape of equal size')
    return torch.ops.gaudi_moe_graph.reshape(tensor, list(shape))


def flatten_rows(tensor):
    if tensor.ndim < 2:
        raise ValueError('MoE inputs require at least two dimensions')
    return graph_reshape(tensor, (math.prod(tensor.shape[:-1]), tensor.shape[-1]))


def ids_rows(ids):
    import torch
    # Never remove this cast based on metadata Long being "usually" int32 on
    # Gaudi. The logical reshape itself preserves the returned logical dtype.
    return flatten_rows(ids.to(torch.int32))


def restore_output(output, *shape):
    return graph_reshape(output, shape)


def _lazy_contract():
    if os.environ.get('PT_HPU_LAZY_MODE') != '1':
        raise RuntimeError('MoE clone ablation requires explicit PT_HPU_LAZY_MODE=1')
    if os.environ.get('PT_ENABLE_INT64_SUPPORT', 'false').lower() not in ('0', 'false'):
        raise RuntimeError('MoE clone ablation requires the audited Lazy int32 device-index contract')


def prepare_inputs_without_clones(x, ids, routing, *, precise, mode):
    """Replace only batch_ops' three initial clones; retain both dtype casts.

    Call only behind GK_MOE_REMOVE_CLONES=1 after the producer-source rewrite.
    Sorted-mode materializations are separate and deliberately not removed.
    Kernel glue still validates actual DATA_I32, independently of Long metadata.
    """
    import torch
    _lazy_contract()
    if not _source_ready or os.environ.get('GK_MOE_GRAPH_VIEWS') != '1':
        raise RuntimeError('producer graph-view rewrite must precede clone ablation')
    if mode not in ('compact', 'broadcast'):
        raise ValueError('clone ablation only admits compact/broadcast; sorted is unchanged')
    if (x.ndim != 2 or x.dtype != torch.bfloat16 or x.shape[1] != 6144
            or ids.ndim != 2 or tuple(ids.shape) != tuple(routing.shape)
            or ids.shape[0] != x.shape[0] or not 1 <= ids.shape[1] <= 384):
        raise ValueError('MoE activation/route geometry mismatch')
    ids = ids.to(torch.int32)
    routing = routing.to(torch.float32 if precise else torch.bfloat16)
    return x, ids, routing


def rewrite_moe_source(source, namespace, *, extension_path=None):
    """Explicit import-hook action, AFTER existing precision/output rewrites.

    Load a separate public bridge once and bind helpers into the module being
    executed. Does not inspect/re-execute an already loaded function's original
    file, which would lose the executor's other intentional source rewrites.
    """
    global _source_ready
    _lazy_contract()
    replacements = {
        'x = x.view(-1, x.shape[-1])': 'x = _gk_moe_flatten(x)',
        'topk_ids = topk_ids.view(-1, topk_ids.shape[-1])': 'topk_ids = _gk_moe_ids_rows(topk_ids)',
        'topk_weights = topk_weights.view(-1, topk_weights.shape[-1])': 'topk_weights = _gk_moe_flatten(topk_weights)',
    }
    for old in replacements:
        if source.count(old) != 1:
            raise ValueError(f'pinned MoE input view site changed: {old}')
    raw_returns = source.count('return output.view(')
    executor_returns = source.count('return _comm_output_view(output, ')
    if (raw_returns, executor_returns) not in ((2, 0), (0, 2)):
        raise ValueError('expected exactly two raw or executor-rewritten MoE returns')
    rewritten = source
    for old, new in replacements.items():
        rewritten = rewritten.replace(old, new)
    if raw_returns:
        rewritten = rewritten.replace('return output.view(', 'return _gk_moe_restore(output, ')
    else:
        rewritten = rewritten.replace('return _comm_output_view(output, ', 'return _gk_moe_restore(output, ')
    # DP-produced views/collective ownership are not covered by the first gate.
    needle = '        input_shape = x.shape'
    if rewritten.count(needle) != 1:
        raise ValueError('pinned apply_monolithic input-shape site changed')
    rewritten = rewritten.replace(needle,
        "        if layer.moe_config.dp_size != 1:\n            raise ValueError('MoE graph-view candidate currently requires DP1')\n"
        "        if getattr(layer, 'custom_routing_function', None) is not None:\n            raise ValueError('Custom-routing output ownership is not audited for clone removal')\n"+needle)
    compile(rewritten, '<gaudi-moe-graph-views>', 'exec')
    import torch
    path = str(Path(extension_path).resolve(strict=True)) if extension_path is not None else None
    if path is not None:
        if hasattr(torch.ops.gaudi_moe_graph, 'reshape') and path not in torch.ops.loaded_libraries:
            raise RuntimeError('MoE reshape already registered elsewhere; use extension_path=None')
        torch.ops.load_library(path)
    if not hasattr(torch.ops.gaudi_moe_graph, 'reshape'):
        raise RuntimeError('load gaudi_moe_graph_views.so before enabling producer rewrites')
    namespace.update(_gk_moe_flatten=flatten_rows, _gk_moe_ids_rows=ids_rows,
                     _gk_moe_restore=restore_output)
    _source_ready = True
    _rewrites.append({'before_sha256': hashlib.sha256(source.encode()).hexdigest(),
                      'after_sha256': hashlib.sha256(rewritten.encode()).hexdigest(),
                      'input_sites': 3, 'output_sites': 2, 'extension_path': path,
                      'device_validated': False})
    return rewritten


def rewrite_batch_source(source):
    """Return a narrow opt-in patch; flag-off leaves the old three clones."""
    old = '    x=x.clone();ids=ids.to(torch.int32).clone();routing=routing.to(torch.float32 if precise else torch.bfloat16).clone()'
    if source.splitlines().count(old) != 1:
        raise ValueError('pinned batch_ops three-clone site changed')
    replacement = """    if os.getenv('GK_MOE_REMOVE_CLONES','0')=='1':
        from gaudi_kernels.moe_graph_views import prepare_inputs_without_clones
        x,ids,routing=prepare_inputs_without_clones(x,ids,routing,precise=precise,mode=mode)
    else:
""" + '    ' + old
    result = source.replace(old, replacement)
    compile(result, '<gaudi-moe-clone-ablation>', 'exec')
    return result


def rewrite_sitecustomize_source(source):
    """Prepare an isolated executor copy; never rewrite the running file."""
    anchors = {
        "MODE = os.environ.get('COMM_EXP_MODE')": """MODE = os.environ.get('COMM_EXP_MODE')
if os.getenv('GK_MOE_REMOVE_CLONES') == '1' and os.getenv('GK_MOE_GRAPH_VIEWS') != '1':
    raise RuntimeError('clone removal requires GK_MOE_GRAPH_VIEWS=1')
if os.getenv('GK_MOE_GRAPH_VIEWS') == '1' and MODE != 'skip-moe-output-view':
    raise RuntimeError('MoE source candidate requires COMM_EXP_MODE=skip-moe-output-view')""",
        "                targets.add('vllm_gaudi.ops.hpu_mxfp4')": """                targets.add('vllm_gaudi.ops.hpu_mxfp4')
            if os.getenv('GK_MOE_GRAPH_VIEWS') == '1':
                targets.add('batch_ops')""",
        "                    if fullname == 'vllm_gaudi.ops.hpu_mxfp4' and MODE == 'skip-moe-output-view':": """                    if fullname == 'batch_ops' and os.getenv('GK_MOE_GRAPH_VIEWS') == '1':
                        from gaudi_kernels.moe_graph_views import rewrite_batch_source
                        source = rewrite_batch_source(original.get_source(fullname))
                        exec(compile(source, spec.origin, 'exec'), module.__dict__)
                    elif fullname == 'vllm_gaudi.ops.hpu_mxfp4' and MODE == 'skip-moe-output-view':""",
        "                        module.__dict__['_comm_output_view'] = output_view": """                        module.__dict__['_comm_output_view'] = output_view
                        if os.getenv('GK_MOE_GRAPH_VIEWS') == '1':
                            from gaudi_kernels.moe_graph_views import rewrite_moe_source
                            source = rewrite_moe_source(source, module.__dict__,
                                extension_path=os.environ['GK_MOE_GRAPH_LIBRARY'])""",
    }
    if 'GK_MOE_GRAPH_VIEWS' in source or 'GK_MOE_REMOVE_CLONES' in source:
        raise ValueError('executor already contains MoE graph-view integration')
    for old in anchors:
        if source.count(old) != 1:
            raise ValueError('pinned executor hook site changed: ' + old)
    for old, new in anchors.items():
        source = source.replace(old, new)
    compile(source, '<gaudi-moe-sitecustomize-candidate>', 'exec')
    return source


def snapshot():
    return {'producer_source_ready': _source_ready, 'rewrites': list(_rewrites),
            'remove_clones_requested': os.environ.get('GK_MOE_REMOVE_CLONES') == '1',
            'qualification': 'offline candidate; device graph, DATA_I32 and DMA removal unmeasured'}
