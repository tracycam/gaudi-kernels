"""CPU-only source/edge audit. Does NOT emulate Lazy capture or MoE kernels."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python'))
from gaudi_kernels import moe_graph_views as candidate


def function(source, name, namespace, *, boundary=False):
    matches = [n for n in ast.walk(ast.parse(source))
               if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(matches) == 1, (name, len(matches))
    node = copy.deepcopy(matches[0])
    node.decorator_list = []
    node.returns = None
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        arg.annotation = None
    if node.args.vararg:
        node.args.vararg.annotation = None
    if node.args.kwarg:
        node.args.kwarg.annotation = None
    if boundary:
        # Run the real batch_ops validation/mask/cast/clone prelude, ending
        # immediately before the first expert kernel dispatch. Kernels are
        # neither mocked as numerical evidence nor called on this CPU gate.
        start = next(i for i, n in enumerate(node.body)
                     if isinstance(n, ast.If) and ast.unparse(n.test) == "mode == 'compact'")
        node.body = node.body[:start] + [ast.Return(value=ast.Tuple(
            elts=[ast.Name(id=n, ctx=ast.Load()) for n in ('x', 'ids', 'routing')], ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, '<cpu-source-gate>', 'exec'), namespace)
    return namespace[name]


def expect_error(fn, text):
    try:
        fn()
    except (RuntimeError, ValueError) as error:
        assert text in str(error), str(error)
        return str(error)
    raise AssertionError('expected rejection: ' + text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plugin', type=Path, required=True)
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--router', type=Path, required=True)
    p.add_argument('--sitecustomize', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = {'status': 'STARTED', 'device_used': False, 'torch': torch.__version__,
              'scope': 'CPU source, shape, dtype, value and explicit clone audit; no Lazy/HPU or MoE MAC claim'}
    try:
        os.environ.update(PT_HPU_LAZY_MODE='1', PT_ENABLE_INT64_SUPPORT='false', GK_MOE_GRAPH_VIEWS='1')
        sources = {n: getattr(args, n).read_text() for n in ('plugin', 'batch', 'router')}
        result['sources'] = {}
        for name, source in sources.items():
            (args.output_dir / (name + '.py')).write_text(source)
            result['sources'][name] = {'path': str(getattr(args, name).resolve()),
                                      'sha256': hashlib.sha256(source.encode()).hexdigest()}
        original = sources['plugin']
        assert original.count('return output.view(') == 2
        assert original.count('topk_weights = topk_weights.to(x.dtype)') == 2
        # Reproduce the active executor's earlier rewrites. The candidate must
        # preserve the precise-routing expression when rewriting current text.
        original = original.replace('return output.view(', 'return _comm_output_view(output, ')
        original = original.replace('topk_weights = topk_weights.to(x.dtype)',
                                    'topk_weights = topk_weights.to(_precision_route_dtype(x))')
        graph_calls = []
        lib = torch.library.Library('gaudi_moe_graph', 'DEF')
        lib.define('reshape(Tensor input, int[] shape) -> Tensor')
        def cpu_reshape(tensor, shape):
            graph_calls.append({'input': list(tensor.shape), 'output': shape, 'dtype': str(tensor.dtype)})
            return tensor.reshape(shape)
        lib.impl('reshape', cpu_reshape, 'CPU')
        helper_namespace = {}
        rewritten = candidate.rewrite_moe_source(original, helper_namespace)
        batch_new = candidate.rewrite_batch_source(sources['batch'])
        (args.output_dir / 'plugin-rewritten.py').write_text(rewritten)
        (args.output_dir / 'batch-rewritten.py').write_text(batch_new)
        assert rewritten.count('topk_weights.to(_precision_route_dtype(x))') == 2
        assert 'topk_ids = topk_ids.to(torch.int32)' in rewritten
        assert batch_new.count('sorted_ids=values.reshape(ids.shape).clone()') == 1
        assert batch_new.count('permutation=permutation.to(torch.int32).reshape(ids.shape).clone()') == 1
        grouped = function(sources['router'], '_grouped_topk', {'torch': torch})
        def select(layer, x, logits):
            return grouped(logits, layer.top_k, True, 2, 1, 'sigmoid', 2.5,
                           torch.linspace(-0.1, 0.1, logits.shape[-1]))
        namespace = dict(torch=torch, F=F, select_experts_from_routed=select,
                         _normalize_moe_activation=lambda x: x,
                         _comm_output_view=lambda out, *shape: out if tuple(out.shape) == shape else out.view(*shape),
                         _precision_route_dtype=lambda x: torch.float32 if os.getenv('UNIFIED_PRECISION_ROUTER') == '1' else x.dtype)
        before = function(original, 'apply_monolithic', dict(namespace))
        after = function(rewritten, 'apply_monolithic', dict(namespace, **helper_namespace))
        torch.manual_seed(927)
        rows = []
        for shape in ((1, 6144), (2, 6144), (1, 1, 6144), (1, 2, 6144), (2, 3, 6144)):
            for route in ('standard', 'grouped', 'gpt_oss'):
                for precise in (False, True):
                    os.environ['UNIFIED_PRECISION_ROUTER'] = str(int(precise))
                    x = torch.randn(shape).bfloat16() + torch.randn(shape).bfloat16()
                    logits = torch.randn(x.numel() // 6144, 32)
                    captures = []
                    def moe_op(x, ids, routing, **kwargs):
                        captures.append((x, ids, routing))
                        # Deterministic consumer checks all route values and x.
                        # This is deliberately not an MoE numerical oracle.
                        return (x.float() * routing.float().sum(-1, keepdim=True)
                                + (ids.float() * routing.float()).sum(-1, keepdim=True)).bfloat16()
                    layer = SimpleNamespace(top_k=4, use_grouped_topk=route == 'grouped',
                                            custom_routing_function=None, activation='silu',
                                            moe_config=SimpleNamespace(dp_size=1), moe_op=moe_op)
                    method = SimpleNamespace(model_type=route)
                    expected = before(method, layer, x, logits)
                    begin = len(graph_calls)
                    actual = after(method, layer, x, logits)
                    assert torch.equal(expected, actual)
                    assert tuple(actual.shape) == shape
                    assert all(torch.equal(a, b) for a, b in zip(*captures))
                    assert captures[1][1].dtype == torch.int32
                    assert captures[1][2].dtype == (torch.float32 if precise else torch.bfloat16)
                    if len(shape) == 2:
                        assert captures[1][0] is x
                        assert len(graph_calls) == begin  # no identity custom reshape either
                    else:
                        assert len(graph_calls) == begin + 2  # flatten and restore
                    rows.append(dict(shape=shape, route=route, precise=precise,
                                     graph_reshapes=len(graph_calls)-begin, values_equal=True))
        result['apply_cases'] = rows
        prep0 = function(sources['batch'], 'moe', {'torch': torch, 'os': os}, boundary=True)
        prep1 = function(batch_new, 'moe', {'torch': torch, 'os': os}, boundary=True)
        cast_rows = []
        original_clone = torch.Tensor.clone
        clone_count = [0]
        def counted_clone(tensor, *a, **kw):
            clone_count[0] += 1
            return original_clone(tensor, *a, **kw)
        try:
            torch.Tensor.clone = counted_clone
            for mode in ('compact', 'broadcast'):
                for precise in (False, True):
                    for masked in (False, True):
                        os.environ['UNIFIED_PRECISION_ROUTER'] = str(int(precise))
                        x = torch.randn(2, 6144).bfloat16() + 1
                        ids = torch.tensor([[1, 3, 5], [8, 2, 7]], dtype=torch.int64)
                        weights = torch.tensor([[.1234567, .3, .2], [.1, .2789012, .3]], dtype=torch.float32)
                        mask = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.bool) if masked else None
                        call_args = (x, ids, weights, None, None, None, None, None, None)
                        clone_count[0] = 0
                        base = prep0(*call_args, mode=mode, route_mask=mask)
                        base_clones = clone_count[0]
                        os.environ['GK_MOE_REMOVE_CLONES'] = '0'
                        clone_count[0] = 0
                        off = prep1(*call_args, mode=mode, route_mask=mask)
                        off_clones = clone_count[0]
                        os.environ['GK_MOE_REMOVE_CLONES'] = '1'
                        clone_count[0] = 0
                        on = prep1(*call_args, mode=mode, route_mask=mask)
                        on_clones = clone_count[0]
                        assert base_clones == off_clones == 3 and on_clones == 0
                        assert all(torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(base, on))
                        assert all(torch.equal(a, b) for a, b in zip(base, off))
                        assert on[0] is x and on[1].dtype == torch.int32
                        assert on[2].dtype == (torch.float32 if precise else torch.bfloat16)
                        cast_rows.append(dict(mode=mode, precise=precise, masked=masked,
                                              baseline_explicit_clones=base_clones, flag_off_explicit_clones=off_clones,
                                              flag_on_explicit_clones=on_clones, values_equal=True))
        finally:
            torch.Tensor.clone = original_clone
        result['batch_boundary_cases'] = cast_rows
        dtype_cases = []
        for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64):
            x = torch.arange(24).to(dtype).reshape(2, 3, 4)
            y = candidate.flatten_rows(x)
            assert y.dtype == dtype and torch.equal(y, x.reshape(6, 4))
            dtype_cases.append(str(dtype))
        result['dtype_preservation'] = dtype_cases
        errors = []
        if args.sitecustomize:
            site = args.sitecustomize.read_text()
            rewritten_site = candidate.rewrite_sitecustomize_source(site)
            (args.output_dir/'sitecustomize-before.py').write_text(site)
            (args.output_dir/'sitecustomize-after.py').write_text(rewritten_site)
            assert rewritten_site.index("source=source.replace(route_needle") < rewritten_site.index('source = rewrite_moe_source')
            assert "targets.add('batch_ops')" in rewritten_site
            errors.append(expect_error(lambda: candidate.rewrite_sitecustomize_source(rewritten_site), 'already contains'))
            tree = ast.parse(rewritten_site)
            # Run only imports and leading flag guards, before installing any
            # meta-path finder. This process never patches a running executor.
            prefix = []
            for node in tree.body:
                if isinstance(node, ast.If) and ast.unparse(node.test).startswith('MODE in'):
                    break
                prefix.append(node)
            code = compile(ast.Module(body=prefix, type_ignores=[]), '<hook-flag-gate>', 'exec')
            os.environ.update(COMM_EXP_MODE='skip-moe-output-view', GK_MOE_GRAPH_VIEWS='1', GK_MOE_REMOVE_CLONES='1')
            exec(code, {})
            os.environ['GK_MOE_GRAPH_VIEWS'] = '0'
            errors.append(expect_error(lambda: exec(code, {}), 'clone removal requires'))
            os.environ.update(GK_MOE_GRAPH_VIEWS='1', COMM_EXP_MODE='baseline')
            errors.append(expect_error(lambda: exec(code, {}), 'requires COMM_EXP_MODE'))
            os.environ['COMM_EXP_MODE'] = 'skip-moe-output-view'
            result['executor_hook'] = {'before_sha256': hashlib.sha256(site.encode()).hexdigest(),
                                       'after_sha256': hashlib.sha256(rewritten_site.encode()).hexdigest(),
                                       'preserves_precision_before_view_rewrite': True,
                                       'flag_guards_pass': True}
        layer.moe_config.dp_size = 2
        errors.append(expect_error(lambda: after(method, layer, x, logits), 'DP1'))
        layer.moe_config.dp_size = 1
        layer.custom_routing_function = lambda: None
        errors.append(expect_error(lambda: after(method, layer, x, logits), 'Custom-routing'))
        errors.append(expect_error(lambda: candidate.rewrite_moe_source(rewritten, {}), 'input view site changed'))
        errors.append(expect_error(lambda: candidate.rewrite_batch_source(batch_new), 'three-clone site changed'))
        ax = torch.zeros(1, 6144, dtype=torch.bfloat16)
        ids = torch.zeros(1, 1, dtype=torch.int64)
        rw = torch.ones(1, 1)
        invoke = lambda mode='compact': candidate.prepare_inputs_without_clones(ax, ids, rw, precise=True, mode=mode)
        errors.append(expect_error(lambda: invoke('sorted'), 'compact/broadcast'))
        os.environ['GK_MOE_GRAPH_VIEWS'] = '0'
        errors.append(expect_error(invoke, 'producer graph-view'))
        os.environ['GK_MOE_GRAPH_VIEWS'] = '1'
        os.environ['PT_ENABLE_INT64_SUPPORT'] = 'true'
        errors.append(expect_error(invoke, 'Lazy int32'))
        os.environ['PT_ENABLE_INT64_SUPPORT'] = 'false'
        os.environ['PT_HPU_LAZY_MODE'] = '0'
        errors.append(expect_error(invoke, 'LAZY_MODE=1'))
        os.environ['PT_HPU_LAZY_MODE'] = '1'
        errors.append(expect_error(lambda: candidate.graph_reshape(torch.zeros(2, 3).t(), (6,)), 'contiguous'))
        errors.append(expect_error(lambda: candidate.graph_reshape(torch.zeros(2, 3), (5,)), 'equal size'))
        result.update(status='PASS_CPU_ONLY', rejection_cases=errors, snapshot=candidate.snapshot())
        print(json.dumps({k: result[k] for k in ('status', 'scope')}))
        print('apply cases:', len(rows), 'batch prelude cases:', len(cast_rows), 'rejections:', len(errors))
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        (args.output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
