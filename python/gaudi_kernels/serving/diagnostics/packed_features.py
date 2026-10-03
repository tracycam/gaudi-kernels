# SPDX-License-Identifier: Apache-2.0
"""Audit actual pre-normalization decoder features for drafter consumers."""
import json


def on_worker(worker, feature_layers):
    import torch
    from gaudi_kernels.engine.context import context
    from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
    from gaudi_kernels.serving.executor.packed_target import PackedTargetExecutor
    runner = worker.model_runner
    if runner.input_batch.num_reqs or context().native.active:
        raise ValueError('Feature audit requires an idle scheduler and no native serving plan')
    layers = tuple(feature_layers)
    target = PackedTargetExecutor(runner, kv_capacity=32, feature_layers=layers)
    baseline = PackedTargetExecutor(runner, kv_capacity=32)
    batch = TokenBatch((RequestTokens('a', (101,), 0, 'decode'),
                        RequestTokens('b', (102, 103, 104, 105), 0, 'verify'),
                        RequestTokens('c', tuple(range(111, 119)), 0, 'prefill', tuple(range(8)))))
    observed, handles = {}, []

    def observer(index):
        def hook(module, args, output):
            # An independent decoder-module boundary; do not infer a feature
            # from the final model norm or from a future query's hidden state.
            if not isinstance(output, tuple) or len(output) != 2:
                raise ValueError('Decoder has no explicit hidden/residual output boundary')
            hidden, residual = output
            observed[index] = (hidden + residual if residual is not None else hidden).clone()
        return hook

    old = tuple(target.feature_holder.aux_hidden_state_layers)
    try:
        for index in layers:
            handles.append(target.feature_holder.layers[index].register_forward_hook(observer(index)))
        result = target.execute(batch)
        feature_cpu = tuple(t.cpu().clone() for t in result.features)
        independent_cpu = tuple(observed[i].cpu().clone() for i in layers)
        hidden, logits = result.hidden.cpu().clone(), result.logits.cpu().clone()
    finally:
        for handle in handles:
            handle.remove()
    expected = baseline.execute(batch)
    report = {'rank': worker.rank, 'valid_rows': batch.num_tokens, 'feature_layers': layers,
              'scope': 'actual target pre-final-norm feature boundaries; no drafter or acceptance/TPS',
              'features': [{'layer': i, 'shape': list(t.shape), 'dtype': str(t.dtype),
                            'finite': bool(torch.isfinite(t).all()), 'boundary_bitwise': torch.equal(t, ref)}
                           for i, t, ref in zip(layers, feature_cpu, independent_cpu)],
              'capture_policy_restored': tuple(target.feature_holder.aux_hidden_state_layers) == old,
              'hidden_bitwise_unchanged': torch.equal(hidden, expected.hidden.cpu()),
              'logits_bitwise_unchanged': torch.equal(logits, expected.logits.cpu())}
    report['pass'] = (report['capture_policy_restored'] and report['hidden_bitwise_unchanged'] and
                      report['logits_bitwise_unchanged'] and len(report['features']) == len(layers) and
                      all(r['finite'] and r['boundary_bitwise'] for r in report['features']))
    root = context().startup.run_dir/'packed-target-features'
    root.mkdir(exist_ok=True)
    torch.save({'features': feature_cpu, 'independent': independent_cpu,
                'hidden': hidden, 'logits': logits}, root/f'rank{worker.rank}.pt')
    (root/f'rank{worker.rank}.json').write_text(json.dumps(report, indent=2)+'\n')
    target.abort(result)
    baseline.abort(expected)
    return report
