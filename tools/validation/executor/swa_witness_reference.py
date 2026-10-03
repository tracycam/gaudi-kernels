# SPDX-License-Identifier: Apache-2.0
"""Independently evaluate trusted, locally captured native SWA operands in FP32.

This reconstructs request-relative causal/window visibility from the page
descriptors, without invoking the native consumer or its tiled implementation.
Numerical differences are reported; no FP64 or precision-admission gate.
"""
import argparse
import json
from pathlib import Path


def compare(witness):
    import torch
    query = witness['query'].reshape(witness['output'].shape[0], witness['sinks'].numel(), -1).float()
    key, value = witness['key'].float(), witness['value'].float()
    if key.shape[1] != 1 or value.shape[1] != 1:
        raise ValueError('This witness decoder describes the qualified single-KV-head native consumer')
    pages, groups, positions = (witness[k].tolist() for k in ('pages', 'groups', 'positions'))
    reference = []
    for row, position in enumerate(positions):
        descriptors = [page for page, group in zip(pages, groups) if group == row]
        if len(descriptors) != 2:
            raise ValueError('Each query must own exactly two ordered page descriptors')
        logical_begin = max(0, position-127)//128*128
        physical = []
        for index, page in enumerate(descriptors):
            if page < 0:
                continue
            physical.extend(page*128+offset for offset in range(128)
                            if position-127 <= logical_begin+index*128+offset <= position)
        selected = torch.tensor(physical, dtype=torch.int64)
        # Discard masked/padded storage before arithmetic, including NaNs.
        keys, values = key.index_select(0, selected)[:, 0], value.index_select(0, selected)[:, 0]
        scores = query[row] @ keys.T * witness['scale']
        probabilities = torch.cat((scores, witness['sinks'].float()[:, None]), -1).softmax(-1)
        reference.append(probabilities[:, :-1] @ values)
    expected = torch.stack(reference)
    actual = witness['output'].float()
    delta = actual-expected
    return dict(layer=witness['layer'], rows=query.shape[0], reference='FP32 scores/softmax/AV, BF16-stored inputs',
                finite=bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()),
                relative_l2=float(delta.norm()/expected.norm().clamp(min=1e-20)),
                max_abs=float(delta.abs().max()),
                per_query_relative_l2=(delta.flatten(1).norm(dim=1)/expected.flatten(1).norm(dim=1).clamp(min=1e-20)).tolist())


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.inputs:
        # Files come from our own trusted torch.save diagnostics. Their older
        # pickle protocol is unsupported by the local weights-only loader.
        witnesses = torch.load(path, map_location='cpu', weights_only=False)
        reports.append(dict(file=path.name, comparisons=[compare(w) for w in witnesses]))
    args.out.write_text(json.dumps(dict(scope='offline operand reference; no model-quality or throughput claim',
                                      reports=reports), indent=2)+'\n')


if __name__ == '__main__':
    main()
