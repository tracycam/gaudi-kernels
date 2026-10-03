# SPDX-License-Identifier: Apache-2.0
"""Owned intermediate snapshots, with no CPU tensor reads inside model hooks."""
import re

import torch


class BoundaryTrace:
    suffixes = ('.input_layernorm', '.post_attention_layernorm', '.self_attn.qkv_proj',
                '.self_attn.attn', '.self_attn.o_proj', '.mlp.gate', '.mlp', '.moe_op')

    def __init__(self, model, layers):
        self.handles, self.frames, self.rows = [], {}, 0
        selected = set(layers)
        for name, module in model.named_modules():
            found = re.search(r'\.layers\.(\d+)(\.|$)', name)
            if found and int(found[1]) in selected and (name.endswith(self.suffixes) or name.endswith('.'+found[1])):
                self.handles.append(module.register_forward_hook(self._observer(name)))

    def _observer(self, name):
        def hook(module, args, output):
            if not self.rows:
                return
            if name in self.frames:
                raise ValueError('Repeated operator boundary in one traced forward: '+name)
            def tensors(value):
                values = value if isinstance(value, (tuple, list)) else (value,)
                return tuple(t.clone() for t in values if isinstance(t, torch.Tensor) and
                             t.ndim > 0 and t.shape[0] == self.rows)
            self.frames[name] = {'input': tensors(args[:2]), 'output': tensors(output)}
        return hook

    def begin(self, rows):
        self.frames, self.rows = {}, rows

    def freeze(self):
        self.rows = 0
        return {name: {kind: tuple(t.cpu().clone() for t in values) for kind, values in frame.items()}
                for name, frame in self.frames.items()}

    def close(self):
        for handle in self.handles:
            handle.remove()


def compare_boundaries(candidate, sequential):
    report = []
    for name, frame in candidate.items():
        for kind, tensors in frame.items():
            for index, actual in enumerate(tensors):
                parts = [value[name][kind][index] for value in sequential]
                expected = torch.cat(parts)
                if actual.shape != expected.shape:
                    raise ValueError('Incomplete packed/reference operator boundary: '+name)
                delta = actual.float() - expected.float()
                report.append({'module': name, 'kind': kind, 'tensor': index,
                    'shape': list(actual.shape), 'bitwise': torch.equal(actual, expected),
                    'different_rows': torch.nonzero(delta.reshape(delta.shape[0], -1).abs().amax(1) > 0).flatten().tolist(),
                    'max_abs': delta.abs().max().item(),
                    'relative_l2': (delta.norm()/expected.float().norm().clamp_min(1e-30)).item()})
    return report
