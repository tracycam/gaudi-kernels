# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from gaudi_kernels.serving.diagnostics.packed_boundaries import BoundaryTrace, compare_boundaries


class PackedBoundaryTests(unittest.TestCase):
    def test_owned_frames_match_per_query_and_do_not_accumulate_other_forwards(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = torch.nn.Identity()
                self.self_attn = torch.nn.Module()
                self.self_attn.qkv_proj = torch.nn.Identity()

            def forward(self, x):
                return self.self_attn.qkv_proj(self.input_layernorm(x)) + 1

        root = torch.nn.Module()
        root.model = torch.nn.Module()
        root.model.layers = torch.nn.ModuleList([Layer(), Layer()])
        def run(x):
            for layer in root.model.layers:
                x = layer(x)
            return x

        trace = BoundaryTrace(root, (1,))
        x = torch.arange(12).reshape(3, 4).float()
        trace.begin(3)
        run(x)
        candidate = trace.freeze()
        self.assertTrue(all('.layers.1' in name for name in candidate))
        self.assertTrue(candidate)
        sequential = []
        for row in x:
            trace.begin(1)
            run(row[None])
            sequential.append(trace.freeze())
        self.assertTrue(all(r['bitwise'] for r in compare_boundaries(candidate, sequential)))
        # Another call cannot change frozen CPU frames or append a second run.
        run(x*0)
        self.assertTrue(all(r['bitwise'] for r in compare_boundaries(candidate, sequential)))
        trace.close()

    def test_missing_reference_operator_fails_closed(self):
        with self.assertRaises(KeyError):
            compare_boundaries({'layer': {'input': (torch.ones(1, 2),)}}, [{}])
