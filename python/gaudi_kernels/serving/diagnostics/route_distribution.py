# SPDX-License-Identifier: Apache-2.0
"""Actual target top-k witnesses, copied on device and inspected after delivery.

Snapshots add nodes and invalidate service-throughput claims for that run.
Recorded replays update the captured snapshot storage without invoking hooks.
"""
from collections import Counter


class RouteDistribution:
    def __init__(self, model):
        self.values, self.handles = {}, []
        for name, module in model.named_modules():
            if type(module).__name__ != 'NativeExpertTP':
                continue
            def capture(module, args, kwargs, name=name):
                self.values[name] = (args[1] if len(args) > 1 else kwargs['topk_ids']).clone()
            self.handles.append(module.register_forward_pre_hook(capture, with_kwargs=True))
        if not self.handles:
            raise ValueError('No actual native MoE consumers available for route witnesses')

    def summarize(self, schedule):
        result = []
        for name, tensor in sorted(self.values.items()):
            rows = tensor.cpu().tolist()
            if len(rows) != schedule.num_tokens or any(len(set(row)) != len(row) or any(i < 0 for i in row) for row in rows):
                raise ValueError('Target route witness extent/unique top-k IDs invalid')
            counts = Counter(i for row in rows for i in row)
            per_request = [len({i for row in rows[begin:end] for i in row})
                           for begin, end in zip(schedule.query_start_loc, schedule.query_start_loc[1:])]
            result.append(dict(layer=name, query_route_ids=rows, unique_experts=len(counts),
                expert_m_histogram=dict(sorted(Counter(counts.values()).items())),
                unique_experts_per_request=per_request))
        if len(result) != len(self.handles):
            raise ValueError('Not every native MoE layer supplied actual route IDs')
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
