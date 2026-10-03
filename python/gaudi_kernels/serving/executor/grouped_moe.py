"""A bounded graph cache per layer, retaining original packed weight owners."""
import torch
from .grouped_moe_runtime import forward


class GroupedMoE(torch.nn.Module):
    def __init__(self, owner):
        super().__init__()
        for name in ('gp', 'gs', 'dp', 'ds', 'table', 'directions'):
            # register_buffer keeps the same tensor/storage; no layout copy.
            self.register_buffer(name, getattr(owner, name), persistent=False)

    def forward(self, x, ids, routing):
        return forward(x, ids, routing, self.gp, self.gs, self.dp, self.ds,
                       self.table, self.directions)
