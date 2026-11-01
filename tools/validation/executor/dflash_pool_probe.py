"""Minimal device gate for changing accepted counts and request-owned slots."""
import argparse
import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from gaudi_kernels.serving.draft.proposer import ProposalProgram
from gaudi_kernels.serving.draft.requests import ContextPool


class TinyDraft(nn.Module):
    spec = SimpleNamespace(hidden=128, layers=1, window=8, kv_heads=1, dim=128,
                           target_layers=(0,), block=4)

    def __init__(self, device):
        super().__init__()
        self.register_buffer('weight', torch.eye(128, device=device, dtype=torch.bfloat16))

    def project_context(self, features, positions):
        value = torch.nn.functional.linear(features, self.weight).unsqueeze(2)
        return ((value, value.clone()),)

    def noise(self, anchors):
        return torch.cat((anchors[:, None], torch.zeros(anchors.shape[0], 3, 128,
                                                       dtype=anchors.dtype, device=anchors.device)), 1)

    def forward(self, noise, positions, context, context_positions, valid):
        keys = context[0][0][:, :, 0, 0].float()
        # Both stored values and absolute positions affect the output. A stale
        # slot or accidentally committed rejected row changes the proposals.
        checksum = ((keys + 2 * context_positions.float()) * valid).sum(1)
        values = (noise[:, :1, 0].float() + checksum[:, None] + positions.float()) % 128
        return values[:, :, None].expand(-1, -1, 128).to(torch.bfloat16)


class TinyTarget(nn.Module):
    def embed_input_ids(self, ids):
        return ids[:, None].expand(-1, 128).to(torch.bfloat16)

    def compute_logits(self, hidden):
        return -(torch.arange(128, device=hidden.device)[None] - hidden[:, :1].float()).abs()


def main():
    import habana_frameworks.torch as ht
    import habana_frameworks.torch.core as hc
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = dict(status='RUNNING', cases=[])

    def save():
        (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        with torch.inference_mode():
            cpu_draft, hpu_draft = TinyDraft('cpu'), TinyDraft('hpu')
            cpu_pool, hpu_pool = ContextPool(cpu_draft, 2, 'cpu'), ContextPool(hpu_draft, 2, 'hpu')
            for pool in (cpu_pool, hpu_pool):
                assert [pool.slot(x) for x in ('a', 'b')] == [0, 1]
            cpu = ProposalProgram(cpu_draft, cpu_pool, TinyTarget(), 3)
            hpu = ht.hpu.wrap_in_hpu_graph(ProposalProgram(hpu_draft, hpu_pool, TinyTarget(), 3),
                                          disable_tensor_cache=False, asynchronous=False)
            cases = [([[0, 1, 2, 3], [10, 11, 12, 13]], [4, 2], [0, 1], [20, 30], [4, 12]),
                     ([[12, 13, 14, 15], [4, 5, 6, 7]], [1, 3], [1, 0], [31, 21], [13, 7]),
                     ([[7, 8, 9, 10], [13, 14, 15, 16]], [4, 0], [0, 1], [22, 32], [11, 13]),
                     ([[13, 14, 15, 16], [0, 1, 2, 3]], [1, 1], [1, 0], [33, 23], [14, 1])]
            for index, (positions, counts, slots, anchors, next_positions) in enumerate(cases):
                if index == 3:
                    for pool in (cpu_pool, hpu_pool):
                        pool.release(['a'])
                        assert pool.slot('c') == 0
                hc.mark_step()
                pos = torch.tensor(positions, dtype=torch.int32)
                features = pos[:, :, None].expand(-1, -1, 128).to(torch.bfloat16).contiguous()
                inputs = (features, pos, *(torch.tensor(x, dtype=torch.int32)
                                          for x in (counts, slots, anchors, next_positions)))
                chosen = (3, 1, 3, 0)[index]
                expected = cpu(*inputs, chosen)
                actual = hpu(*(value.to('hpu') for value in inputs), chosen)
                hc.mark_step()
                torch.hpu.synchronize()
                assert torch.equal(actual.cpu(), expected), (index, actual.cpu(), expected)
                assert torch.equal(hpu_pool.storage.positions.cpu(), cpu_pool.storage.positions), index
                for kind in ('key_0', 'value_0'):
                    assert torch.equal(getattr(hpu_pool.storage, kind).cpu(), getattr(cpu_pool.storage, kind)), index
                result['cases'].append(dict(index=index, output=actual.cpu().tolist(),
                                             positions=hpu_pool.storage.positions.cpu().tolist()))
                save()
            result['plans'] = [cached.graph.execution_info() for cached in inspect.getclosurevars(hpu.forward).nonlocals['cache'].values()]
            result['status'] = 'FUNCTIONAL_PASS'
            save()
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
