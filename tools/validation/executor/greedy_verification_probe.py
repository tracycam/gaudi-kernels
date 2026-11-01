"""Compare device verification with the legacy host-read path on one HPU."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch


PLACEHOLDER_TOKEN_ID = -1


def rejection_sample_device(
    draft_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    num_seqs: int,
    cu_num_draft_tokens: torch.Tensor,
) -> torch.Tensor:
    """Fixed-shape greedy verification with no intermediate host reads.

    Padding cannot reject a candidate or select a bonus. The caller transfers
    the final accepted-token matrix once for the scheduler's existing contract.
    """
    device = target_token_ids.device
    target = target_token_ids.reshape(num_seqs, -1).to(torch.int32)
    draft = draft_token_ids.to(device=device, dtype=torch.int32).reshape_as(target)
    bonus = bonus_token_ids.to(device=device, dtype=torch.int32).reshape(num_seqs, 1)
    width = target.shape[1]
    if width == 0:
        return bonus
    cumulative = cu_num_draft_tokens.to(device)
    counts = cumulative - torch.cat((torch.zeros_like(cumulative[:1]), cumulative[:-1]))
    pos = torch.arange(width, device=device)
    mismatch = (draft != target) & (pos[None] < counts[:, None])
    first = torch.where(mismatch, pos[None], width).amin(1)
    rejected = first < counts
    accepted = torch.where(rejected, first + 1, counts)
    columns = torch.arange(width + 1, device=device)[None]
    targets = torch.cat((target, torch.zeros_like(bonus)), 1)
    bonus_mask = (columns == counts[:, None]) & (~rejected[:, None])
    return torch.where(columns < accepted[:, None], targets, torch.where(bonus_mask, bonus, PLACEHOLDER_TOKEN_ID))


def main():
    import habana_frameworks.torch.core as hc
    from vllm_gaudi.v1.sample.hpu_rejection_sampler import rejection_sample_pytorch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=100)
    args = parser.parse_args()
    result = dict(status='RUNNING', cases=[], timings=[])

    def save():
        args.out.write_text(json.dumps(result, indent=2) + '\n')

    save()
    try:
        with torch.inference_mode():
            for counts in ([7], [3, 3], [1, 1, 1], [0, 2, 3], [3, 0, 1]):
                batch, width = len(counts), max(max(counts), 1)
                draft = torch.arange(batch * width, dtype=torch.int32).reshape(batch, width)
                target = draft.clone()
                if batch > 1:
                    target[1, 0] = 99
                for row, count in enumerate(counts):
                    target[row, count:] = 100
                bonus = torch.arange(batch, dtype=torch.int32)[:, None] + 101
                cumulative = torch.tensor(counts, dtype=torch.int32).cumsum(0)
                expected = rejection_sample_pytorch(draft, target, bonus, counts, cumulative)
                tensors = [value.to('hpu') for value in (draft, target, bonus, cumulative)]
                actual = rejection_sample_device(*tensors[:3], batch, tensors[3]).cpu()
                assert torch.equal(expected, actual), (counts, expected, actual)
                result['cases'].append(dict(counts=counts, output=actual.tolist()))
                save()

            for counts in ([7], [3, 3], [1, 1, 1]):
                batch, width = len(counts), max(counts)
                torch.manual_seed(42)
                logits = torch.randn(batch, width, 131072).to('hpu')
                draft = torch.randint(0, 131072, (batch, width), dtype=torch.int32).to('hpu')
                bonus = torch.randint(0, 131072, (batch, 1), dtype=torch.int32).to('hpu')
                cumulative = torch.tensor(counts, dtype=torch.int32).cumsum(0).to('hpu')
                hc.mark_step()
                torch.hpu.synchronize()

                def run(device):
                    target = logits.argmax(-1)
                    return (rejection_sample_device(draft, target, bonus, batch, cumulative).cpu()
                            if device else rejection_sample_pytorch(draft, target, bonus, counts, cumulative))

                assert torch.equal(run(False), run(True))
                for mode in (False, True):
                    for _ in range(5):
                        run(mode)
                samples = {False: [], True: []}
                # ABBA, same inputs, output materialization included each call.
                for _ in range(args.repeats):
                    for mode in (False, True, True, False):
                        start = time.perf_counter_ns()
                        run(mode)
                        samples[mode].append((time.perf_counter_ns() - start) / 1000)
                result['timings'].append(dict(counts=counts, legacy_median_us=statistics.median(samples[False]),
                                               device_median_us=statistics.median(samples[True]),
                                               legacy_samples_us=samples[False], device_samples_us=samples[True]))
                save()
            result['status'] = 'FUNCTIONAL_PASS'
    except BaseException as error:
        result.update(status='FAIL', error=repr(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
