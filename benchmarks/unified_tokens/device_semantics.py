"""One-card semantic probe; neither native replay nor full-model TPS validation."""
import json
import os
from pathlib import Path
import torch
import habana_frameworks.torch.core as ht
from gaudi_kernels.engine.token_batch import RequestTokens, TokenBatch
from gaudi_kernels.serving.executor.target_reference import fixture, packed_target, sequential_target


def sync():
    ht.mark_step()
    torch.hpu.synchronize()


def main():
    root = Path(os.environ['PROBE_OUT'])
    result = {'scope': 'synthetic BF16 packed target-layer semantics on one HPU; no native/full-model or traffic claim',
              'cases': [], 'pass': False}
    try:
        with torch.inference_mode():
            for window in (0, 128):
                cpu = fixture(sliding_window=window)
                device = fixture(device='hpu', sliding_window=window)
                batch, hidden, norm, qkv, out, histories, spec = device
                expected = sequential_target(*cpu).hidden.float()
                actual = packed_target(*device)
                sync()
                value = actual.hidden.cpu().float()
                delta = value - expected
                relative_l2 = float(delta.norm() / expected.norm().clamp_min(1e-30))
                torch.testing.assert_close(value, expected, rtol=.03, atol=.03)
                assert torch.isfinite(value).all()
                changed = hidden.clone()
                changed[3:5] = -changed[3:5].flip(-1)
                changed[133:] = float('nan')
                mutated = packed_target(batch, changed, norm, qkv, out, histories, spec)
                sync()
                mutated_value = mutated.hidden.cpu()
                assert torch.equal(actual.hidden.cpu()[:3], mutated_value[:3])
                assert torch.equal(actual.hidden.cpu()[5:], mutated_value[5:])
                assert not torch.equal(actual.hidden.cpu()[3:5], mutated_value[3:5])
                commits = []
                cpu_actual = packed_target(*cpu)
                for count in (0, 2, 4):
                    next_batch = TokenBatch((RequestTokens('verify', (99,), 126 + count, 'decode'),))
                    committed = ((torch.cat((histories[1][0], actual.candidate_keys[1][:count])),
                                  torch.cat((histories[1][1], actual.candidate_values[1][:count]))),)
                    cpu_committed = ((torch.cat((cpu[5][1][0], cpu_actual.candidate_keys[1][:count])),
                                      torch.cat((cpu[5][1][1], cpu_actual.candidate_values[1][:count]))),)
                    following = packed_target(next_batch, hidden[:1], norm, qkv, out, committed, spec)
                    reference = sequential_target(next_batch, cpu[1][:1], cpu[2], cpu[3], cpu[4], cpu_committed, spec)
                    sync()
                    torch.testing.assert_close(following.hidden.cpu().float(), reference.hidden.float(),
                                               rtol=.03, atol=.03)
                    commits.append({'committed_input_queries': count, 'continued_decode_checked': True})
                record = {'sliding_window': window, 'valid_rows': batch.num_tokens,
                          'capacity_rows': batch.capacity.token_rows,
                          'query_lengths': batch.query_lengths, 'query_start_loc': batch.query_start_loc,
                          'relative_l2': relative_l2, 'max_absolute': float(delta.abs().max()),
                          'qkv_calls': actual.projection_calls['qkv'], 'out_calls': actual.projection_calls['out'],
                          'future_query_isolation': True, 'nan_padding_excluded': True, 'commit_cases': commits}
                result['cases'].append(record)
                torch.save({'expected': expected, 'actual': value, 'mutated': mutated_value},
                           root / f'window-{window}.pt')
                (root / 'semantic-result.json').write_text(json.dumps(result, indent=2) + '\n')
            result['pass'] = True
    except BaseException as error:
        result['error'] = repr(error)
        raise
    finally:
        (root / 'semantic-result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
