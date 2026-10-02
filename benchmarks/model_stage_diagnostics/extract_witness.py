"""Retain an equal-input QKV witness from our own frozen stage captures."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--case', type=Path, required=True)
p.add_argument('--reference', required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--layer', type=int, required=True)
p.add_argument('--rank', type=int, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(1)
paths = [a.case / 'quality-logits' / f'{tag}-teacher-0-stages-rank{a.rank}.pt'
         for tag in (a.reference, a.candidate)]
caps = [torch.load(path, map_location='cpu', weights_only=False) for path in paths]
for key in ('input_ids', 'positions'):
    assert torch.equal(caps[0][key], caps[1][key]), key
prefix = f'model.language_model.model.layers.{a.layer}.'
inputs = [cap['stages'][prefix + 'input_layernorm'] for cap in caps]
assert torch.equal(*inputs), 'Local kernel attribution requires equal observed inputs'
outputs = [cap['stages'][prefix + 'self_attn.qkv_proj'] for cap in caps]
assert outputs[0].dtype == outputs[1].dtype == torch.bfloat16
assert outputs[0].shape == outputs[1].shape
indices = (outputs[0].view(torch.int16) != outputs[1].view(torch.int16)).nonzero()
assert len(indices), 'No witness at this layer/rank'
a.out.mkdir(parents=True, exist_ok=True)
records = []
for index in indices.tolist():
    values = [output[tuple(index)] for output in outputs]
    record = dict(layer=a.layer, rank=a.rank, output_index=index[-1],
                  index=index, input_equal=True, reference_policy=a.reference,
                  candidate_policy=a.candidate,
                  reference_value=float(values[0]), actual_value=float(values[1]),
                  reference_bits=int(values[0].view(torch.int16)) & 65535,
                  actual_bits=int(values[1].view(torch.int16)) & 65535)
    name = f'witness-layer{a.layer}-rank{a.rank}-row{index[-1]}.pt'
    torch.save(dict(input_bf16=inputs[0][index[0]:index[0]+1],
                    reference_qkv=outputs[0], actual_qkv=outputs[1], record=record), a.out/name)
    records.append(dict(**record, witness=name))
report = dict(sources=[dict(path=str(path), bytes=path.stat().st_size,
                          sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths],
              records=records, device_accessed=False, thresholds_changed=False)
(a.out/'witnesses.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(records, indent=2))
