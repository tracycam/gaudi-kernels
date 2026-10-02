"""Append score stores to disassembled ELF; keep every existing issue packet.

The added address setup precedes QK. The two stores precede softmax and extend
that boundary by two packets; this diagnostic is never used for performance.
Original full-output ELF correctness is always checked separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('source', type=Path)
p.add_argument('--variant', choices=['baseline', 'quad'], required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
original, active = [], False
for raw in a.source.read_text().splitlines():
    label = re.fullmatch(r'[0-9a-f]+ (main|\.LBB\w+):', raw.strip())
    if label:
        original.append(label[1] + ':')
        active = True
    elif active:
        hit = re.match(r'\s*[0-9a-f]+:\s+(.*)', raw)
        if hit:
            original.append(hit[1].split('//', 1)[0].strip())
        elif raw.startswith(('Disassembly of section', '.section', 'TPC compiler')):
            break
assert not any(re.search(r'\bI1[01]\b', s) for s in original), 'debug index registers must be unused'
label, first, second = (11, 5, 9) if a.variant == 'baseline' else (16, 9, 10)
inserts = {'.LBB0_5:': [
    'set_indx I10, b00010, S32; nop; nop; nop',
    'set_indx I10, b11101, 0x0; nop; nop; nop',
    'set_indx I11, b00010, S32; nop; nop; nop',
    'set_indx I11, b11101, 0x0; nop; nop; nop',
    'set_indx I11, b00001, 0x40; nop; nop; nop'],
    f'.LBB0_{label}:': [f'nop; nop; nop; st_tnsr 0x8, I10, V{first}',
                       f'nop; nop; nop; st_tnsr 0x8, I11, V{second}']}
result, original_positions = [], []
for line in original:
    original_positions.append(len(result))
    result.append(line)
    result.extend(inserts.get(line, []))
assert [result[i] for i in original_positions] == original
assert len(result) - len(original) == 7
a.output.write_text('.text\n.globl main\n' + '\n'.join(result) + '\n')
a.output.with_suffix('.json').write_text(json.dumps({
    'input_disassembly_sha256': hashlib.sha256(a.source.read_bytes()).hexdigest(),
    'variant': a.variant, 'existing_packets_modified': 0,
    'setup_packets_before_qk': 5, 'score_store_packets_before_softmax': 2,
    'original_sequence_preserved': True, 'performance_variant': False}, indent=2) + '\n')
