"""Static packet accounting from actual archived ELF disassembly, not latency."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--simulator', type=Path, required=True)
p.add_argument('--unroll', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
report = dict(scope='actual ELF VLIW packet counts; no hardware timing/cache/transaction inference', kernels={})
for candidate, base in [('rolled', a.simulator), ('old_unroll4', a.unroll)]:
    result = json.loads((base/'result.json').read_text())
    for mode in ('norm', 'fused', 'block'):
        directory = base/'random'/mode
        text = (directory/'kernel.dis').read_text().split('.section .KernelInfo')[0]
        packets = []
        labels = {}
        for line in text.splitlines():
            label = re.match(r'^([0-9a-f]+) (\S+):$', line)
            if label:
                labels[label[2]] = int(label[1], 16)
            item = re.match(r'^\s*([0-9a-f]+):\s+(.*)', line)
            if item:
                packets.append((int(item[1], 16), item[2].split('//')[0].strip()))
        loops = []
        for address, body in packets:
            if not body.startswith('loop '):
                continue
            target = body.rsplit(', ', 1)[1]
            following = sorted(v for v in labels.values() if address < v < labels[target])
            if not following:
                continue
            begin = following[0]
            body_packets = [v for addr, v in packets if begin <= addr < labels[target]]
            ops = Counter(op.strip().split()[0] for v in body_packets for op in v.split(';'))
            loops.append(dict(instruction=body, begin=hex(begin), end_exclusive=hex(labels[target]),
                              packets=len(body_packets), all_nop_packets=sum(all(o.strip()=='nop' for o in v.split(';')) for v in body_packets),
                              slot_ops=dict(ops)))
        record = next(r for r in result['records'] if r['name']=='random')['modes'][mode]
        report['kernels'][candidate+'/'+mode] = dict(
            elf_sha256=hashlib.sha256((directory/'kernel.o').read_bytes()).hexdigest(),
            static_packets=len(packets), max_vector_register=max(map(int, re.findall(r'\bV(\d+)\b', text))),
            simulator_M1_H6144_packets=record['instructions'], loops=loops)

report['current_M1_H6144'] = dict(stats_packets_per_block=24, apply_packets_per_block=31,
    fused_apply_quant_packets_per_block=250, blocks=48, norm_fixed_packets=252,
    fused_fixed_packets=255, norm_total_packets=252+48*(24+31),
    fused_total_packets=255+48*(24+250), standalone_quant_grid=[48, 1], fused_grid=[1],
    vlm_row_bytes=16384, requested_norm_intermediate_store_bytes=12288,
    standalone_block_quant_requested_norm_read_bytes=12288)
assert report['kernels']['rolled/norm']['simulator_M1_H6144_packets']==report['current_M1_H6144']['norm_total_packets']
assert report['kernels']['rolled/fused']['simulator_M1_H6144_packets']==report['current_M1_H6144']['fused_total_packets']
report['grid24_design_only'] = dict(compiled=False, grid=[24], blocks_per_task=2,
    approximate_inherited_packets_per_task=252+48*24+2*250,
    excludes='new ownership predicates, changed scheduling, local storage changes, memory latency',
    activation_requested_read_bytes=24*6144*2*2, baseline_activation_requested_read_bytes=6144*2*2,
    gamma_requested_read_bytes=6144*2, residual_requested_store_bytes=6144*2,
    q_requested_store_bytes=6144, scale_requested_store_bytes=48*4,
    weight_read_change_bytes=0, actual_HBM_transactions='unmeasured')
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report['current_M1_H6144']))
