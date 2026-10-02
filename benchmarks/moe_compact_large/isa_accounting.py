"""Offline audit of the exact unchanged libraries used by the large-M probe.

Counts decoded scheduled packets, including each half of compressed encoding.
This does not turn packet counts, logical bytes, or SoC telemetry into cycles,
physical bandwidth, utilization, or predicted latency.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

from elftools.elf.elffile import ELFFile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def extract(path, symbol, name, output):
    raw = path.read_bytes()
    elf = ELFFile(io.BytesIO(raw))
    symbols = {s.name: s for s in elf.get_section_by_name('.symtab').iter_symbols()}
    start = symbols['_binary_' + symbol + '_o_start']
    end = symbols['_binary_' + symbol + '_o_end']
    section = elf.get_section(start['st_shndx'])
    offset = section['sh_offset'] + start['st_value'] - section['sh_addr']
    blob = raw[offset:offset + end['st_value'] - start['st_value']]
    text = ELFFile(io.BytesIO(blob)).get_section_by_name('.text').data()
    target = output / (name + '.elf')
    target.write_bytes(blob)
    (output / (name + '.text')).write_bytes(text)
    command = ['/usr/bin/tpc-llvm-objdump', '-d', '--triple=tpc', '--mcpu=gaudi2',
               '--no-show-raw-insn', str(target)]
    disassembly = subprocess.check_output(command, text=True)
    (output / (name + '.dis')).write_text(disassembly)
    return disassembly, dict(library=str(path), library_sha256=sha(raw),
                             elf_sha256=sha(blob), text_sha256=sha(text), command=command)


def audit(disassembly):
    rows = []
    for line in disassembly.splitlines():
        match = re.match(r'\s*([0-9a-f]+):\s+(.*)', line)
        if match:
            rows.append((int(match[1], 16), match[2].split('//')[0].strip()))
    loops = [(i, address, op) for i, (address, op) in enumerate(rows)
             if op.startswith('loop')]
    assert len(loops) == 2
    ranges = []
    for index, address, op in loops:
        end = address + int(re.search(r'<, (\d+)', op)[1])
        # Hardware-loop delay packet precedes the first body packet. End is
        # inclusive (the final FP32 scale MAC in the inner K32 loop).
        ranges.append((rows[index + 2][0], end))
    outer, inner = ranges
    body = [(address, op) for address, op in rows if inner[0] <= address <= inner[1]]
    setup = [(address, op) for address, op in rows
             if outer[0] <= address <= outer[1] and not inner[0] <= address <= inner[1]]
    assert body[-1][1].split(';')[2].strip().startswith('mac.f32')
    split = [[s.strip() for s in op.split(';')] for _, op in body]
    assert all(len(slots) == 4 for slots in split)
    slots = [Counter(row[i].split()[0] for row in split if row[i] != 'nop') for i in range(4)]
    tensor_loads = Counter()
    for row in split:
        if row[0].startswith('ld_tnsr'):
            tensor = re.search(r'V\d+, 0x([0-9a-f]+),', row[0])
            assert tensor
            tensor_loads[int(tensor[1], 16)] += 1
    assert tensor_loads[0] == 32 and tensor_loads[1] == 2
    assert slots[0]['lookup_2c'] == 64 and slots[2]['mac.bf16'] == 128
    assert slots[2]['mac.f32'] == 8
    # Raw operand classes are checked as well as the op-name histogram.
    assert all('acc_fp32' in row[2] for row in split if row[2].startswith('mac.bf16'))
    local_non_mmio = [op for _, op in rows
                     if re.search(r'\b(?:ld_l|st_l)(?:_v)?\b', op) and 'mmio' not in op]
    assert not local_non_mmio, local_non_mmio
    body_signature = '\n'.join(' ; '.join(op.split()) for _, op in body)
    return dict(inner_start_hex=hex(inner[0]), inner_last_hex=hex(inner[1]),
                decoded_body_sha256=sha(body_signature.encode()),
                k32_packets=len(body), outer_task_overhead_packets=len(setup),
                slot_instruction_counts=[dict(s) for s in slots],
                slot_non_nop=[sum(s.values()) for s in slots],
                tensor_loads_by_operand=dict(tensor_loads),
                scalar_activation_loads_per_k32=slots[0]['ld_g'],
                activation_address_generations_per_k32=slots[3]['gen_addr'],
                vector_activation_loads_per_k32=tensor_loads[2],
                activation_shuffle_per_k32=slots[2]['shuffle.u8'],
                no_local_spill_instructions=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    c = args.canonical
    runtime = c / 'artifacts/builds/moe-compact8/final-a/runtime-a'
    production = c / 'artifacts/builds/production-integration/model-runs/production-fp32-70-d/source'
    libraries = {
        'nm': (runtime / 'executor/tpc/libnative_tpc.so', 'gemv'),
        'gp': (production / 'production-runtime/gp-scale-tail/tpc.so', 'candidate'),
        'down': (runtime / 'libraries/down/libgaudi_down_activation_tpc.so', 'direct_down'),
        'scalar_gp': (runtime / 'tpc/libbatch_tpc.so', 'direct_gp'),
        'scalar_down': (runtime / 'tpc/libbatch_tpc.so', 'direct_down'),
        'folded_down': (runtime / 'libraries/folded/libgaudi_moe_activation_folded_tpc.so', 'folded_down'),
    }
    kernels = {}
    for name, (path, symbol) in libraries.items():
        disassembly, identity = extract(path, symbol, name, args.output)
        kernels[name] = dict(identity=identity, **audit(disassembly))
    assert [kernels[x]['k32_packets'] for x in libraries] == [230, 247, 289, 235, 230, 269]
    assert kernels['nm']['decoded_body_sha256'] == kernels['scalar_down']['decoded_body_sha256']
    def pre_inner(text):
        rows = []
        for line in text.splitlines():
            match = re.match(r'\s*[0-9a-f]+:\s+(.*)', line)
            if not match:
                continue
            op = match[1].split('//')[0].strip()
            if op.startswith('loop 0,'):
                break
            if op.startswith('loop'):
                # Only branch extent differs as the inner schedule shrinks.
                op = re.sub(r'(<, )\d+', r'\1<end>', op)
            rows.append(op)
        return rows
    prefix_checks = {}
    for scalar, vector in [('scalar_gp', 'gp'), ('scalar_down', 'down')]:
        old = pre_inner((args.output / (scalar + '.dis')).read_text())
        new = pre_inner((args.output / (vector + '.dis')).read_text())
        assert old == new
        prefix_checks[scalar] = dict(compared_to=vector, decoded_packets=len(old),
                                    equal_except_outer_loop_extent=True)
    assert kernels['nm']['identity']['library_sha256'] == 'b0c3382922a66098974b190b37775f31098517f0d26836a5100e77c944237f71'
    assert kernels['gp']['identity']['library_sha256'] == 'f45f038808d6ec6ba14ac99445fe7bdce35a523c6f63271a3f37624e57130e07'
    assert kernels['down']['identity']['library_sha256'] == '9675942ebd61768cfa16064f9a034ef34f11e42238b0dcdfdca1a5811d73c47a'
    shapes = []
    for m in (1, 2, 8, 32, 128, 512, 513):
        tasks = {'gp': m * 8 * 3, 'down': m * 8 * 12}
        repeats = {'gp': 64, 'down': 8}
        groups = sum(tasks[k] * repeats[k] for k in tasks)
        schedules = {}
        for name, selection in {'broadcast': ('nm', 'nm'), 'compact': ('gp', 'down'),
                                'compact_scalar_both': ('scalar_gp', 'scalar_down'),
                                'compact_scalar_down': ('gp', 'scalar_down'),
                                'compact_folded_down': ('gp', 'folded_down')}.items():
            detail = {}
            for stage, variant in zip(('gp', 'down'), selection):
                kernel = kernels[variant]
                body = tasks[stage] * repeats[stage] * kernel['k32_packets']
                overhead = tasks[stage] * kernel['outer_task_overhead_packets']
                detail[stage] = dict(body_packets=body, task_overhead_packets=overhead,
                                     vpu_instructions=tasks[stage] * repeats[stage] * kernel['slot_non_nop'][2])
            schedules[name] = dict(stages=detail,
                body_packets=sum(x['body_packets'] for x in detail.values()),
                body_plus_task_packets=sum(x['body_packets'] + x['task_overhead_packets'] for x in detail.values()),
                vpu_instructions=sum(x['vpu_instructions'] for x in detail.values()))
        shapes.append(dict(M=m, topk=8, tasks=tasks,
            logical_tasks_per_core_at_24={s: tasks[s] / 24 for s in tasks},
            task_balance_scope='Arithmetic only; no observed wave or clock claim',
            k32_groups=groups, schedules=schedules,
            logical_weight_scale_requested_bytes=groups * (32 * 256 + 2 * 256),
            scalar_activation_requested_bytes=groups * 32 * 2,
            vector_activation_requested_bytes=groups * 256,
            broadcast_gp_materialized_bytes=tasks['gp'] * 2048 * 2,
            broadcast_down_materialized_bytes=tasks['down'] * 256 * 2,
            compact_activation_owner_bytes=m * 6144 * 2,
            compact_gate_owner_bytes=m * 8 * 256 * 2,
            scope='Requested operand bytes, overlapping vector windows included; not physical HBM transactions. Mismatched padded lanes are not consumed.'))
    result = dict(status='PASS_STATIC_ISA_ACCOUNTING', device_accessed=False,
                  kernels=kernels, compact_mapping_prefix_checks=prefix_checks, shapes=shapes,
                  scope='Scheduled packets and requested bytes only; no inferred cycles, utilization, clock, bandwidth or model gain. Frozen large-M probe DLL identities are checked.')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {k: value[k] for k in ('k32_packets','outer_task_overhead_packets','slot_non_nop')}
                      for name, value in kernels.items()}))


if __name__ == '__main__':
    main()
