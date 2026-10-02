"""Scheduled packet and conservative register liveness audit of actual assembly.

D<n> aliases V<n>,V<n+1>. Partial writes and predicated writes conservatively
read their entire old destination. This may overstate lane-level liveness; it
does not turn register numbers or packet counts into physical latency claims.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re


def parse(path):
    blocks = collections.defaultdict(list)
    label, bundle = 0, None
    for line, raw in enumerate(path.read_text().splitlines(), 1):
        match = re.match(r'(?:\.LBB0_|// %bb\.)(\d+)', raw.strip())
        if match:
            label = int(match[1])
            continue
        s = raw.split('//')[0].strip()
        if not s or s.startswith('.') or s.startswith('$'):
            continue
        if s == '{':
            bundle = []
        elif s == '}':
            blocks[label].append(bundle)
            bundle = None
        elif bundle is None:
            blocks[label].append([(line, s)])
        else:
            bundle.append((line, s))
    return blocks


def expand(register):
    kind, number = register[0], int(register[1:])
    return {number, number + 1} if kind == 'D' else {number}


def def_use(instruction):
    registers = re.findall(r'%([VD]\d+)', instruction)
    if not registers:
        return set(), set()
    op = instruction.split()[0]
    if op.startswith('st_'):
        return set(), set().union(*(expand(r) for r in registers))
    defs = expand(registers[0])
    uses = set().union(*(expand(r) for r in registers[1:])) if len(registers) > 1 else set()
    if (op.startswith('mac.') or '%SP' in instruction or '%VP' in instruction
            or op.startswith('unpack.') or 'mov_dg unpack' in instruction
            or 'ld_tnsr partial' in instruction):
        uses |= defs
    return defs, uses


def path_record(blocks, ids, boundary):
    packets = [packet for block in ids for packet in blocks[block]]
    operations = collections.Counter(instruction.split()[0] for packet in packets for _, instruction in packet)
    ud = []
    for packet in packets:
        defs, uses = set(), set()
        for _, instruction in packet:
            d, u = def_use(instruction)
            defs |= d
            uses |= u
        ud.append((defs, uses))
    # The selected paths are complete QK loop iterations, including score
    # insertion and the loop edge. Iterate the back edge until fixed point.
    # Query3, scores2, lane IDs, scale and three shuffle controls survive the
    # whole token loop, including the score vector untouched on this path.
    live_in = set(boundary)
    for _ in range(100):
        live = live_in.copy()
        for defs, uses in reversed(ud):
            live = (live - defs) | uses
        if live == live_in:
            break
        live_in |= live
    else:
        raise RuntimeError('liveness did not converge')
    live, points = live_in.copy(), []
    for index in reversed(range(len(packets))):
        after = live.copy()
        defs, uses = ud[index]
        live = (live - defs) | uses
        points.append({'packet': index, 'source_lines': [line for line, _ in packets[index]],
                       'before': sorted(live), 'after': sorted(after),
                       'defs': sorted(defs), 'uses': sorted(uses)})
    points.reverse()
    peak = max(points, key=lambda point: len(set(point['before']) | set(point['defs'])))
    registers = set().union(*(d | u for d, u in ud))
    return {'blocks': ids, 'packets': len(packets),
            'nop_only_packets': sum(all(s.lower() == 'nop' for _, s in packet) for packet in packets),
            'opcodes': dict(operations), 'vector_registers': sorted(registers),
            'persistent_query_scores_controls': boundary,
            'loop_carried_registers_conservative': sorted(live_in),
            'peak_live_plus_destinations_conservative': len(set(peak['before']) | set(peak['defs'])),
            'peak_packet': peak, 'liveness_by_packet': points}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = {'baseline': {'token': [6, 7]},
             'head_pair': {'lower64': [9, 10, 11, 14],
                           'cross64': [9, 10, 13, 14],
                           'upper64': [9, 12, 13, 14]},
             'head_quad': {'lower64': [9, 10, 11, 12, 13, 18],
                           'upper64': [9, 14, 15, 16, 17, 18],
                           'cross63': [9, 10, 15, 16, 17, 18],
                           'cross62': [9, 10, 11, 16, 17, 18],
                           'cross61': [9, 10, 11, 12, 17, 18]},
             'head_quad_select': {'all_score_indices': [9]}}
    boundary = {'baseline': [0, 1, 2, 3, 4, 6, 7, 8, 9, 11],
                'head_pair': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10],
                'head_quad': [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
                'head_quad_select': [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]}
    result = {'scope': 'scheduled packets and conservative whole-register backward liveness; no physical latency',
              'records': {}}
    for name, choices in paths.items():
        assembly = a.build / (name + '.s')
        blocks = parse(assembly)
        text = assembly.read_text()
        memory = [line.strip() for line in text.splitlines()
                  if re.search(r'\b(?:ld_l|st_l)\b', line) and 'mmio' not in line]
        result['records'][name] = {
            'assembly_sha256': hashlib.sha256(assembly.read_bytes()).hexdigest(),
            'non_mmio_local_memory_instructions': memory,
            'paths': {key: path_record(blocks, value, boundary[name]) for key, value in choices.items()}}
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    for name, value in result['records'].items():
        print(name, {key: {field: rec[field] for field in (
            'packets', 'nop_only_packets', 'vector_registers',
            'loop_carried_registers_conservative', 'peak_live_plus_destinations_conservative')}
                     for key, rec in value['paths'].items()})


if __name__ == '__main__':
    main()
