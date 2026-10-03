"""Count decoded TPC loop issue packets separately from the LOOP delay packet.

Use actual TPC objdump output, not embedded C or stale basic-block labels.
Counts each decoded compressed half as its own instruction; no cycle-counter
or physical bandwidth claim. The pinned compiler inserts one LOOP delay slot.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def analyze(disassembly, loop_index):
    rows = []
    for line in disassembly.splitlines():
        match = re.match(r'\s*([0-9a-f]+):\s+(.*)', line)
        if match:
            rows.append((int(match[1],16),match[2].split('//')[0].strip()))
    loops = [i for i, (_, op) in enumerate(rows) if op.startswith('loop ')]
    if loop_index not in range(len(loops)):
        raise ValueError('Missing selected hardware loop')
    i = loops[loop_index]
    address, instruction = rows[i]
    end = address + int(re.search(r'<,\s*(\d+)',instruction)[1])
    delay, first = rows[i+1], rows[i+2]
    body = [op for pc,op in rows if first[0] <= pc <= end]
    slots = [[part.strip() for part in op.split(';')] for op in body]
    if not slots or any(len(row)!=4 for row in slots):
        raise ValueError('Malformed TPC loop body')
    histograms = [Counter(row[s].split()[0] for row in slots if row[s]!='nop') for s in range(4)]
    return dict(loop_address=hex(address),delay_address=hex(delay[0]),delay_instruction=delay[1],
        first_repeated_instruction=hex(first[0]),last_repeated_instruction=hex(end),
        repeated_decoded_packets=len(body),slot_non_nop=[sum(h.values())for h in histograms],
        slot_instruction_counts=[dict(h)for h in histograms],
        scope='Static decoded issue instructions, not runtime cycles. LOOP delay is excluded from repeated body; compressed halves remain separate instructions.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('disassembly',type=Path)
    p.add_argument('--loop-index',type=int,required=True)
    p.add_argument('--output',type=Path)
    a=p.parse_args();result=json.dumps(analyze(a.disassembly.read_text(),a.loop_index),indent=2)+'\n'
    if a.output:a.output.write_text(result)
    else:print(result,end='')
