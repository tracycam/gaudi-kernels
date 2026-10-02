"""Audit actual sealed VLIW schedules; static packets are not device cycles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit(path):
    raw=path.read_bytes()
    body=raw.decode().split('.LBB0_3:\n',1)[1].split('.LBB0_6:',1)[0]
    rows=[[slot.strip() for slot in line.split(';')]
          for line in body.splitlines() if ';' in line]
    assert rows and all(len(row)==4 for row in rows)
    slots=[Counter(row[i].split()[0] for row in rows
                   if row[i].lower()!='nop') for i in range(4)]
    vpu=sum(slots[2].values())
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),
                packets=len(rows),slot_instructions=[dict(x) for x in slots],
                slot_non_nop=[sum(x.values()) for x in slots],
                unchanged_vpu_assignment_packet_compression_ceiling=1-vpu/len(rows),
                scope='Static emitted loop; instruction latencies, cache misses, dynamic stalls, other loops and clocks are not measured. Ceiling assumes the VPU slot assignment stays unchanged; it is not a hardware or algorithm lower bound.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('assembly',type=Path,nargs='+')
    a=p.parse_args();print(json.dumps([audit(x) for x in a.assembly],indent=2))
