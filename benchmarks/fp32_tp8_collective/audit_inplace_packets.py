"""Read-only exact/runtime HCL packet evidence; never contact a device."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p = argparse.ArgumentParser()
p.add_argument('--case', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
status = json.loads((a.case / 'exit.json').read_text())
assert status['runner_exit_code'] == 0 and status['packet_only']
assert status['comparison'] == 'ag_inplace'
records = []
for kind in ['exact', 'ordinary']:
    for rank in range(8):
        path = a.case / kind / f'logs-rank{rank}' / 'hcl.log'
        text = path.read_text()
        version = re.search(r'Version:\s*([^\s]+)', text).group(1)
        lines = text.splitlines()
        starts = [i for i, line in enumerate(lines)
                  if 'Running another iteration, collectiveOp=5[eHCLAllGather]' in line]
        assert len(starts) == 4, (kind, rank, len(starts))
        for phase, begin in enumerate(starts):
            end = starts[phase + 1] if phase < 3 else len(lines)
            section = lines[begin:end]
            copies = [line for line in section if 'Counts for memcpy: op 5[eHCLAllGather]' in line]
            packets = [line for line in section if 'serializeDmaCommand with arc_cmd_nic_edma_lin_ops' in line]
            signals = [line for line in section if 'signal 3[EDMA_MEMCOPY] should add' in line]
            totals = [int(m.group(1)) for line in section
                      if 'Calculating requirements for event 0[GENERAL_COMPLETION_EVENT]' in line
                      for m in [re.search(r'generates (\d+) signals', line)] if m]
            inplace = phase in (1, 2)
            assert len(copies) == len(packets) == len(signals) == (0 if inplace else 1), (kind, rank, phase)
            assert totals == [43 if inplace else 46], (kind, rank, phase, totals)
            if not inplace:
                assert 'count 6144,' in copies[0] and 'transfer_size:24576,' in packets[0]
                assert 'should add 3' in signals[0]
            records.append(dict(fixture=kind, rank=rank, phase=phase, in_place=inplace,
                                runtime_version=version, file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                collective_start_line=begin + 1, local_copy_commands=len(packets),
                                local_copy_bytes=0 if inplace else 24576,
                                total_completion_signals=totals[0],
                                copy_and_signal_evidence=copies + packets + signals))
report = dict(status='PASS_SERIALIZED_PACKET_AUDIT', records=records,
              out_of_place_local_copy_commands=sum(r['local_copy_commands'] for r in records if not r['in_place']),
              in_place_local_copy_commands=sum(r['local_copy_commands'] for r in records if r['in_place']),
              scope='Actual runtime serialized packets and completion accounting, not physical HBM counters or a latency decomposition')
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'records'}))
