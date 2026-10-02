"""Audit emitted HCL handoff work, without treating packet counts as latency."""
import argparse,collections,hashlib,json,re
from pathlib import Path
from stream_audit import check
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
status=json.loads((a.case/'exit.json').read_text());assert status['runner_exit_code']==0 and status['packet_only'] and status['comparison']=='ag_two_stream'
records=[]
for kind in ('exact','ordinary'):
 for rank in range(8):
  path=a.case/kind/f'logs-rank{rank}/hcl.log';lines=path.read_text().splitlines()
  starts=[i for i,l in enumerate(lines)if 'Running another iteration, collectiveOp=5[eHCLAllGather]'in l];assert len(starts)==4
  phases=[]
  for phase,b in enumerate(starts):
   section=lines[b:starts[phase+1]if phase<3 else len(lines)]
   packets=dict(collections.Counter(m.group(1)for l in section for m in [re.search(r'Packets \| (serialize\w+)',l)]if m))
   waits=[l for l in section if 'Adding stream wait on LSO:'in l]
   copy=[l for l in section if 'serializeDmaCommand with arc_cmd_nic_edma_lin_ops'in l]
   signals=[int(m.group(1))for l in section if 'Calculating requirements for event 0[GENERAL_COMPLETION_EVENT]'in l for m in [re.search(r'generates (\d+) signals',l)]if m]
   assert len(copy)==1 and 'transfer_size:24576,'in copy[0] and signals==[46]
   assert len(waits)==5 and packets['serializeFenceDecCommand']==10 and packets['serializeAllocBarrierCommand']==6
   tag=('agdual'if phase in (1,2)else'agserial')+str(phase)
   api=[json.loads(l)for l in (a.case/kind/f'rank{rank}-{tag}-api.jsonl').read_text().splitlines()]
   flow=check(api,1,phase in (1,2))
   rec=dict(fixture=kind,rank=rank,phase=phase,api=flow,serialized_packet_counts=packets,
            producer_long_SO_waits=len(waits),local_copy_bytes=24576,total_completion_signals=signals[0],
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),start_line=b+1,evidence=waits+copy)
   phases.append(rec);records.append(rec)
  assert phases[1]['serialized_packet_counts']==phases[2]['serialized_packet_counts']==phases[3]['serialized_packet_counts'],(kind,rank)
result=dict(status='PASS_NO_HANDOFF_PACKET_REMOVAL',segments=len(records),records=records,
            warm_packet_classes_equal=True,production_promoted=False,
            scope='Public stream/event dependency audit plus actual serialized HCL command classes. First serial phase retains cold context setup. Not full Synapse firmware packet equivalence, NIC/HBM counters, or latency attribution.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k!='records'}))
