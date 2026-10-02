"""Count actual scheduled inner-loop packets; never call these elapsed cycles."""
import argparse
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('builds',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('isa_parse',Path(__file__).resolve().parents[1]/'swa128_ilp/isa_audit.py')
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
records=[]
for build in a.builds:
    for name in ('gp4','down4'):
        path=build/(name+'.s')
        if not path.is_file():continue
        blocks=parser.parse(path);label=None;inner=[]
        for line in path.read_text().splitlines():
            match=re.match(r'(?:\.LBB0_|// %bb\.)(\d+)',line.strip())
            if match:label=int(match[1])
            if 'This Inner Loop Header' in line:inner.append(label)
        loops=[]
        for block in inner:
            packets=blocks[block];ops=Counter();macs=defaultdict(list)
            for index,packet in enumerate(packets):
                for _,op in packet:
                    ops[op.split()[0].lower()]+=1
                    if op.startswith('mac.bf16'):
                        macs[re.search(r'%D\d+',op)[0]].append(index)
            if not macs:continue
            gaps=[b-a for indices in macs.values()for a,b in zip(indices,indices[1:])]
            loops.append(dict(block=block,packets=len(packets),nop_only_packets=sum(all(s.lower()=='nop'for _,s in b)for b in packets),
                bf16_macs=ops['mac.bf16'],accumulator_pairs=sorted(macs),same_accumulator_packet_gaps=dict(Counter(gaps)),
                local_vector_loads=ops['ld_l_v'],local_vector_stores=ops['st_l_v'],opcodes=dict(ops)))
        records.append(dict(build=str(build),kernel=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),loops=loops))
result=dict(scope='Static scheduled packets in actual compiled inner loops, not measured cycles, issue counters, or dynamic spill totals.',records=records)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
for r in records:
    print(r['build'],r['kernel'])
    for loop in r['loops']:print({k:v for k,v in loop.items()if k!='opcodes'})
