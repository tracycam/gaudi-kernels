"""Count actual disassembled VLIW packets, separately from encoded byte size."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
from elftools.elf.elffile import ELFFile


def text(path):return ELFFile(io.BytesIO(path.read_bytes())).get_section_by_name('.text').data()


def main():
    p=argparse.ArgumentParser();p.add_argument('--builds',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows={}
    for factor in (1,2,4):
        root=a.builds/f'u{factor}';source=(root/'graph_decode.s').read_text()
        assert not re.search(r'\b(?:ld_l|st_l)_v\b',source)
        with(root/'graph_decode.dis').open('w')as f:
            subprocess.run(['tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2',
                            '--no-show-raw-insn',str(root/'graph_decode.o')],stdout=f,check=True)
        dis=(root/'graph_decode.dis').read_text()
        labels={name:int(address,16)for address,name in re.findall(r'^([0-9a-f]+) (\.LBB\d+_\d+):',dis,re.M)}
        packets=[]
        for address,body in re.findall(r'^\s*([0-9a-f]+):\s+(.+)$',dis,re.M):
            parts=[s.strip()for s in body.split('//')[0].split(';')]
            packets.append((int(address,16),parts))
        starts=re.findall(r'^(\.LBB\d+_\d+):\s*// %for.body51(?:\.1)?$',source,re.M)
        ends=re.findall(r'^(\.LBB\d+_\d+):\s*// %for.cond.cleanup50(?:\.1)?$',source,re.M)
        assert len(starts)==len(ends)==2
        loops=[]
        for start,end in zip(starts,ends):
            values=[(addr,parts)for addr,parts in packets if labels[start]<=addr<labels[end]]
            assert all(len(parts)==4 for _,parts in values)
            loads=[i for i,(_,parts) in enumerate(values) if parts[0].startswith('ld_tnsr')]
            lookups=[i for i,(_,parts) in enumerate(values) if parts[0].startswith('lookup_2c')]
            assert len(loads)==len(lookups)==factor
            counts=Counter(s.split()[0]for _,parts in values for s in parts if s!='nop')
            loops.append(dict(label=start,packet_count=len(values),packets_per_K=len(values)/factor,
                              encoded_bytes=labels[end]-labels[start],
                              all_nop_packets=sum(all(s=='nop'for s in parts)for _,parts in values),
                              slot_non_nop=[sum(parts[i]!='nop'for _,parts in values)for i in range(4)],
                              load_packet_indices=loads,lookup_packet_indices=lookups,
                              operations=dict(counts)))
        unchanged=[]
        for old in (a.builds/'u1').glob('*.o'):
            if old.name.endswith('_x86.o')or old.stem=='graph_decode':continue
            assert text(old)==text(root/old.name),old.name;unchanged.append(old.stem)
        rows[str(factor)]=dict(loops=loops,max_V_register=max(map(int,re.findall(r'%V(\d+)',source))),
                              no_vector_spill=True,unchanged_kernel_text=unchanged,
                              actual_text_sha256=hashlib.sha256(text(root/'graph_decode.o')).hexdigest())
    report=dict(status='PASS_STATIC_UNROLL_AUDIT',variants=rows,
                slots=['LOAD','SPU','VPU','STORE'],
                scope='Actual disassembly counts each compressed half as one scheduled packet. '
                      'Active K-loop path includes zero initialization, branch, loads, lookup, scale and stores. '
                      'No clock, stall or device-time inference.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows))


if __name__=='__main__':main()
