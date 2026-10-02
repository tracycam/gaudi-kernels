"""Check changed decoder ISA and unchanged other core kernel .text."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
from elftools.elf.elffile import ELFFile


def section(path):
    return ELFFile(io.BytesIO(path.read_bytes())).get_section_by_name('.text').data()


def instructions(source):
    return [line.strip().split()[0] for line in source.splitlines()
            if line.startswith('\t') and line.strip()
            and not line.strip().startswith(('.', '//', '{', '}'))]


def main():
    p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True)
    p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();kernels={}
    for old in sorted(a.old.glob('*.o')):
        if old.name.endswith('_x86.o'):continue
        new=a.new/old.name;ot=section(old);nt=section(new)
        if old.stem!='graph_decode':assert ot==nt,old.name
        kernels[old.stem]=dict(old_text_sha256=hashlib.sha256(ot).hexdigest(),
                              new_text_sha256=hashlib.sha256(nt).hexdigest(),identical=ot==nt)
    assert len(kernels)==7
    rows={}
    for label,root in [('old',a.old),('new',a.new)]:
        source=(root/'graph_decode.s').read_text()
        assert not re.search(r'\b(?:ld_l|st_l)_v\b',source)
        loops=[]
        for first,last in [('7','9'),('13','15')]:
            block=source[source.index('.LBB0_'+first+':'):source.index('.LBB0_'+last+':')]
            counts=Counter(instructions(block))
            if label=='new':
                assert 'or.i16'not in block and '0x8000'not in block
                assert counts['and.i16']==2 and counts['sel_eq.u16']==2
            loops.append(dict(block=first,assembly_instructions=dict(counts)))
        rows[label]=dict(loops=loops,
                        whole_function_instructions=len(instructions(source.split('.Lfunc_end0')[0])),
                        max_V_register=max(map(int,re.findall(r'%V(\d+)',source))),
                        local_vector_spill=False)
    report=dict(status='PASS_ISA_SCALE_BITS',kernels=kernels,decoder=rows,
                scope='Per active inner K step processes 256 decoded BF16 values. '
                      'New scale setup has two extra masks outside K, amortized over K32. '
                      'No static-to-device timing inference.')
    a.output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'])


if __name__=='__main__':main()
