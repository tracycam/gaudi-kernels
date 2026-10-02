"""Hash real TPC ELF text separately from source/command metadata."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def text_section(path):
    data=path.read_bytes()
    assert data[:6]==b'\x7fELF\x01\x01', 'TPC ELF32 little-endian expected'
    offset=struct.unpack_from('<I',data,32)[0]
    size,count,names=struct.unpack_from('<HHH',data,46)
    sections=[struct.unpack_from('<10I',data,offset+i*size)for i in range(count)]
    table=sections[names];strings=data[table[4]:table[4]+table[5]]
    for section in sections:
        name=strings[section[0]:].split(b'\0',1)[0]
        if name==b'.text':return data[section[4]:section[4]+section[5]]
    raise AssertionError('missing .text')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    records=[]
    for obj in sorted(a.root.glob('*/builds/*/mxfp4_decode.o')):
        asm=obj.with_suffix('.s')
        if not asm.is_file():continue
        source=asm.read_text().split('\t.type\ttpc_compiler')[0]
        regs=[int(x)+(kind=='D') for kind,x in re.findall(r'%(V|D)(\d+)',source)]
        local=[l.strip()for l in source.splitlines()if re.search(r'\b(?:ld|st)_l(?:_v)?\b',l)and'mmio'not in l]
        text=text_section(obj)
        records.append(dict(path=str(obj),elf_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),
                            text_sha256=hashlib.sha256(text).hexdigest(),text_bytes=len(text),
                            compiler_assembly_sha256=hashlib.sha256(asm.read_bytes()).hexdigest(),
                            max_vector_register=max(regs),non_mmio_local_instructions=local))
    indexed={r['path']:r for r in records}
    control=indexed[str(a.root/'pragma/builds/control/mxfp4_decode.o')]
    pragma=indexed[str(a.root/'pragma/builds/u4/mxfp4_decode.o')]
    assert control['text_sha256']==pragma['text_sha256']
    assert not any(r['non_mmio_local_instructions']for r in records)
    result=dict(records=records,pragma_did_not_change_executable_text=True,
                no_non_mmio_local_spills_in_compiler_assembly=True,
                scope='Static executable text and register use; no issue-cycle claim')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
