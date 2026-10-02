"""Repair stale C-template symbols after replacing pinned literal .text.

The qualified reference retained two obsolete C loop labels and the original
C function size. These are debug symbols, not TPC metadata. Do not leave them
pointing into unrelated newly scheduled instructions in disassembly.
"""
import argparse,json,struct
from pathlib import Path


def inspect(data):
    assert data[:7]==b'\x7fELF\x01\x01\x01'
    offset=struct.unpack_from('<I',data,32)[0];size,count,names=struct.unpack_from('<HHH',data,46)
    sections=[struct.unpack_from('<10I',data,offset+i*size)for i in range(count)]
    sym=next(s for s in sections if s[1]==2);string=sections[sym[6]];strings=data[string[4]:string[4]+string[5]]
    labels={}
    for pos in range(sym[4],sym[4]+sym[5],sym[9]):
        n,v,length,info,other,index=struct.unpack_from('<IIIBBH',data,pos)
        name=strings[n:strings.index(0,n)].decode()
        if name=='main'or name.startswith('.LBB'):labels[name]=(pos,v)
    ns=sections[names];table=data[ns[4]:ns[4]+ns[5]]
    text=next(s for s in sections if table[s[0]:table.index(0,s[0])]==b'.text')
    return labels,text[5]


p=argparse.ArgumentParser();p.add_argument('target',type=Path);p.add_argument('assembled',type=Path);a=p.parse_args()
blob=bytearray(a.target.read_bytes());old,_=inspect(blob);new,length=inspect(a.assembled.read_bytes())
assert new.keys()<=old.keys()and old.keys()-new.keys()<=set(['.LBB0_4','.LBB0_5'])
for name,(pos,value)in old.items():
    if name not in new:struct.pack_into('<IIIBBH',blob,pos,0,0,0,0,0,0)
    else:
        struct.pack_into('<I',blob,pos+4,new[name][1])
        if name=='main':struct.pack_into('<I',blob,pos+8,length)
a.target.write_bytes(blob)
print(json.dumps({'removed_stale_debug_labels':sorted(old.keys()-new.keys()),'main_text_bytes':length}))
