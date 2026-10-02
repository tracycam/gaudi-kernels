"""Keep .text replacement debug labels accurate; changes no instruction/metadata bytes."""
import argparse,struct
from pathlib import Path

def symbols(blob):
 assert blob[:7]==b'\x7fELF\x01\x01\x01'
 offset=struct.unpack_from('<I',blob,32)[0];size,count=struct.unpack_from('<HH',blob,46)
 sections=[struct.unpack_from('<10I',blob,offset+i*size) for i in range(count)]
 sym=next(s for s in sections if s[1]==2);strings=sections[sym[6]];table=blob[strings[4]:strings[4]+strings[5]];result={}
 for pos in range(sym[4],sym[4]+sym[5],sym[9]):
  name,value,length,info,other,shndx=struct.unpack_from('<IIIBBH',blob,pos)
  label=table[name:table.index(0,name)].decode()
  if label=='main' or label.startswith('.LBB'):result[label]=(pos,value,length)
 return result
p=argparse.ArgumentParser();p.add_argument('target',type=Path);p.add_argument('assembled',type=Path);a=p.parse_args()
blob=bytearray(a.target.read_bytes());new=symbols(a.assembled.read_bytes());old=symbols(blob)
assert new.keys()==old.keys()
for name,(pos,value,length) in old.items():struct.pack_into('<I',blob,pos+4,new[name][1])
a.target.write_bytes(blob)
