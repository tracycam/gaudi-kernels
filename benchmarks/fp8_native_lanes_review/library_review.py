"""Compare all embedded kernels by actual ELF sections and TPC-fork IR."""
import argparse
import difflib
import hashlib
import io
import json
from pathlib import Path
import subprocess
from elftools.elf.elffile import ELFFile

p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
def sha(b):return hashlib.sha256(b).hexdigest()
def embedded(path,key):
 raw=path.read_bytes();e=ELFFile(io.BytesIO(raw));symbols={s.name:s for s in e.get_section_by_name('.symtab').iter_symbols()};first=symbols['_binary_'+key+'_o_start'];last=symbols['_binary_'+key+'_o_end'];section=e.get_section(first['st_shndx']);offset=section['sh_offset']+first['st_value']-section['sh_addr'];return raw[offset:offset+last['st_value']-first['st_value']]
records={};commands=[]
for key in ['block_quant','block_decode','block_decode_fast','block_reduce','block_finish']:
 raw={};sections={};normalized={};symbols={}
 for mode,path in [('old',a.old),('new',a.new)]:
  blob=embedded(path,key);raw[mode]=blob;target=a.output/(key+'-'+mode+'.elf');target.write_bytes(blob);elf=ELFFile(io.BytesIO(blob));sections[mode]={s.name:s.data()for s in elf.iter_sections()}
  symbols[mode]={s.name:dict(value=s['st_value'],size=s['st_size'],section=s['st_shndx'],type=s['st_info']['type'])for s in elf.get_section_by_name('.symtab').iter_symbols()}
  if key=='block_decode':continue
  bc=a.output/(key+'-'+mode+'.bc');bc.write_bytes(sections[mode]['.llvmbc']);ll=a.output/(key+'-'+mode+'.ll')
  cmd=['/usr/bin/tpc-clang','-S','-emit-llvm','-x','ir','-mcpu=gaudi2','-O0',str(bc),'-o',str(ll)];commands.append(cmd)
  with(a.output/(key+'-'+mode+'-read.log')).open('w')as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
  lines=[]
  for line in ll.read_text().splitlines():
   if line.startswith(('; ModuleID','@llvm.embedded.module','@tpc_compiler','@llvm.compiler.used')):continue
   if line.startswith('source_filename = '):line='source_filename = "csrc/'+line.split('/csrc/',1)[1]
   lines.append(line)
  normalized[mode]='\n'.join(lines)+'\n';(a.output/(key+'-'+mode+'-normalized.ll')).write_text(normalized[mode])
 names=set(sections['old'])|set(sections['new']);different=[n for n in sorted(names)if sections['old'].get(n)!=sections['new'].get(n)]
 oldmeta,newmeta=sections['old']['.tpc_metadata'],sections['new']['.tpc_metadata'];assert len(oldmeta)==len(newmeta)==262
 offsets=[i for i,(x,y)in enumerate(zip(oldmeta,newmeta))if x!=y]
 symbol_differences={name:{mode:symbols[mode].get(name)for mode in ('old','new')}for name in set(symbols['old'])|set(symbols['new'])if symbols['old'].get(name)!=symbols['new'].get(name)}
 if key!='block_decode':
  assert set(different)=={'.llvmbc','.symtab','.tpc_compiler','.tpc_metadata'},different
  assert set(offsets)<=set(range(18,22)),offsets
  assert normalized['old']==normalized['new'],key
  assert set(symbol_differences)<={'llvm.embedded.module','tpc_compiler'},symbol_differences
 records[key]=dict(whole_elf_equal=raw['old']==raw['new'],different_sections=different,
   old_elf_sha256=sha(raw['old']),new_elf_sha256=sha(raw['new']),
   sections={mode:{name:sha(data)for name,data in sec.items()}for mode,sec in sections.items()},
   metadata_changed_offsets=offsets,metadata_hashedISA={mode:int.from_bytes(sec['.tpc_metadata'][18:22],'little')for mode,sec in sections.items()},
   symbol_differences=symbol_differences,normalized_reemitted_IR_equal=None if key=='block_decode'else True)
report=dict(status='PASS_OTHER_FOUR_EXECUTABLE_AND_DATA_SECTIONS',libraries={mode:dict(path=str(path.resolve()),sha256=sha(path.read_bytes()))for mode,path in [('old',a.old),('new',a.new)]},kernels=records,commands=commands,
 scope='Other four .text/.source/KernelInfo/data sections identical. Raw ELF/bitcode/compiler records differ. Metadata changes only compiler hashedISA; TPC-fork re-emitted IR compared after removing path and compiler-owned embedding globals. This is not a claim that entire ELF/metadata bytes match.')
(a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'])
