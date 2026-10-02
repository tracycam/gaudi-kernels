#!/usr/bin/env python3
"""Verify/extract bound reducer ELFs from sealed libraries without loading them."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from elftools.elf.elffile import ELFFile
from gaudi_kernels.fp32_artifact_binding import _CORE, _REDUCERS

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,required=True,help='sealed 70-c source directory')
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False)
paths={
 'GK_BLOCK_FP8_TPC_LIBRARY':'production-runtime/libraries/libgaudi_block_fp8_framework_tpc.so',
 'GK_BLOCK_FP8_TORCH_LIBRARY':'production-runtime/libraries/gaudi_block_fp8_torch.so',
 'GK_BLOCK_REDUCE_TPC_LIBRARY':'production-runtime/libraries/libblock_fp8_reduce_experiment.so',
 'GK_BLOCK_REDUCE_TORCH_LIBRARY':'production-runtime/libraries/block_fp8_reduce_experiment.so',
 'GK_BLOCK_REDUCE_ISA_TPC_LIBRARY':'production-runtime/qkv-neumaier-isa/tpc.so',
 'GK_BLOCK_REDUCE_ISA_TORCH_LIBRARY':'production-runtime/qkv-neumaier-isa/torch.so'}
entries=[('sequential','GK_BLOCK_FP8_TPC_LIBRARY','_binary_block_reduce_o'),
         ('neumaier_fp32','GK_BLOCK_REDUCE_TPC_LIBRARY','_binary_reduce_neumaier_o'),
         ('neumaier_isa_fp32','GK_BLOCK_REDUCE_ISA_TPC_LIBRARY','_binary_neumaier_handschedule_o')]
records=[]
for reduction,key,symbol in entries:
 entry=_REDUCERS[reduction];out=args.out/reduction;out.mkdir()
 libraries={}
 for name,digest in {**_CORE,**entry['libraries']}.items():
  origin=args.source/paths[name];blob=origin.read_bytes()
  assert hashlib.sha256(blob).hexdigest()==digest
  target=out/(name+'.so');target.write_bytes(blob)
  libraries[name]=dict(source=paths[name],sha256=digest)
 blob=(args.source/paths[key]).read_bytes();elf=ELFFile(io.BytesIO(blob))
 symbols={s.name:s for s in elf.get_section_by_name('.symtab').iter_symbols()}
 start,end=symbols[symbol+'_start'],symbols[symbol+'_end']
 section=elf.get_section(start['st_shndx']);offset=section['sh_offset']+start['st_value']-section['sh_addr']
 code=blob[offset:offset+end['st_value']-start['st_value']]
 code_sha=hashlib.sha256(code).hexdigest();assert code_sha==entry['elf_sha256']
 inner=ELFFile(io.BytesIO(code));text=inner.get_section_by_name('.text').data()
 text_sha=hashlib.sha256(text).hexdigest();assert text_sha==entry['text_sha256']
 (out/'reducer.elf').write_bytes(code);(out/'actual.text').write_bytes(text)
 command=['/usr/bin/tpc-llvm-objdump','--triple=tpc','--mcpu=gaudi2','-d',str(out/'reducer.elf')]
 result=subprocess.run(command,capture_output=True,text=True);(out/'objdump.log').write_text(result.stderr)
 (out/'actual.disasm').write_text(result.stdout);result.check_returncode()
 records.append(dict(reduction=reduction,libraries=libraries,elf_sha256=code_sha,text_sha256=text_sha,command=command))
(args.out/'result.json').write_text(json.dumps(dict(status='PASS_OFFLINE_LIBRARY_ELF_BINDING',records=records,
 scope='exact embedded ELF/text identity; no runtime recipe extraction, device acquisition or new performance qualification'),indent=2)+'\n')
print('verified',len(records),'exact reducer ELF/text bindings')
