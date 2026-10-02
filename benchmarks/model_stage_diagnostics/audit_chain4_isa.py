#!/usr/bin/env python3
"""Extract actual archived chain4 text and count four-slot packets offline."""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
from elftools.elf.elffile import ELFFile
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
base=a.repo/'artifacts/builds/block-fp8-reduce-experiment/reduce-24af322'
library=base/'target-24af322/builds/tpc/libblock_fp8_reduce_experiment.so'
blob=library.read_bytes();elf=ELFFile(io.BytesIO(blob));symbols={s.name:s for s in elf.get_section_by_name('.symtab').iter_symbols()}
start,end=symbols['_binary_reduce_chain4_o_start'],symbols['_binary_reduce_chain4_o_end']
section=elf.get_section(start['st_shndx']);offset=section['sh_offset']+start['st_value']-section['sh_addr'];code=blob[offset:offset+end['st_value']-start['st_value']]
(a.out/'chain4.o').write_bytes(code);text=ELFFile(io.BytesIO(code)).get_section_by_name('.text').data();(a.out/'actual.text').write_bytes(text)
command=['/usr/bin/tpc-llvm-objdump','-d','--triple=tpc','--mcpu=gaudi2','--no-show-raw-insn',str(a.out/'chain4.o')]
completed=subprocess.run(command,capture_output=True,text=True,check=True);(a.out/'actual.dis').write_text(completed.stdout);(a.out/'objdump.log').write_text(completed.stderr)
source=a.repo/'benchmarks/reduce_neumaier_isa/audit.py';spec=importlib.util.spec_from_file_location('slot_audit',source);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
report=audit.audit(a.out/'actual.dis');report.update(groups_per_inner_loop=4,elf_sha256=hashlib.sha256(code).hexdigest(),text_sha256=hashlib.sha256(text).hexdigest(),library_sha256=hashlib.sha256(blob).hexdigest(),command=command)
for path in [library,source,base/'reduce_simulator-v2',a.repo/'benchmarks/model_stage_diagnostics/reduce_simulator.cpp',a.repo/'csrc/tpc/block_fp8_reduce_experiment/reduce_chains.c',a.repo/'csrc/host/block_fp8_reduce_experiment_glue.cpp',base/'target-24af322/builds/tpc/reduce_chain4.s']:
 shutil.copy2(path,a.out/path.name)
report['retained_sources_and_binaries']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file()}
report['scope']='actual archived ELF, four-slot count and conservative architectural liveness; no device cycles, stalls or speedup inferred'
(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({key:report[key] for key in ('inner_loop_packets','groups_per_inner_loop','inner_loop_full_nops','occupied_slots','vector_register_count','vector_local_spill_instructions')},indent=2))
