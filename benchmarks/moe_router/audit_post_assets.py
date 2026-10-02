"""CPU-only audit of deployed ELF payloads and scheduled instruction sites."""
import argparse
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from elftools.elf.elffile import ELFFile

p = argparse.ArgumentParser()
p.add_argument('--build', type=Path, required=True)
p.add_argument('--reference', type=Path, required=True)
p.add_argument('--vendor', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
sha = lambda b: hashlib.sha256(b).hexdigest()
def section(data, name):
    return ELFFile(io.BytesIO(data)).get_section_by_name(name).data()
library = (a.build / 'libgaudi_router_post8_tpc.so').read_bytes()
elf = ELFFile(io.BytesIO(library))
symbols = elf.get_section_by_name('.symtab')
def embedded(name):
    begin = symbols.get_symbol_by_name('_binary_' + name + '_o_start')[0]
    end = symbols.get_symbol_by_name('_binary_' + name + '_o_end')[0]
    data = elf.get_section(begin['st_shndx'])
    offset = begin['st_value'] - data['sh_addr']
    return data.data()[offset:offset + end['st_value'] - begin['st_value']]
report = {'status': 'STARTED', 'device_used': False,
          'library_sha256': sha(library), 'kernels': {}, 'vendor': {}}
def census(path):
    asm = subprocess.check_output(['tpc-llvm-objdump', '--triple=tpc', '-d', str(path)], text=True)
    path.with_suffix('.audit.objdump').write_text(asm)
    packets = [' '.join(line.split('\t')[1:]).split('//')[0].strip() for line in asm.splitlines()
               if re.match(r'^\s*[0-9a-f]+:', line) and len(line.split('\t')) >= 3]
    # Each compressed half is a scheduled packet, just as each full-width line.
    ops = [op.strip() for packet in packets for op in packet.split(';')]
    vregs = [int(v) for op in ops for v in re.findall(r'\bV(\d+)\b', op)]
    count = lambda pattern: sum(bool(re.search(pattern, op)) for op in ops)
    return {'scheduled_packets_whole_function': len(packets),
            'max_general_vreg': max(vregs, default=-1),
            'ld_g_sites': count(r'^ld_g\b'), 'gen_addr_sites': count(r'^gen_addr\b'),
            'ld_tnsr_sites': count(r'^ld_tnsr'),
            'local_vector_loads': count(r'^ld_l_v\b'),
            'local_vector_stores': count(r'^st_l_v\b'),
            'local_scalar_non_mmio': count(r'^(?:ld_l|st_l)\s+(?!mmio\b)')}
for mode in ('scalar', 'vector'):
    data = (a.build / (mode + '.o')).read_bytes()
    reference = (a.reference / (mode + '.o')).read_bytes()
    assert embedded(mode) == data, 'embedded ELF differs from built file'
    actual_text, reference_text = section(data, '.text'), section(reference, '.text')
    assert actual_text == reference_text, 'remote/local ISA differs: repeat actual ELF SIM'
    record = census(a.build / (mode + '.o'))
    record.update(elf_sha256=sha(data), text_sha256=sha(actual_text),
                  embedded_elf_exact=True, reference_text_exact=True)
    assert 0 <= record['max_general_vreg'] < 40, 'register parse/alias requires review'
    assert record['ld_g_sites'] == (16 if mode == 'scalar' else 0)
    assert record['ld_tnsr_sites'] == (0 if mode == 'scalar' else 7)
    assert not any(record[n] for n in ('local_vector_loads', 'local_vector_stores', 'local_scalar_non_mmio'))
    report['kernels'][mode] = record
for name in ('gather', 'reduce', 'divide', 'multiply'):
    path = a.vendor / (name + '.o')
    report['vendor'][name] = dict(census(path), elf_sha256=sha(path.read_bytes()))
report['status'] = 'EMBEDDED_ELF_AND_ISA_PASS_NOT_DEVICE_QUALIFIED'
a.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report))
