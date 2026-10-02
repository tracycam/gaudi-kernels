"""Instruction placement and unchanged-kernel audit; no predicted device speed."""
import argparse,hashlib,io,json,re
from pathlib import Path
from elftools.elf.elffile import ELFFile
p=argparse.ArgumentParser();p.add_argument('--builds',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--old-core',type=Path);p.add_argument('--old-tiles',type=Path);a=p.parse_args()
sha=lambda d:hashlib.sha256(d).hexdigest()
def text(data):return ELFFile(io.BytesIO(data)).get_section_by_name('.text').data()
def embedded(path,name):
    elf=ELFFile(io.BytesIO(path.read_bytes()));tab=elf.get_section_by_name('.symtab')or elf.get_section_by_name('.dynsym');s=tab.get_symbol_by_name('_binary_'+name+'_o_start')[0];e=tab.get_symbol_by_name('_binary_'+name+'_o_end')[0];section=elf.get_section(s['st_shndx']);lo=s['st_value']-section['sh_addr'];return section.data()[lo:lo+e['st_value']-s['st_value']]
report=dict(status='PASS_STATIC_HOIST_PLACEMENT',kernels={},same_unmodified_text=[],baseline_matches_deployed={},scope='Actual .text and compiler basic-block placement only; no physical memory/cycles claim.')
for kind,names in [('core',('graph_count','graph_plan','graph_gather','graph_decode','graph_gate_rows','graph_combine','graph_literal_gate')),('tiles',('gather','gate','combine'))]:
    for name in names:
        paths={v:a.builds/(kind+'-'+v)/(name+'.o')for v in ('control','hoist')};data={v:p.read_bytes()for v,p in paths.items()};texts={v:text(d)for v,d in data.items()};changed=name in ('graph_decode','gather')
        if not changed:assert texts['control']==texts['hoist'];report['same_unmodified_text'].append(name)
        for v in ('control','hoist'):
            src=(paths[v].with_suffix('.s')).read_text();assert not re.search(r'\b(?:ld_l|st_l)_v\b',src)
        report['kernels'][name]={v:dict(elf_sha256=sha(data[v]),text_sha256=sha(texts[v]))for v in data}
        old=a.old_core if kind=='core'else a.old_tiles
        if old is not None:assert text(embedded(old,name))==texts['control'],name;report['baseline_matches_deployed'][name]=True
old=(a.builds/'core-control/graph_decode.s').read_text();new=(a.builds/'core-hoist/graph_decode.s').read_text();pattern=r'ld_l mmio[^\n]*0x158'
assert len(re.findall(pattern,old))==4 and len(re.findall(pattern,new))==1
assert new.index('0x158')<new.index('.LBB0_2:')
for first,last in [('7','10'),('13','16')]:
    loop=new[new.index('.LBB0_'+first+':'):new.index('.LBB0_'+last+':')];assert 'mmio'not in loop and 'mul.i32'not in loop
oldg=(a.builds/'tiles-control/gather.s').read_text();newg=(a.builds/'tiles-hoist/gather.s').read_text()
assert 'udiv' in oldg[oldg.index('.LBB0_4:'):oldg.index('.LBB0_7:')]
assert 'udiv'not in newg[newg.index('.LBB0_15:'):newg.index('.LBB0_18:')]
assert newg.index('udiv')<newg.index('.LBB0_15:')
report.update(K_dimension_MMIO_sites={'old':4,'new':1,'new_outside_all_loops':True},decoder_K_loops_no_MMIO_or_source_multiply=True,gather_division_outside_block_loop=True,gather_power_of_two_path='runtime R1/2/4/8 shift or identity, R3/5/6/7 signed quotient fallback; actual ISA all R1..8 gate')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'])
