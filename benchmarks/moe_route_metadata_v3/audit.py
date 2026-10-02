"""Pin deployed ISA to simulated text, count valid writes and explicit bridge."""
import argparse,hashlib,io,json,re
from pathlib import Path
from elftools.elf.elffile import ELFFile
p=argparse.ArgumentParser();p.add_argument('--local-build',type=Path,required=True);p.add_argument('--deployed-build',type=Path,required=True);p.add_argument('--v2-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sha=lambda x:hashlib.sha256(x).hexdigest()
def text(p):return ELFFile(io.BytesIO(p.read_bytes())).get_section_by_name('.text').data()
records={}
for name in ('count','prefix','inverse','row_map'):
    local=a.local_build/(name+'.o');target=a.deployed_build/(name+'.o');s=(a.deployed_build/(name+'.s')).read_text();assert not re.search(r'\b(?:ld_l|st_l)_v\b',s)
    assert text(local)==text(target),(name,'deployed instruction mismatch')
    if name in ('count','inverse'):assert re.search(r'ld_tnsr\s+%V\d+, 0x1,',s)and not re.search(r'ld_g\s+%V',s)
    records[name]=dict(ELF_sha256=sha(target.read_bytes()),text_sha256=sha(text(target)),simulated_text_equal=True,no_VLM_spill=True)
assert text(a.deployed_build/'prefix.o')==text(a.v2_build/'prefix.o')
report=dict(status='PASS_DEPLOYED_V3_ISA',kernels=records,prefix_text_unchanged=True,domain={'T_max':513,'R_max':8,'E_max':384,'C_max':32},
 logical_request_account=dict(T=513,R=8,E=384,C=16,B=616,offset_shape=[384,66],offset_bytes=101376,v2_chunk_count_bytes=99840,
 count_full64_loads=384*64,count_tail_scalar_loads=384*8,inverse_full_vector_loads=4104,inverse_scalar_expert_prefix_offset_loads=3*4104,inverse_integer_writes=4104,row_map_valid_writes=4104,row_map_padding_writes=616*16-4104,row_map_total_writes=616*16),
 scope='Logical requests and exact output ownership, not physical HBM/cache counters or cycles. Prefix receives identical arithmetic; low-level tensors must be actual preceding stage outputs.')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(report['status'])
