"""Static assembly facts; instruction counts are not device timings."""
import argparse
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser();p.add_argument('build',type=Path);p.add_argument('historical',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
rows=[]
for path in sorted(a.build.glob('*.s')):
    code=path.read_text().split('$func_end0:')[0]
    instructions=[line.strip() for line in code.splitlines() if line.strip() and not line.lstrip().startswith(('.', '//', '{', '}', 'main:'))]
    maxima={kind:max([int(x) for x in re.findall(r'%?'+kind+r'(\d+)\b',code)] or [-1]) for kind in ('S','V','I','AD')}
    floating=[line for line in instructions if re.search(r'\b(?:add|sub|mul|mac|madd|convert)\.(?:f32|bf16)\b',line)]
    if path.name.startswith('mx_exact_'):
        assert not floating
    rows.append({'assembly':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'highest_register_index':maxima,
                 'scalar_local_load_mentions':len(re.findall(r'\bld_l\s+(?!mmio)',code)),
                 'scalar_local_store_mentions':len(re.findall(r'\bst_l\s+(?!mmio)',code)),
                 'floating_arithmetic_sites':len(floating),'integer_global_stores':len(re.findall(r'\bst_g\s',code)),
                 'note':'SLM holds explicit exact state and compiler bookkeeping; not a spill-free or cycle claim'})
original=a.historical.read_text().split('.LBB0_3:')[1].split('.LBB0_6:')[0]
conditional=(a.build/'mx_dispatch_history.s').read_text();body=conditional.split('.LBB0_3:')[1].split('.LBB0_6:')[0]
assert original==body
safe=(a.build/'mx_dispatch_finish_bias.s').read_text().split('.LBB0_86:')[1].split('.LBB0_90:')[0]
assert 'loop 0, 512, 64' in safe and 'add.f32' in safe and 'st_tnsr' in safe
assert not re.search(r'\b(?:ld_l|st_l|st_g)\b',safe)
result={'status':'static_ISA_only_no_device_validation','all_static_checks_pass':True,'historical_mac_body_unchanged':True,
        'historical_body_sha256':hashlib.sha256(body.encode()).hexdigest(),
        'safe_finish':{'labels':'LBB0_86 through LBB0_89','columns_per_iteration':64,'iterations':8,'scalar_accumulator_or_SLM_accesses_in_branch':0,'common_bookkeeping_slots':['0x10c','0x110'],'whole_kernel_slm_free':False},'kernels':rows}
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'kernels':len(rows),'all_static_checks_pass':True}))
