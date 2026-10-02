"""Read-only real C32 regression plus explicitly synthetic fault injections."""
import argparse,copy,json
from pathlib import Path
from audit import audit_graph,OLD_META,WIDE_META
p=argparse.ArgumentParser();p.add_argument('--post-graph',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
g=next(g for g in json.loads(a.post_graph.read_text())['graphs']if any(n['guid']=='gk_mxfp4_graph_decode_historical'for n in g['nodes']))
original=audit_graph(g,str(a.post_graph));assert original['placement_pass']
renamed=copy.deepcopy(g)
for n in renamed['nodes']:
 if n['guid']in OLD_META:n['guid']=WIDE_META[OLD_META.index(n['guid'])]
 elif n['guid'].startswith('gk_route_tile_'):n['guid']=n['guid'].replace('gk_route_tile_','gk_route_wide_tile_')
wide=audit_graph(renamed,'synthetic GUID rename only; not a C64 device graph');assert wide['placement_pass']and wide['namespace']=='wide'
fields=('decoder_nodes','compute_nodes','MME_nodes','workspace_bytes','decoded_weight_SRAM_address_union_bytes','all_SRAM_address_union_bytes')
assert all(original[k]==wide[k]for k in fields)
rejected=[]
for kind in('expanded_dram','expanded_persistent','owner_bf16','mme_swap','metadata_missing','metadata_physical_long','mixed_C'):
 changed=copy.deepcopy(renamed);ts={t['name']:t for t in changed['tensors']};ns=changed['nodes'];dec=next(n for n in ns if n['guid']=='gk_mxfp4_graph_decode_historical')
 if kind=='expanded_dram':ts[dec['output_tensors'][0]]['allocation']='DRAM'
 if kind=='expanded_persistent':ts[dec['output_tensors'][0]]['persistent']=True
 if kind=='owner_bf16':ts[dec['input_tensors'][0]]['dtype']='bf16'
 if kind=='mme_swap':
  n=next(n for n in ns if n['guid']=='batch_gemm');n['input_tensors'][:2]=reversed(n['input_tensors'][:2])
 if kind=='metadata_missing':ns[:]=[n for n in ns if n['guid']!=WIDE_META[0]]
 if kind=='metadata_physical_long':
  n=next(n for n in ns if n['guid']==WIDE_META[0]);ts[n['output_tensors'][0]]['dtype']='int64'
 if kind=='mixed_C':
  n=next(n for n in ns if n['guid']=='gk_route_wide_tile_gather_bf16_v1');n['params'][4]=64
 r=audit_graph(changed,'synthetic fault');assert not r['placement_pass'],kind;rejected.append(dict(fault=kind,issues=r['issues']))
report=dict(status='PASS_OLD_C32_AUDIT_AND_SYNTHETIC_FAILURES',real_old_C32={k:original[k]for k in fields},wide_GUID_parser_checked=True,synthetic_renaming_not_device_qualification=True,rejected=rejected)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status=report['status'],rejections=len(rejected))))
