"""Audit actual post-compile allocation and declared disjoint TPC read ranges.

Tensor descriptors bind the entire packed expert to each decode node. The
kernel's block-offset parameter and glue access mapping restrict actual logical
reads to its N tile. Do not count each descriptor as a full physical HBM read,
or present this static access audit as a physical bus transaction measurement.
"""
import argparse, collections, json, math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
plan=next(json.loads(line) for line in (a.case/'run.log').read_text().splitlines() if line.startswith('{') and json.loads(line).get('stage')=='plan')
graphs=json.loads((a.case/'post_graph.json').read_text())['graphs'];assert len(graphs)==1
g=graphs[0];tensors={t['name']:t for t in g['tensors']}
decoder_guids={'gk_mxfp4_decode_bf16_v1','gk_mxfp4_decode_ordered_bf16_v1'}
decodes=[n for n in g['nodes'] if n['guid'] in decoder_guids]
mmes=[n for n in g['nodes'] if n['engine']=='MME']
assert len(decodes)==plan['decode_nodes'] and len(mmes)==plan['mme_nodes'],'compiler duplicated or removed expected nodes'
ranges=collections.defaultdict(list);details=[];scratch=[];mme_details=[]
for node in decodes:
    w,s,lut=(tensors[name] for name in node['input_tensors'][:3])
    out=tensors[node['output_tensors'][0]]
    assert w['dtype']==s['dtype']=='uint8' and w['persistent'] and s['persistent']
    assert w['allocation']==s['allocation']=='DRAM'
    assert out['dtype']=='bf16' and out['allocation']=='SRAM' and out['rmw_section'] and not out['persistent']
    offset=int.from_bytes(bytes(node['params']),'little',signed=True)
    n,k=out['max_shape'];blocks=(n+255)//256
    assert k==plan['K']
    for tensor in [w,s]:
        stride=tensor['strides'][2];region=[offset*stride,(offset+blocks)*stride]
        ranges[tensor['name']].append(region)
    scratch.append({'name':out['name'],'allocation':out['allocation'],'persistent':out['persistent'],
                    'bytes':math.prod(out['max_shape'])*2,'address':out['offset']})
    details.append({'node':node['name'],'block_offset':offset,'decoded_N':n,'K':k,
                    'packed_interval':ranges[w['name']][-1],'scale_interval':ranges[s['name']][-1],
                    'physical_hbm_bytes':None,'tpc_working_engines':node.get('tpc_working_engines')})
for name,regions in ranges.items():
    cursor=0
    for start,end in sorted(regions):
        assert start==cursor,(name,regions)
        cursor=end
    assert cursor==math.prod(tensors[name]['max_shape']),name
for node in mmes:
    x,w=(tensors[name] for name in node['input_tensors']);y=tensors[node['output_tensors'][0]]
    assert x['dtype']=='bf16' and w['dtype']=='bf16' and w['allocation']=='SRAM' and y['dtype']=='float32'
    mme_details.append({'node':node['name'],'activation_shape':x['max_shape'],'weight_shape':w['max_shape'],
                        'output_shape':y['max_shape'],'strategy':node.get('mme_node_strategy'),
                        'compiler_estimated_compute_utilization':node.get('mme_compute_utilization'),
                        'compiler_expected_compute_cycles':node.get('mme_expected_compute_cycles'),'rollups':node.get('rollups')})
decoded_names={n['output_tensors'][0] for n in decodes}
for node in g['nodes']:
    if node['engine']!='MME' and node['guid'] not in decoder_guids:
        assert not decoded_names.intersection(node['input_tensors']),'unexpected consumer/copy of expanded weights'
record={'case':a.case.name,'pass':bool(decodes),'all_decoded_weights_nonpersistent_sram':True,
        'packed_and_scales_persistent_hbm':True,'no_compiler_decode_duplication':True,
        'logical_declared_intervals_cover_once':True,'logical_weight_read_bytes':sum(math.prod(tensors[name]['max_shape']) for name in ranges),
        'all_scratch_reuses_one_address':len({s['address'] for s in scratch})==1,
        'max_scratch_bytes':max(s['bytes'] for s in scratch),'scratch':scratch,'reads':details,'mme':mme_details,
        'physical_hbm_bytes':None,'scope':'compiler placement and static kernel access ranges, not bus counters or dynamic MME routing'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['scratch','reads','mme']}))
