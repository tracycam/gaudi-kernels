"""Check the captured outer fixture keeps original cache sections and logical edges."""
import argparse,collections,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=[]
for graph in json.loads(a.input.read_text())['graphs']:
    nodes=graph['nodes'];attention=[n for n in nodes if n['guid']=='gk_swa128_window_head_fast_fp32_v0']
    if not attention:continue
    assert len(attention)==1
    physical=[n for n in nodes if not n['is_logical']]
    assert len([n for n in physical if n['guid']=='index_copy_fwd_bf16'])==2
    assert all(n['engine']=='TPC' for n in physical)
    # The fixture's only other physical nodes are its three pointwise producers
    # and pointwise context consumer. Built-in reshape entries are logical.
    assert len(physical)==7
    caches={}
    for width in (192,128):
        ts=[t for t in graph['tensors'] if t['max_shape']==[width,1,1024]]
        assert len(ts)>=2 and all(t['persistent'] for t in ts)
        sections={(tuple(t['user_mem_section_index']),t['offset']) for t in ts}
        assert len(sections)==1,(width,sections)
        caches[str(width)]={'tensor_names':[t['name'] for t in ts],'section_and_offset':list(sections)[0]}
    tmap={t['name']:t for t in graph['tensors']}
    metadata=[tmap[n] for n in attention[0]['input_tensors'][3:6]]
    assert all(t['dtype']=='int32' for t in metadata)
    records.append({'graph':graph['name'],'physical_nodes':len(physical),
                    'logical_reshape_nodes':sum(n['is_logical'] for n in nodes),
                    'physical_guids':[n['guid'] for n in physical],'cache_aliases':caches,
                    'metadata_native_types':[t['dtype'] for t in metadata]})
assert len(records)==2,'expected distinct 2D and 3D candidate capture graphs'
a.output.write_text(json.dumps({'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'status':'PASS_OUTER_GRAPH_STRUCTURE','records':records,'scope':'PostGraph alias/physical-node evidence plus separate runtime cache payload/address checks; not HBM transaction counters'},indent=2)+'\n')
